"""Full GPU Embryo simulation driver (Phase 3, Pass C).

``EmbryoModel`` wires the reusable GPU engine + every steppable into one driver,
the GPU-resident equivalent of the CC3D Embryo project (Embryo.xml + all
``EmbryoSteppables.py`` steppables). It reuses -- never duplicates -- the engine:

  engine.GPUEngine            CPM Volume+Contact sweep (+ FPP spring energy seam)
  engine.FPPLinks             device link CSR (tissue + lamellipodia + substrate)
  engine.LamellipodiaSteppable  lamellipodia link + cohesotaxis + Poisson turnover
  embryo.TissueLinkSteppable    tissue links + intercalation turnover (NEW)
  embryo.PassiveSubstrateSteppable  passive<->substrate adhesion links (NEW)
  embryo.ClosureSteppable       windowed floor-free-area / closure observable (NEW)

NOTE on the FPP link law (verified against the vendored CC3D source, read-only):
Embryo.xml nests ``<LinkConstituentLaw><Formula>Lambda*Length</Formula>`` inside the
Leading-Substrate ``<Parameters>`` block. But ``FocalPointPlasticityPlugin::init``
reads ``LinkConstituentLaw`` via ``_xmlData->getFirstElement`` over the plugin's
DIRECT children only (CC3DXMLElement.cpp:313, non-recursive), so a grandchild is
never found -> the plugin falls back to ``elasticLinkConstituentLaw`` =
``lambda*(L-target)^2`` for ALL links. The nested custom formula is dead config.
Likewise the per-pair ``ActivationEnergy=-50`` never enters the sweep: auto-junction
creation is gated on ``>= maxNumberOfJunctions`` which defaults to 0 (no
MaxNumberOfJunctions in the XML), so ``tryAddingNewJunction`` always early-outs.
=> The engine's quadratic ``fpp_delta_cell`` is the faithful Embryo FPP law.

NOTE on the Contact NeighborOrder. Embryo.xml uses ``Contact NeighborOrder=4``. The
GPU sweep's 8-color (2x2x2) flip checkerboard is proven race-safe only for neighbor
reads within the Moore (order<=3) shell -- two same-color voxels can sit at axial
distance 2 (an order-4 contact neighbor), so reading order-4 contact during the
parallel sweep needs a 27-color scheme (unimplemented; documented next step). This
driver therefore runs ``contact_neighbor_order=3`` (the Phase-2-validated config).
Phase-3 validation quantified the order-3-vs-order-4 gap (mesenchyme cells are ~11%
more compact at order 4) AND showed the GPU reproduces order-4 contact within MC
noise when configured for it (contact order is independent of the flip-coloring
cap) -- see ``test_order4_physics_gap_and_gpu_reproduction``.
"""

from __future__ import annotations

import math

import numpy as np

from engine import EngineConfig, GPUEngine
from engine.state import state_from_id_lattice
from engine.fpp import FPPLinks
from engine.steppables import SteppableManager, LamellipodiaSteppable
from engine.geometry import (
    _LatticeBuilder, create_hollow_sphere, mesendoderm_sphere_circumference,
    LEADING, PASSIVE, RING, SUBSTRATE, MEDIUM,
)

from .params import EmbryoParams, DEFAULT
from .steppables import TissueLinkSteppable, PassiveSubstrateSteppable, ClosureSteppable


# ---------------------------------------------------------------------------
# Initial-condition builders
# ---------------------------------------------------------------------------
def _embryo_cfg(cube_size, temperature, seed):
    n_types = 5
    target_volume = np.array([0.0, 125.0, 125.0, 125.0, 1.0])
    lambda_volume = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    contact = np.full((n_types, n_types), 10.0)
    return EngineConfig(
        Lx=cube_size, Ly=cube_size, Lz=cube_size,
        temperature=temperature, n_types=n_types,
        target_volume=target_volume, lambda_volume=lambda_volume, contact=contact,
        frozen=np.array([0, SUBSTRATE], dtype=np.int32),
        contact_neighbor_order=3, flip_neighbor_order=3, tracker_neighbor_order=1,
        seed=seed,
    )


def build_scaled_embryo(cube_size: int = 100, temperature: float = 10.0, seed: int = 12345):
    """The full Embryo IC (``engine.geometry.build_embryo_start`` ring radii) scaled
    to ``cube_size`` (100 = the exact reference geometry). Ring radii/levels scale by
    cube_size/100. Returns (EngineState, info). At cube_size=100 this is voxel-exact
    vs EmbryoSteppables (60 Leading / 618 Passive / 62333 Substrate)."""
    b = _LatticeBuilder(cube_size)
    sc = cube_size / 100.0

    def rs(v):  # ring radius / z-level scaler
        return v * sc

    radius = 50 * sc
    tolerance = 1
    voxel_spacing = 1
    region_spacing = 0

    ring_counts = {}
    ring_counts["leading"] = mesendoderm_sphere_circumference(
        b, cube_size, rs(58), int(rs(50)), voxel_spacing, region_spacing, LEADING)
    passive_blocks = 0
    for r, z in [(52, 48), (58, 45), (52, 45), (46, 45), (57, 40), (52, 40),
                 (46, 40), (40, 40), (56, 35), (51, 35), (46, 35), (40, 35)]:
        passive_blocks += mesendoderm_sphere_circumference(
            b, cube_size, rs(r), int(rs(z)), voxel_spacing, region_spacing, PASSIVE)
    ring_counts["passive"] = passive_blocks

    n_before = b._next_id
    create_hollow_sphere(b, cube_size, radius, tolerance, open_cap=None)
    ring_counts["substrate_voxels"] = b._next_id - n_before

    cell_type = b.finalize_types()
    cfg = _embryo_cfg(cube_size, temperature, seed)
    state = state_from_id_lattice(cfg, b.ids, cell_type)
    type_counts = {t: int(np.sum(cell_type == t)) for t in range(5)}
    return state, {"ring_counts": ring_counts, "type_counts": type_counts,
                   "n_cells": state.n_cells}


def build_closure_scene(L: int = 24, temperature: float = 10.0, seed: int = 1,
                        n_leaders: int = 8, ring_radius=None, n_passive_ring: int = 8):
    """A reduced wound-closure scene capturing the Embryo physics essentials at a
    scale the NumPy CPU reference can run in ~seconds (the in-gate statistical gate,
    like Phase 2's 24^3 reduced gate):

      * a flat Substrate floor of 1-voxel frozen cells at z=0,1 (the ectoderm shell
        flattened locally -- leaders crawl ON it);
      * a ring of LEADING cell blocks around a central opening at z=2..3 (the
        closing leading edge);
      * a ring of PASSIVE cell blocks just inside the leaders (the follower
        mesendoderm).

    The leaders' exposed (medium-facing) boundary pixels sit beside the floor, so
    the cohesotaxis lamellipodia pipeline finds upward substrate targets; tissue
    links connect adjacent leaders/passives (intercalation); the floor free area
    (opening) closes as the ring contracts.

    Returns (EngineState, info) with the leader/passive id lists + the floor window.
    """
    ids = np.zeros((L, L, L), dtype=np.int32)
    cell_type = [0]

    def new_cell(t):
        cell_type.append(int(t))
        return len(cell_type) - 1

    # substrate floor (1-voxel frozen cells), single layer at z=0 (cells crawl ON it)
    for y in range(L):
        for x in range(L):
            ids[0, y, x] = new_cell(SUBSTRATE)

    cx = cy = L // 2
    if ring_radius is None:
        ring_radius = max(4, L // 4)

    # leader/passive cells sit on the floor at z=1,2 forming a ring around a central
    # opening; as the ring contracts the opening (exposed floor) closes.
    lead_ids = []
    for i in range(n_leaders):
        a = 2.0 * math.pi * i / n_leaders
        px = int(round(cx + ring_radius * math.cos(a)))
        py = int(round(cy + ring_radius * math.sin(a)))
        cid = new_cell(LEADING)
        x0, x1 = max(1, px - 1), min(L - 1, px + 2)
        y0, y1 = max(1, py - 1), min(L - 1, py + 2)
        ids[1:3, y0:y1, x0:x1] = cid
        lead_ids.append(cid)

    # passive ring just inside the leaders
    passive_ids = []
    pr = max(2, ring_radius - 3)
    for i in range(n_passive_ring):
        a = 2.0 * math.pi * i / n_passive_ring + (math.pi / n_passive_ring)
        px = int(round(cx + pr * math.cos(a)))
        py = int(round(cy + pr * math.sin(a)))
        cid = new_cell(PASSIVE)
        x0, x1 = max(1, px - 1), min(L - 1, px + 2)
        y0, y1 = max(1, py - 1), min(L - 1, py + 2)
        # only fill voxels still Medium so passive doesn't clobber leaders
        block = ids[1:3, y0:y1, x0:x1]
        block[block == 0] = cid
        ids[1:3, y0:y1, x0:x1] = block
        passive_ids.append(cid)

    cfg = _embryo_cfg(L, temperature, seed)
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, temperature=temperature, n_types=5,
        target_volume=np.array([0.0, 18.0, 18.0, 18.0, 1.0]),  # leader/passive ~ block vol
        lambda_volume=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
        contact=np.full((5, 5), 10.0),
        frozen=np.array([0, SUBSTRATE], dtype=np.int32),
        contact_neighbor_order=3, flip_neighbor_order=3, tracker_neighbor_order=1,
        seed=seed,
    )
    state = state_from_id_lattice(cfg, ids, np.asarray(cell_type, np.int32))
    # floor window covering the opening (centered)
    w = (cx - ring_radius - 2, cx + ring_radius + 3, cy - ring_radius - 2, cy + ring_radius + 3)
    info = {
        "leaders": np.array(lead_ids, dtype=np.int64),
        "passives": np.array(passive_ids, dtype=np.int64),
        "window": w,
        "n_cells": state.n_cells,
        "type_counts": {t: int(np.sum(np.asarray(cell_type) == t)) for t in range(5)},
    }
    return state, info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class EmbryoModel:
    """Complete GPU Embryo simulation: engine + FPP links + all steppables.

    ``enable`` selects which steppables run (default: all). The driver owns one
    ``FPPLinks`` inventory shared by every link kind (tissue / lamellipodia /
    substrate), exactly like CC3D's single FocalPointPlasticity plugin.
    """

    def __init__(self, state, params: EmbryoParams = DEFAULT, device: str = "cuda:0",
                 enable=("lamellipodia", "tissue", "passive_substrate", "closure"),
                 closure_window=None, csr_method: str = "device",
                 link_backend: str = "device"):
        self.state = state
        self.params = params
        self.csr_method = csr_method  # "device" (on-GPU compaction) or "host"
        # Phase 7: "device" = port the link create/delete loops to device kernels that
        # read the resident neighbor-CSR handles + device inventory directly and DROP
        # the per-MCS host CSR copyback; "host" = the Phase-3 host loops (kept behind
        # this flag for the differential link-set tests).
        self.link_backend = link_backend
        self.engine = GPUEngine(state, device=device)
        self.links = FPPLinks(self.engine,
                              target_length_default=params.tissue_target,
                              lambda_default=params.tissue_lambda,
                              max_length_default=params.tissue_max)
        self.engine.attach_fpp(self.links)
        self.manager = SteppableManager(self.engine)
        self.enable = set(enable)
        self.steppables = {}

        lb = self.link_backend
        if "lamellipodia" in self.enable:
            s = LamellipodiaSteppable(
                self.engine, self.links, leading_type=LEADING, substrate_type=SUBSTRATE,
                passive_type=PASSIVE, lamellipodia_distance=params.lamellipodia_distance,
                link_backend=lb)
            self.manager.register(s)
            self.steppables["lamellipodia"] = s
        if "tissue" in self.enable:
            # Leading uses MaxNeighborNum+1 cap; Passive uses MaxNeighborNum. CC3D
            # has both; here both Leading+Passive get tissue links. Use the Leading
            # (+1) cap for leaders and the Passive cap for passives via two managers.
            # In device mode ONLY the Leading manager runs the tissue Poisson delete
            # over the shared inventory (one draw per tissue link per MCS); Passive
            # only recreates (see TissueLinkSteppable.device_poisson_delete).
            s_lead = TissueLinkSteppable(
                self.engine, self.links, cell_types=(LEADING,), params=params,
                link_cap_offset=1, substrate_type=SUBSTRATE, link_backend=lb,
                device_poisson_delete=True)
            s_pas = TissueLinkSteppable(
                self.engine, self.links, cell_types=(PASSIVE,), params=params,
                link_cap_offset=0, substrate_type=SUBSTRATE, link_backend=lb,
                device_poisson_delete=False)
            self.manager.register(s_lead)
            self.manager.register(s_pas)
            self.steppables["tissue_leading"] = s_lead
            self.steppables["tissue_passive"] = s_pas
        if "passive_substrate" in self.enable:
            s = PassiveSubstrateSteppable(
                self.engine, self.links, passive_type=PASSIVE, substrate_type=SUBSTRATE,
                params=params, link_backend=lb)
            self.manager.register(s)
            self.steppables["passive_substrate"] = s
        if "closure" in self.enable:
            s = ClosureSteppable(self.engine, substrate_type=SUBSTRATE, window=closure_window)
            self.manager.register(s)
            self.steppables["closure"] = s

    def _inject_shared_csr(self):
        """Build the order-1 neighbor CSR ONCE per MCS and make it available to the
        steppables that need the neighbor relation.

        DEVICE backend (Phase 7): publish ONLY the resident device CSR handles
        (``engine.neighbor_csr_*_dev``) -- the steppable device kernels read those
        directly, so the whole contact graph is NEVER copied back to the host on the
        hot path (the copyback this phase removes). HOST backend: build + copy the host
        CSR arrays (the Phase-3 path) and share them, for the differential tests."""
        if self.link_backend == "device":
            self.engine.publish_neighbor_csr_device(order=1)
            return
        csr = self.engine.neighbor_contact_csr(order=1, method=self.csr_method)
        for key in ("tissue_leading", "tissue_passive", "passive_substrate"):
            s = self.steppables.get(key)
            if s is not None:
                s.shared_csr = csr

    def start(self):
        self._inject_shared_csr()
        self.manager.start()
        return self

    def run(self, n_mcs: int, mcs_offset: int = 0):
        import warp as wp
        for m in range(n_mcs):
            mcs = mcs_offset + m
            self.engine.step_mcs(mcs)
            self._inject_shared_csr()  # one CSR build per MCS, shared
            for s in self.manager.steppables:
                if mcs % s.frequency == 0:
                    s.step(mcs)
        wp.synchronize()
        return self

    # observables ------------------------------------------------------------
    def closure_history(self):
        s = self.steppables.get("closure")
        return s.history if s is not None else []

    def active_link_lengths(self):
        return self.links.active_link_lengths()

    def num_active_links(self):
        return self.links.num_active()
