"""Remaining Embryo steppable ports (Phase 3, Pass C).

Ports the FocalPointPlasticity TISSUE-link dynamics + INTERCALATION turnover from
``LeadingEdgeSteppable`` / ``PassiveSteppable`` and the passive-cell-substrate
links from ``PassiveSteppable``, plus the windowed floor-free-area / closure
observable from ``SubstrateSteppable`` -- the parts Passes A/B did not cover.

These reuse the existing engine seams (do NOT duplicate them):
  * ``engine.fpp.FPPLinks`` create/delete/rebuild (the per-MCS topology boundary);
  * ``engine.GPUEngine.neighbor_contact_csr(order=1)`` for the order-1 common-surface
    neighbor relation == CC3D ``get_cell_neighbor_data_list`` (NeighborTracker
    NeighborOrder=1, verified in NeighborTrackerPlugin.cpp);
  * ``engine.cohesotaxis`` Poisson turnover kernel for the on-device Bernoulli
    delete decisions (keyed Philox, the sanctioned reproducible GPU draw).

CC3D semantics preserved (read-only ref ``EmbryoSteppables.py``):
  * Tissue links: ``new_fpp_link(a,b,TissueLambda=600,TTarget=5,Tmax=10)`` between a
    cell and each non-Substrate order-1 neighbor lacking a link; capped so a cell
    holds ``< MaxNeighborNum(+1)`` links. Each MCS every tissue link is deleted with
    probability ``1-exp(-TissueRate)`` (the intercalation neighbor-exchange driver),
    then missing neighbor links are recreated.
  * Passive-substrate links: a passive cell next to Substrate with no substrate link
    makes one to a random Substrate neighbor
    ``new_fpp_link(...,SLinkLambda=10,SLink_TargetDist=1,SLinkMaxDist=5)``; deleted
    each MCS with probability ``1-exp(-SubLinkRate)``.

RNG note (carry-forward, sanctioned): per-MCS Bernoulli turnover uses keyed Philox
(``poisson_turnover_kernel``) rather than CC3D's single shared sequential stream.
Reproducibility is per (mcs, cell, stream, seed); statistical fidelity (the rate)
is what the gate checks. Distinct ``stream`` offsets keep tissue / substrate /
lamellipodia turnover independent.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from engine.engine import GPUEngine
from engine.steppables import GPUSteppable, CellDict
from engine import cohesotaxis as CT

from .params import EmbryoParams, DEFAULT

wp.init()


# Stream offsets so independent turnover processes draw from disjoint Philox keys.
_STREAM_TISSUE = 1
_STREAM_SUBLINK = 2
_STREAM_LAMELLAE = 0  # LamellipodiaSteppable already uses the base stream (offset 0)


@wp.kernel
def _poisson_turnover_stream_kernel(
    n: wp.int32,
    prob: wp.float32,
    mcs: wp.int32,
    base_seed: wp.int32,
    stream: wp.int32,
    decision: wp.array(dtype=wp.int32),
):
    """Per-item Bernoulli(prob) keyed by (mcs, item, stream, base_seed). 1 = act."""
    c = wp.tid()
    if c >= n:
        return
    seed = base_seed + mcs * 131072 + c * 16384 + stream * 257
    state = wp.rand_init(seed, c + stream * 7919)
    u = wp.randf(state)
    if u < prob:
        decision[c] = 1
    else:
        decision[c] = 0


def _bernoulli(n, prob, mcs, base_seed, stream, device):
    dec = wp.zeros(int(n), dtype=wp.int32, device=device)
    wp.launch(_poisson_turnover_stream_kernel,
              dim=int(n),
              inputs=[int(n), float(prob), int(mcs), int(base_seed), int(stream), dec],
              device=device)
    wp.synchronize()
    return dec.numpy()


def neighbor_adjacency(engine: GPUEngine, exclude_types=(), csr=None):
    """Per-cell list of order-1 common-surface neighbor cell ids (excluding Medium
    and any ``exclude_types``), from the engine's scalable hashed neighbor CSR.

    This is the GPU analogue of iterating ``get_cell_neighbor_data_list(cell)``.
    Returns a list ``adj`` of length n_cells+1; ``adj[c]`` is an int array of
    neighbor ids. ``csr`` (indptr,indices,data) may be passed to reuse a
    once-per-MCS build (the EmbryoModel shares one across all tissue steppables).
    """
    if csr is None:
        indptr, indices, _data = engine.neighbor_contact_csr(order=1)
    else:
        indptr, indices, _data = csr
    cell_type = engine.cell_type.numpy()
    n1 = engine.n_cells + 1
    excl = set(int(t) for t in exclude_types)
    adj = [np.zeros(0, dtype=np.int64) for _ in range(n1)]
    for c in range(1, n1):
        lo, hi = int(indptr[c]), int(indptr[c + 1])
        nb = indices[lo:hi]
        nb = nb[nb != 0]  # drop Medium
        if excl:
            nb = nb[~np.isin(cell_type[nb], list(excl))]
        adj[c] = nb.astype(np.int64)
    return adj


class TissueLinkSteppable(GPUSteppable):
    """Port of the TISSUE-link + INTERCALATION dynamics shared by
    ``LeadingEdgeSteppable`` and ``PassiveSteppable``.

    For the cells of ``cell_types`` (Leading and Passive in the Embryo):
      start(): create a tissue FPP link to every non-Substrate order-1 neighbor
        lacking one (lambda=TissueLambda, target=TTarget, max=Tmax).
      step(mcs): delete each existing tissue link with prob 1-exp(-TissueRate)
        (intercalation), then recreate links to neighbors while the cell holds
        fewer than ``max_links`` links.

    Links are mutated on the host ``FPPLinks`` topology at the per-MCS boundary; the
    engine rebuilds the device CSR once per MCS (never inside the Metropolis loop).
    ``link_cap_offset`` is +1 for Leading (CC3D ``MaxNeighborNum+1``) and 0 for
    Passive (``MaxNeighborNum``), matching the two call sites.
    """

    def __init__(self, engine: GPUEngine, links, cell_types, params: EmbryoParams = DEFAULT,
                 link_cap_offset: int = 1, substrate_type: int = 4, frequency: int = 1):
        super().__init__(engine, frequency)
        self.links = links
        self.p = params
        self.substrate_type = int(substrate_type)
        self.cell_types = set(int(t) for t in cell_types)
        self.max_links = params.max_neighbor_num + int(link_cap_offset)
        ctype = engine.cell_type.numpy()
        self.managed = np.nonzero(np.isin(ctype, list(self.cell_types)))[0].astype(np.int64)
        # set of undirected tissue links we own (so we only Poisson-delete ours, not
        # lamellipodia/substrate links). Stored as frozenset({a,b}).
        self._tissue = set()
        # optional once-per-MCS shared neighbor CSR (set by EmbryoModel)
        self.shared_csr = None

    def _key(self, a, b):
        return (a, b) if a < b else (b, a)

    def _current_link_map(self):
        """Map cell -> set of partners over the WHOLE current FPP inventory (all
        link kinds), used for the per-cell link-count cap + dedup guards."""
        a = self.links._a
        b = self.links._b
        m = {}
        for i in range(a.shape[0]):
            ai, bi = int(a[i]), int(b[i])
            m.setdefault(ai, set()).add(bi)
            m.setdefault(bi, set()).add(ai)
        return m

    def start(self):
        adj = neighbor_adjacency(self.engine, exclude_types=(self.substrate_type,),
                                 csr=self.shared_csr)
        link_map = self._current_link_map()
        for c in self.managed:
            c = int(c)
            for nb in adj[c]:
                nb = int(nb)
                if nb in link_map.get(c, ()):  # get_fpp_link_by_cells(...) is None guard
                    continue
                self.links.create_link(c, nb, lam=self.p.tissue_lambda,
                                       target=self.p.tissue_target, maxlen=self.p.tissue_max)
                self._tissue.add(self._key(c, nb))
                link_map.setdefault(c, set()).add(nb)
                link_map.setdefault(nb, set()).add(c)
        return len(self._tissue)

    def step(self, mcs: int):
        p = self.p
        # --- (1) Poisson-delete each existing tissue link (intercalation) ---
        tissue_list = list(self._tissue)
        if tissue_list:
            dec = _bernoulli(len(tissue_list), p.tissue_delete_prob, mcs,
                             self.engine.base_seed, _STREAM_TISSUE, self.engine.device)
            for i, (a, b) in enumerate(tissue_list):
                if dec[i] == 1:
                    self.links.delete_link(a, b)
                    self._tissue.discard((a, b))

        # --- (2) recreate tissue links to neighbors under the per-cell cap ---
        adj = neighbor_adjacency(self.engine, exclude_types=(self.substrate_type,),
                                 csr=self.shared_csr)
        link_map = self._current_link_map()
        for c in self.managed:
            c = int(c)
            partners = link_map.get(c, set())
            if len(partners) >= self.max_links:
                continue
            for nb in adj[c]:
                nb = int(nb)
                if len(partners) >= self.max_links:
                    break
                if nb in partners:
                    continue
                self.links.create_link(c, nb, lam=p.tissue_lambda,
                                       target=p.tissue_target, maxlen=p.tissue_max)
                self._tissue.add(self._key(c, nb))
                partners.add(nb)
                link_map.setdefault(nb, set()).add(c)
        return len(self._tissue)


class PassiveSubstrateSteppable(GPUSteppable):
    """Port of ``PassiveSteppable`` passive-cell <-> Substrate adhesion links.

    Per passive cell next to the Substrate with no current substrate link, create
    one to a (deterministically chosen) Substrate order-1 neighbor
    (lambda=SLinkLambda=10, target=SLink_TargetDist=1, max=SLinkMaxDist=5); each MCS
    delete it with prob 1-exp(-SubLinkRate). ``cell.dict['link']`` (the linked
    substrate id, 0=none) is mirrored in a per-cell SoA, like CC3D's cell.dict.

    CC3D picks the substrate partner via ``random.choice`` of the neighbor list; on
    the GPU we pick deterministically (smallest neighbor id) -- the *which* substrate
    cell is not a measured observable (all substrate cells are frozen, identical
    1-voxel cells), only that the adhesion link exists at the model rate. This is the
    sanctioned reproducible-GPU substitution (documented).
    """

    def __init__(self, engine: GPUEngine, links, passive_type: int = 2,
                 substrate_type: int = 4, params: EmbryoParams = DEFAULT, frequency: int = 1):
        super().__init__(engine, frequency)
        self.links = links
        self.p = params
        self.passive_type = int(passive_type)
        self.substrate_type = int(substrate_type)
        ctype = engine.cell_type.numpy()
        self.passive = np.nonzero(ctype == self.passive_type)[0].astype(np.int64)
        self.cell_dict.register("sub_link", "int32", 0)  # cell.dict['link'] mirror
        self.shared_csr = None

    def _substrate_neighbor_map(self):
        if self.shared_csr is None:
            indptr, indices, _ = self.engine.neighbor_contact_csr(order=1)
        else:
            indptr, indices, _ = self.shared_csr
        ctype = self.engine.cell_type.numpy()
        out = {}
        for c in self.passive:
            c = int(c)
            lo, hi = int(indptr[c]), int(indptr[c + 1])
            nb = indices[lo:hi]
            nb = nb[(nb != 0)]
            sub = nb[ctype[nb] == self.substrate_type]
            out[c] = sub
        return out

    def start(self):
        if not self.p.if_passive_substrate:
            return 0
        # CC3D PassiveSteppable.start only sets up tissue links + next_to_substrate
        # flags; the substrate link itself is created in step(). Nothing to create
        # here for substrate links.
        return 0

    def step(self, mcs: int):
        if not self.p.if_passive_substrate:
            return 0
        p = self.p
        sl = self.cell_dict.get("sub_link")
        sub_nb = self._substrate_neighbor_map()

        # (a) create a substrate link for passive cells next-to-substrate w/o one
        for c in self.passive:
            c = int(c)
            if sl[c] != 0:
                continue
            sub = sub_nb.get(c, np.zeros(0, dtype=np.int64))
            if sub.size == 0:
                continue
            target = int(sub.min())  # deterministic pick (see class docstring)
            self.links.create_link(c, target, lam=p.slink_lambda,
                                   target=p.slink_target, maxlen=p.slink_max)
            sl[c] = target

        # (b) Poisson-delete existing substrate links
        have = np.nonzero(sl[self.passive] != 0)[0]
        if have.size:
            cells = self.passive[have]
            dec = _bernoulli(cells.size, p.sub_link_delete_prob, mcs,
                             self.engine.base_seed, _STREAM_SUBLINK, self.engine.device)
            for i, c in enumerate(cells):
                c = int(c)
                if dec[i] == 1:
                    self.links.delete_link(c, int(sl[c]))
                    sl[c] = 0

        self.cell_dict.set("sub_link", sl)
        return int(np.count_nonzero(sl[self.passive]))


class ClosureSteppable(GPUSteppable):
    """Port of ``SubstrateSteppable`` floor-free-area / closure observable.

    Faithful to CC3D: a fixed ``CellArray`` of substrate floor cells (those at z=0
    inside the window ``x in [x0,x1), y in [y0,y1)``) is recorded at start; each MCS
    ``FloorFreeArea = sum(cell.volume for c in CellArray if cell_field[xCOM,yCOM,zCOM+1]
    is Medium)`` and ``radius = sqrt(FloorFreeArea/pi)``. ``CloseTime`` is the first
    MCS where ``FloorFreeArea < 25``. The window is scaled to the lattice.

    Computed on device by a per-voxel kernel that, for each window-floor substrate
    cell, tests whether the voxel directly above its COM is Medium -- the exact CC3D
    test ``cell_field[cell.xCOM, cell.yCOM, cell.zCOM+1]``.
    """

    def __init__(self, engine: GPUEngine, substrate_type: int = 4,
                 window=None, frequency: int = 1):
        super().__init__(engine, frequency)
        self.substrate_type = int(substrate_type)
        # CC3D default window: xint=arange(20,70), yint=arange(25,70) on a 100^3 box.
        # Scale to this lattice so the metric is meaningful at reduced scale.
        if window is None:
            sx = engine.Lx / 100.0
            sy = engine.Ly / 100.0
            window = (int(20 * sx), int(70 * sx), int(25 * sy), int(70 * sy))
        self.x0, self.x1, self.y0, self.y1 = window
        # build the fixed CellArray: substrate cells whose z=0 voxel lies in-window.
        ids = engine.get_ids()  # (Lz,Ly,Lx)
        ctype = engine.cell_type.numpy()
        cellset = set()
        z0 = 0
        for y in range(self.y0, min(self.y1, engine.Ly)):
            for x in range(self.x0, min(self.x1, engine.Lx)):
                cid = int(ids[z0, y, x])
                if cid != 0 and ctype[cid] == self.substrate_type:
                    cellset.add(cid)
        self.cell_array = np.array(sorted(cellset), dtype=np.int32)
        self._cell_array_w = wp.array(self.cell_array, dtype=wp.int32, device=engine.device)
        self.history = []          # (mcs, free_area, radius)
        self.close_time = None

    def free_area(self) -> int:
        eng = self.engine
        if self.cell_array.size == 0:
            return 0
        out = wp.zeros(1, dtype=wp.int32, device=eng.device)
        wp.launch(
            _closure_free_area_kernel,
            dim=int(self.cell_array.size),
            inputs=[
                self._cell_array_w, int(self.cell_array.size),
                eng.ids, eng.cell_type, eng.volume, eng.xsum, eng.ysum, eng.zsum,
                eng.Lx, eng.Ly, eng.Lz, out,
            ],
            device=eng.device,
        )
        wp.synchronize()
        return int(out.numpy()[0])

    def step(self, mcs: int):
        fa = self.free_area()
        radius = float(np.sqrt(fa / np.pi)) if fa > 0 else 0.0
        self.history.append((mcs, fa, radius))
        if fa < 25 and self.close_time is None and mcs > 0:
            self.close_time = mcs
        return fa


@wp.kernel
def _closure_free_area_kernel(
    cell_array: wp.array(dtype=wp.int32),
    n_array: wp.int32,
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    out_free: wp.array(dtype=wp.int32),
):
    """For each floor cell in CellArray, if the voxel directly above its COM is
    Medium (id 0), add its volume to the free-area accumulator. One thread per
    CellArray entry. Mirrors SubstrateSteppable: cellAbove = cell_field[xCOM,yCOM,
    zCOM+1]; if not cellAbove: FloorFreeArea += cell.volume."""
    i = wp.tid()
    if i >= n_array:
        return
    cid = cell_array[i]
    v = volume[cid]
    if v <= 0.0:
        return
    cx = wp.int32(wp.float64(xsum[cid]) / wp.float64(v))
    cy = wp.int32(wp.float64(ysum[cid]) / wp.float64(v))
    cz = wp.int32(wp.float64(zsum[cid]) / wp.float64(v))
    zz = cz + 1
    above = wp.int32(0)
    if zz >= 0 and zz < Lz and cx >= 0 and cx < Lx and cy >= 0 and cy < Ly:
        above = ids[(zz * Ly + cy) * Lx + cx]
    if above == 0:
        wp.atomic_add(out_free, 0, wp.int32(v))
