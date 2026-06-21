"""GPU-resident steppable API skeleton (Phase 2).

CC3D steppables hold per-cell scratch in ``cell.dict`` and run arbitrary per-MCS
Python. The GPU port replaces ``cell.dict`` with **registered Structure-of-Arrays**
on the device (one ``wp.array`` of length n_cells+1 per scratch field), and the hot
per-cell/per-pixel logic with ``@wp.kernel`` functions that read/write those arrays
and the engine's GPU-resident state directly -- no host round-trip.

This module provides:

* ``CellDict`` -- registry of named per-cell SoA scratch arrays (the cell.dict
  equivalent), with host get/set helpers.
* ``GPUSteppable`` -- base class with ``start()`` / ``step(mcs)`` / ``finish()``,
  holding a reference to the ``GPUEngine`` and a ``CellDict``.
* ``SteppableManager`` -- runs registered steppables around each engine MCS, the
  GPU analogue of CC3D's steppable scheduling.
* ``FloorFreeAreaSteppable`` -- a worked **non-FPP** example end-to-end: it ports
  the Embryo ``SubstrateSteppable`` "floor free area" metric (sum of substrate-cell
  volumes whose voxel directly above is Medium) as an on-device kernel over the
  id-lattice + per-cell SoA, with the result reduced on the GPU. This demonstrates
  the full path: per-cell SoA scratch + a Warp kernel computing a steppable
  observable with no per-object SWIG calls.

FPP link dynamics and the cohesotaxis pipeline are explicitly OUT of Phase 2
(Phase 3) -- this skeleton is the API/seam they will plug into.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from .engine import GPUEngine

wp.init()


# ---------------------------------------------------------------------------
# cell.dict equivalent: named per-cell SoA scratch arrays
# ---------------------------------------------------------------------------
class CellDict:
    """Registry of per-cell scratch arrays (length n_cells+1), the GPU analogue of
    CC3D ``cell.dict``. Supports int32 / float32 / int64 fields."""

    _DTYPES = {"int32": wp.int32, "float32": wp.float32, "int64": wp.int64}
    _NP = {"int32": np.int32, "float32": np.float32, "int64": np.int64}

    def __init__(self, n_cells: int, device: str = "cuda:0"):
        self.n1 = n_cells + 1
        self.device = device
        self._arrays: dict[str, wp.array] = {}
        self._dtypes: dict[str, str] = {}

    def register(self, name: str, dtype: str = "float32", fill=0):
        if dtype not in self._DTYPES:
            raise ValueError(f"unsupported dtype {dtype}")
        npd = self._NP[dtype]
        host = np.full(self.n1, fill, dtype=npd)
        self._arrays[name] = wp.array(host, dtype=self._DTYPES[dtype], device=self.device)
        self._dtypes[name] = dtype
        return self._arrays[name]

    def arr(self, name: str) -> wp.array:
        return self._arrays[name]

    def get(self, name: str) -> np.ndarray:
        return self._arrays[name].numpy().copy()

    def set(self, name: str, values: np.ndarray):
        dtype = self._dtypes[name]
        self._arrays[name] = wp.array(
            np.asarray(values, dtype=self._NP[dtype]),
            dtype=self._DTYPES[dtype], device=self.device,
        )


# ---------------------------------------------------------------------------
# steppable base + manager
# ---------------------------------------------------------------------------
class GPUSteppable:
    def __init__(self, engine: GPUEngine, frequency: int = 1):
        self.engine = engine
        self.frequency = frequency
        self.cell_dict = CellDict(engine.n_cells, device=engine.device)

    def start(self):
        pass

    def step(self, mcs: int):
        pass

    def finish(self):
        pass


class SteppableManager:
    """Runs steppables around the engine MCS loop (CC3D-style scheduling)."""

    def __init__(self, engine: GPUEngine):
        self.engine = engine
        self.steppables: list[GPUSteppable] = []

    def register(self, steppable: GPUSteppable):
        self.steppables.append(steppable)
        return steppable

    def start(self):
        for s in self.steppables:
            s.start()

    def run(self, n_mcs: int, mcs_offset: int = 0):
        for m in range(n_mcs):
            mcs = mcs_offset + m
            self.engine.step_mcs(mcs)
            for s in self.steppables:
                if mcs % s.frequency == 0:
                    s.step(mcs)
        wp.synchronize()

    def finish(self):
        for s in self.steppables:
            s.finish()


# ---------------------------------------------------------------------------
# worked non-FPP example: floor-free-area (Embryo SubstrateSteppable port)
# ---------------------------------------------------------------------------
@wp.kernel
def _floor_free_area_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    substrate_type: wp.int32,
    exposed_above: wp.array(dtype=wp.int32),   # per-cell: count of voxels with Medium directly above
):
    """For each substrate voxel whose neighbor at z+1 is Medium, increment its
    cell's exposed-voxel count. Sum over a cell == that cell's contribution to the
    floor free area (the Embryo SubstrateSteppable counts ``cell.volume`` of floor
    cells with no cell directly above; here we count exposed voxels per cell, an
    equivalent on-device formulation). One thread per voxel."""
    i = wp.tid()
    cid = ids[i]
    if cid == 0:
        return
    if cell_type[cid] != substrate_type:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    zz = z + 1
    above = wp.int32(0)
    if zz < Lz:
        above = ids[(zz * Ly + y) * Lx + x]
    else:
        above = wp.int32(0)
    if above == 0:
        wp.atomic_add(exposed_above, cid, 1)


class LamellipodiaSteppable(GPUSteppable):
    """GPU port of the Embryo ``LeadingEdgeSteppable`` lamellipodia dynamics
    (``EmbryoSteppables.py``): the ``ifCohesotaxis`` link target selection +
    Poisson link turnover, driven entirely by on-device kernels at the per-MCS
    steppable boundary (the FPP create/delete seam from Pass A).

    Per MCS, for each LEADING cell:
      * if it has a lamellipodia link, delete it with Poisson prob
        ``1-exp(-LamellaeRate)`` (on-device Bernoulli keyed by (mcs,cell,seed));
      * if it then has no lamellipodia link, run the cohesotaxis pipeline to select
        a substrate target and create a new lamellipodia link (per-link
        LamellipodiaLambda / LLTargetDist / LLMaxDist) via ``FPPLinks.create_link``.

    ``self.link_target[cid]`` mirrors CC3D ``cell.dict['link']`` (the substrate id a
    leader is currently linked to; 0 = none). Links are created/deleted on the host
    topology at this boundary, then the device CSR is rebuilt once per MCS by the
    engine -- never inside the Metropolis inner loop.
    """

    def __init__(self, engine: GPUEngine, links, leading_type: int,
                 substrate_type: int, passive_type: int,
                 lamellipodia_distance: int | None = None, frequency: int = 1,
                 link_backend: str = "host", cohesotaxis_backend: str = "fused"):
        super().__init__(engine, frequency)
        from . import cohesotaxis as CT
        self._CT = CT
        self.links = links
        self.leading_type = int(leading_type)
        self.substrate_type = int(substrate_type)
        self.link_backend = link_backend  # "device" (no copyback) or "host" (tests)
        # cohesotaxis_backend: "fused" (Phase-8 single persistent-buffer device pipeline
        # -- canonical-order, deterministic, no inter-stage host glue; the hot path) or
        # "staged" (the Phase-3 staged pipeline, kept reachable for the differential
        # test). Both are deterministic; "staged" uses canonical order to match "fused".
        self.cohesotaxis_backend = cohesotaxis_backend
        ld = CT.LAMELLIPODIA_DISTANCE if lamellipodia_distance is None else lamellipodia_distance
        if cohesotaxis_backend == "fused":
            from .cohesotaxis_fused import FusedCohesotaxisPipeline
            self.pipe = FusedCohesotaxisPipeline(
                engine, leading_type=leading_type, substrate_type=substrate_type,
                passive_type=passive_type, lamellipodia_distance=ld)
        else:
            self.pipe = CT.CohesotaxisPipeline(
                engine, leading_type=leading_type, substrate_type=substrate_type,
                passive_type=passive_type, lamellipodia_distance=ld, canonical=True)
        # cell.dict['link'] equivalent: substrate id each leader is linked to (0=none)
        # (HOST path only; the device path derives 'has a link' from the inventory)
        self.cell_dict.register("link_target", "int32", 0)
        self.poisson_rate = CT.LAMELLAE_RATE
        self.lamellipodia_lambda = CT.LAMELLIPODIA_LAMBDA
        self.lamellae_delete_prob = float(1.0 - np.exp(-self.poisson_rate))

    def _leaders(self):
        return self.pipe.lead_ids

    def start(self):
        """Create the initial lamellipodia link for every leader (the
        ``LeadingEdgeSteppable.start`` ``create_lamellipodia_link`` call)."""
        created = self.pipe.create_lamellipodia_links(self.links, mcs=0)
        lt = self.cell_dict.get("link_target")
        for cell, tgt in created.items():
            lt[cell] = tgt
        self.cell_dict.set("link_target", lt)
        return created

    def step(self, mcs: int):
        if self.link_backend == "device":
            return self._step_device(mcs)
        return self._step_host(mcs)

    # ------------------------------------------------------------- device path
    def _step_device(self, mcs: int):
        """Lamellipodia turnover with NO host link round-trip: Poisson-delete existing
        lamellipodia links via the device keep-mask + compact (keyed per leader cell,
        the same decisions as the host path), then recreate for leaders now lacking a
        link -- the 'need' set derived from the inventory (not a host SoA). The
        cohesotaxis create pipeline is already on-device."""
        # --- device Poisson delete (keep/compact seam), per-leader keyed ---
        self.links.poisson_delete_device_by_cell(
            self.lamellipodia_lambda, self.lamellae_delete_prob, mcs,
            self.engine.base_seed, self.engine, self.substrate_type)
        # --- recreate for leaders that now lack a lamellipodia link ---
        has = self.links.cells_with_kind_link(
            self.lamellipodia_lambda, self.engine, self.substrate_type)
        leaders = self._leaders()
        has_host = has.numpy()                 # (n1,) ints -- a per-cell flag, not the graph
        need = {int(c) for c in leaders if has_host[int(c)] == 0}
        if need:
            self.pipe.create_lamellipodia_links(self.links, mcs=mcs, only_cells=need)
        return int(self.links.n_pairs)

    # ------------------------------------------------------------- host path
    def _step_host(self, mcs: int):
        CT = self._CT
        lt = self.cell_dict.get("link_target")
        leaders = self._leaders()

        # --- Poisson delete of existing lamellipodia links (on-device decisions) ---
        n1 = self.engine.n_cells + 1
        dec = CT.poisson_delete_decisions(n_cells=n1, mcs=mcs,
                                          base_seed=self.engine.base_seed,
                                          rate=self.poisson_rate, device=self.engine.device)
        to_delete = []
        for cell in leaders:
            tgt = int(lt[cell])
            if tgt != 0 and dec[cell] == 1:
                to_delete.append((int(cell), tgt))
                lt[cell] = 0
        if to_delete:  # one vectorized compaction for the whole batch
            self.links.delete_links_bulk(to_delete)

        # --- recreate lamellipodia links for leaders now lacking one ---
        need = {int(c) for c in leaders if int(lt[c]) == 0}
        if need:
            created = self.pipe.create_lamellipodia_links(self.links, mcs=mcs,
                                                          only_cells=need)
            for cell, tgt in created.items():
                lt[cell] = tgt

        self.cell_dict.set("link_target", lt)
        # links are static within the next sweep; the engine rebuilds the CSR at the
        # MCS boundary (step_mcs), so no explicit rebuild() needed here.
        return int(np.count_nonzero(lt[leaders]))


class FloorFreeAreaSteppable(GPUSteppable):
    """Non-FPP example: tracks the substrate floor free area on the GPU each MCS.

    ``self.history`` accumulates (mcs, free_area). ``free_area`` = number of
    substrate voxels with Medium directly above, computed entirely on device.
    Demonstrates a per-cell SoA scratch ('exposed') + a Warp kernel + a GPU
    reduction, the steppable path with no per-object host calls.
    """

    def __init__(self, engine: GPUEngine, substrate_type: int, frequency: int = 1):
        super().__init__(engine, frequency)
        self.substrate_type = int(substrate_type)
        self.cell_dict.register("exposed", "int32", 0)
        self.history: list[tuple[int, int]] = []

    def step(self, mcs: int):
        eng = self.engine
        exposed = self.cell_dict.arr("exposed")
        exposed.zero_()
        wp.launch(
            _floor_free_area_kernel,
            dim=eng.cfg.n_voxels,
            inputs=[
                eng.ids, eng.cell_type, eng.Lx, eng.Ly, eng.Lz,
                self.substrate_type, exposed,
            ],
            device=eng.device,
        )
        wp.synchronize()
        free_area = int(self.cell_dict.get("exposed").sum())
        self.history.append((mcs, free_area))
        return free_area
