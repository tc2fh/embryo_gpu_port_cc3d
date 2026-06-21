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
