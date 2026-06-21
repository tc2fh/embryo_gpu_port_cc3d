"""Host-side initial-state construction for the GPU CPM engine.

``EngineState`` holds NumPy arrays describing the initial condition:

* ``ids``    : (Lz, Ly, Lx) int32 id-lattice, 0 = Medium, 1..n_cells = cell ids.
               This is the single source of truth; the GPU engine copies it to a
               flat ``int32`` device array.
* ``cell_type``      : (n_cells+1,) int32, per-cell type id (index 0 = Medium = 0).
* ``volume``         : (n_cells+1,) float64, voxel count per cell.
* ``xsum/ysum/zsum`` : (n_cells+1,) int64, sum of pixel coords per cell. COM is
               ``sum/volume`` -- exactly CC3D's ``xCM/volume`` for non-periodic BC
               (see CenterOfMassPlugin::field3DChange, the no-boundary branch).

Per-cell **target_volume / lambda_volume** are derived from the per-type config
arrays at engine-build time (kept on the config to stay close to CC3D's
``VolumeEnergyParameters`` by-cell-type semantics).

The integer coordinate sums make COM bit-exact and reproducible (no float
non-associativity); volume is an exact voxel count. The accompanying assert in
the engine checks ``volume == bincount(ids)`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EngineConfig


@dataclass
class EngineState:
    cfg: EngineConfig
    ids: np.ndarray                       # (Lz,Ly,Lx) int32
    cell_type: np.ndarray                 # (n_cells+1,) int32
    volume: np.ndarray                    # (n_cells+1,) float64
    xsum: np.ndarray                      # (n_cells+1,) int64
    ysum: np.ndarray
    zsum: np.ndarray
    n_cells: int = 0
    # optional per-cell scratch (the GPU steppable cell.dict-equivalents live in
    # the engine; this is just initial-condition metadata if a builder needs it)
    meta: dict = field(default_factory=dict)

    def coms(self) -> np.ndarray:
        """(n_cells+1, 3) COM array (Medium row = 0)."""
        v = np.where(self.volume > 0, self.volume, 1.0)
        return np.stack([self.xsum / v, self.ysum / v, self.zsum / v], axis=1)


def com_accumulators(ids: np.ndarray, n_cells: int):
    """Compute (xsum, ysum, zsum, volume) per cell from an id-lattice (Lz,Ly,Lx).

    xsum/ysum/zsum are int64 sums of integer pixel coordinates; volume is float64
    voxel count. Index 0 (Medium) is present but unused by the energy.
    """
    xsum = np.zeros(n_cells + 1, dtype=np.int64)
    ysum = np.zeros(n_cells + 1, dtype=np.int64)
    zsum = np.zeros(n_cells + 1, dtype=np.int64)
    volume = np.zeros(n_cells + 1, dtype=np.float64)
    zz, yy, xx = np.nonzero(ids)
    cids = ids[zz, yy, xx]
    np.add.at(volume, cids, 1.0)
    np.add.at(xsum, cids, xx.astype(np.int64))
    np.add.at(ysum, cids, yy.astype(np.int64))
    np.add.at(zsum, cids, zz.astype(np.int64))
    return xsum, ysum, zsum, volume


def state_from_id_lattice(
    cfg: EngineConfig,
    ids: np.ndarray,
    cell_type: np.ndarray,
) -> EngineState:
    """Build an ``EngineState`` from an explicit id-lattice and per-cell types.

    ``ids`` is (Lz,Ly,Lx) int32 with 0 = Medium. ``cell_type`` is (n_cells+1,)
    with cell_type[0] = 0. n_cells = max id in the lattice (== len(cell_type)-1).
    """
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    assert ids.shape == (cfg.Lz, cfg.Ly, cfg.Lx), (
        f"id-lattice shape {ids.shape} != (Lz,Ly,Lx) {(cfg.Lz, cfg.Ly, cfg.Lx)}"
    )
    cell_type = np.asarray(cell_type, dtype=np.int32)
    n_cells = int(len(cell_type) - 1)
    assert ids.max(initial=0) <= n_cells, "id-lattice has ids beyond cell_type length"
    xsum, ysum, zsum, volume = com_accumulators(ids, n_cells)
    return EngineState(
        cfg=cfg,
        ids=ids,
        cell_type=cell_type,
        volume=volume,
        xsum=xsum,
        ysum=ysum,
        zsum=zsum,
        n_cells=n_cells,
    )


def build_grid_state(cfg: EngineConfig, cells_per_axis: int = 3) -> EngineState:
    """A simple reproducible initial state: cubic cells on a regular grid.

    Used by the statistical-equivalence tests (a clean, well-defined IC shared by
    the GPU engine and the CPU reference). All non-medium cells are type 1.
    """
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    n = cells_per_axis
    ids = np.zeros((Lz, Ly, Lx), dtype=np.int32)
    block = int(round(float(cfg.target_volume[1]) ** (1.0 / 3.0)))
    block = max(2, block)
    spacing_x, spacing_y, spacing_z = Lx // n, Ly // n, Lz // n
    mx = (spacing_x - block) // 2
    my = (spacing_y - block) // 2
    mz = (spacing_z - block) // 2
    cid = 0
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                cid += 1
                x0, y0, z0 = ix * spacing_x + mx, iy * spacing_y + my, iz * spacing_z + mz
                ids[z0:z0 + block, y0:y0 + block, x0:x0 + block] = cid
    n_cells = n ** 3
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    cell_type[1:] = 1
    return state_from_id_lattice(cfg, ids, cell_type)
