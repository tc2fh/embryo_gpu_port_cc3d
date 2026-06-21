"""CPU (NumPy) reference CPM with Volume + Contact -- the statistical ground truth.

Plain single-site Metropolis mirroring ``Potts3D::metropolisFast`` semantics:

* One MCS = ``flip2DimRatio * Lx*Ly*Lz`` flip attempts.
* Each attempt: pick a random *source* pixel ``pt`` (its cell is ``cell``), pick a
  random neighbor ``changePixel`` from the flip-target shell; ``changePixel`` may
  flip to ``cell`` (i.e. newCell = cell at pt, oldCell = cell at changePixel).
  Frozen-type cells are never a source or target (``checkIfFrozen``).
* Energy delta = Volume (changeEnergyByCellType, incremental) + Contact
  (changeEnergy over the contact shell, id-inequality form). Accept if dE<=0 else
  with prob exp(-dE/T) (DefaultAcceptanceFunction).

Independent RNG (NumPy PCG) from the GPU's Philox stream, so agreement is
*distributional* within Monte-Carlo noise, not bit-identical (per the plan's
verification section). This reference uses the SAME neighbor orders as the GPU
engine config so the two compare on identical physics.
"""

from __future__ import annotations

import numpy as np

from .config import EngineConfig, neighbor_offsets
from .state import EngineState


class CPUReference:
    def __init__(self, state: EngineState):
        self.cfg: EngineConfig = state.cfg
        self.Lx, self.Ly, self.Lz = self.cfg.Lx, self.cfg.Ly, self.cfg.Lz
        self.ids = state.ids.copy()                       # (Lz,Ly,Lx)
        self.cell_type = state.cell_type.copy()
        self.volume = state.volume.copy().astype(np.float64)
        self.xsum = state.xsum.copy().astype(np.float64)
        self.ysum = state.ysum.copy().astype(np.float64)
        self.zsum = state.zsum.copy().astype(np.float64)
        self.n_cells = state.n_cells

        n1 = self.n_cells + 1
        self.tv = self.cfg.target_volume[self.cell_type].astype(np.float64)
        self.lv = self.cfg.lambda_volume[self.cell_type].astype(np.float64)
        self.J = self.cfg.contact.astype(np.float64)
        self.frozen_type = self.cfg.frozen_mask().astype(bool)

        self.contact_off = neighbor_offsets(self.cfg.contact_neighbor_order)
        self.flip_off = neighbor_offsets(self.cfg.flip_neighbor_order)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.T = float(self.cfg.temperature)
        self.attempts_per_mcs = int(round(self.cfg.flip2_dim_ratio * self.Lx * self.Ly * self.Lz))

    def _d_volume(self, new_id, old_id) -> float:
        e = 0.0
        if new_id != 0:
            e += self.lv[new_id] * (1.0 + 2.0 * (self.volume[new_id] - self.tv[new_id]))
        if old_id != 0:
            e += self.lv[old_id] * (1.0 - 2.0 * (self.volume[old_id] - self.tv[old_id]))
        return e

    def _d_contact(self, x, y, z, new_id, old_id) -> float:
        new_t = self.cell_type[new_id]
        old_t = self.cell_type[old_id]
        e = 0.0
        for dx, dy, dz in self.contact_off:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= self.Lx or ny >= self.Ly or nz >= self.Lz:
                ncell = 0
            else:
                ncell = int(self.ids[nz, ny, nx])
            nt = self.cell_type[ncell]
            if ncell != old_id:
                e -= self.J[old_t, nt]
            if ncell != new_id:
                e += self.J[new_t, nt]
        return e

    def step_mcs(self):
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        nflip = len(self.flip_off)
        for _ in range(self.attempts_per_mcs):
            # source pixel pt; its cell is the candidate value
            px = int(self.rng.integers(0, Lx))
            py = int(self.rng.integers(0, Ly))
            pz = int(self.rng.integers(0, Lz))
            src_id = int(self.ids[pz, py, px])
            # Medium (id 0) is never frozen (CC3D gates checkIfFrozen on non-null cell)
            if src_id != 0 and self.frozen_type[self.cell_type[src_id]]:
                continue
            # target pixel = a random neighbor of pt
            k = int(self.rng.integers(0, nflip))
            dx, dy, dz = self.flip_off[k]
            tx, ty, tz = px + dx, py + dy, pz + dz
            if tx < 0 or ty < 0 or tz < 0 or tx >= Lx or ty >= Ly or tz >= Lz:
                continue
            tgt_id = int(self.ids[tz, ty, tx])
            if tgt_id == src_id:
                continue
            if tgt_id != 0 and self.frozen_type[self.cell_type[tgt_id]]:
                continue
            # target flips to source's cell: newCell=src_id, oldCell=tgt_id at (tx,ty,tz)
            de = (
                self._d_volume(src_id, tgt_id)
                + self._d_contact(tx, ty, tz, src_id, tgt_id)
            )
            if de <= 0.0:
                accept = True
            else:
                accept = self.rng.random() < np.exp(-de / self.T)
            if accept:
                self._apply(tx, ty, tz, src_id, tgt_id)

    def _apply(self, x, y, z, new_id, old_id):
        self.ids[z, y, x] = new_id
        if old_id != 0:
            self.volume[old_id] -= 1.0
            self.xsum[old_id] -= x
            self.ysum[old_id] -= y
            self.zsum[old_id] -= z
        if new_id != 0:
            self.volume[new_id] += 1.0
            self.xsum[new_id] += x
            self.ysum[new_id] += y
            self.zsum[new_id] += z

    def run(self, n_mcs: int):
        for _ in range(n_mcs):
            self.step_mcs()

    # observables (parallel to GPUEngine)
    def volumes(self) -> np.ndarray:
        return self.volume[1:].copy()

    def coms(self) -> np.ndarray:
        v = np.where(self.volume > 0, self.volume, 1.0)
        return np.stack([self.xsum / v, self.ysum / v, self.zsum / v], axis=1)[1:]

    def surface_areas(self, order: int | None = None) -> np.ndarray:
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        surf = np.zeros(self.n_cells + 1, dtype=np.int64)
        zz, yy, xx = np.nonzero(self.ids)
        cids = self.ids[zz, yy, xx]
        for dx, dy, dz in off:
            nx, ny, nz = xx + dx, yy + dy, zz + dz
            valid = (nx >= 0) & (ny >= 0) & (nz >= 0) & (nx < self.Lx) & (ny < self.Ly) & (nz < self.Lz)
            ncell = np.zeros_like(cids)
            ncell[valid] = self.ids[nz[valid], ny[valid], nx[valid]]
            diff = ncell != cids
            np.add.at(surf, cids[diff], 1)
        return surf[1:].copy()

    def total_energy(self) -> float:
        e_vol = float(np.sum(self.lv[1:] * (self.volume[1:] - self.tv[1:]) ** 2))
        # contact: 0.5 * sum over voxels of differing-neighbor J
        e_contact = 0.0
        zz, yy, xx = np.nonzero(np.ones_like(self.ids))  # all voxels incl medium
        # iterate compactly over all voxels
        ids = self.ids
        types = self.cell_type
        e = 0.0
        for dx, dy, dz in self.contact_off:
            nx, ny, nz = xx + dx, yy + dy, zz + dz
            valid = (nx >= 0) & (ny >= 0) & (nz >= 0) & (nx < self.Lx) & (ny < self.Ly) & (nz < self.Lz)
            self_id = ids[zz, yy, xx]
            ncell = np.zeros_like(self_id)
            ncell[valid] = ids[nz[valid], ny[valid], nx[valid]]
            diff = ncell != self_id
            e += float(np.sum(self.J[types[self_id[diff]], types[ncell[diff]]]))
        e_contact = 0.5 * e
        return e_vol + e_contact
