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

        # FocalPointPlasticity (optional; enabled via enable_fpp). The link
        # topology is persistent; the *active* set (length <= max) is recomputed
        # per MCS, exactly mirroring the GPU FPPLinks rebuild seam.
        self._fpp_pairs = None          # (M,2)
        self._fpp_lambda = None         # (M,)
        self._fpp_target = None         # (M,)
        self._fpp_max = None            # (M,)
        self._fpp_adj = None            # per-cell list of (other, lambda, target)
        self._fpp_keep = None           # bool mask of currently-active links

    # --------------------------------------------------------------------- FPP
    def enable_fpp(self, pairs, lam, target, maxlen):
        """Attach FPP spring links (per-link lambda/target/max). ``pairs`` is
        (M,2). Energy per link = lambda*(L - target)^2, L = ||COM_a - COM_b||."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        m = pairs.shape[0]
        self._fpp_pairs = pairs
        self._fpp_lambda = np.broadcast_to(np.asarray(lam, np.float64), (m,)).copy()
        self._fpp_target = np.broadcast_to(np.asarray(target, np.float64), (m,)).copy()
        self._fpp_max = np.broadcast_to(np.asarray(maxlen, np.float64), (m,)).copy()
        self._rebuild_fpp_links()

    def create_fpp_link(self, a, b, lam, target, maxlen):
        """Append an undirected link a-b with per-link params (mirrors the GPU
        ``FPPLinks.create_link`` topology edit at the steppable boundary). Effective
        at the next ``_rebuild_fpp_links`` (run once per MCS in ``step_mcs``)."""
        if self._fpp_pairs is None:
            self._fpp_pairs = np.zeros((0, 2), dtype=np.int64)
            self._fpp_lambda = np.zeros(0, dtype=np.float64)
            self._fpp_target = np.zeros(0, dtype=np.float64)
            self._fpp_max = np.zeros(0, dtype=np.float64)
        self._fpp_pairs = np.vstack([self._fpp_pairs, [int(a), int(b)]]).astype(np.int64)
        self._fpp_lambda = np.append(self._fpp_lambda, float(lam))
        self._fpp_target = np.append(self._fpp_target, float(target))
        self._fpp_max = np.append(self._fpp_max, float(maxlen))

    def delete_fpp_link(self, a, b):
        """Remove every undirected link matching {a,b} (mirrors the GPU
        ``FPPLinks.delete_link``)."""
        if self._fpp_pairs is None or self._fpp_pairs.shape[0] == 0:
            return
        a, b = int(a), int(b)
        pa = self._fpp_pairs[:, 0]; pb = self._fpp_pairs[:, 1]
        match = ((pa == a) & (pb == b)) | ((pa == b) & (pb == a))
        if np.any(match):
            keep = ~match
            self._fpp_pairs = self._fpp_pairs[keep]
            self._fpp_lambda = self._fpp_lambda[keep]
            self._fpp_target = self._fpp_target[keep]
            self._fpp_max = self._fpp_max[keep]

    def has_link(self, a, b) -> bool:
        if self._fpp_pairs is None or self._fpp_pairs.shape[0] == 0:
            return False
        a, b = int(a), int(b)
        pa = self._fpp_pairs[:, 0]; pb = self._fpp_pairs[:, 1]
        return bool(np.any(((pa == a) & (pb == b)) | ((pa == b) & (pb == a))))

    def links_of_cell(self, c) -> np.ndarray:
        """Indices into the link arrays touching cell ``c``."""
        if self._fpp_pairs is None or self._fpp_pairs.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        c = int(c)
        pa = self._fpp_pairs[:, 0]; pb = self._fpp_pairs[:, 1]
        return np.nonzero((pa == c) | (pb == c))[0]

    def _rebuild_fpp_links(self):
        """Recompute the active link set (COM length <= max) and per-cell
        adjacency with the kept links' params."""
        if self._fpp_pairs is None:
            return
        v = np.where(self.volume > 0, self.volume, 1.0)
        cx, cy, cz = self.xsum / v, self.ysum / v, self.zsum / v
        a = self._fpp_pairs[:, 0]; b = self._fpp_pairs[:, 1]
        d = np.sqrt((cx[a] - cx[b]) ** 2 + (cy[a] - cy[b]) ** 2 + (cz[a] - cz[b]) ** 2)
        keep = d <= self._fpp_max
        self._fpp_keep = keep
        adj = [[] for _ in range(self.n_cells + 1)]
        for i in np.nonzero(keep)[0]:
            ai, bi = int(a[i]), int(b[i])
            adj[ai].append((bi, self._fpp_lambda[i], self._fpp_target[i]))
            adj[bi].append((ai, self._fpp_lambda[i], self._fpp_target[i]))
        self._fpp_adj = [
            (np.array([t[0] for t in lst], dtype=np.int64),
             np.array([t[1] for t in lst], dtype=np.float64),
             np.array([t[2] for t in lst], dtype=np.float64))
            for lst in adj
        ]

    def _fpp_cell_delta(self, cid, dvol, dxs, dys, dzs) -> float:
        """dE for shifting cell cid's COM by (dvol, dxs, dys, dzs) over its links
        (read from the int64-equivalent xsum/ysum/zsum / volume COM)."""
        if cid == 0 or self._fpp_adj is None:
            return 0.0
        others, lam, tgt = self._fpp_adj[cid]
        if others.size == 0:
            return 0.0
        v0 = self.volume[cid]
        if v0 <= 0:
            return 0.0
        v1 = v0 + dvol
        if v1 <= 0:
            return 0.0
        cx0 = self.xsum[cid] / v0; cy0 = self.ysum[cid] / v0; cz0 = self.zsum[cid] / v0
        cx1 = (self.xsum[cid] + dxs) / v1
        cy1 = (self.ysum[cid] + dys) / v1
        cz1 = (self.zsum[cid] + dzs) / v1
        vo = np.where(self.volume[others] > 0, self.volume[others], 1.0)
        ox = self.xsum[others] / vo; oy = self.ysum[others] / vo; oz = self.zsum[others] / vo
        lb = np.sqrt((cx0 - ox) ** 2 + (cy0 - oy) ** 2 + (cz0 - oz) ** 2)
        la = np.sqrt((cx1 - ox) ** 2 + (cy1 - oy) ** 2 + (cz1 - oz) ** 2)
        return float(np.sum(lam * ((la - tgt) ** 2 - (lb - tgt) ** 2)))

    def _d_fpp(self, x, y, z, new_id, old_id) -> float:
        if self._fpp_adj is None:
            return 0.0
        e = 0.0
        if new_id != 0:
            e += self._fpp_cell_delta(new_id, 1.0, x, y, z)
        if old_id != 0:
            e += self._fpp_cell_delta(old_id, -1.0, -x, -y, -z)
        return e

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
        # FPP links are static within a sweep; rebuild the active set once at the
        # MCS boundary (mirrors the GPU FPPLinks.rebuild() seam).
        if self._fpp_pairs is not None:
            self._rebuild_fpp_links()
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
                + self._d_fpp(tx, ty, tz, src_id, tgt_id)
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

    def active_link_lengths(self) -> np.ndarray:
        """COM-to-COM length of the currently-active FPP links (kept set)."""
        if self._fpp_pairs is None:
            return np.zeros(0)
        self._rebuild_fpp_links()
        v = np.where(self.volume > 0, self.volume, 1.0)
        cx, cy, cz = self.xsum / v, self.ysum / v, self.zsum / v
        a = self._fpp_pairs[self._fpp_keep, 0]
        b = self._fpp_pairs[self._fpp_keep, 1]
        return np.sqrt((cx[a] - cx[b]) ** 2 + (cy[a] - cy[b]) ** 2 + (cz[a] - cz[b]) ** 2)

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
