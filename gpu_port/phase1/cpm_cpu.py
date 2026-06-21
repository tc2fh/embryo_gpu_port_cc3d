"""CPU (NumPy) reference for the synthetic CPM + FPP model.

Plain, readable single-site Metropolis. This is the *ground truth* the GPU Warp
prototype is compared against (statistically, not bit-for-bit -- the RNG streams
differ by construction).

Energy terms exactly mirror the vendored CC3D semantics (see ``model.py`` header):

* dE_volume  = lambda_V*(1 + 2*(V_new - V_t)) + lambda_V*(1 - 2*(V_old - V_t))
               evaluated per non-medium participant -- the incremental form of
               lambda*(V - V_t)^2 from ``VolumePlugin::changeEnergyByCellType``.
* dE_contact = sum_{Moore neighbors} [ J(new, nCell) - J(old, nCell) ]
               skipping the neighbor that equals the cell itself, exactly like
               ``ContactPlugin::changeEnergy``.
* dE_fpp     = sum over the moving cell's links of
               lambda*(L_after - L_t)^2 - lambda*(L_before - L_t)^2
               where L is the COM-to-COM Euclidean distance and the moving cell's
               COM shifts by +-1 voxel coordinate (``potentialFunction`` +
               ``distInvariantCM``). offset cancels in the delta.

The FPP link list is rebuilt each MCS from the persistent topology: links whose
COM length exceeds ``fpp_max_length`` are dropped (delete), and re-added when they
shrink back (create). This exercises the dynamic create/delete lifecycle that the
GPU CSR machinery must reproduce.
"""

from __future__ import annotations

import numpy as np

from model import (
    MOORE_OFFSETS,
    VONNEUMANN_OFFSETS,
    ModelConfig,
    ModelState,
)


class CPUEngine:
    def __init__(self, state: ModelState):
        self.cfg = state.cfg
        self.L = state.cfg.L
        self.ids = state.ids.copy()
        self.types = state.types.copy()
        self.target_volume = state.target_volume.copy()
        self.lambda_volume = state.lambda_volume.copy()
        self.xsum = state.xsum.copy()
        self.ysum = state.ysum.copy()
        self.zsum = state.zsum.copy()
        self.volume = state.volume.copy()
        self.J = self.cfg.contact_matrix()
        # persistent (full) link topology; the *active* set is recomputed per MCS
        self.link_pairs_all = state.link_pairs.copy()
        self.active_links = self.link_pairs_all.copy()
        # per-cell adjacency for fast dE_fpp (list of (other_cell) per cell)
        self._rebuild_adjacency()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.T = self.cfg.temperature

    # -- COM helpers ---------------------------------------------------------
    def com(self, cid: int):
        v = self.volume[cid]
        if v <= 0:
            return 0.0, 0.0, 0.0
        return self.xsum[cid] / v, self.ysum[cid] / v, self.zsum[cid] / v

    def _rebuild_adjacency(self):
        n = self.cfg.n_cells
        adj = [[] for _ in range(n + 1)]
        for a, b in self.active_links:
            adj[a].append(b)
            adj[b].append(a)
        self.adj = [np.asarray(x, dtype=np.int32) for x in adj]

    def rebuild_links(self):
        """Dynamic link rebuild: keep links whose current COM length <= max."""
        if len(self.link_pairs_all) == 0:
            self.active_links = self.link_pairs_all
            self._rebuild_adjacency()
            return
        vol = np.where(self.volume > 0, self.volume, 1.0)
        cx = self.xsum / vol
        cy = self.ysum / vol
        cz = self.zsum / vol
        a = self.link_pairs_all[:, 0]
        b = self.link_pairs_all[:, 1]
        dx = cx[a] - cx[b]
        dy = cy[a] - cy[b]
        dz = cz[a] - cz[b]
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        keep = length <= self.cfg.fpp_max_length
        self.active_links = self.link_pairs_all[keep]
        self._rebuild_adjacency()

    # -- energy deltas -------------------------------------------------------
    def _d_volume(self, gaining: int, losing: int) -> float:
        e = 0.0
        if gaining != 0:
            lv = self.lambda_volume[gaining]
            tv = self.target_volume[gaining]
            e += lv * (1.0 + 2.0 * (self.volume[gaining] - tv))
        if losing != 0:
            lv = self.lambda_volume[losing]
            tv = self.target_volume[losing]
            e += lv * (1.0 - 2.0 * (self.volume[losing] - tv))
        return e

    def _d_contact(self, x, y, z, new_id, old_id) -> float:
        L = self.L
        new_t = self.types[new_id]
        old_t = self.types[old_id]
        e = 0.0
        for dx, dy, dz in MOORE_OFFSETS:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= L or ny >= L or nz >= L:
                # out of bounds neighbor is treated as Medium (id 0, type 0)
                ncell = 0
            else:
                ncell = self.ids[nz, ny, nx]
            nt = self.types[ncell]
            if ncell != old_id:
                e -= self.J[old_t, nt]
            if ncell != new_id:
                e += self.J[new_t, nt]
        return e

    def _fpp_cell_energy_delta(self, cid: int, dxs, dys, dzs) -> float:
        """dE for shifting cell `cid` COM by (dxs,dys,dzs)/V_after over its links.

        We compute the moving cell's COM before and after the volume/sum change.
        Caller passes the volume *after* change as v_after.
        """
        if cid == 0:
            return 0.0
        neigh = self.adj[cid]
        if neigh.size == 0:
            return 0.0
        lam = self.cfg.fpp_lambda
        tgt = self.cfg.fpp_target_length
        # before
        v0 = self.volume[cid]
        cx0 = self.xsum[cid] / v0
        cy0 = self.ysum[cid] / v0
        cz0 = self.zsum[cid] / v0
        # after
        v1 = v0 + dxs[0]  # dxs[0] is the volume delta (+1 / -1)
        if v1 <= 0:
            return 0.0
        cx1 = (self.xsum[cid] + dxs[1]) / v1
        cy1 = (self.ysum[cid] + dys[1]) / v1
        cz1 = (self.zsum[cid] + dzs[1]) / v1
        # other endpoints' COM (unchanged this flip)
        vol = np.where(self.volume > 0, self.volume, 1.0)
        ox = self.xsum[neigh] / vol[neigh]
        oy = self.ysum[neigh] / vol[neigh]
        oz = self.zsum[neigh] / vol[neigh]
        lb = np.sqrt((cx0 - ox) ** 2 + (cy0 - oy) ** 2 + (cz0 - oz) ** 2)
        la = np.sqrt((cx1 - ox) ** 2 + (cy1 - oy) ** 2 + (cz1 - oz) ** 2)
        de = lam * ((la - tgt) ** 2 - (lb - tgt) ** 2)
        return float(np.sum(de))

    def _d_fpp(self, x, y, z, new_id, old_id) -> float:
        """FPP delta: gaining cell gains voxel (x,y,z); losing cell loses it."""
        e = 0.0
        # gaining cell: volume +1, sums + (x,y,z)
        if new_id != 0:
            e += self._fpp_cell_energy_delta(
                new_id, (1.0, x), (1.0, y), (1.0, z)
            )
        # losing cell: volume -1, sums - (x,y,z)
        if old_id != 0:
            e += self._fpp_cell_energy_delta(
                old_id, (-1.0, -x), (-1.0, -y), (-1.0, -z)
            )
        return e

    # -- Metropolis ----------------------------------------------------------
    def step_mcs(self, flips_per_mcs: int | None = None):
        """One Monte Carlo step = L^3 flip attempts (one sweep)."""
        L = self.L
        n_sites = L * L * L
        if flips_per_mcs is None:
            flips_per_mcs = n_sites
        self.rebuild_links()
        for _ in range(flips_per_mcs):
            x = int(self.rng.integers(0, L))
            y = int(self.rng.integers(0, L))
            z = int(self.rng.integers(0, L))
            old_id = int(self.ids[z, y, x])
            # pick a face neighbor to copy FROM (its id becomes the new id)
            k = int(self.rng.integers(0, len(VONNEUMANN_OFFSETS)))
            dx, dy, dz = VONNEUMANN_OFFSETS[k]
            sx, sy, sz = x + dx, y + dy, z + dz
            if sx < 0 or sy < 0 or sz < 0 or sx >= L or sy >= L or sz >= L:
                continue
            new_id = int(self.ids[sz, sy, sx])
            if new_id == old_id:
                continue
            de = (
                self._d_volume(new_id, old_id)
                + self._d_contact(x, y, z, new_id, old_id)
                + self._d_fpp(x, y, z, new_id, old_id)
            )
            if de <= 0.0:
                accept = True
            else:
                accept = self.rng.random() < np.exp(-de / self.T)
            if accept:
                self._apply(x, y, z, new_id, old_id)

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

    # -- observables ---------------------------------------------------------
    def link_lengths(self) -> np.ndarray:
        from model import link_lengths_from_state

        return link_lengths_from_state(
            self.xsum, self.ysum, self.zsum, self.volume, self.active_links
        )

    def volumes(self) -> np.ndarray:
        return self.volume[1:].copy()

    def coms(self) -> np.ndarray:
        from model import cell_coms

        return cell_coms(self.xsum, self.ysum, self.zsum, self.volume)[1:]


def run_cpu(cfg: ModelConfig, n_mcs: int, state: ModelState | None = None) -> CPUEngine:
    from model import build_state

    if state is None:
        state = build_state(cfg)
    eng = CPUEngine(state)
    eng.run(n_mcs)
    return eng
