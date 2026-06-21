"""FocalPointPlasticity device-authoritative link inventory (Phase 3, Pass A).

CC3D's FocalPointPlasticity holds, per cell, a set of spring **links** to other
cells, each created by ``new_fpp_link(cell_a, cell_b, lambda, targetDist, maxDist)``
and carrying its own ``lambda`` / ``target length`` / ``max length``. A link's
energy is ``lambda*(L - target)^2`` with ``L = || COM_a - COM_b ||_2``; CC3D drops
a link once the cell COMs separate beyond ``maxDistance``.

``FPPLinks`` mirrors that on the GPU with the **proven Phase 1 pattern**
(``gpu_port/phase1/cpm_gpu.py``): a device-authoritative per-cell **link CSR**
built by

  1. a flag/degree-count kernel that keeps links whose current COM length <= their
     own max_length (this is the *delete* -- links that grew too long are dropped),
  2. a host exclusive prefix-sum of the per-cell degree -> ``link_ptr`` (n_cells is
     tiny and off the hot per-flip path), and
  3. an atomic-append fill kernel that writes each kept undirected link into BOTH
     endpoints' CSR ranges, carrying its per-link lambda/target (the *create*).

Links are **static within a color sweep**: create/delete happen at the per-MCS
steppable boundary (the engine ``recompute_trackers`` seam), never inside the
Metropolis inner loop, so the kernel can read the CSR race-free. The energy
kernel (``kernels.fpp_delta_cell``) reads link length from the engine int64
``xsum/ysum/zsum / volume`` COM directly -- the exact, reproducible single source
of truth (no separate COM tracker, no float drift).

Topology mutation API (the steppable boundary):
  * ``create_link(a, b, ...)`` / ``delete_link(a, b)`` edit the host topology.
  * ``set_topology(pairs, lambdas, targets, maxlens)`` sets it wholesale.
  * ``rebuild()`` rebuilds the device CSR from the current topology + live COMs.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from . import kernels as K

wp.init()


def grid_graph_links(cells_per_axis: int) -> np.ndarray:
    """Undirected (a,b) pairs (a<b, sorted) for axis-adjacent cells on a
    cells_per_axis^3 cell grid (the Phase 1 synthetic spring network). Cell ids
    are 1-based in lattice raster order (matches ``build_grid_state``)."""
    n = cells_per_axis

    def cid(ix, iy, iz):
        return iz * n * n + iy * n + ix + 1

    pairs = set()
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                a = cid(ix, iy, iz)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if jx < n and jy < n and jz < n:
                        b = cid(jx, jy, jz)
                        lo, hi = (a, b) if a < b else (b, a)
                        pairs.add((lo, hi))
    return np.asarray(sorted(pairs), dtype=np.int32)


class FPPLinks:
    """Device-authoritative FPP link inventory + per-cell CSR for one engine.

    Plugs into the engine via ``engine.attach_fpp(self)``; the engine passes the
    CSR arrays (``link_ptr``, ``link_other``, ``link_lambda``, ``link_target``)
    into ``metropolis_color_kernel`` and calls ``rebuild()`` once per MCS through
    ``recompute_trackers``.
    """

    def __init__(self, engine, target_length_default: float = 5.0,
                 lambda_default: float = 1.0, max_length_default: float = 10.0):
        self.engine = engine
        self.device = engine.device
        self.n_cells = engine.n_cells
        self.n1 = self.n_cells + 1
        self.target_default = float(target_length_default)
        self.lambda_default = float(lambda_default)
        self.max_default = float(max_length_default)

        # host topology (the editable source of truth for create/delete). Each
        # entry i is an undirected link (a_i, b_i) with per-link params; a==-1
        # marks a tombstoned (deleted) slot, compacted away on the next set.
        self._a = np.zeros(0, dtype=np.int32)
        self._b = np.zeros(0, dtype=np.int32)
        self._lam = np.zeros(0, dtype=np.float32)
        self._tgt = np.zeros(0, dtype=np.float32)
        self._max = np.zeros(0, dtype=np.float32)

        # device CSR (rebuilt each MCS). Allocated lazily in rebuild().
        self.link_ptr = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        self.link_other = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.link_lambda = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.link_target = wp.zeros(1, dtype=wp.float32, device=self.device)

        # device topology scratch (rebuilt when the host topology changes)
        self._dev_dirty = True
        self._pa = self._pb = self._pmax = None
        self._plam = self._ptgt = None
        self._keep = self._degree = self._cursor = None
        self._n_active = 0

    # ----------------------------------------------------------- topology edits
    def set_topology(self, pairs, lambdas=None, targets=None, maxlens=None):
        """Replace the entire link topology. ``pairs`` is (M,2) int (a,b)."""
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        m = pairs.shape[0]
        self._a = pairs[:, 0].astype(np.int32).copy()
        self._b = pairs[:, 1].astype(np.int32).copy()
        self._lam = (np.full(m, self.lambda_default, np.float32) if lambdas is None
                     else np.asarray(lambdas, np.float32).copy())
        self._tgt = (np.full(m, self.target_default, np.float32) if targets is None
                     else np.asarray(targets, np.float32).copy())
        self._max = (np.full(m, self.max_default, np.float32) if maxlens is None
                     else np.asarray(maxlens, np.float32).copy())
        self._dev_dirty = True

    def set_max_lengths(self, maxlens):
        self._max = np.asarray(maxlens, np.float32).copy()
        self._dev_dirty = True

    def create_links_bulk(self, a, b, lam=None, target=None, maxlen=None):
        """Append MANY undirected links in a single allocation.

        This is the batched form of ``create_link``: k separate ``create_link``
        calls each ``np.append`` the whole inventory (O(M) per call -> O(M*k) for a
        steppable's per-MCS relink loop, the dominant host cost at full Embryo
        scale). Here the k new links are concatenated once -> O(M + k).

        ``a`` / ``b`` are length-k id arrays. Each per-link param accepts a scalar
        (broadcast to all k -- the common case, one lambda/target/max for the whole
        batch), a length-k array, or None (-> the class default). Links are appended
        in the given order, so a sequence of ``create_link``/``create_links_bulk``
        calls yields the same inventory as the equivalent single calls. Effective at
        the next ``rebuild()``."""
        a = np.ascontiguousarray(a, dtype=np.int32).ravel()
        b = np.ascontiguousarray(b, dtype=np.int32).ravel()
        k = a.shape[0]
        if k == 0:
            return
        if b.shape[0] != k:
            raise ValueError("create_links_bulk: a and b must have equal length")

        def _col(v, default):
            if v is None:
                return np.full(k, default, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32).ravel()
            if v.shape[0] == 1:                       # scalar -> broadcast to k
                return np.full(k, float(v[0]), dtype=np.float32)
            if v.shape[0] != k:
                raise ValueError("create_links_bulk: per-link param length must be 1 or k")
            return np.ascontiguousarray(v, dtype=np.float32)

        self._a = np.concatenate([self._a, a])
        self._b = np.concatenate([self._b, b])
        self._lam = np.concatenate([self._lam, _col(lam, self.lambda_default)])
        self._tgt = np.concatenate([self._tgt, _col(target, self.target_default)])
        self._max = np.concatenate([self._max, _col(maxlen, self.max_default)])
        self._dev_dirty = True

    def delete_links_bulk(self, pairs):
        """Remove EVERY link whose unordered endpoints match any {a,b} in ``pairs``,
        in one vectorized pass over the inventory.

        The batched form of ``delete_link``: k separate ``delete_link`` calls each
        scan + recompact the whole inventory (O(M*k) for a per-MCS Poisson-delete
        loop). Here all k targets are matched with a single ``np.isin`` and one
        compaction. ``pairs`` is an iterable of (a, b)."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.shape[0] == 0 or self._a.shape[0] == 0:
            return
        mult = np.int64(self.n_cells + 2)             # pack {lo,hi} -> one int64 key
        ea = self._a.astype(np.int64)
        eb = self._b.astype(np.int64)
        ekeys = np.minimum(ea, eb) * mult + np.maximum(ea, eb)
        dkeys = np.unique(np.minimum(pairs[:, 0], pairs[:, 1]) * mult
                          + np.maximum(pairs[:, 0], pairs[:, 1]))
        match = np.isin(ekeys, dkeys)
        if match.any():
            keep = ~match
            self._a = self._a[keep]
            self._b = self._b[keep]
            self._lam = self._lam[keep]
            self._tgt = self._tgt[keep]
            self._max = self._max[keep]
            self._dev_dirty = True

    def create_link(self, a: int, b: int, lam=None, target=None, maxlen=None):
        """Add ONE undirected link a-b (no dedup; CC3D guards dups at the call site).
        Effective at the next ``rebuild()``. Hot per-MCS loops should batch their
        edits through ``create_links_bulk`` (one allocation instead of O(M) each)."""
        self.create_links_bulk(
            [a], [b],
            lam=None if lam is None else [lam],
            target=None if target is None else [target],
            maxlen=None if maxlen is None else [maxlen])

    def delete_link(self, a: int, b: int):
        """Remove every undirected link matching {a,b}. Hot per-MCS loops should
        batch their edits through ``delete_links_bulk``."""
        self.delete_links_bulk([(int(a), int(b))])

    @property
    def n_pairs(self) -> int:
        return int(self._a.shape[0])

    def has_links(self) -> bool:
        return self.n_pairs > 0

    def num_active(self) -> int:
        """Number of links kept (within max_length) at the last rebuild."""
        return int(self._n_active)

    # --------------------------------------------------------------- device sync
    def _sync_device_topology(self):
        m = self.n_pairs
        self._pa = wp.array(self._a, dtype=wp.int32, device=self.device)
        self._pb = wp.array(self._b, dtype=wp.int32, device=self.device)
        self._plam = wp.array(self._lam, dtype=wp.float32, device=self.device)
        self._ptgt = wp.array(self._tgt, dtype=wp.float32, device=self.device)
        self._pmax = wp.array(self._max, dtype=wp.float32, device=self.device)
        self._keep = wp.zeros(max(1, m), dtype=wp.int32, device=self.device)
        self._degree = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        self._cursor = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        # CSR holds 2 directed entries per undirected link (max possible)
        self.link_other = wp.zeros(max(1, 2 * m), dtype=wp.int32, device=self.device)
        self.link_lambda = wp.zeros(max(1, 2 * m), dtype=wp.float32, device=self.device)
        self.link_target = wp.zeros(max(1, 2 * m), dtype=wp.float32, device=self.device)
        self._dev_dirty = False

    def rebuild(self):
        """Rebuild the per-cell CSR from the current topology and live COMs
        (delete-by-flag + atomic-append). Call at the per-MCS boundary."""
        eng = self.engine
        m = self.n_pairs
        if m == 0:
            self.link_ptr.zero_()
            self._n_active = 0
            return
        if self._dev_dirty:
            self._sync_device_topology()

        self._degree.zero_()
        self._cursor.zero_()
        wp.launch(
            K.fpp_count_active_links_kernel,
            dim=m,
            inputs=[
                self._pa, self._pb, self._pmax,
                eng.xsum, eng.ysum, eng.zsum, eng.volume,
                self._keep, self._degree,
            ],
            device=self.device,
        )
        # exclusive prefix sum of degree -> link_ptr (host; n_cells is tiny and
        # this is off the hot per-flip path -- the Phase 1 approach).
        deg = self._degree.numpy()
        ptr = np.zeros(self.n1 + 1, dtype=np.int32)
        ptr[1:] = np.cumsum(deg[: self.n1])
        self.link_ptr = wp.array(ptr, dtype=wp.int32, device=self.device)
        self._n_active = int(self._keep.numpy().sum())

        self._cursor.zero_()
        wp.launch(
            K.fpp_fill_csr_kernel,
            dim=m,
            inputs=[
                self._pa, self._pb, self._plam, self._ptgt, self._keep,
                self.link_ptr, self._cursor, self.link_other,
                self.link_lambda, self.link_target,
            ],
            device=self.device,
        )
        wp.synchronize()

    # --------------------------------------------------------------- observables
    def active_link_lengths(self) -> np.ndarray:
        """COM-to-COM length of each currently-kept link (host copy).

        Ensures the device keep-flags reflect the CURRENT topology: if the host
        topology was edited (create/delete) since the last rebuild -- e.g. a
        steppable mutated links after the per-MCS ``step_mcs`` rebuild -- ``_keep``
        would be stale (a different length than ``self._a``); rebuild first so the
        kept set + lengths are consistent with the live topology and COMs."""
        if self.n_pairs == 0:
            return np.zeros(0)
        if self._dev_dirty or self._keep is None or self._keep.shape[0] < self.n_pairs:
            self.rebuild()
        coms = np.zeros((self.n1, 3))
        coms[1:] = self.engine.coms()
        keep = self._keep.numpy().astype(bool)[: self.n_pairs]
        a = self._a[keep]
        b = self._b[keep]
        d = coms[a] - coms[b]
        return np.sqrt((d * d).sum(axis=1))
