"""FocalPointPlasticity device-authoritative link inventory (Phase 3 core; Phase 6
moves the inventory storage + edits onto the GPU).

CC3D's FocalPointPlasticity holds, per cell, a set of spring **links** to other
cells, each created by ``new_fpp_link(cell_a, cell_b, lambda, targetDist, maxDist)``
and carrying its own ``lambda`` / ``target length`` / ``max length``. A link's
energy is ``lambda*(L - target)^2`` with ``L = || COM_a - COM_b ||_2``; CC3D drops
a link once the cell COMs separate beyond ``maxDistance``.

``FPPLinks`` mirrors that on the GPU with the **proven Phase 1 pattern**
(``gpu_port/phase1/cpm_gpu.py``): a device-authoritative per-cell **link CSR**
built each MCS by

  1. a flag/degree-count kernel that keeps links whose current COM length <= their
     own max_length (this is the *delete* -- links that grew too long are dropped),
  2. a DEVICE exclusive prefix-sum of the per-cell degree -> ``link_ptr`` (Phase 6;
     was a host ``np.cumsum`` + per-step ``wp.array`` realloc), and
  3. an atomic-append fill kernel that writes each kept undirected link into BOTH
     endpoints' CSR ranges, carrying its per-link lambda/target (the *create*).

Phase 6 also makes the **link inventory itself** device-authoritative: the
``a/b/lam/tgt/max`` arrays live on the GPU (``_a_dev`` ...); ``create_links_bulk``
appends a contiguous device block (order-preserving), ``delete_links_bulk`` and the
new ``compact_with_keep_mask`` run a device keep/compact driven by a decision mask,
and ``rebuild()`` reads the device inventory directly -- no host topology roundtrip.
The legacy ``_a/_b/_lam/_tgt/_max`` names remain as **read-only properties** that
return the host (numpy) view of the device inventory, so every existing consumer
(``embryo`` steppables, the validation tests) that reads them is unchanged.

Links are **static within a color sweep**: create/delete happen at the per-MCS
steppable boundary (the engine ``recompute_trackers`` seam), never inside the
Metropolis inner loop, so the kernel can read the CSR race-free. The energy
kernel (``kernels.fpp_delta_cell``) reads link length from the engine int64
``xsum/ysum/zsum / volume`` COM directly -- the exact, reproducible single source
of truth (no separate COM tracker, no float drift).

Topology mutation API (the steppable boundary):
  * ``create_link(a, b, ...)`` / ``delete_link(a, b)`` edit the topology.
  * ``create_links_bulk`` / ``delete_links_bulk`` -- batched edits (one device op).
  * ``compact_with_keep_mask(keep_dev)`` -- device-mask-driven keep/compact (the
    Poisson-delete seam Phase 7 wires in directly from device).
  * ``set_topology(pairs, lambdas, targets, maxlens)`` sets it wholesale.
  * ``rebuild()`` rebuilds the device CSR from the current inventory + live COMs.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from . import kernels as K
from . import scan as S

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

        # --- device-authoritative link inventory (Phase 6) ---
        # ``_a_dev/_b_dev`` are undirected endpoints, ``_lam/_tgt/_max_dev`` the
        # per-link params. ``_m`` live links occupy ``[0:_m]`` of capacity ``_cap``;
        # create appends a contiguous block (order-preserving), delete compacts.
        self._m = 0
        self._cap = 0
        self._a_dev = wp.zeros(0, dtype=wp.int32, device=self.device)
        self._b_dev = wp.zeros(0, dtype=wp.int32, device=self.device)
        self._lam_dev = wp.zeros(0, dtype=wp.float32, device=self.device)
        self._tgt_dev = wp.zeros(0, dtype=wp.float32, device=self.device)
        self._max_dev = wp.zeros(0, dtype=wp.float32, device=self.device)

        # set True by every inventory mutation, cleared by rebuild(): tells
        # active_link_lengths() the cached _keep flags no longer match the live
        # inventory (the old _dev_dirty semantics) so it rebuilds first.
        self._csr_stale = True

        # host mirror cache for the read-only _a/_b/... properties. PER-ARRAY lazy
        # sync: the hot steppable path reads only _a/_b each MCS, so reading _a copies
        # back just a/b (not lam/tgt/max). Each entry is (cached_array, dirty_flag).
        self._host_cache = {
            "a": (np.zeros(0, dtype=np.int32), True),
            "b": (np.zeros(0, dtype=np.int32), True),
            "lam": (np.zeros(0, dtype=np.float32), True),
            "tgt": (np.zeros(0, dtype=np.float32), True),
            "max": (np.zeros(0, dtype=np.float32), True),
        }

        # device CSR (rebuilt each MCS). ``link_ptr`` is a stable reused buffer
        # (Phase 6: was realloc'd each step); payload arrays grow with the inventory.
        self.link_ptr = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        self.link_other = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.link_lambda = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.link_target = wp.zeros(1, dtype=wp.float32, device=self.device)

        # device CSR scratch (sized to the inventory; grown when it grows)
        self._csr_cap = 0
        self._keep = None
        self._degree = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        self._cursor = wp.zeros(self.n1 + 1, dtype=wp.int32, device=self.device)
        self._n_active = 0

        # compaction scratch (the keep/compact primitive)
        self._compact_pos = None      # exclusive prefix-sum of the keep mask
        self._scratch_a = self._scratch_b = None
        self._scratch_lam = self._scratch_tgt = self._scratch_max = None

    # ------------------------------------------------------- device capacity mgmt
    def _ensure_capacity(self, need: int):
        """Grow the device inventory arrays to hold at least ``need`` links,
        preserving the live ``[0:_m]`` contents. Geometric growth amortizes the
        realloc over a sequence of appends."""
        if need <= self._cap:
            return
        new_cap = max(need, max(1, self._cap) * 2, 8)

        def grow_i32(old):
            new = wp.zeros(new_cap, dtype=wp.int32, device=self.device)
            if self._m > 0:
                wp.copy(new[0:self._m], old[0:self._m])
            return new

        def grow_f32(old):
            new = wp.zeros(new_cap, dtype=wp.float32, device=self.device)
            if self._m > 0:
                wp.copy(new[0:self._m], old[0:self._m])
            return new

        self._a_dev = grow_i32(self._a_dev)
        self._b_dev = grow_i32(self._b_dev)
        self._lam_dev = grow_f32(self._lam_dev)
        self._tgt_dev = grow_f32(self._tgt_dev)
        self._max_dev = grow_f32(self._max_dev)
        self._cap = new_cap
        # invalidate compaction/CSR scratch tied to the old capacity
        self._compact_pos = None
        self._scratch_a = None
        self._keep = None
        self._csr_cap = 0

    def _ensure_csr_scratch(self):
        """Size the per-link CSR scratch (keep flags + payload) to the inventory."""
        m = self._m
        if self._keep is None or self._csr_cap < m:
            cap = max(1, m)
            self._keep = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self.link_other = wp.zeros(max(1, 2 * cap), dtype=wp.int32, device=self.device)
            self.link_lambda = wp.zeros(max(1, 2 * cap), dtype=wp.float32, device=self.device)
            self.link_target = wp.zeros(max(1, 2 * cap), dtype=wp.float32, device=self.device)
            self._csr_cap = cap

    def _invalidate_host(self):
        for k, (arr, _) in self._host_cache.items():
            self._host_cache[k] = (arr, True)
        self._csr_stale = True

    def _host_view(self, name: str, dev, empty_dtype) -> np.ndarray:
        """Lazily copy back one inventory column from the device (cached until the
        next edit). Reading the hot _a/_b each MCS copies only those columns."""
        arr, dirty = self._host_cache[name]
        if not dirty:
            return arr
        m = self._m
        arr = (np.zeros(0, dtype=empty_dtype) if m == 0 else dev[:m].numpy().copy())
        self._host_cache[name] = (arr, False)
        return arr

    # ---- read-only host views of the device inventory (legacy attribute names) --
    @property
    def _a(self) -> np.ndarray:
        return self._host_view("a", self._a_dev, np.int32)

    @property
    def _b(self) -> np.ndarray:
        return self._host_view("b", self._b_dev, np.int32)

    @property
    def _lam(self) -> np.ndarray:
        return self._host_view("lam", self._lam_dev, np.float32)

    @property
    def _tgt(self) -> np.ndarray:
        return self._host_view("tgt", self._tgt_dev, np.float32)

    @property
    def _max(self) -> np.ndarray:
        return self._host_view("max", self._max_dev, np.float32)

    # ----------------------------------------------------------- topology edits
    def set_topology(self, pairs, lambdas=None, targets=None, maxlens=None):
        """Replace the entire link inventory. ``pairs`` is (M,2) int (a,b)."""
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        m = pairs.shape[0]
        a = pairs[:, 0].astype(np.int32).copy()
        b = pairs[:, 1].astype(np.int32).copy()
        lam = (np.full(m, self.lambda_default, np.float32) if lambdas is None
               else np.asarray(lambdas, np.float32).reshape(-1).copy())
        tgt = (np.full(m, self.target_default, np.float32) if targets is None
               else np.asarray(targets, np.float32).reshape(-1).copy())
        mx = (np.full(m, self.max_default, np.float32) if maxlens is None
              else np.asarray(maxlens, np.float32).reshape(-1).copy())
        self._m = 0
        self._ensure_capacity(m)
        if m > 0:
            wp.copy(self._a_dev[0:m], wp.array(a, dtype=wp.int32, device=self.device))
            wp.copy(self._b_dev[0:m], wp.array(b, dtype=wp.int32, device=self.device))
            wp.copy(self._lam_dev[0:m], wp.array(lam, dtype=wp.float32, device=self.device))
            wp.copy(self._tgt_dev[0:m], wp.array(tgt, dtype=wp.float32, device=self.device))
            wp.copy(self._max_dev[0:m], wp.array(mx, dtype=wp.float32, device=self.device))
        self._m = m
        self._invalidate_host()

    def set_max_lengths(self, maxlens):
        mx = np.asarray(maxlens, np.float32).reshape(-1).copy()
        if mx.shape[0] != self._m:
            raise ValueError(
                f"set_max_lengths: got {mx.shape[0]} values for {self._m} links")
        if self._m > 0:
            wp.copy(self._max_dev[0:self._m],
                    wp.array(mx, dtype=wp.float32, device=self.device))
        self._invalidate_host()

    def create_links_bulk(self, a, b, lam=None, target=None, maxlen=None):
        """Append MANY undirected links in a single device block write.

        ``a`` / ``b`` are length-k id arrays. Each per-link param accepts a scalar
        (broadcast to all k -- the common case, one lambda/target/max for the whole
        batch), a length-k array, or None (-> the class default). Links are appended
        in the given order into a contiguous device block ``[_m, _m+k)`` -- the order
        is preserved, so a sequence of ``create_link``/``create_links_bulk`` calls
        yields the same inventory layout as the equivalent single calls. Effective at
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

        lam = _col(lam, self.lambda_default)
        tgt = _col(target, self.target_default)
        mx = _col(maxlen, self.max_default)

        base = self._m
        self._ensure_capacity(base + k)
        # contiguous block append (atomic in the sense of one reserved [base,base+k)
        # region; each batch element lands at a deterministic offset -> stable order)
        wp.copy(self._a_dev[base:base + k], wp.array(a, dtype=wp.int32, device=self.device))
        wp.copy(self._b_dev[base:base + k], wp.array(b, dtype=wp.int32, device=self.device))
        wp.copy(self._lam_dev[base:base + k], wp.array(lam, dtype=wp.float32, device=self.device))
        wp.copy(self._tgt_dev[base:base + k], wp.array(tgt, dtype=wp.float32, device=self.device))
        wp.copy(self._max_dev[base:base + k], wp.array(mx, dtype=wp.float32, device=self.device))
        self._m = base + k
        self._invalidate_host()

    def delete_links_bulk(self, pairs):
        """Remove EVERY link whose unordered endpoints match any {a,b} in ``pairs``,
        in one device keep/compact pass over the inventory.

        The unordered packed delete keys are computed on host (they come from a host
        pair-list), uploaded once, and a device kernel marks each inventory slot
        keep/drop via binary search; the device keep/compact then rewrites the
        survivors in their original order. ``pairs`` is an iterable of (a, b)."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.shape[0] == 0 or self._m == 0:
            return
        mult = np.int64(self.n_cells + 2)             # pack {lo,hi} -> one int64 key
        dkeys = np.unique(np.minimum(pairs[:, 0], pairs[:, 1]) * mult
                          + np.maximum(pairs[:, 0], pairs[:, 1]))
        m = self._m
        del_dev = wp.array(dkeys.astype(np.int64), dtype=wp.int64, device=self.device)
        self._ensure_csr_scratch()
        keep = self._keep
        keep.zero_()
        wp.launch(
            K.fpp_mark_keep_by_key_kernel,
            dim=m,
            inputs=[self._a_dev, self._b_dev, m, wp.int64(int(mult)),
                    del_dev, int(dkeys.shape[0]), keep],
            device=self.device,
        )
        self._compact_in_place(keep, m)

    def compact_with_keep_mask(self, keep_dev):
        """Device-mask-driven keep/compact (the Phase-7 Poisson-delete seam).

        ``keep_dev`` is a device ``wp.array(int32)`` of length >= current ``n_pairs``;
        slot i with ``keep_dev[i]==1`` survives, ``==0`` is dropped. Survivors are
        rewritten in their original (insertion) order via a stable scatter -> the
        downstream CSR offsets stay deterministic. Returns the new link count."""
        m = self._m
        if m == 0:
            return 0
        if keep_dev.shape[0] < m:
            raise ValueError(
                f"compact_with_keep_mask: mask len {keep_dev.shape[0]} < n_pairs {m}")
        self._compact_in_place(keep_dev, m)
        return self._m

    def _compact_in_place(self, keep, m):
        """Stable device compaction: dst = exclusive-scan(keep); scatter survivors
        into scratch; swap scratch in as the live inventory. Updates ``_m``."""
        if self._compact_pos is None or self._compact_pos.shape[0] < m:
            self._compact_pos = wp.zeros(max(1, self._cap), dtype=wp.int32, device=self.device)
            self._scratch_a = wp.zeros(max(1, self._cap), dtype=wp.int32, device=self.device)
            self._scratch_b = wp.zeros(max(1, self._cap), dtype=wp.int32, device=self.device)
            self._scratch_lam = wp.zeros(max(1, self._cap), dtype=wp.float32, device=self.device)
            self._scratch_tgt = wp.zeros(max(1, self._cap), dtype=wp.float32, device=self.device)
            self._scratch_max = wp.zeros(max(1, self._cap), dtype=wp.float32, device=self.device)
        pos = self._compact_pos
        # exclusive prefix-sum of keep -> each survivor's compacted destination index
        wp.utils.array_scan(keep[0:m], pos[0:m], False)
        wp.launch(
            K.fpp_compact_links_kernel,
            dim=m,
            inputs=[keep, pos, m,
                    self._a_dev, self._b_dev, self._lam_dev, self._tgt_dev, self._max_dev,
                    self._scratch_a, self._scratch_b, self._scratch_lam,
                    self._scratch_tgt, self._scratch_max],
            device=self.device,
        )
        # new live count = total kept (host read of one scan total; tiny + at the
        # steppable boundary, not the hot per-flip path)
        wp.synchronize()
        keep_np = keep[:m].numpy()
        n_keep = int(keep_np.sum())
        # swap scratch in as the authoritative inventory (the scratch becomes live;
        # the old arrays become next round's scratch -- no extra copy back)
        self._a_dev, self._scratch_a = self._scratch_a, self._a_dev
        self._b_dev, self._scratch_b = self._scratch_b, self._b_dev
        self._lam_dev, self._scratch_lam = self._scratch_lam, self._lam_dev
        self._tgt_dev, self._scratch_tgt = self._scratch_tgt, self._tgt_dev
        self._max_dev, self._scratch_max = self._scratch_max, self._max_dev
        self._m = n_keep
        self._invalidate_host()

    def create_link(self, a: int, b: int, lam=None, target=None, maxlen=None):
        """Add ONE undirected link a-b (no dedup; CC3D guards dups at the call site).
        Effective at the next ``rebuild()``. Hot per-MCS loops should batch their
        edits through ``create_links_bulk`` (one device block instead of one each)."""
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
        return int(self._m)

    def has_links(self) -> bool:
        return self._m > 0

    def num_active(self) -> int:
        """Number of links kept (within max_length) at the last rebuild."""
        return int(self._n_active)

    # --------------------------------------------------------------------- rebuild
    def rebuild(self):
        """Rebuild the per-cell CSR from the current inventory and live COMs
        (delete-by-flag + device scan + atomic-append), fully on-device. Call at the
        per-MCS boundary."""
        eng = self.engine
        m = self._m
        if m == 0:
            self.link_ptr.zero_()
            self._n_active = 0
            self._csr_stale = False
            return
        self._ensure_csr_scratch()

        self._degree.zero_()
        self._cursor.zero_()
        wp.launch(
            K.fpp_count_active_links_kernel,
            dim=m,
            inputs=[
                self._a_dev, self._b_dev, self._max_dev,
                eng.xsum, eng.ysum, eng.zsum, eng.volume,
                self._keep, self._degree,
            ],
            device=self.device,
        )
        # exclusive prefix sum of degree -> link_ptr, ON DEVICE (Phase 6: replaces
        # the host np.cumsum + per-step wp.array realloc). link_ptr is a stable
        # reused buffer; the device scan is byte-identical to the prior cumsum.
        self.link_ptr.zero_()
        S.exclusive_scan_to_ptr_i32(self._degree, self.n1, self.link_ptr, self.device)
        # number kept (host reduction of the keep flags; tiny, off the hot path)
        wp.synchronize()
        self._n_active = int(self._keep[:m].numpy().sum())

        self._cursor.zero_()
        wp.launch(
            K.fpp_fill_csr_kernel,
            dim=m,
            inputs=[
                self._a_dev, self._b_dev, self._lam_dev, self._tgt_dev, self._keep,
                self.link_ptr, self._cursor, self.link_other,
                self.link_lambda, self.link_target,
            ],
            device=self.device,
        )
        wp.synchronize()
        self._csr_stale = False

    # --------------------------------------------------------------- observables
    def active_link_lengths(self) -> np.ndarray:
        """COM-to-COM length of each currently-kept link (host copy).

        Ensures the device keep-flags reflect the CURRENT inventory: if it was edited
        (create/delete) since the last rebuild -- e.g. a steppable mutated links after
        the per-MCS ``step_mcs`` rebuild -- ``_keep`` would be stale (a different
        length than the inventory); rebuild first so the kept set + lengths are
        consistent with the live inventory and COMs."""
        if self._m == 0:
            return np.zeros(0)
        if self._csr_stale or self._keep is None or self._keep.shape[0] < self._m:
            self.rebuild()
        coms = np.zeros((self.n1, 3))
        coms[1:] = self.engine.coms()
        keep = self._keep[:self._m].numpy().astype(bool)
        a = self._a[keep]
        b = self._b[keep]
        d = coms[a] - coms[b]
        return np.sqrt((d * d).sum(axis=1))
