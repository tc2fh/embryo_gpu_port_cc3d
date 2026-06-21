"""Batched FocalPointPlasticity link inventory (Phase 5, Tier 1).

``BatchedFPPLinks`` is the replica-axis analogue of ``engine.fpp.FPPLinks``: it
holds, for R replicas that SHARE one undirected link topology, a per-replica link
CSR rebuilt each MCS from each replica's own COM. The energy-relevant per-link
parameters -- ``lambda`` / ``target length`` / ``max length`` -- are PER REPLICA, so
a sweep varies the FPP spring stiffness / rest length / break length across the
batch while every replica runs the same network. This is the engine-layer half of
batched FPP; the dynamic create/delete + cohesotaxis dynamics (which differ the
topology per replica) are the steppable-layer Tier 2.

Layout (replica-major, matching ``BatchedGPUEngine``):
  * topology   : ``_a`` / ``_b`` are (M,) shared cell-id pairs.
  * per-link   : ``_lam`` / ``_tgt`` / ``_max`` are (R, M) -> flat (R*M,).
  * link CSR   : ``link_ptr`` is (R*(n1+1),) of per-replica LOCAL offsets;
                 ``link_other`` / ``link_lambda`` / ``link_target`` are
                 (R*pay_stride,) with ``pay_stride = 2*M`` (max directed entries).

The flat slot for an appended directed entry is
``r*pay_stride + link_ptr[r*(n1+1)+cell] + cursor`` -- so the metropolis kernel's
``fpp_delta_cell_b`` reads ``link_ptr[ptr_base+cid]`` (local) then adds ``pay_base``.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from . import kernels as K

wp.init()


class BatchedFPPLinks:
    """Per-replica FPP link CSR over R replicas sharing one undirected topology.

    Plugs into ``BatchedGPUEngine`` via ``engine.attach_fpp(self)``; the engine passes
    ``link_ptr / link_other / link_lambda / link_target`` (+ ``link_pay_stride``) into
    ``metropolis_color_batched_kernel`` and calls ``rebuild()`` once per MCS.
    """

    def __init__(self, engine, target_length_default: float = 5.0,
                 lambda_default: float = 1.0, max_length_default: float = 10.0):
        self.engine = engine
        self.device = engine.device
        self.R = engine.R
        self.n_cells = engine.n_cells
        self.n1 = engine.n1
        self.target_default = float(target_length_default)
        self.lambda_default = float(lambda_default)
        self.max_default = float(max_length_default)

        # PER-REPLICA topology + params, all (R, M) with -1 tombstones in _a/_b for
        # unused / deleted slots. A shared topology (Tier 1) is just every row equal.
        self._a = np.full((self.R, 0), -1, dtype=np.int32)
        self._b = np.full((self.R, 0), -1, dtype=np.int32)
        self._lam = np.zeros((self.R, 0), dtype=np.float32)
        self._tgt = np.zeros((self.R, 0), dtype=np.float32)
        self._max = np.zeros((self.R, 0), dtype=np.float32)

        # device CSR (rebuilt each MCS). Allocated lazily in rebuild().
        self.link_ptr = wp.zeros(self.R * (self.n1 + 1), dtype=wp.int32, device=self.device)
        self.link_other = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.link_lambda = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.link_target = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.link_pay_stride = 1

        self._dev_dirty = True
        self._pa = self._pb = None
        self._plam = self._ptgt = self._pmax = None
        self._keep = self._degree = self._cursor = None

    # ----------------------------------------------------------- topology setup
    def _broadcast(self, v, default, m):
        """Coerce a per-link param to a (R, M) float32 array. Accepts None (-> the
        class default), a scalar, a length-M vector (shared across replicas), or an
        already (R, M) array (the swept case -- different value per replica)."""
        if v is None:
            return np.full((self.R, m), default, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 0:
            return np.full((self.R, m), float(v), dtype=np.float32)
        if v.shape == (m,):
            return np.broadcast_to(v, (self.R, m)).astype(np.float32).copy()
        if v.shape == (self.R, m):
            return v.astype(np.float32).copy()
        raise ValueError(f"per-link param shape {v.shape} != scalar / (M={m},) / (R={self.R}, M={m})")

    def set_topology(self, pairs, lambdas=None, targets=None, maxlens=None):
        """Replace the topology with a SHARED undirected network (Tier 1 sweep case):
        every replica gets the same ``pairs`` (M,2), broadcast to (R, M). Each per-link
        param may be a scalar / (M,) shared / (R,M) per-replica array (or None ->
        default)."""
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        m = pairs.shape[0]
        self._a = np.broadcast_to(pairs[:, 0], (self.R, m)).astype(np.int32).copy()
        self._b = np.broadcast_to(pairs[:, 1], (self.R, m)).astype(np.int32).copy()
        self._lam = self._broadcast(lambdas, self.lambda_default, m)
        self._tgt = self._broadcast(targets, self.target_default, m)
        self._max = self._broadcast(maxlens, self.max_default, m)
        self._dev_dirty = True

    def set_per_replica_pairs(self, pairs_list, lambdas=None, targets=None, maxlens=None):
        """Replace the topology with a DIFFERENT undirected network per replica (Tier 2
        dynamic case). ``pairs_list`` is a length-R list of (m_r, 2) int arrays; rows
        are padded to ``M = max(m_r)`` with -1 tombstones. ``lambdas`` / ``targets`` /
        ``maxlens`` are matching length-R lists of (m_r,) arrays (or None -> default)."""
        R = self.R
        assert len(pairs_list) == R, f"pairs_list len {len(pairs_list)} != R {R}"
        ps = [np.asarray(p, dtype=np.int32).reshape(-1, 2) for p in pairs_list]
        M = max((p.shape[0] for p in ps), default=0)
        a = np.full((R, M), -1, dtype=np.int32)
        b = np.full((R, M), -1, dtype=np.int32)
        lam = np.full((R, M), self.lambda_default, dtype=np.float32)
        tgt = np.full((R, M), self.target_default, dtype=np.float32)
        mx = np.full((R, M), self.max_default, dtype=np.float32)
        for r in range(R):
            m = ps[r].shape[0]
            if m == 0:
                continue
            a[r, :m] = ps[r][:, 0]
            b[r, :m] = ps[r][:, 1]
            if lambdas is not None:
                lam[r, :m] = np.asarray(lambdas[r], dtype=np.float32).reshape(-1)
            if targets is not None:
                tgt[r, :m] = np.asarray(targets[r], dtype=np.float32).reshape(-1)
            if maxlens is not None:
                mx[r, :m] = np.asarray(maxlens[r], dtype=np.float32).reshape(-1)
        self._a, self._b, self._lam, self._tgt, self._max = a, b, lam, tgt, mx
        self._dev_dirty = True

    @property
    def n_pairs(self) -> int:
        """Padded per-replica topology width M (slots, including tombstones)."""
        return int(self._a.shape[1])

    def has_links(self) -> bool:
        return self.n_pairs > 0 and bool((self._a >= 0).any())

    # --------------------------------------------------------------- device sync
    def _sync_device_topology(self):
        m = self.n_pairs
        R = self.R
        # all per-replica arrays flattened replica-major (r*M + i) to match the kernels
        self._pa = wp.array(self._a.reshape(-1), dtype=wp.int32, device=self.device)
        self._pb = wp.array(self._b.reshape(-1), dtype=wp.int32, device=self.device)
        self._plam = wp.array(self._lam.reshape(-1), dtype=wp.float32, device=self.device)
        self._ptgt = wp.array(self._tgt.reshape(-1), dtype=wp.float32, device=self.device)
        self._pmax = wp.array(self._max.reshape(-1), dtype=wp.float32, device=self.device)
        self._keep = wp.zeros(max(1, R * m), dtype=wp.int32, device=self.device)
        self._degree = wp.zeros(R * (self.n1 + 1), dtype=wp.int32, device=self.device)
        self._cursor = wp.zeros(R * (self.n1 + 1), dtype=wp.int32, device=self.device)
        self.link_pay_stride = 2 * m
        n_pay = max(1, R * self.link_pay_stride)
        self.link_other = wp.zeros(n_pay, dtype=wp.int32, device=self.device)
        self.link_lambda = wp.zeros(n_pay, dtype=wp.float32, device=self.device)
        self.link_target = wp.zeros(n_pay, dtype=wp.float32, device=self.device)
        self._dev_dirty = False

    def rebuild(self):
        """Rebuild the per-replica link CSR from the shared topology and each
        replica's live COMs (delete-by-flag + atomic-append). Call per MCS."""
        eng = self.engine
        m = self.n_pairs
        R = self.R
        if m == 0:
            self.link_ptr.zero_()
            return
        if self._dev_dirty:
            self._sync_device_topology()

        self._degree.zero_()
        self._cursor.zero_()
        wp.launch(
            K.fpp_count_active_links_batched_kernel,
            dim=R * m,
            inputs=[
                R, m, self.n1,
                self._pa, self._pb, self._pmax,
                eng.xsum, eng.ysum, eng.zsum, eng.volume,
                self._keep, self._degree,
            ],
            device=self.device,
        )
        wp.synchronize()

        # per-replica exclusive prefix sum of degree -> LOCAL link_ptr (host; the
        # n1 axis is tiny and off the hot per-flip path -- the Phase 1/3 approach).
        deg = self._degree.numpy().reshape(R, self.n1 + 1)
        ptr = np.zeros((R, self.n1 + 1), dtype=np.int32)
        ptr[:, 1:] = np.cumsum(deg[:, : self.n1], axis=1)
        self.link_ptr = wp.array(ptr.reshape(-1), dtype=wp.int32, device=self.device)

        self._cursor.zero_()
        wp.launch(
            K.fpp_fill_csr_batched_kernel,
            dim=R * m,
            inputs=[
                R, m, self.n1, self.link_pay_stride,
                self._pa, self._pb, self._plam, self._ptgt, self._keep,
                self.link_ptr, self._cursor,
                self.link_other, self.link_lambda, self.link_target,
            ],
            device=self.device,
        )
        wp.synchronize()

    def num_active(self) -> np.ndarray:
        """(R,) number of links kept (within max_length) at the last rebuild."""
        if self._keep is None:
            return np.zeros(self.R, dtype=np.int64)
        return self._keep.numpy().reshape(self.R, self.n_pairs).sum(axis=1)
