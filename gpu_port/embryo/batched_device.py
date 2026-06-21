"""Device-backed batched Embryo driver (Phase 8, Objective 2).

``BatchedDeviceEmbryoModel`` collapses the per-replica HOST loops of
``batched_steppables.BatchedEmbryoModel`` (the pre-phase O(R)-host path) into batched
device launches + per-replica device-kernel calls that reuse Phase 7's device link
kernels with the SAME keying as the single ``EmbryoModel``. The result is BIT-IDENTICAL
per replica to R independent single ``EmbryoModel.run``s (Volume + Contact + FPP link
inventory), and removes all per-MCS host data movement (the contact-graph copyback +
the Python adjacency dicts the host path paid each replica each MCS).

How the collapse is structured:

  * **One batched keyed-global radix-sort CSR** (``BatchedGPUEngine.publish_neighbor_
    csr_device``): all R replicas' contact CSR in a single sort (replica id in the
    key's high bits -> disjoint per-replica ranges) + one global scan, published as
    device handles. No per-replica compaction loop, no host copyback.
  * **Per-replica device link inventory**: each replica gets its OWN device-authoritative
    ``FPPLinks`` bound to a lightweight ``_ReplicaView`` that points the Phase-7 device
    kernels at replica r's slice of the global CSR (``csr_indptr_dev[r*n1:]`` +
    the global ``csr_indices_dev``) and replica r's COM/volume slice. The tissue
    cap-ordered relink, substrate min-id create, and the Poisson keep-mask/compact run
    EXACTLY as in the single engine -> identical keying -> bit-identical inventory.
  * **One batched fused cohesotaxis** (``BatchedFusedCohesotaxisPipeline``): a single
    classify launch over R*nvox, one global key-sort, one select launch over R*n_lead,
    keyed per (r, cell) by ``base_seed_r[r]`` -> the per-replica lamellipodia targets,
    bit-identical to R single fused pipelines.
  * **Device combine**: each replica's per-kind device inventory blocks are concatenated
    on device into the padded (R, M) batched FPP CSR the engine's spring energy reads;
    no per-replica Python emit/concat/upload.

The batched FPP spring energy reads the per-sweep COM SNAPSHOT (Phase 8), so the
batched FPP-driven trajectory is bit-reproducible per replica.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from engine.batched import BatchedGPUEngine
from engine.fpp import FPPLinks
from engine.batched_fpp import BatchedFPPLinks
from engine.cohesotaxis_fused import BatchedFusedCohesotaxisPipeline
from engine import cohesotaxis as CT
from engine import link_kernels as LK
from engine import batched_link_kernels as BLK
from engine.geometry import LEADING, PASSIVE, SUBSTRATE

from .params import EmbryoParams, DEFAULT
from .steppables import _STREAM_TISSUE, _STREAM_SUBLINK

wp.init()


def _next_pow2(n: int) -> int:
    cap = 1
    while cap < n:
        cap <<= 1
    return cap


class BatchedDeviceFPP:
    """Per-replica device-authoritative link inventory over R replicas (Phase 8). Holds
    ``a/b/lam/tgt/max`` as flat ``(R*cap)`` blocks + a per-replica live count ``m[R]`` on
    device, and runs the batched link kernels (tissue cap-relink, substrate min-id,
    lamellipodia create, Poisson keep-mask + compact) in SINGLE launches over R -- no
    per-replica host sync. Feeds the engine via an inner ``BatchedFPPLinks`` whose padded
    (R,M) topology is set directly from these device blocks (the device combine)."""

    def __init__(self, engine: BatchedGPUEngine, params: EmbryoParams):
        self.engine = engine
        self.device = engine.device
        self.R = engine.R
        self.n1 = engine.n1
        self.p = params
        self._cap = 0
        self._a = self._b = self._lam = self._tgt = self._max = None
        self.m = wp.zeros(self.R, dtype=wp.int32, device=self.device)   # per-replica live count
        # scratch
        self._degree = wp.zeros(self.R * self.n1, dtype=wp.int32, device=self.device)
        self._has = wp.zeros(self.R * self.n1, dtype=wp.int32, device=self.device)
        self._keep = None
        self._table = None
        self._tab_cap = 0
        self._emit_a = self._emit_b = None
        self._emit_cap = 0
        # inner BatchedFPPLinks (per-replica CSR rebuild + the spring energy seam)
        self._bfpp = BatchedFPPLinks(
            engine, target_length_default=params.tissue_target,
            lambda_default=params.tissue_lambda, max_length_default=params.tissue_max)

    def _ensure_cap(self, need_per_replica: int):
        if self._cap >= need_per_replica and self._a is not None:
            return
        cap = max(8, int(need_per_replica))
        R = self.R
        def grow_i(old):
            new = wp.full(R * cap, -1, dtype=wp.int32, device=self.device)
            return new
        def grow_f(old):
            return wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        # preserve existing live contents per replica
        old_cap = self._cap
        na = wp.full(R * cap, -1, dtype=wp.int32, device=self.device)
        nb = wp.full(R * cap, -1, dtype=wp.int32, device=self.device)
        nl = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        nt = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        nm = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        if self._a is not None and old_cap > 0:
            for r in range(R):
                wp.copy(na[r * cap: r * cap + old_cap], self._a[r * old_cap:(r + 1) * old_cap])
                wp.copy(nb[r * cap: r * cap + old_cap], self._b[r * old_cap:(r + 1) * old_cap])
                wp.copy(nl[r * cap: r * cap + old_cap], self._lam[r * old_cap:(r + 1) * old_cap])
                wp.copy(nt[r * cap: r * cap + old_cap], self._tgt[r * old_cap:(r + 1) * old_cap])
                wp.copy(nm[r * cap: r * cap + old_cap], self._max[r * old_cap:(r + 1) * old_cap])
        self._a, self._b, self._lam, self._tgt, self._max = na, nb, nl, nt, nm
        self._cap = cap
        self._keep = wp.zeros(R * cap, dtype=wp.int32, device=self.device)

    def _ensure_table(self, need_entries):
        cap = _next_pow2(max(8, int(need_entries) * 2))
        if self._tab_cap < cap:
            self._table = wp.zeros(self.R * cap, dtype=wp.int64, device=self.device)
            self._tab_cap = cap
        return self._table, self._tab_cap

    def _ensure_emit(self, n_passive):
        need = self.R * max(1, int(n_passive))
        if self._emit_cap < need:
            self._emit_a = wp.zeros(need, dtype=wp.int32, device=self.device)
            self._emit_b = wp.zeros(need, dtype=wp.int32, device=self.device)
            self._emit_cap = need

    @property
    def mult(self):
        return wp.int64(self.n1 + 1)

    def max_m(self):
        return int(self.m.numpy().max()) if self._cap else 0

    # ----------------------------------------------------------------- ops
    def tissue_relink(self, managed_dev, n_managed, cap_deg, substrate_type,
                      n_contacts_total):
        R, n1 = self.R, self.n1
        # headroom: current max live + a bound on this pass's claims (<= contacts)
        max_new = max(1, int(n_contacts_total))
        self._ensure_cap(self.max_m() + max_new)
        self._degree.zero_()
        wp.launch(BLK.degree_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, R, self._cap, n1, self.m, self._degree],
                  device=self.device)
        table, tab_cap = self._ensure_table(self.max_m() + max_new)
        table.fill_(wp.int64(-1))
        wp.launch(BLK.hash_fill_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, R, self._cap, self.m, self.mult, tab_cap, table],
                  device=self.device)
        wp.launch(
            BLK.tissue_relink_batched_kernel, dim=R,
            inputs=[R, self._cap, n1, managed_dev, int(n_managed), int(cap_deg),
                    int(substrate_type), self.engine.cell_type,
                    self.engine.csr_indptr_dev, self.engine.csr_indices_dev,
                    self._degree, table, tab_cap, self.mult,
                    self._a, self._b, self._lam, self._tgt, self._max, self.m,
                    float(self.p.tissue_lambda), float(self.p.tissue_target),
                    float(self.p.tissue_max)],
            device=self.device)

    def substrate_create(self, passive_dev, n_passive, substrate_type):
        R, n1 = self.R, self.n1
        self._ensure_cap(self.max_m() + int(n_passive))
        self._has.zero_()
        wp.launch(BLK.substrate_has_link_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, self._lam, R, self._cap, n1, self.m,
                          float(self.p.slink_lambda), self.engine.cell_type,
                          int(substrate_type), self._has],
                  device=self.device)
        self._ensure_emit(n_passive)
        wp.launch(BLK.substrate_emit_batched_kernel, dim=R * int(n_passive),
                  inputs=[R, n1, passive_dev, int(n_passive), self._has,
                          self.engine.cell_type, int(substrate_type),
                          self.engine.csr_indptr_dev, self.engine.csr_indices_dev,
                          self._emit_a, self._emit_b],
                  device=self.device)
        wp.launch(BLK.substrate_append_batched_kernel, dim=R,
                  inputs=[R, self._cap, int(n_passive), self._emit_a, self._emit_b,
                          self._a, self._b, self._lam, self._tgt, self._max, self.m,
                          float(self.p.slink_lambda), float(self.p.slink_target),
                          float(self.p.slink_max)],
                  device=self.device)

    def lam_has_link(self, kind_lambda, substrate_type):
        R, n1 = self.R, self.n1
        self._has.zero_()
        wp.launch(BLK.lam_has_link_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, self._lam, R, self._cap, n1, self.m,
                          float(kind_lambda), self.engine.cell_type, int(substrate_type),
                          self._has],
                  device=self.device)
        return self._has

    def lamellipodia_create(self, targets_dev, lead_ids_dev, n_lead, kind_lambda,
                            substrate_type, only_need):
        R, n1 = self.R, self.n1
        self._ensure_cap(self.max_m() + int(n_lead))
        if only_need:
            self.lam_has_link(kind_lambda, substrate_type)
        else:
            self._has.zero_()
        wp.launch(
            BLK.lamellipodia_append_batched_kernel, dim=R,
            inputs=[R, self._cap, n1, int(n_lead), targets_dev, lead_ids_dev,
                    self._lam, self._a, self._b, self._lam, self._tgt, self._max, self.m,
                    self._has, 1 if only_need else 0,
                    float(kind_lambda), float(CT.LL_TARGET_DIST), float(CT.LL_MAX_DIST)],
            device=self.device)

    def poisson_delete(self, kind_lambda, prob_dev, mcs, stream):
        R = self.R
        if self._cap == 0:
            return
        self._keep.zero_()
        wp.launch(BLK.poisson_keep_mask_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, self._lam, R, self._cap, self.m,
                          float(kind_lambda), prob_dev, int(mcs),
                          self.engine.base_seed_r, int(stream), self._keep],
                  device=self.device)
        wp.launch(BLK.compact_batched_kernel, dim=R,
                  inputs=[R, self._cap, self._keep, self.m,
                          self._a, self._b, self._lam, self._tgt, self._max],
                  device=self.device)

    def poisson_delete_by_cell(self, kind_lambda, prob_dev, mcs, substrate_type):
        R = self.R
        if self._cap == 0:
            return
        self._keep.zero_()
        wp.launch(BLK.poisson_keep_mask_by_cell_batched_kernel, dim=R * self._cap,
                  inputs=[self._a, self._b, self._lam, R, self._cap, self.m,
                          float(kind_lambda), self.engine.cell_type, int(substrate_type),
                          prob_dev, int(mcs), self.engine.base_seed_r, self._keep],
                  device=self.device)
        wp.launch(BLK.compact_batched_kernel, dim=R,
                  inputs=[R, self._cap, self._keep, self.m,
                          self._a, self._b, self._lam, self._tgt, self._max],
                  device=self.device)

    # ----------------------------------------------------- combine -> engine seam
    def build_csr_and_attach(self):
        """Hand the per-replica device blocks to the inner BatchedFPPLinks as the padded
        (R, M) topology (M = max live count) and attach to the engine. No host concat."""
        M = self.max_m()
        if M == 0:
            self._bfpp._a = np.full((self.R, 0), -1, dtype=np.int32)
            self._bfpp._b = np.full((self.R, 0), -1, dtype=np.int32)
            self._bfpp._lam = np.zeros((self.R, 0), dtype=np.float32)
            self._bfpp._tgt = np.zeros((self.R, 0), dtype=np.float32)
            self._bfpp._max = np.zeros((self.R, 0), dtype=np.float32)
            self._bfpp._dev_dirty = True
            self._bfpp._device_topology = False
            self.engine.attach_fpp(self._bfpp)
            return
        # the inner BatchedFPPLinks expects flat (R*M) replica-major; our blocks are
        # (R*cap) with cap >= M. Gather the live [0:M] of each replica into a packed
        # (R*M) view via per-replica device copies (cheap; R small device->device copies).
        R, cap = self.R, self._cap
        if getattr(self, "_packed_cap", 0) < M:
            self._pa = wp.full(R * M, -1, dtype=wp.int32, device=self.device)
            self._pb = wp.full(R * M, -1, dtype=wp.int32, device=self.device)
            self._pl = wp.zeros(R * M, dtype=wp.float32, device=self.device)
            self._pt = wp.zeros(R * M, dtype=wp.float32, device=self.device)
            self._pm = wp.zeros(R * M, dtype=wp.float32, device=self.device)
            self._packed_cap = M
        # ALWAYS reset to tombstones: each replica's block may have only m[r] (< M) live
        # links and STALE (non-tombstone) data in [m[r]:cap) left by a prior compact.
        # Copying only the live [0:m[r]] keeps the padding [m[r]:M] tombstoned (else the
        # stale slots become phantom links in the FPP energy).
        self._pa.fill_(wp.int32(-1)); self._pb.fill_(wp.int32(-1))
        ms = self.m.numpy()
        for r in range(R):
            mr = int(ms[r])
            if mr <= 0:
                continue
            wp.copy(self._pa[r * M: r * M + mr], self._a[r * cap: r * cap + mr])
            wp.copy(self._pb[r * M: r * M + mr], self._b[r * cap: r * cap + mr])
            wp.copy(self._pl[r * M: r * M + mr], self._lam[r * cap: r * cap + mr])
            wp.copy(self._pt[r * M: r * M + mr], self._tgt[r * cap: r * cap + mr])
            wp.copy(self._pm[r * M: r * M + mr], self._max[r * cap: r * cap + mr])
        self._bfpp._set_device_topology(self._pa, self._pb, self._pl, self._pt, self._pm, M)
        self.engine.attach_fpp(self._bfpp)

    def link_arrays(self):
        """(host) per-replica (a, b, lam) of the live links, for tests."""
        ms = self.m.numpy()
        a = self._a.numpy(); b = self._b.numpy(); lam = self._lam.numpy()
        out = []
        for r in range(self.R):
            mr = int(ms[r])
            out.append((a[r * self._cap: r * self._cap + mr].copy(),
                        b[r * self._cap: r * self._cap + mr].copy(),
                        lam[r * self._cap: r * self._cap + mr].copy()))
        return out


class _ReplicaView:
    """Single-engine facade over a ``BatchedGPUEngine`` replica ``r`` exposing exactly
    the surface ``FPPLinks`` + the Phase-7 device link kernels read: the shared
    ``cell_type``, the replica's COM/volume DEVICE SLICES (alias the live batched
    arrays), and replica r's slice of the batched global neighbor CSR (the global
    indptr sliced from ``r*n1`` so ``indptr[c]`` is the global offset, paired with the
    full global indices array -> ``indices[indptr[c]:indptr[c+1]]`` is r's cell-c row).
    """

    def __init__(self, engine: BatchedGPUEngine, r: int):
        n1, nvox = engine.n1, engine.nvox
        self._engine = engine
        self.r = r
        self.device = engine.device
        self.cell_type = engine.cell_type          # shared across replicas
        self.n_cells = engine.n_cells
        self.Lx, self.Ly, self.Lz = engine.Lx, engine.Ly, engine.Lz
        self.cfg = engine.cfg
        self.base_seed = int(engine._seeds_np[r])
        self.ids = engine.ids[r * nvox:(r + 1) * nvox]
        self.xsum = engine.xsum[r * n1:(r + 1) * n1]
        self.ysum = engine.ysum[r * n1:(r + 1) * n1]
        self.zsum = engine.zsum[r * n1:(r + 1) * n1]
        self.volume = engine.volume[r * n1:(r + 1) * n1]
        # neighbor CSR handles refreshed each MCS by _refresh_csr()
        self.neighbor_csr_indptr_dev = None
        self.neighbor_csr_indices_dev = None
        self.neighbor_csr_n_contacts = 0

    def coms(self) -> np.ndarray:
        vol = self.volume.numpy().astype(np.float64)
        v = np.where(vol > 0, vol, 1.0)
        cx = self.xsum.numpy().astype(np.float64) / v
        cy = self.ysum.numpy().astype(np.float64) / v
        cz = self.zsum.numpy().astype(np.float64) / v
        return np.stack([cx, cy, cz], axis=1)[1:]

    def refresh_csr(self):
        """Point this view's neighbor-CSR handles at replica r's slice of the engine's
        batched global CSR (published once per MCS by the engine). The global indptr
        slice keeps the global offsets, so it pairs with the full global indices."""
        eng = self._engine
        n1 = eng.n1
        r = self.r
        # slice the global indptr from r*n1 (so indptr[c] == global offset of (r,c));
        # length n1+1 covers rows [r*n1, r*n1+n1]. The last entry is the start of the
        # next replica's block == end of this replica's last row.
        self.neighbor_csr_indptr_dev = eng.csr_indptr_dev[r * n1: r * n1 + n1 + 1]
        self.neighbor_csr_indices_dev = eng.csr_indices_dev
        # per-replica contact count (for the relink claim upper bound) = block size
        ip = eng.csr_indptr_dev
        # read just two scalars (cheap; off the hot Metropolis path)
        lo = int(ip[r * n1: r * n1 + 1].numpy()[0])
        hi = int(ip[r * n1 + n1: r * n1 + n1 + 1].numpy()[0])
        self.neighbor_csr_n_contacts = hi - lo


class BatchedDeviceEmbryoModel:
    """Bit-identical-per-replica, device-collapsed batched Embryo driver.

    ``tissue_delete_prob`` / ``sub_delete_prob`` / ``lamellae_delete_rate`` are optional
    length-R sweep axes (TissueRate / SubLinkRate / LamellaeRate). With all replicas at
    the default rate and seeded ``base_seed + r*stride``, replica r is bit-identical to a
    single ``EmbryoModel`` seeded the same way."""

    def __init__(self, engine: BatchedGPUEngine, params: EmbryoParams = DEFAULT,
                 enable=("lamellipodia", "tissue", "passive_substrate"),
                 leading_type: int = LEADING, passive_type: int = PASSIVE,
                 substrate_type: int = SUBSTRATE,
                 tissue_delete_prob=None, sub_delete_prob=None, lamellae_delete_rate=None,
                 fast: bool = True):
        self.engine = engine
        self.p = params
        self.R = engine.R
        self.enable = set(enable)
        self.leading_type = int(leading_type)
        self.passive_type = int(passive_type)
        self.substrate_type = int(substrate_type)
        # fast: use the BATCHED device link inventory (single launches over R, no per-
        # replica host sync). fast=False uses R per-replica FPPLinks (R device launches);
        # both are bit-identical per replica, fast is the throughput path.
        self.fast = bool(fast)
        ctype = engine.cell_type.numpy()
        self._ctype = ctype

        # managed-cell device id arrays (ASCENDING -> the load-bearing cap order)
        lead = np.sort(np.nonzero(ctype == self.leading_type)[0]).astype(np.int32)
        pas = np.sort(np.nonzero(ctype == self.passive_type)[0]).astype(np.int32)
        self.lead_dev = wp.array(lead, dtype=wp.int32, device=engine.device)
        self.pas_dev = wp.array(pas, dtype=wp.int32, device=engine.device)
        self.n_lead = int(lead.shape[0])
        self.n_pas = int(pas.shape[0])
        self.max_lead = params.max_neighbor_num + 1
        self.max_pas = params.max_neighbor_num

        # batched fused cohesotaxis (one pipeline over all replicas)
        if "lamellipodia" in self.enable:
            self.cohesotaxis = BatchedFusedCohesotaxisPipeline(
                engine, leading_type=leading_type, substrate_type=substrate_type,
                passive_type=passive_type, lamellipodia_distance=params.lamellipodia_distance,
                sigma=params.sigma)
            self.lead_ids = self.cohesotaxis.lead_ids
        else:
            self.cohesotaxis = None
            self.lead_ids = lead

        def _rate_vec(v, default):
            if v is None:
                return np.full(self.R, default, dtype=np.float64)
            return np.broadcast_to(np.asarray(v, np.float64), (self.R,)).copy()
        self.tissue_dp = _rate_vec(tissue_delete_prob, params.tissue_delete_prob)
        self.sub_dp = _rate_vec(sub_delete_prob, params.sub_link_delete_prob)
        self.lam_dr = _rate_vec(lamellae_delete_rate, params.lamellae_rate)
        # per-replica delete-probability DEVICE arrays (the batched Poisson kernels read
        # these per replica -> the sweep varies the rate across the batch axis)
        self._tissue_dp_dev = wp.array(self.tissue_dp.astype(np.float32),
                                       dtype=wp.float32, device=engine.device)
        self._sub_dp_dev = wp.array(self.sub_dp.astype(np.float32),
                                    dtype=wp.float32, device=engine.device)
        self._lam_dp_dev = wp.array((1.0 - np.exp(-self.lam_dr)).astype(np.float32),
                                    dtype=wp.float32, device=engine.device)

        if self.fast:
            self.bfpp = BatchedDeviceFPP(engine, params)
            self.lead_ids_dev = wp.array(self.lead_ids.astype(np.int32),
                                         dtype=wp.int32, device=engine.device)
        else:
            # per-replica FPPLinks fallback (R device launches; bit-identical)
            self.views = [_ReplicaView(engine, r) for r in range(self.R)]
            self.links = [FPPLinks(self.views[r], target_length_default=params.tissue_target,
                                   lambda_default=params.tissue_lambda,
                                   max_length_default=params.tissue_max)
                          for r in range(self.R)]
            self._batched_fpp = _CombinedBatchedFPP(engine, params)

    # ---------------------------------------------------- per-replica device ops
    def _publish_csr(self):
        self.engine.publish_neighbor_csr_device(order=1)
        if not self.fast:
            for v in self.views:
                v.refresh_csr()

    def _tissue_relink_all(self, cap_lead, cap_pas):
        for r in range(self.R):
            lk = self.links[r]
            lk.tissue_relink_device(
                self.views[r], self.lead_dev, self.n_lead, cap=cap_lead,
                substrate_type=self.substrate_type, lam=self.p.tissue_lambda,
                target=self.p.tissue_target, maxlen=self.p.tissue_max)
            lk.tissue_relink_device(
                self.views[r], self.pas_dev, self.n_pas, cap=cap_pas,
                substrate_type=self.substrate_type, lam=self.p.tissue_lambda,
                target=self.p.tissue_target, maxlen=self.p.tissue_max)

    def _substrate_create_all(self):
        if "passive_substrate" not in self.enable or not self.p.if_passive_substrate:
            return
        for r in range(self.R):
            self.links[r].substrate_relink_device(
                self.views[r], self.pas_dev, self.n_pas, substrate_type=self.substrate_type,
                slink_lambda=self.p.slink_lambda, slink_target=self.p.slink_target,
                slink_max=self.p.slink_max)

    def _lamellipodia_create_all(self, mcs, only_need=True):
        if self.cohesotaxis is None:
            return
        targets = self.cohesotaxis.targets(mcs)          # (R, n_lead) device->host once
        lam = self.p.lamellipodia_lambda
        for r in range(self.R):
            lk = self.links[r]
            if only_need:
                has = lk.cells_with_kind_link(lam, self.views[r], self.substrate_type)
                has_host = has.numpy()
            new_a, new_b = [], []
            for s, cell in enumerate(self.lead_ids):
                tgt = int(targets[r, s])
                if tgt < 0:
                    continue
                if only_need and has_host[int(cell)] != 0:
                    continue
                new_a.append(int(cell))
                new_b.append(tgt)
            if new_a:
                lk.create_links_bulk(new_a, new_b, lam=lam,
                                     target=CT.LL_TARGET_DIST, maxlen=CT.LL_MAX_DIST)

    # --------------------------------------------------------- FAST (batched) path
    def _start_fast(self):
        self._publish_csr()
        nct = int(self.engine.csr_n_contacts)
        if "tissue" in self.enable:
            self.bfpp.tissue_relink(self.lead_dev, self.n_lead, FPPLinks_NO_CAP,
                                    self.substrate_type, nct)
            self.bfpp.tissue_relink(self.pas_dev, self.n_pas, FPPLinks_NO_CAP,
                                    self.substrate_type, nct)
        if self.cohesotaxis is not None:
            tgt_dev, _ = self.cohesotaxis.targets_device(0)
            self.bfpp.lamellipodia_create(tgt_dev, self.lead_ids_dev, self.n_lead,
                                          self.p.lamellipodia_lambda, self.substrate_type,
                                          only_need=False)
        self.bfpp.build_csr_and_attach()
        return self

    def _run_fast(self, n_mcs, mcs_offset):
        nct_attr = "csr_n_contacts"
        for mm in range(n_mcs):
            mcs = mcs_offset + mm
            self.engine.step_mcs(mcs)
            self._publish_csr()
            nct = int(getattr(self.engine, nct_attr))
            if "tissue" in self.enable:
                self.bfpp.poisson_delete(self.p.tissue_lambda, self._tissue_dp_dev,
                                         mcs, _STREAM_TISSUE)
                self.bfpp.tissue_relink(self.lead_dev, self.n_lead, self.max_lead,
                                        self.substrate_type, nct)
                self.bfpp.tissue_relink(self.pas_dev, self.n_pas, self.max_pas,
                                        self.substrate_type, nct)
            if "passive_substrate" in self.enable and self.p.if_passive_substrate:
                self.bfpp.substrate_create(self.pas_dev, self.n_pas, self.substrate_type)
                self.bfpp.poisson_delete(self.p.slink_lambda, self._sub_dp_dev,
                                         mcs, _STREAM_SUBLINK)
            if self.cohesotaxis is not None:
                self.bfpp.poisson_delete_by_cell(self.p.lamellipodia_lambda,
                                                 self._lam_dp_dev, mcs, self.substrate_type)
                tgt_dev, _ = self.cohesotaxis.targets_device(mcs)
                self.bfpp.lamellipodia_create(tgt_dev, self.lead_ids_dev, self.n_lead,
                                              self.p.lamellipodia_lambda,
                                              self.substrate_type, only_need=True)
            self.bfpp.build_csr_and_attach()
        return self

    def start(self):
        if self.fast:
            return self._start_fast()
        self._publish_csr()
        if "tissue" in self.enable:
            self._tissue_relink_all(FPPLinks_NO_CAP, FPPLinks_NO_CAP)
        self._lamellipodia_create_all(mcs=0, only_need=False)
        self._combine_and_attach()
        return self

    def run(self, n_mcs: int, mcs_offset: int = 0):
        if self.fast:
            return self._run_fast(n_mcs, mcs_offset)
        for m in range(n_mcs):
            mcs = mcs_offset + m
            self.engine.step_mcs(mcs)                     # batched CPM + FPP (snapshot)
            self._publish_csr()
            # (1) tissue: Poisson delete (Leading owns the draw) + cap-ordered relink
            if "tissue" in self.enable:
                for r in range(self.R):
                    self.links[r].poisson_delete_device(
                        self.p.tissue_lambda, float(self.tissue_dp[r]), mcs,
                        self.views[r].base_seed, _STREAM_TISSUE)
                self._tissue_relink_all(self.max_lead, self.max_pas)
            # (2) substrate: create min-id + Poisson delete
            if "passive_substrate" in self.enable and self.p.if_passive_substrate:
                self._substrate_create_all()
                for r in range(self.R):
                    self.links[r].poisson_delete_device(
                        self.p.slink_lambda, float(self.sub_dp[r]), mcs,
                        self.views[r].base_seed, _STREAM_SUBLINK)
            # (3) lamellipodia: Poisson delete (per leader) + recreate for those lacking
            if self.cohesotaxis is not None:
                lam = self.p.lamellipodia_lambda
                for r in range(self.R):
                    prob = float(1.0 - np.exp(-self.lam_dr[r]))
                    self.links[r].poisson_delete_device_by_cell(
                        lam, prob, mcs, self.views[r].base_seed, self.views[r],
                        self.substrate_type)
                self._lamellipodia_create_all(mcs=mcs, only_need=True)
            self._combine_and_attach()
        return self

    # ------------------------------------------------------------ device combine
    def _combine_and_attach(self):
        """Concatenate every replica's device inventory into the padded (R,M) batched
        FPP CSR on device, and (re)attach it to the engine for the next sweep."""
        self._batched_fpp.combine_from(self.links)
        self.engine.attach_fpp(self._batched_fpp)

    # convenience: per-replica link inventory snapshot (host) for tests
    def link_arrays(self):
        if self.fast:
            return self.bfpp.link_arrays()
        return [(lk._a.copy(), lk._b.copy(), lk._lam.copy()) for lk in self.links]


# cap "off" sentinel matching TissueLinkSteppable._NO_CAP
FPPLinks_NO_CAP = 1 << 30


# ---------------------------------------------------------------------------
# Device combine: assemble the per-replica device inventories into ONE padded
# (R, M) batched FPP CSR via device gathers (no per-replica host concat/upload).
# ---------------------------------------------------------------------------
@wp.kernel
def _combine_gather_kernel(
    R: wp.int32, M: wp.int32,
    src_a: wp.array(dtype=wp.int32), src_b: wp.array(dtype=wp.int32),
    src_lam: wp.array(dtype=wp.float32), src_tgt: wp.array(dtype=wp.float32),
    src_max: wp.array(dtype=wp.float32),
    r: wp.int32, m_r: wp.int32, row_base: wp.int32,
    out_a: wp.array(dtype=wp.int32), out_b: wp.array(dtype=wp.int32),
    out_lam: wp.array(dtype=wp.float32), out_tgt: wp.array(dtype=wp.float32),
    out_max: wp.array(dtype=wp.float32),
):
    """Copy replica r's ``m_r`` live links into row r of the padded (R,M) arrays
    (flat r*M + i). One thread per link; unused slots stay tombstoned (-1)."""
    i = wp.tid()
    if i >= m_r:
        return
    dst = r * M + i
    out_a[dst] = src_a[row_base + i]
    out_b[dst] = src_b[row_base + i]
    out_lam[dst] = src_lam[row_base + i]
    out_tgt[dst] = src_tgt[row_base + i]
    out_max[dst] = src_max[row_base + i]


class _CombinedBatchedFPP:
    """A ``BatchedFPPLinks``-compatible object that the ``BatchedGPUEngine`` Metropolis
    kernel reads, assembled by DEVICE-gathering each replica's device inventory into the
    padded (R, M) arrays (no host concat/upload). Mirrors the per-replica link-CSR
    rebuild contract of ``engine.batched_fpp.BatchedFPPLinks``.

    We reuse ``BatchedFPPLinks`` for the per-replica CSR build (rebuild over the padded
    (R,M) topology + per-replica COM); this class only owns the device combine that
    fills that topology from the per-replica ``FPPLinks._a_dev`` blocks."""

    def __init__(self, engine: BatchedGPUEngine, params: EmbryoParams):
        from engine.batched_fpp import BatchedFPPLinks
        self.engine = engine
        self.R = engine.R
        self._bfpp = BatchedFPPLinks(
            engine, target_length_default=params.tissue_target,
            lambda_default=params.tissue_lambda, max_length_default=params.tissue_max)
        self.device = engine.device
        # padded device topology (R, M) flat; grown on demand
        self._cap_M = 0
        self._a = self._b = self._lam = self._tgt = self._max = None

    def _ensure(self, M):
        if self._cap_M >= M and self._a is not None:
            return
        cap = max(8, int(M))
        R = self.R
        self._a = wp.full(R * cap, -1, dtype=wp.int32, device=self.device)
        self._b = wp.full(R * cap, -1, dtype=wp.int32, device=self.device)
        self._lam = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        self._tgt = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        self._max = wp.zeros(R * cap, dtype=wp.float32, device=self.device)
        self._cap_M = cap

    def combine_from(self, links_list):
        """Gather each replica's live device inventory into the padded (R,M) arrays via
        device kernels, then hand them to the inner BatchedFPPLinks as the topology."""
        R = self.R
        ms = [int(lk.n_pairs) for lk in links_list]
        M = max(ms) if ms else 0
        if M == 0:
            # empty inventory -> M=0 topology
            self._bfpp._a = np.full((R, 0), -1, dtype=np.int32)
            self._bfpp._b = np.full((R, 0), -1, dtype=np.int32)
            self._bfpp._lam = np.zeros((R, 0), dtype=np.float32)
            self._bfpp._tgt = np.zeros((R, 0), dtype=np.float32)
            self._bfpp._max = np.zeros((R, 0), dtype=np.float32)
            self._bfpp._dev_dirty = True
            return
        self._ensure(M)
        # reset padding to tombstones
        self._a.fill_(wp.int32(-1))
        self._b.fill_(wp.int32(-1))
        for r in range(R):
            lk = links_list[r]
            m_r = int(lk.n_pairs)
            if m_r == 0:
                continue
            wp.launch(
                _combine_gather_kernel,
                dim=m_r,
                inputs=[R, M, lk._a_dev, lk._b_dev, lk._lam_dev, lk._tgt_dev, lk._max_dev,
                        r, m_r, 0,
                        self._a, self._b, self._lam, self._tgt, self._max],
                device=self.device,
            )
        # feed the combined padded topology to the inner BatchedFPPLinks (device arrays)
        self._bfpp._set_device_topology(self._a, self._b, self._lam, self._tgt, self._max, M)

    # ---- BatchedFPPLinks pass-through (the engine reads these) ----
    @property
    def link_ptr(self):
        return self._bfpp.link_ptr

    @property
    def link_other(self):
        return self._bfpp.link_other

    @property
    def link_lambda(self):
        return self._bfpp.link_lambda

    @property
    def link_target(self):
        return self._bfpp.link_target

    @property
    def link_pay_stride(self):
        return self._bfpp.link_pay_stride

    def has_links(self):
        return self._bfpp.has_links()

    def rebuild(self):
        self._bfpp.rebuild()

    def num_active(self):
        return self._bfpp.num_active()

    @property
    def n_pairs(self):
        return self._bfpp.n_pairs
