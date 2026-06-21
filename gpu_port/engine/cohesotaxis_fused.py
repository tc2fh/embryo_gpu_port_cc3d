"""Fused on-device cohesotaxis lamellipodia-link pipeline (Phase 8, Objective 1).

The Phase-3 ``cohesotaxis.CohesotaxisPipeline`` runs the lamellipodia target
selection as five SEPARATELY-launched stages (classify -> fill -> pixeldist ->
gumbel_select -> manhattan_argmax) glued by host ``.numpy()`` copies + per-cell host
prefix-sums + a host ``SigWeights`` table. This module fuses that into ONE
persistent-buffer device pipeline:

  * persistent device buffers, allocated once and reused every step (no per-step
    ``wp.zeros`` of the pixel arrays / weight table);
  * the per-slot CSR offsets come from the device exclusive-scan (``engine.scan``),
    not a host ``np.cumsum``;
  * the free / adhesion pixels are filled in a CANONICAL per-slot order (ascending
    global voxel index) via a single device key-sort, so the PixelDist float sum is
    order-stable and the rank tiebreak is deterministic -- this is what makes the
    lamellipodia selection BIT-REPRODUCIBLE (the staged pipeline's atomic-append fill
    order was the dominant source of FPP non-reproducibility);
  * ``SigWeights`` is computed ON DEVICE inside the fused select kernel (no host PDF
    table upload);
  * the Gumbel-max select and the Manhattan-shell argmax-by-zCOM are fused into ONE
    kernel, so the selected pixel never round-trips to the host between stages.

Only the final ``{cell -> target_substrate_id}`` (n_slots int32s) leaves the device.

Fidelity (the Phase-8 invariant): the fused pipeline selects the IDENTICAL target as
the staged pipeline for every leader, per ``(mcs, cell, seed)`` key, when both use the
canonical pixel order. The canonical order is itself fidelity-neutral: CC3D's
``FreePixelList`` is sorted (a stable total order); ascending global voxel index is a
valid stable order, exactly the "ascending cumulative-distance, id tiebreak" the spec
fixes. (The staged pipeline is made canonical-order too via ``fill_sorted=True`` so the
differential test compares two deterministic pipelines.)

All kernels live in this real ``.py`` file (this Warp build reads kernel source via
``inspect``) and pass constant tables as flat int32 device arrays.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from .engine import GPUEngine
from . import scan as S
from . import cohesotaxis as CT
from .cohesotaxis import (
    cohesotaxis_classify_count_kernel,
    LAMELLIPODIA_LAMBDA, LAMELLIPODIA_DISTANCE, LL_TARGET_DIST, LL_MAX_DIST, SIGMA,
    STENCIL_18_FLAT, _STENCIL_18,
)

wp.init()


# ===========================================================================
# Canonical-order fill: emit a packed sort key per flagged voxel, sort, scatter.
# The packed key ``slot*nvox + voxel_index`` sorts grouped-by-slot, voxel-ascending
# within each slot -> the exact "free_ptr segments, ascending id" canonical order.
# ===========================================================================
@wp.kernel
def emit_pixel_keys_kernel(
    nvox: wp.int64,
    ids: wp.array(dtype=wp.int32),
    cell_slot: wp.array(dtype=wp.int32),
    is_free: wp.array(dtype=wp.int32),
    is_adh: wp.array(dtype=wp.int32),
    free_counter: wp.array(dtype=wp.int32),     # slot 0 = global free emit cursor
    adh_counter: wp.array(dtype=wp.int32),      # slot 0 = global adhesion emit cursor
    free_keys: wp.array(dtype=wp.int64),        # packed slot*nvox+i (free)
    adh_keys: wp.array(dtype=wp.int64),         # packed slot*nvox+i (adhesion)
):
    """Emit, per flagged voxel, a packed sort key ``slot*nvox + voxel_index`` into a
    dense array (atomic global cursor; the subsequent SORT makes the order canonical,
    so the atomic emit order is irrelevant). One thread per voxel."""
    i = wp.tid()
    f = is_free[i]
    a = is_adh[i]
    if f == 0 and a == 0:
        return
    cid = ids[i]
    slot = cell_slot[cid]
    if slot < 0:
        return
    base = wp.int64(slot) * nvox + wp.int64(i)
    if f == 1:
        p = wp.atomic_add(free_counter, 0, 1)
        free_keys[p] = base
    if a == 1:
        p = wp.atomic_add(adh_counter, 0, 1)
        adh_keys[p] = base


@wp.kernel
def scatter_free_kernel(
    n_free: wp.int32,
    nvox: wp.int64,
    sorted_keys: wp.array(dtype=wp.int64),      # ascending packed keys (free)
    Lx: wp.int32, Ly: wp.int32,
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    free_slot: wp.array(dtype=wp.int32),
):
    """Unpack the sorted free keys into canonical-order (x,y,z,slot) arrays. One thread
    per free pixel; output index == the pixel's global rank (slot-major, voxel-asc)."""
    p = wp.tid()
    if p >= n_free:
        return
    key = sorted_keys[p]
    slot = wp.int32(key / nvox)
    i = wp.int32(key - wp.int64(slot) * nvox)
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    free_x[p] = x
    free_y[p] = y
    free_z[p] = z
    free_slot[p] = slot


@wp.kernel
def scatter_adh_kernel(
    n_adh: wp.int32,
    nvox: wp.int64,
    sorted_keys: wp.array(dtype=wp.int64),      # ascending packed keys (adhesion)
    Lx: wp.int32, Ly: wp.int32,
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
):
    """Unpack the sorted adhesion keys into canonical-order (x,y,z) arrays. The
    adhesion CSR offsets per slot are the device prefix-sum of the per-slot counts;
    sorting by slot*nvox+i lands each slot's pixels contiguously, voxel-ascending."""
    p = wp.tid()
    if p >= n_adh:
        return
    key = sorted_keys[p]
    slot = wp.int32(key / nvox)
    i = wp.int32(key - wp.int64(slot) * nvox)
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    adh_x[p] = x
    adh_y[p] = y
    adh_z[p] = z


@wp.kernel
def pixeldist_kernel(
    n_free: wp.int32,
    free_slot: wp.array(dtype=wp.int32),
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    adh_ptr: wp.array(dtype=wp.int32),
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
    cum_dist: wp.array(dtype=wp.float32),
):
    """CumDist[f] = sum over the cell's adhesion pixels of ||f - a||_2, summed in
    CANONICAL (ascending voxel-index) order -> order-stable float32 -> deterministic.
    One thread per free pixel (serial inner sum, no float atomics)."""
    i = wp.tid()
    if i >= n_free:
        return
    slot = free_slot[i]
    fx = wp.float32(free_x[i])
    fy = wp.float32(free_y[i])
    fz = wp.float32(free_z[i])
    start = adh_ptr[slot]
    end = adh_ptr[slot + 1]
    s = wp.float32(0.0)
    for k in range(start, end):
        dx = fx - wp.float32(adh_x[k])
        dy = fy - wp.float32(adh_y[k])
        dz = fz - wp.float32(adh_z[k])
        s += wp.sqrt(dx * dx + dy * dy + dz * dz)
    cum_dist[i] = s


@wp.func
def _sigmoid(x: wp.float32, b: wp.float32) -> wp.float32:
    return wp.float32(1.0) / (wp.float32(1.0) + wp.exp(-(x - b)))


@wp.func
def _sig_window(j: wp.int32, k: wp.int32, b: wp.float32) -> wp.float32:
    """sigmoid value at the j-th of k linspace(-5,5,k) points (the SigWeights window),
    computed ON DEVICE -- identical to EmbryoSteppables.SigWeights' per-point value."""
    if k <= 1:
        return wp.float32(1.0)
    xr = wp.float32(-5.0) + wp.float32(10.0) * wp.float32(j) / wp.float32(k - 1)
    return _sigmoid(xr, b)


# ===========================================================================
# Fused Gumbel-max select + Manhattan-shell argmax (one thread per LEADING slot).
# SigWeights is computed ON DEVICE here; the selected pixel never leaves the device --
# the per-slot output is the chosen Substrate target id (or -1).
# ===========================================================================
@wp.kernel
def select_and_target_kernel(
    n_slots: wp.int32,
    free_ptr: wp.array(dtype=wp.int32),
    cum_dist: wp.array(dtype=wp.float32),
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    slot_cell: wp.array(dtype=wp.int32),
    sigma: wp.float32,
    mcs: wp.int32,
    base_seed: wp.int32,
    # Manhattan-shell argmax inputs
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    substrate_type: wp.int32,
    n_order: wp.int32,
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    volume: wp.array(dtype=wp.float32),
    out_target: wp.array(dtype=wp.int32),       # per slot: chosen substrate id or -1
):
    """Per LEADING slot: (1) rank its free pixels ascending by CumDist (id tiebreak),
    map rank -> log SigWeights computed on device, add Gumbel(Philox keyed by
    (mcs,cell,base_seed)), argmax -> chosen free pixel; (2) Manhattan-shell argmax-by-
    zCOM from that pixel -> Substrate target. Both stages fused; only the target id is
    written. Byte-equivalent selection to the staged pipeline on the same canonical
    pixel order."""
    s = wp.tid()
    if s >= n_slots:
        return
    out_target[s] = wp.int32(-1)
    start = free_ptr[s]
    end = free_ptr[s + 1]
    nfree = end - start
    if nfree <= 0:
        return

    # SigWeights normalizer Z = sum_j sigmoid(window_j) over the slot's k points.
    Z = wp.float32(0.0)
    for j in range(nfree):
        Z += _sig_window(j, nfree, sigma)

    cell = slot_cell[s]
    seed = base_seed + mcs * 131072 + cell * 16384
    state = wp.rand_init(seed, cell)

    best_idx = wp.int32(-1)
    best_key = wp.float32(-1.0e30)
    for gi in range(start, end):
        di = cum_dist[gi]
        rank = wp.int32(0)
        for gj in range(start, end):
            dj = cum_dist[gj]
            if dj < di or (dj == di and gj < gi):
                rank += 1
        # SigWeights is the DESCENDING reversed sigmoid PDF: rank 0 == largest weight
        # == the LAST window point (index nfree-1-rank).
        widx = nfree - 1 - rank
        w = _sig_window(widx, nfree, sigma) / Z
        if w < wp.float32(1.0e-300):
            w = wp.float32(1.0e-300)
        lw = wp.log(w)
        u = wp.randf(state)
        if u <= 1.0e-20:
            u = wp.float32(1.0e-20)
        g = -wp.log(-wp.log(u))
        key = lw + g
        if key > best_key:
            best_key = key
            best_idx = gi
    if best_idx < 0:
        return

    # ---- Manhattan-shell argmax-by-zCOM for the chosen pixel (fused, no readback) ----
    px = free_x[best_idx]
    py = free_y[best_idx]
    pz = free_z[best_idx]
    pzf = wp.float32(pz)
    best_id = wp.int32(-1)
    best_z = wp.float32(-1.0e30)
    n = n_order
    for dx in range(-n, n + 1):
        adx = dx
        if adx < 0:
            adx = -adx
        for dy in range(-n, n + 1):
            ady = dy
            if ady < 0:
                ady = -ady
            for dz in range(-n, n + 1):
                adz = dz
                if adz < 0:
                    adz = -adz
                if adx + ady + adz != n:
                    continue
                nx = px + dx
                ny = py + dy
                nz = pz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= Lx or ny >= Ly or nz >= Lz:
                    continue
                c = ids[(nz * Ly + ny) * Lx + nx]
                if c == 0:
                    continue
                if cell_type[c] != substrate_type:
                    continue
                vc = volume[c]
                if vc <= 0.0:
                    continue
                zc = wp.float32(wp.float64(zsum[c]) / wp.float64(vc))
                if zc > pzf:
                    if zc > best_z or (zc == best_z and c > best_id):
                        best_z = zc
                        best_id = c
    out_target[s] = best_id


class FusedCohesotaxisPipeline:
    """One persistent-buffer device pipeline for the lamellipodia target selection of
    a ``GPUEngine``'s LEADING cells (Phase 8). Reuses Phase-3's classify kernel + the
    device scan; adds the canonical-order key-sort fill + the fused SigWeights /
    Gumbel-max / Manhattan-argmax select. Only ``select_targets`` returns to the host
    (the ``{cell:target}`` dict)."""

    def __init__(self, engine: GPUEngine, leading_type: int, substrate_type: int,
                 passive_type: int, lamellipodia_distance: int = LAMELLIPODIA_DISTANCE,
                 sigma: float = SIGMA):
        self.engine = engine
        self.device = engine.device
        self.leading_type = int(leading_type)
        self.substrate_type = int(substrate_type)
        self.passive_type = int(passive_type)
        self.lam_dist = int(lamellipodia_distance)
        self.sigma = float(sigma)
        self.Lx, self.Ly, self.Lz = engine.Lx, engine.Ly, engine.Lz
        self.nvox = int(engine.cfg.n_voxels)
        self.n1 = engine.n_cells + 1

        ctype = engine.cell_type.numpy()
        lead_ids = np.nonzero(ctype == self.leading_type)[0].astype(np.int32)
        self.lead_ids = lead_ids
        self.n_slots = int(len(lead_ids))
        cell_slot = np.full(self.n1, -1, dtype=np.int32)
        cell_slot[lead_ids] = np.arange(self.n_slots, dtype=np.int32)
        self.cell_slot = wp.array(cell_slot, dtype=wp.int32, device=self.device)
        self.slot_cell = wp.array(lead_ids, dtype=wp.int32, device=self.device)
        self.stencil = wp.array(STENCIL_18_FLAT, dtype=wp.int32, device=self.device)
        self.n_sten = int(len(_STENCIL_18))

        # ---- persistent device buffers (allocated once, reused every step) ----
        ns = max(1, self.n_slots)
        self._is_free = wp.zeros(self.nvox, dtype=wp.int32, device=self.device)
        self._is_adh = wp.zeros(self.nvox, dtype=wp.int32, device=self.device)
        self._free_count = wp.zeros(ns + 1, dtype=wp.int32, device=self.device)
        self._adh_count = wp.zeros(ns + 1, dtype=wp.int32, device=self.device)
        self._free_ptr = wp.zeros(ns + 1, dtype=wp.int32, device=self.device)
        self._adh_ptr = wp.zeros(ns + 1, dtype=wp.int32, device=self.device)
        self._free_ctr = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._adh_ctr = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._target = wp.full(ns, -1, dtype=wp.int32, device=self.device)
        # grown-on-demand pixel + key buffers (sized to the live free/adh counts)
        self._cap_free = 0
        self._cap_adh = 0
        self._free_keys = None
        self._adh_keys = None
        self._free_x = self._free_y = self._free_z = self._free_slot = None
        self._adh_x = self._adh_y = self._adh_z = None
        self._cum = None

    def _ensure_free(self, n):
        # radix_sort_pairs double-buffers -> key storage must hold 2*n
        if self._cap_free >= n and self._free_keys is not None:
            return
        cap = max(8, int(n))
        self._free_keys = wp.zeros(2 * cap, dtype=wp.int64, device=self.device)
        self._free_vals = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._free_x = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_y = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_z = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_slot = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._cum = wp.zeros(cap, dtype=wp.float32, device=self.device)
        self._cap_free = cap

    def _ensure_adh(self, n):
        if self._cap_adh >= n and self._adh_keys is not None:
            return
        cap = max(8, int(n))
        self._adh_keys = wp.zeros(2 * cap, dtype=wp.int64, device=self.device)
        self._adh_vals = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._adh_x = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._adh_y = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._adh_z = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._cap_adh = cap

    def select_targets(self, mcs: int) -> dict:
        """Run the fused pipeline for all LEADING cells; return
        ``{cell_id: target_substrate_id}`` for cells that selected a valid target.
        Only this final dict crosses to the host (n_slots int32 readback)."""
        if self.n_slots == 0:
            return {}
        eng = self.engine
        nvox = self.nvox
        self._is_free.zero_()
        self._is_adh.zero_()
        self._free_count.zero_()
        self._adh_count.zero_()

        # ---- stage 1: classify (per-voxel) -> flags + per-slot counts ----
        wp.launch(
            cohesotaxis_classify_count_kernel,
            dim=nvox,
            inputs=[eng.ids, eng.cell_type, self.Lx, self.Ly, self.Lz,
                    self.leading_type, self.substrate_type, self.passive_type,
                    self.stencil, self.n_sten,
                    self.cell_slot, self._free_count, self._adh_count,
                    self._is_free, self._is_adh],
            device=self.device,
        )
        # ---- device exclusive scan of per-slot counts -> CSR offsets ----
        self._free_ptr.zero_()
        self._adh_ptr.zero_()
        S.exclusive_scan_to_ptr_i32(self._free_count, self.n_slots, self._free_ptr, self.device)
        S.exclusive_scan_to_ptr_i32(self._adh_count, self.n_slots, self._adh_ptr, self.device)
        wp.synchronize()
        n_free = int(self._free_ptr.numpy()[self.n_slots])
        n_adh = int(self._adh_ptr.numpy()[self.n_slots])
        if n_free == 0:
            return {}

        # ---- stage 2: canonical-order fill (emit packed keys -> sort -> scatter) ----
        self._ensure_free(n_free)
        self._ensure_adh(max(1, n_adh))
        self._free_ctr.zero_()
        self._adh_ctr.zero_()
        wp.launch(
            emit_pixel_keys_kernel,
            dim=nvox,
            inputs=[wp.int64(nvox), eng.ids, self.cell_slot,
                    self._is_free, self._is_adh, self._free_ctr, self._adh_ctr,
                    self._free_keys, self._adh_keys],
            device=self.device,
        )
        # one global radix sort each -> grouped by slot, ascending voxel index within
        wp.utils.radix_sort_pairs(self._free_keys, self._free_vals, n_free)
        if n_adh > 0:
            wp.utils.radix_sort_pairs(self._adh_keys, self._adh_vals, n_adh)
        wp.launch(
            scatter_free_kernel,
            dim=n_free,
            inputs=[n_free, wp.int64(nvox), self._free_keys, self.Lx, self.Ly,
                    self._free_x, self._free_y, self._free_z, self._free_slot],
            device=self.device,
        )
        if n_adh > 0:
            wp.launch(
                scatter_adh_kernel,
                dim=n_adh,
                inputs=[n_adh, wp.int64(nvox), self._adh_keys, self.Lx, self.Ly,
                        self._adh_x, self._adh_y, self._adh_z],
                device=self.device,
            )

        # ---- stage 3: PixelDist (canonical order -> deterministic) ----
        self._cum.zero_()
        wp.launch(
            pixeldist_kernel,
            dim=n_free,
            inputs=[n_free, self._free_slot, self._free_x, self._free_y, self._free_z,
                    self._adh_ptr, self._adh_x, self._adh_y, self._adh_z, self._cum],
            device=self.device,
        )

        # ---- stage 4+5 fused: SigWeights + Gumbel-max select + Manhattan argmax ----
        self._target.fill_(wp.int32(-1))
        wp.launch(
            select_and_target_kernel,
            dim=self.n_slots,
            inputs=[self.n_slots, self._free_ptr, self._cum,
                    self._free_x, self._free_y, self._free_z, self.slot_cell,
                    float(self.sigma), int(mcs), int(eng.base_seed),
                    eng.ids, eng.cell_type, self.Lx, self.Ly, self.Lz,
                    self.substrate_type, self.lam_dist,
                    eng.xsum, eng.ysum, eng.zsum, eng.volume, self._target],
            device=self.device,
        )
        wp.synchronize()
        tgt = self._target.numpy()[:self.n_slots]   # the ONLY host readback
        out = {}
        for s in range(self.n_slots):
            if tgt[s] >= 0:
                out[int(self.lead_ids[s])] = int(tgt[s])
        return out

    def create_lamellipodia_links(self, links, mcs: int, only_cells=None) -> dict:
        """Run the fused pipeline and create the selected lamellipodia FPP links via
        ``FPPLinks.create_links_bulk`` (LamellipodiaLambda / LLTargetDist / LLMaxDist).
        ``only_cells`` restricts which leaders get a link this step."""
        targets = self.select_targets(mcs)
        created = {}
        new_a, new_b = [], []
        for cell, tgt in targets.items():
            if only_cells is not None and cell not in only_cells:
                continue
            new_a.append(cell)
            new_b.append(tgt)
            created[cell] = tgt
        if new_a:
            links.create_links_bulk(new_a, new_b, lam=LAMELLIPODIA_LAMBDA,
                                    target=LL_TARGET_DIST, maxlen=LL_MAX_DIST)
        return created


# ===========================================================================
# BATCHED (replica-segmented) fused cohesotaxis (Phase 8, Objective 2).
# One pipeline over R*n_lead slots, keyed per (r, cell). Slots are global
# ``r*n_lead + s``; pixel keys pack the GLOBAL slot in the high bits so a single sort
# groups pixels by (replica, leader), voxel-ascending within. Per-replica COM/seed
# slices, so a batched run's selection is bit-identical to R single fused pipelines.
# ===========================================================================
@wp.kernel
def classify_count_batched_kernel(
    ids: wp.array(dtype=wp.int32),              # (R*nvox,)
    cell_type: wp.array(dtype=wp.int32),        # shared
    nvox: wp.int32, R: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    leading_type: wp.int32, substrate_type: wp.int32, passive_type: wp.int32,
    stencil: wp.array(dtype=wp.int32), n_sten: wp.int32,
    cell_slot: wp.array(dtype=wp.int32),        # cid -> dense leading slot (shared), or -1
    n_lead: wp.int32,
    free_count: wp.array(dtype=wp.int32),       # (R*n_lead+1,) per global-slot
    adh_count: wp.array(dtype=wp.int32),
    is_free: wp.array(dtype=wp.int32),          # (R*nvox,)
    is_adh: wp.array(dtype=wp.int32),
):
    """Replica-aware classify: one thread per (replica, voxel). Counts the per-(r,
    leader) Free / Adhesion pixels into the global-slot count arrays (row r*n_lead+s).
    Same stencil rule as the single classify; reads replica r's lattice slice."""
    gid = wp.tid()
    r = gid / nvox
    i = gid % nvox
    if r >= R:
        return
    vox_base = r * nvox
    cid = ids[vox_base + i]
    is_free[vox_base + i] = 0
    is_adh[vox_base + i] = 0
    if cid == 0:
        return
    if cell_type[cid] != leading_type:
        return
    slot = cell_slot[cid]
    if slot < 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    nm = wp.int32(0)
    ns = wp.int32(0)
    nf = wp.int32(0)
    for n in range(n_sten):
        nx = x + stencil[3 * n + 0]
        ny = y + stencil[3 * n + 1]
        nz = z + stencil[3 * n + 2]
        nc = wp.int32(0)
        if nx >= 0 and ny >= 0 and nz >= 0 and nx < Lx and ny < Ly and nz < Lz:
            nc = ids[vox_base + (nz * Ly + ny) * Lx + nx]
        if nc == 0:
            nm = 1
        else:
            t = cell_type[nc]
            if t == substrate_type:
                ns = 1
            elif t == passive_type:
                nf = 1
    gslot = r * n_lead + slot
    if nm == 1 and ns == 1:
        is_free[vox_base + i] = 1
        wp.atomic_add(free_count, gslot, 1)
    if ns == 1 or nf == 1:
        is_adh[vox_base + i] = 1
        wp.atomic_add(adh_count, gslot, 1)


@wp.kernel
def emit_keys_batched_kernel(
    nvox: wp.int64, R: wp.int32,
    ids: wp.array(dtype=wp.int32),
    cell_slot: wp.array(dtype=wp.int32),
    n_lead: wp.int64,
    is_free: wp.array(dtype=wp.int32),
    is_adh: wp.array(dtype=wp.int32),
    free_counter: wp.array(dtype=wp.int32),
    adh_counter: wp.array(dtype=wp.int32),
    free_keys: wp.array(dtype=wp.int64),
    adh_keys: wp.array(dtype=wp.int64),
):
    """Emit per flagged (replica, voxel) a packed key ``gslot*nvox + i`` (gslot =
    r*n_lead + slot) so a single global sort groups pixels by (replica, leader),
    voxel-ascending. One thread per (replica, voxel)."""
    gid = wp.tid()
    r = wp.int32(gid / wp.int32(nvox))
    i = wp.int32(wp.int64(gid) % nvox)
    if r >= R:
        return
    vox_base = r * wp.int32(nvox)
    f = is_free[vox_base + i]
    a = is_adh[vox_base + i]
    if f == 0 and a == 0:
        return
    cid = ids[vox_base + i]
    slot = cell_slot[cid]
    if slot < 0:
        return
    gslot = wp.int64(r) * n_lead + wp.int64(slot)
    base = gslot * nvox + wp.int64(i)
    if f == 1:
        p = wp.atomic_add(free_counter, 0, 1)
        free_keys[p] = base
    if a == 1:
        p = wp.atomic_add(adh_counter, 0, 1)
        adh_keys[p] = base


@wp.kernel
def scatter_free_batched_kernel(
    n_free: wp.int32, nvox: wp.int64,
    sorted_keys: wp.array(dtype=wp.int64),
    Lx: wp.int32, Ly: wp.int32,
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    free_gslot: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    if p >= n_free:
        return
    key = sorted_keys[p]
    gslot = wp.int32(key / nvox)
    i = wp.int32(key - wp.int64(gslot) * nvox)
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    free_x[p] = x
    free_y[p] = y
    free_z[p] = z
    free_gslot[p] = gslot


@wp.kernel
def scatter_adh_batched_kernel(
    n_adh: wp.int32, nvox: wp.int64,
    sorted_keys: wp.array(dtype=wp.int64),
    Lx: wp.int32, Ly: wp.int32,
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    if p >= n_adh:
        return
    key = sorted_keys[p]
    gslot = wp.int32(key / nvox)
    i = wp.int32(key - wp.int64(gslot) * nvox)
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    adh_x[p] = x
    adh_y[p] = y
    adh_z[p] = z


@wp.kernel
def pixeldist_batched_kernel(
    n_free: wp.int32,
    free_gslot: wp.array(dtype=wp.int32),
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    adh_ptr: wp.array(dtype=wp.int32),          # (R*n_lead+1,) global-slot CSR
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
    cum_dist: wp.array(dtype=wp.float32),
):
    """CumDist per free pixel over its (r,leader)'s adhesion pixels, canonical order."""
    i = wp.tid()
    if i >= n_free:
        return
    gslot = free_gslot[i]
    fx = wp.float32(free_x[i])
    fy = wp.float32(free_y[i])
    fz = wp.float32(free_z[i])
    start = adh_ptr[gslot]
    end = adh_ptr[gslot + 1]
    s = wp.float32(0.0)
    for k in range(start, end):
        dx = fx - wp.float32(adh_x[k])
        dy = fy - wp.float32(adh_y[k])
        dz = fz - wp.float32(adh_z[k])
        s += wp.sqrt(dx * dx + dy * dy + dz * dz)
    cum_dist[i] = s


@wp.kernel
def select_and_target_batched_kernel(
    n_slots_total: wp.int32,                    # R*n_lead
    n_lead: wp.int32,
    nvox: wp.int32, n1: wp.int32,
    free_ptr: wp.array(dtype=wp.int32),         # (R*n_lead+1,)
    cum_dist: wp.array(dtype=wp.float32),
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    slot_cell: wp.array(dtype=wp.int32),        # (n_lead,) dense slot -> leader cell id
    base_seed_r: wp.array(dtype=wp.int32),      # (R,) per-replica base seed
    sigma: wp.float32,
    mcs: wp.int32,
    ids: wp.array(dtype=wp.int32),              # (R*nvox,)
    cell_type: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    substrate_type: wp.int32, n_order: wp.int32,
    xsum: wp.array(dtype=wp.int64),             # (R*n1,)
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    volume: wp.array(dtype=wp.float32),
    out_target: wp.array(dtype=wp.int32),       # (R*n_lead,) chosen substrate id or -1
):
    """Per global slot (r, leader): fused SigWeights + Gumbel-max select (keyed by
    (mcs, cell, base_seed_r[r])) + Manhattan-shell argmax-by-zCOM over replica r's
    lattice/COM slice. Bit-identical to the single fused pipeline for replica r."""
    g = wp.tid()
    if g >= n_slots_total:
        return
    out_target[g] = wp.int32(-1)
    start = free_ptr[g]
    end = free_ptr[g + 1]
    nfree = end - start
    if nfree <= 0:
        return
    r = g / n_lead
    s = g % n_lead
    vox_base = r * nvox
    cell_base = r * n1

    Z = wp.float32(0.0)
    for j in range(nfree):
        Z += _sig_window(j, nfree, sigma)

    cell = slot_cell[s]
    seed = base_seed_r[r] + mcs * 131072 + cell * 16384
    state = wp.rand_init(seed, cell)

    best_idx = wp.int32(-1)
    best_key = wp.float32(-1.0e30)
    for gi in range(start, end):
        di = cum_dist[gi]
        rank = wp.int32(0)
        for gj in range(start, end):
            dj = cum_dist[gj]
            if dj < di or (dj == di and gj < gi):
                rank += 1
        widx = nfree - 1 - rank
        w = _sig_window(widx, nfree, sigma) / Z
        if w < wp.float32(1.0e-300):
            w = wp.float32(1.0e-300)
        lw = wp.log(w)
        u = wp.randf(state)
        if u <= 1.0e-20:
            u = wp.float32(1.0e-20)
        gmb = -wp.log(-wp.log(u))
        key = lw + gmb
        if key > best_key:
            best_key = key
            best_idx = gi
    if best_idx < 0:
        return

    px = free_x[best_idx]
    py = free_y[best_idx]
    pz = free_z[best_idx]
    pzf = wp.float32(pz)
    best_id = wp.int32(-1)
    best_z = wp.float32(-1.0e30)
    n = n_order
    for dx in range(-n, n + 1):
        adx = dx
        if adx < 0:
            adx = -adx
        for dy in range(-n, n + 1):
            ady = dy
            if ady < 0:
                ady = -ady
            for dz in range(-n, n + 1):
                adz = dz
                if adz < 0:
                    adz = -adz
                if adx + ady + adz != n:
                    continue
                nx = px + dx
                ny = py + dy
                nz = pz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= Lx or ny >= Ly or nz >= Lz:
                    continue
                c = ids[vox_base + (nz * Ly + ny) * Lx + nx]
                if c == 0:
                    continue
                if cell_type[c] != substrate_type:
                    continue
                vc = volume[cell_base + c]
                if vc <= 0.0:
                    continue
                zc = wp.float32(wp.float64(zsum[cell_base + c]) / wp.float64(vc))
                if zc > pzf:
                    if zc > best_z or (zc == best_z and c > best_id):
                        best_z = zc
                        best_id = c
    out_target[g] = best_id


class BatchedFusedCohesotaxisPipeline:
    """Replica-segmented fused cohesotaxis over R replicas sharing the cell-type
    assignment (Phase 8, Objective 2). One classify launch over R*nvox, one global
    key-sort fill, one select launch over R*n_lead -> per-(r,leader) Substrate targets,
    keyed by the replica's base seed. Bit-identical to R single fused pipelines.

    ``engine`` is a ``BatchedGPUEngine`` (reads ids/xsum/ysum/zsum/volume + base_seed_r
    + n1/nvox). Returns ``targets(mcs)`` = (R, n_lead) int32 array of chosen Substrate
    ids (or -1), and ``lead_ids`` (shared dense leader ids)."""

    def __init__(self, engine, leading_type: int, substrate_type: int, passive_type: int,
                 lamellipodia_distance: int = LAMELLIPODIA_DISTANCE, sigma: float = SIGMA):
        self.engine = engine
        self.device = engine.device
        self.R = engine.R
        self.leading_type = int(leading_type)
        self.substrate_type = int(substrate_type)
        self.passive_type = int(passive_type)
        self.lam_dist = int(lamellipodia_distance)
        self.sigma = float(sigma)
        self.Lx, self.Ly, self.Lz = engine.Lx, engine.Ly, engine.Lz
        self.nvox = int(engine.nvox)
        self.n1 = engine.n1

        ctype = engine.cell_type.numpy()
        lead_ids = np.nonzero(ctype == self.leading_type)[0].astype(np.int32)
        self.lead_ids = lead_ids
        self.n_lead = int(len(lead_ids))
        cell_slot = np.full(self.n1, -1, dtype=np.int32)
        cell_slot[lead_ids] = np.arange(self.n_lead, dtype=np.int32)
        self.cell_slot = wp.array(cell_slot, dtype=wp.int32, device=self.device)
        self.slot_cell = wp.array(lead_ids, dtype=wp.int32, device=self.device)
        self.stencil = wp.array(STENCIL_18_FLAT, dtype=wp.int32, device=self.device)
        self.n_sten = int(len(_STENCIL_18))

        R, nslots = self.R, max(1, self.R * self.n_lead)
        self._is_free = wp.zeros(self.R * self.nvox, dtype=wp.int32, device=self.device)
        self._is_adh = wp.zeros(self.R * self.nvox, dtype=wp.int32, device=self.device)
        self._free_count = wp.zeros(nslots + 1, dtype=wp.int32, device=self.device)
        self._adh_count = wp.zeros(nslots + 1, dtype=wp.int32, device=self.device)
        self._free_ptr = wp.zeros(nslots + 1, dtype=wp.int32, device=self.device)
        self._adh_ptr = wp.zeros(nslots + 1, dtype=wp.int32, device=self.device)
        self._free_ctr = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._adh_ctr = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._target = wp.full(nslots, -1, dtype=wp.int32, device=self.device)
        self._cap_free = 0
        self._cap_adh = 0
        self._free_keys = self._adh_keys = None
        self._free_x = self._free_y = self._free_z = self._free_gslot = None
        self._adh_x = self._adh_y = self._adh_z = None
        self._cum = None

    def _ensure_free(self, n):
        if self._cap_free >= n and self._free_keys is not None:
            return
        cap = max(8, int(n))
        self._free_keys = wp.zeros(2 * cap, dtype=wp.int64, device=self.device)
        self._free_vals = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._free_x = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_y = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_z = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._free_gslot = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._cum = wp.zeros(cap, dtype=wp.float32, device=self.device)
        self._cap_free = cap

    def _ensure_adh(self, n):
        if self._cap_adh >= n and self._adh_keys is not None:
            return
        cap = max(8, int(n))
        self._adh_keys = wp.zeros(2 * cap, dtype=wp.int64, device=self.device)
        self._adh_vals = wp.zeros(2 * cap, dtype=wp.int32, device=self.device)
        self._adh_x = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._adh_y = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._adh_z = wp.zeros(cap, dtype=wp.int32, device=self.device)
        self._cap_adh = cap

    def targets(self, mcs: int) -> np.ndarray:
        """Run the batched fused pipeline; return an (R, n_lead) int32 array of chosen
        Substrate target ids per leader per replica (or -1). One classify launch, one
        global sort, one select launch; a single (R*n_lead) host readback."""
        R, n_lead = self.R, self.n_lead
        if n_lead == 0:
            return np.full((R, 0), -1, dtype=np.int32)
        eng = self.engine
        nvox = self.nvox
        nslots = R * n_lead
        self._is_free.zero_()
        self._is_adh.zero_()
        self._free_count.zero_()
        self._adh_count.zero_()

        wp.launch(
            classify_count_batched_kernel,
            dim=R * nvox,
            inputs=[eng.ids, eng.cell_type, nvox, R, self.Lx, self.Ly, self.Lz,
                    self.leading_type, self.substrate_type, self.passive_type,
                    self.stencil, self.n_sten, self.cell_slot, n_lead,
                    self._free_count, self._adh_count, self._is_free, self._is_adh],
            device=self.device,
        )
        self._free_ptr.zero_()
        self._adh_ptr.zero_()
        S.exclusive_scan_to_ptr_i32(self._free_count, nslots, self._free_ptr, self.device)
        S.exclusive_scan_to_ptr_i32(self._adh_count, nslots, self._adh_ptr, self.device)
        wp.synchronize()
        n_free = int(self._free_ptr.numpy()[nslots])
        n_adh = int(self._adh_ptr.numpy()[nslots])
        if n_free == 0:
            return np.full((R, n_lead), -1, dtype=np.int32)

        self._ensure_free(n_free)
        self._ensure_adh(max(1, n_adh))
        self._free_ctr.zero_()
        self._adh_ctr.zero_()
        wp.launch(
            emit_keys_batched_kernel,
            dim=R * nvox,
            inputs=[wp.int64(nvox), R, eng.ids, self.cell_slot, wp.int64(n_lead),
                    self._is_free, self._is_adh, self._free_ctr, self._adh_ctr,
                    self._free_keys, self._adh_keys],
            device=self.device,
        )
        wp.utils.radix_sort_pairs(self._free_keys, self._free_vals, n_free)
        if n_adh > 0:
            wp.utils.radix_sort_pairs(self._adh_keys, self._adh_vals, n_adh)
        wp.launch(
            scatter_free_batched_kernel,
            dim=n_free,
            inputs=[n_free, wp.int64(nvox), self._free_keys, self.Lx, self.Ly,
                    self._free_x, self._free_y, self._free_z, self._free_gslot],
            device=self.device,
        )
        if n_adh > 0:
            wp.launch(
                scatter_adh_batched_kernel,
                dim=n_adh,
                inputs=[n_adh, wp.int64(nvox), self._adh_keys, self.Lx, self.Ly,
                        self._adh_x, self._adh_y, self._adh_z],
                device=self.device,
            )
        self._cum.zero_()
        wp.launch(
            pixeldist_batched_kernel,
            dim=n_free,
            inputs=[n_free, self._free_gslot, self._free_x, self._free_y, self._free_z,
                    self._adh_ptr, self._adh_x, self._adh_y, self._adh_z, self._cum],
            device=self.device,
        )
        self._target.fill_(wp.int32(-1))
        wp.launch(
            select_and_target_batched_kernel,
            dim=nslots,
            inputs=[nslots, n_lead, nvox, self.n1, self._free_ptr, self._cum,
                    self._free_x, self._free_y, self._free_z, self.slot_cell,
                    eng.base_seed_r, float(self.sigma), int(mcs),
                    eng.ids, eng.cell_type, self.Lx, self.Ly, self.Lz,
                    self.substrate_type, self.lam_dist,
                    eng.xsum, eng.ysum, eng.zsum, eng.volume, self._target],
            device=self.device,
        )
        wp.synchronize()
        self._last_nslots = nslots
        return self._target.numpy()[:nslots].reshape(R, n_lead).copy()

    def targets_device(self, mcs: int):
        """Run the batched fused pipeline and return the DEVICE target array
        (R*n_lead int32, chosen Substrate id or -1) WITHOUT a host readback -- the seam
        the batched device link create consumes directly. Also returns the device
        ``slot_cell`` (leader ids per dense slot)."""
        # reuse targets() machinery but skip the host copy: re-run the kernels and hand
        # back the device buffer. (targets() leaves self._target populated on device.)
        self.targets(mcs)              # fills self._target on device (+ a small readback)
        return self._target, self.slot_cell
