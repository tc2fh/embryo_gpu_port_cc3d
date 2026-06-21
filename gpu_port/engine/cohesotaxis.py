"""On-device cohesotaxis lamellipodia-link pipeline + Poisson link turnover
(Phase 3, Pass B).

This is the GPU port of the CC3D Embryo ``ifCohesotaxis=1`` target selection in
``EmbryoSteppables.create_lamellipodia_link`` (read-only reference:
``Embryo_Model_dev/Embryo/Simulation/EmbryoSteppables.py``) and the lamellipodia
Poisson turnover in ``LeadingEdgeSteppable.step``.

CPU semantics reproduced (per LEADING cell needing a link):

  1. **Stencil classify.** Over the cell's boundary pixels, scan the explicit
     18-offset stencil ``neighbor_pixel_locations``. Flag each pixel
     next-to-Medium / next-to-Substrate / next-to-Passive(=follower). Build
       * FreePixelList     = pixels (next-to-Medium AND next-to-Substrate)
       * CellAdhesionList  = pixels (next-to-Substrate OR next-to-Passive)
  2. **Segmented compaction.** Compact each cell's Free / Adhesion pixels into
     contiguous per-cell segments (atomic-append into ranges fixed by a host
     exclusive prefix-sum of per-cell counts -- the engine's proven seam).
  3. **All-pairs PixelDist reduction.** For each Free pixel f,
     ``CumDist[f] = sum over CellAdhesion a of || f - a ||_2`` (PixelDist).
  4. **Gumbel-max weighted select.** Rank Free pixels ascending by CumDist;
     map rank -> ``SigWeights(n, Sigma)`` (descending sigmoid PDF over [-5,5]).
     CC3D draws ``rng.choice(FreePixelList, p=weights)``; that categorical draw is
     reproduced as ``argmax_r ( log w_r + Gumbel_r )`` with the Gumbel noise from
     Philox keyed by ``(mcs, cell, base_seed)`` -- the engine RNG convention, so
     the choice is reproducible and bit-stable (no float atomics involved).
  5. **Manhattan-shell argmax.** From the selected pixel, take Substrate cells at
     *exact Manhattan distance == LamellipodiaDistance* whose ``zCOM > pixel.z``;
     the chosen target is the one with the greatest ``zCOM`` (walk toward the
     animal pole). Create the lamellipodia FPP link to it via Pass A's
     ``FPPLinks.create_link`` with (LamellipodiaLambda, LLTargetDist, LLMaxDist).

Reproducibility note (carry-forward constraint): GPU float ``atomic_add`` is not
bit-reproducible, so the *selection* uses Gumbel-max-with-keyed-RNG rather than
an atomic weighted sum, and the deterministic stages (PixelDist, Manhattan argmax,
classification) are exact reductions/argmaxes with no float-atomic dependence.

All kernels live in this real ``.py`` file (this Warp build reads kernel source via
``inspect``; no ``exec()``'d kernels) and pass constant offset tables as flat int32
device arrays (no ``wp.mat`` const type in Warp 1.14).
"""

from __future__ import annotations

import numpy as np

import warp as wp

from .engine import GPUEngine

wp.init()


# ---------------------------------------------------------------------------
# Embryo lamellipodia / cohesotaxis parameters (EmbryoSteppables.py module head).
# ---------------------------------------------------------------------------
LAMELLIPODIA_LAMBDA = 800.0
LAMELLIPODIA_DISTANCE = 2          # Manhattan shell order for substrate pick
LL_TARGET_DIST = 1.0
LL_MAX_DIST = 15.0
SIGMA = 8.0                        # SigWeights kappa (cohesotaxis bias slope)

# Poisson lamellipodia turnover: timestep * L_CyclingRate = 5 * (1/180).
_TIMESTEP = 5
_L_CYCLING_RATE = 1.0 / 180.0
LAMELLAE_RATE = _TIMESTEP * _L_CYCLING_RATE

# 18-offset stencil == EmbryoSteppables.neighbor_pixel_locations (flattened to
# int32 [dx,dy,dz, dx,dy,dz, ...] for the kernels).
_STENCIL_18 = (
    (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)
STENCIL_18_FLAT = np.asarray(_STENCIL_18, dtype=np.int32).flatten()


# ---------------------------------------------------------------------------
# SigWeights (host): the CPU selection PDF. Tiny + per-cell-rank, computed once
# per distinct free-pixel count and uploaded; the device select reads it by rank.
# ---------------------------------------------------------------------------
def sig_weights(x: int, b: float = SIGMA) -> np.ndarray:
    """EmbryoSteppables.SigWeights: a length-``x`` descending sigmoid PDF over the
    viewing window [-5,5], shifted by kappa ``b`` and reversed (rank 0 = largest)."""
    if x <= 0:
        return np.zeros(0, dtype=np.float64)
    if x == 1:
        return np.ones(1, dtype=np.float64)
    xrange = np.linspace(-5.0, 5.0, x)
    sig = 1.0 / (1.0 + np.e ** -(xrange - b))
    w = sig / np.sum(sig)
    return w[::-1].copy()


# ===========================================================================
# Stage 1 -- stencil classify (one thread per voxel). Counts the cell-local
# Free / Adhesion pixels per LEADING cell into per-cell-slot degree arrays.
# ===========================================================================
@wp.kernel
def cohesotaxis_classify_count_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    leading_type: wp.int32,
    substrate_type: wp.int32,
    passive_type: wp.int32,
    stencil: wp.array(dtype=wp.int32),      # flat 3*18
    n_sten: wp.int32,
    cell_slot: wp.array(dtype=wp.int32),    # cid -> dense leading slot, or -1
    free_count: wp.array(dtype=wp.int32),   # per-slot
    adh_count: wp.array(dtype=wp.int32),    # per-slot
    is_free: wp.array(dtype=wp.int32),      # per-voxel 0/1
    is_adh: wp.array(dtype=wp.int32),       # per-voxel 0/1
):
    i = wp.tid()
    cid = ids[i]
    if cid == 0:
        is_free[i] = 0
        is_adh[i] = 0
        return
    if cell_type[cid] != leading_type:
        is_free[i] = 0
        is_adh[i] = 0
        return
    slot = cell_slot[cid]
    if slot < 0:
        is_free[i] = 0
        is_adh[i] = 0
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
            nc = ids[(nz * Ly + ny) * Lx + nx]
        if nc == 0:
            nm = 1
        else:
            t = cell_type[nc]
            if t == substrate_type:
                ns = 1
            elif t == passive_type:
                nf = 1
    f = wp.int32(0)
    a = wp.int32(0)
    if nm == 1 and ns == 1:
        f = 1
    if ns == 1 or nf == 1:
        a = 1
    is_free[i] = f
    is_adh[i] = a
    if f == 1:
        wp.atomic_add(free_count, slot, 1)
    if a == 1:
        wp.atomic_add(adh_count, slot, 1)


# ===========================================================================
# Stage 2 -- segmented compaction (atomic-append) into per-slot ranges fixed by
# a host exclusive prefix-sum of the per-slot counts. One thread per voxel.
# ===========================================================================
@wp.kernel
def cohesotaxis_fill_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    cell_slot: wp.array(dtype=wp.int32),
    is_free: wp.array(dtype=wp.int32),
    is_adh: wp.array(dtype=wp.int32),
    free_ptr: wp.array(dtype=wp.int32),
    adh_ptr: wp.array(dtype=wp.int32),
    free_cursor: wp.array(dtype=wp.int32),
    adh_cursor: wp.array(dtype=wp.int32),
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    f = is_free[i]
    a = is_adh[i]
    if f == 0 and a == 0:
        return
    cid = ids[i]
    slot = cell_slot[cid]
    if slot < 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    if f == 1:
        p = wp.atomic_add(free_cursor, slot, 1)
        idx = free_ptr[slot] + p
        free_x[idx] = x
        free_y[idx] = y
        free_z[idx] = z
    if a == 1:
        p = wp.atomic_add(adh_cursor, slot, 1)
        idx = adh_ptr[slot] + p
        adh_x[idx] = x
        adh_y[idx] = y
        adh_z[idx] = z


# ===========================================================================
# Stage 3 -- all-pairs PixelDist reduction (one thread per Free pixel). Sums the
# Euclidean distance from the free pixel to every adhesion pixel of its cell.
# Deterministic (a per-thread serial sum -> no float atomics). EmbryoSteppables
# .PixelDist.
# ===========================================================================
@wp.kernel
def cohesotaxis_pixeldist_kernel(
    n_free: wp.int32,
    free_slot: wp.array(dtype=wp.int32),    # slot of each free pixel (global index)
    free_x: wp.array(dtype=wp.int32),
    free_y: wp.array(dtype=wp.int32),
    free_z: wp.array(dtype=wp.int32),
    adh_ptr: wp.array(dtype=wp.int32),
    adh_x: wp.array(dtype=wp.int32),
    adh_y: wp.array(dtype=wp.int32),
    adh_z: wp.array(dtype=wp.int32),
    cum_dist: wp.array(dtype=wp.float32),
):
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


# ===========================================================================
# Stage 4 -- Gumbel-max weighted select (one thread per LEADING slot). Ranks the
# slot's Free pixels ascending by CumDist (a small selection scan), maps rank ->
# log SigWeights, adds Gumbel noise from Philox keyed by (mcs, cell, base_seed),
# and picks the argmax rank. Output = the chosen free-pixel global index per slot.
#
# log_weights is uploaded as a flat array indexed [weight_ptr[slot] + rank], where
# weight_ptr segments by slot (each slot's PDF has length = its free count). This
# keeps SigWeights host-side (tiny) while the categorical draw is on-device and
# reproducible.
# ===========================================================================
@wp.kernel
def cohesotaxis_gumbel_select_kernel(
    n_slots: wp.int32,
    free_ptr: wp.array(dtype=wp.int32),     # per-slot CSR of free pixels
    cum_dist: wp.array(dtype=wp.float32),   # per free pixel (global index)
    log_weights: wp.array(dtype=wp.float32),  # per (slot,rank), segmented by free_ptr
    slot_cell: wp.array(dtype=wp.int32),    # slot -> cell id (for the RNG key)
    mcs: wp.int32,
    base_seed: wp.int32,
    chosen_free_idx: wp.array(dtype=wp.int32),  # per-slot global free index, or -1
):
    s = wp.tid()
    if s >= n_slots:
        return
    start = free_ptr[s]
    end = free_ptr[s + 1]
    nfree = end - start
    if nfree <= 0:
        chosen_free_idx[s] = -1
        return

    # RNG keyed by (mcs, cell, base_seed) -- engine convention.
    cell = slot_cell[s]
    seed = base_seed + mcs * 131072 + cell * 16384
    state = wp.rand_init(seed, cell)

    # rank r is assigned to the free pixel with the r-th smallest CumDist (ties by
    # smaller global index -> a total order matching a stable ascending sort, the
    # CPU FreePixelList.sort). For each free pixel, count how many strictly precede
    # it -> its rank; then Gumbel-max over rank's log-weight.
    best_idx = wp.int32(-1)
    best_key = wp.float32(-1.0e30)
    for gi in range(start, end):
        di = cum_dist[gi]
        # rank = #{gj : (cum<di) or (cum==di and gj<gi)}
        rank = wp.int32(0)
        for gj in range(start, end):
            dj = cum_dist[gj]
            if dj < di or (dj == di and gj < gi):
                rank += 1
        lw = log_weights[start + rank]
        # Gumbel(0,1) = -log(-log(U)); argmax(lw + g) ~ Categorical(softmax(lw)).
        u = wp.randf(state)
        # guard the log domain
        if u <= 1.0e-20:
            u = 1.0e-20
        g = -wp.log(-wp.log(u))
        key = lw + g
        if key > best_key:
            best_key = key
            best_idx = gi
    chosen_free_idx[s] = best_idx


# ===========================================================================
# Stage 5 -- Manhattan-shell argmax (one thread per query pixel). Among Substrate
# cells at EXACT Manhattan distance == n from the pixel whose zCOM > pixel.z,
# choose the id with the greatest zCOM. EmbryoSteppables.nth_order_neighbors +
# the max-zCOM pick. Deterministic argmax (ties -> larger id, a fixed total order).
# ===========================================================================
@wp.kernel
def cohesotaxis_manhattan_argmax_kernel(
    n_query: wp.int32,
    qx: wp.array(dtype=wp.int32),
    qy: wp.array(dtype=wp.int32),
    qz: wp.array(dtype=wp.int32),
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    substrate_type: wp.int32,
    n_order: wp.int32,
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    volume: wp.array(dtype=wp.float32),
    out_target: wp.array(dtype=wp.int32),   # per query: chosen substrate id or -1
):
    q = wp.tid()
    if q >= n_query:
        return
    px = qx[q]
    py = qy[q]
    pz = qz[q]
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
                    # argmax by zCOM; tie -> larger id (fixed deterministic order)
                    if zc > best_z or (zc == best_z and c > best_id):
                        best_z = zc
                        best_id = c
    out_target[q] = best_id


# ===========================================================================
# Poisson turnover decisions (one thread per cell). Bernoulli(1-exp(-rate)) keyed
# by (mcs, cell, base_seed). LeadingEdgeSteppable.step lamellipodia deletion test
# ``np.random.uniform() < 1-exp(-LamellaeRate)``.
# ===========================================================================
@wp.kernel
def poisson_turnover_kernel(
    n: wp.int32,
    prob: wp.float32,
    mcs: wp.int32,
    base_seed: wp.int32,
    decision: wp.array(dtype=wp.int32),     # 1 = delete this MCS, 0 = keep
):
    c = wp.tid()
    if c >= n:
        return
    seed = base_seed + mcs * 131072 + c * 16384
    state = wp.rand_init(seed, c)
    u = wp.randf(state)
    if u < prob:
        decision[c] = 1
    else:
        decision[c] = 0


# ---------------------------------------------------------------------------
# Host-side result container + orchestration
# ---------------------------------------------------------------------------
class ClassifyResult:
    """Per-LEADING-cell Free / Adhesion pixel segments (host views) from the
    stencil-classify + segmented-compaction stages, plus the optional CumDist."""

    def __init__(self, pipeline):
        self._p = pipeline
        self.free_ptr = None        # (n_slots+1,) int32 CSR over free pixels
        self.adh_ptr = None         # (n_slots+1,) int32 CSR over adhesion pixels
        self.free_xyz = None        # (n_free,3) int32
        self.adh_xyz = None         # (n_adh,3) int32
        self.free_slot = None       # (n_free,) int32 owning slot
        self.cum = None             # (n_free,) float32 (after pixel_dist)

    def _seg(self, ptr, xyz, cid):
        slot = self._p.cell_slot_host[cid]
        if slot < 0:
            return np.zeros((0, 3), dtype=np.int32)
        lo, hi = int(ptr[slot]), int(ptr[slot + 1])
        return xyz[lo:hi]

    def free_pixels(self, cid) -> np.ndarray:
        return self._seg(self.free_ptr, self.free_xyz, cid)

    def adhesion_pixels(self, cid) -> np.ndarray:
        return self._seg(self.adh_ptr, self.adh_xyz, cid)

    def cum_dist(self, cid) -> np.ndarray:
        slot = self._p.cell_slot_host[cid]
        if slot < 0 or self.cum is None:
            return np.zeros(0, dtype=np.float32)
        lo, hi = int(self.free_ptr[slot]), int(self.free_ptr[slot + 1])
        return self.cum[lo:hi]


class CohesotaxisPipeline:
    """Drives the staged on-device cohesotaxis lamellipodia pipeline for the
    LEADING cells of a ``GPUEngine``. Stages run as Warp kernels; the host does
    only the tiny per-cell prefix sums + SigWeights (off the hot path), exactly
    like the FPP CSR rebuild seam.
    """

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
        self.n1 = engine.n_cells + 1

        # dense leading-cell slots: cid -> 0..n_lead-1, else -1
        ctype = engine.cell_type.numpy()
        lead_ids = np.nonzero(ctype == self.leading_type)[0]
        self.lead_ids = lead_ids.astype(np.int32)
        self.n_slots = int(len(lead_ids))
        cell_slot = np.full(self.n1, -1, dtype=np.int32)
        cell_slot[lead_ids] = np.arange(self.n_slots, dtype=np.int32)
        self.cell_slot_host = cell_slot
        self.cell_slot = wp.array(cell_slot, dtype=wp.int32, device=self.device)
        self.slot_cell = wp.array(self.lead_ids, dtype=wp.int32, device=self.device)

        self.stencil = wp.array(STENCIL_18_FLAT, dtype=wp.int32, device=self.device)
        self.n_sten = int(len(_STENCIL_18))

    # --------------------------------------------------------- stage 1 + 2
    def classify(self) -> ClassifyResult:
        eng = self.engine
        nvox = eng.cfg.n_voxels
        ns = max(1, self.n_slots)

        free_count = wp.zeros(ns, dtype=wp.int32, device=self.device)
        adh_count = wp.zeros(ns, dtype=wp.int32, device=self.device)
        is_free = wp.zeros(nvox, dtype=wp.int32, device=self.device)
        is_adh = wp.zeros(nvox, dtype=wp.int32, device=self.device)

        wp.launch(
            cohesotaxis_classify_count_kernel,
            dim=nvox,
            inputs=[
                eng.ids, eng.cell_type, self.Lx, self.Ly, self.Lz,
                self.leading_type, self.substrate_type, self.passive_type,
                self.stencil, self.n_sten,
                self.cell_slot, free_count, adh_count, is_free, is_adh,
            ],
            device=self.device,
        )
        wp.synchronize()

        fc = free_count.numpy()
        ac = adh_count.numpy()
        free_ptr = np.zeros(self.n_slots + 1, dtype=np.int32)
        adh_ptr = np.zeros(self.n_slots + 1, dtype=np.int32)
        free_ptr[1:] = np.cumsum(fc[: self.n_slots])
        adh_ptr[1:] = np.cumsum(ac[: self.n_slots])
        n_free = int(free_ptr[-1])
        n_adh = int(adh_ptr[-1])

        res = ClassifyResult(self)
        res.free_ptr = free_ptr
        res.adh_ptr = adh_ptr

        d_free_ptr = wp.array(free_ptr, dtype=wp.int32, device=self.device)
        d_adh_ptr = wp.array(adh_ptr, dtype=wp.int32, device=self.device)
        free_cursor = wp.zeros(ns, dtype=wp.int32, device=self.device)
        adh_cursor = wp.zeros(ns, dtype=wp.int32, device=self.device)
        fX = wp.zeros(max(1, n_free), dtype=wp.int32, device=self.device)
        fY = wp.zeros(max(1, n_free), dtype=wp.int32, device=self.device)
        fZ = wp.zeros(max(1, n_free), dtype=wp.int32, device=self.device)
        aX = wp.zeros(max(1, n_adh), dtype=wp.int32, device=self.device)
        aY = wp.zeros(max(1, n_adh), dtype=wp.int32, device=self.device)
        aZ = wp.zeros(max(1, n_adh), dtype=wp.int32, device=self.device)

        wp.launch(
            cohesotaxis_fill_kernel,
            dim=nvox,
            inputs=[
                eng.ids, self.Lx, self.Ly, self.Lz, self.cell_slot,
                is_free, is_adh, d_free_ptr, d_adh_ptr,
                free_cursor, adh_cursor,
                fX, fY, fZ, aX, aY, aZ,
            ],
            device=self.device,
        )
        wp.synchronize()

        res.free_xyz = np.stack([fX.numpy()[:n_free], fY.numpy()[:n_free],
                                 fZ.numpy()[:n_free]], axis=1).astype(np.int32) \
            if n_free else np.zeros((0, 3), np.int32)
        res.adh_xyz = np.stack([aX.numpy()[:n_adh], aY.numpy()[:n_adh],
                                aZ.numpy()[:n_adh]], axis=1).astype(np.int32) \
            if n_adh else np.zeros((0, 3), np.int32)
        # owning slot per free pixel (from the CSR segments)
        fslot = np.zeros(n_free, dtype=np.int32)
        for s in range(self.n_slots):
            fslot[free_ptr[s]:free_ptr[s + 1]] = s
        res.free_slot = fslot

        # cache device buffers for later stages
        self._d_free_ptr = d_free_ptr
        self._d_adh_ptr = d_adh_ptr
        self._fX, self._fY, self._fZ = fX, fY, fZ
        self._aX, self._aY, self._aZ = aX, aY, aZ
        self._n_free = n_free
        self._n_adh = n_adh
        return res

    # --------------------------------------------------------- stage 3
    def pixel_dist(self, res: ClassifyResult) -> np.ndarray:
        n_free = self._n_free
        if n_free == 0:
            res.cum = np.zeros(0, dtype=np.float32)
            self._cum = wp.zeros(1, dtype=wp.float32, device=self.device)
            return res.cum
        free_slot = wp.array(res.free_slot, dtype=wp.int32, device=self.device)
        cum = wp.zeros(n_free, dtype=wp.float32, device=self.device)
        wp.launch(
            cohesotaxis_pixeldist_kernel,
            dim=n_free,
            inputs=[
                n_free, free_slot,
                self._fX, self._fY, self._fZ,
                self._d_adh_ptr, self._aX, self._aY, self._aZ, cum,
            ],
            device=self.device,
        )
        wp.synchronize()
        res.cum = cum.numpy()
        self._cum = cum
        return res.cum

    # --------------------------------------------------------- stage 4
    def gumbel_select(self, res: ClassifyResult, mcs: int) -> np.ndarray:
        """Return, per slot, the GLOBAL free-pixel index chosen by Gumbel-max over
        SigWeights (or -1 if the slot has no free pixels)."""
        if res.cum is None:
            self.pixel_dist(res)
        # build segmented log-weights: for each slot, SigWeights(nfree, sigma)
        free_ptr = res.free_ptr
        n_free = self._n_free
        log_w = np.full(max(1, n_free), -1.0e30, dtype=np.float32)
        for s in range(self.n_slots):
            lo, hi = int(free_ptr[s]), int(free_ptr[s + 1])
            k = hi - lo
            if k <= 0:
                continue
            w = sig_weights(k, self.sigma)
            with np.errstate(divide="ignore"):
                lw = np.log(np.maximum(w, 1e-300))
            log_w[lo:hi] = lw.astype(np.float32)
        d_log_w = wp.array(log_w, dtype=wp.float32, device=self.device)

        ns = max(1, self.n_slots)
        chosen = wp.full(ns, -1, dtype=wp.int32, device=self.device)
        cum = self._cum if self._n_free > 0 else wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(
            cohesotaxis_gumbel_select_kernel,
            dim=ns,
            inputs=[
                self.n_slots, self._d_free_ptr, cum, d_log_w,
                self.slot_cell, int(mcs), self.engine.base_seed, chosen,
            ],
            device=self.device,
        )
        wp.synchronize()
        return chosen.numpy()[: self.n_slots] if self.n_slots else np.zeros(0, np.int32)

    # --------------------------------------------------------- stage 5
    def manhattan_argmax_for_pixels(self, pixels) -> np.ndarray:
        """Manhattan-shell substrate argmax-by-zCOM for an explicit list of query
        pixels (n,3). Used both by the full pipeline and the exactness test."""
        eng = self.engine
        pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 3)
        nq = pixels.shape[0]
        if nq == 0:
            return np.zeros(0, dtype=np.int32)
        qx = wp.array(pixels[:, 0].copy(), dtype=wp.int32, device=self.device)
        qy = wp.array(pixels[:, 1].copy(), dtype=wp.int32, device=self.device)
        qz = wp.array(pixels[:, 2].copy(), dtype=wp.int32, device=self.device)
        out = wp.zeros(nq, dtype=wp.int32, device=self.device)
        wp.launch(
            cohesotaxis_manhattan_argmax_kernel,
            dim=nq,
            inputs=[
                nq, qx, qy, qz,
                eng.ids, eng.cell_type, self.Lx, self.Ly, self.Lz,
                self.substrate_type, self.lam_dist,
                eng.xsum, eng.ysum, eng.zsum, eng.volume, out,
            ],
            device=self.device,
        )
        wp.synchronize()
        return out.numpy()

    # --------------------------------------------------------- full pipeline
    def select_targets(self, mcs: int) -> dict:
        """Run stages 1-5 for all LEADING cells; return {cell_id: target_substrate_id}
        for cells that selected a valid lamellipodia target."""
        res = self.classify()
        if self.n_slots == 0 or self._n_free == 0:
            return {}
        self.pixel_dist(res)
        chosen = self.gumbel_select(res, mcs)             # per-slot global free idx
        valid = chosen >= 0
        if not valid.any():
            return {}
        sel_idx = chosen[valid]
        sel_pixels = res.free_xyz[sel_idx]                # (k,3)
        targets = self.manhattan_argmax_for_pixels(sel_pixels)
        out = {}
        slots = np.nonzero(valid)[0]
        for s, tgt in zip(slots, targets):
            if int(tgt) >= 0:
                out[int(self.lead_ids[s])] = int(tgt)
        return out

    def create_lamellipodia_links(self, links, mcs: int,
                                  only_cells=None) -> dict:
        """Run the full pipeline and create the selected lamellipodia FPP links via
        Pass A's ``FPPLinks.create_link`` (LamellipodiaLambda / LLTargetDist /
        LLMaxDist). ``only_cells`` (set of cell ids) restricts which leaders get a
        link this step (LeadingEdgeSteppable only relinks cells lacking a 'link').
        Returns {cell_id: target_id} for links created."""
        targets = self.select_targets(mcs)
        created = {}
        new_a, new_b = [], []
        for cell, tgt in targets.items():
            if only_cells is not None and cell not in only_cells:
                continue
            new_a.append(cell)
            new_b.append(tgt)
            created[cell] = tgt
        if new_a:  # one allocation for the whole batch (vs O(M) per link)
            links.create_links_bulk(new_a, new_b, lam=LAMELLIPODIA_LAMBDA,
                                    target=LL_TARGET_DIST, maxlen=LL_MAX_DIST)
        return created


# ---------------------------------------------------------------------------
# Standalone drivers used by the validation tests (thin wrappers over the
# kernels; keep the statistical/exactness checks independent of a full scene).
# ---------------------------------------------------------------------------
def gumbel_select_rank_histogram(n: int, sigma: float, n_draws: int, mcs: int,
                                 base_seed: int, device: str = "cuda:0") -> np.ndarray:
    """Draw ``n_draws`` independent Gumbel-max selections from ``SigWeights(n,
    sigma)`` (one per synthetic slot, key varied via the cell coordinate) and
    return the chosen-rank histogram (length n). Here CumDist == rank so the chosen
    global index equals the chosen rank, directly comparable to SigWeights."""
    # one slot per draw, each with n free pixels whose cum_dist == rank (0..n-1)
    free_ptr = (np.arange(n_draws + 1, dtype=np.int32) * n)
    cum = np.tile(np.arange(n, dtype=np.float32), n_draws)
    w = sig_weights(n, sigma)
    log_w = np.log(np.maximum(w, 1e-300)).astype(np.float32)
    log_weights = np.tile(log_w, n_draws)
    slot_cell = np.arange(n_draws, dtype=np.int32)

    d_free_ptr = wp.array(free_ptr, dtype=wp.int32, device=device)
    d_cum = wp.array(cum, dtype=wp.float32, device=device)
    d_log_w = wp.array(log_weights, dtype=wp.float32, device=device)
    d_slot_cell = wp.array(slot_cell, dtype=wp.int32, device=device)
    chosen = wp.full(n_draws, -1, dtype=wp.int32, device=device)
    wp.launch(
        cohesotaxis_gumbel_select_kernel,
        dim=n_draws,
        inputs=[n_draws, d_free_ptr, d_cum, d_log_w, d_slot_cell,
                int(mcs), int(base_seed), chosen],
        device=device,
    )
    wp.synchronize()
    ch = chosen.numpy()
    # chosen is a GLOBAL free index; convert to rank within its slot
    ranks = ch - free_ptr[:n_draws]
    return np.bincount(ranks, minlength=n).astype(np.int64)


def poisson_delete_decisions(n_cells: int, mcs: int, base_seed: int,
                             rate: float = LAMELLAE_RATE,
                             device: str = "cuda:0") -> np.ndarray:
    """Per-cell Bernoulli(1-exp(-rate)) delete decisions keyed by (mcs, cell,
    base_seed). Returns an int32 array of 0/1 (1 == delete)."""
    prob = float(1.0 - np.exp(-rate))
    dec = wp.zeros(n_cells, dtype=wp.int32, device=device)
    wp.launch(
        poisson_turnover_kernel,
        dim=n_cells,
        inputs=[int(n_cells), prob, int(mcs), int(base_seed), dec],
        device=device,
    )
    wp.synchronize()
    return dec.numpy()
