"""Device link-management kernels (Phase 7).

Ports the per-cell HOST link-management loops of the Embryo steppables
(``embryo/steppables.py`` ``TissueLinkSteppable`` / ``PassiveSubstrateSteppable``
and ``engine/steppables.py`` ``LamellipodiaSteppable``) to device kernels that
consume Phase 6's resident device CSR handles (``engine.neighbor_csr_*_dev``) and
the device-authoritative FPP link inventory (``FPPLinks._a_dev`` ...) directly --
so the per-MCS contact-graph host copyback and the Python per-cell loops are
eliminated.

Three device primitives live here:

  1. **Tissue cap-ordered relink** (``tissue_relink_serial_kernel``): for the
     managed cells of one type, in ASCENDING id order, scan each cell's neighbor
     CSR row in CSR (ascending-neighbor-id) order and claim a tissue link to every
     non-Substrate neighbor that has no current link, while the cell holds fewer
     than its degree cap (``MaxNeighborNum`` for Passive, ``+1`` for Leading).

     The truncation order is **load-bearing**: CC3D (``EmbryoSteppables.py``
     ``LeadingEdgeSteppable``/``PassiveSteppable``) iterates ``get_cell_neighbor_
     data_list(cell)`` (NeighborOrder-1 == our order-1 CSR row, ascending) and stops
     at the cap, and the cap counts ALL link kinds on the cell and is re-checked each
     neighbor; a link created while processing a smaller-id cell is visible to (and
     consumes the budget of) a larger-id cell processed later in the SAME pass. That
     intra-pass forward dependency (smaller-id endpoint processed first, but a larger
     id still gets a second chance to claim a link the smaller one skipped for budget)
     is an inherently SERIAL greedy over the managed cells. We therefore process the
     managed cells in ascending order in a single-thread kernel that maintains a live
     per-cell degree array (init = full-inventory degree, all kinds) and an
     open-addressed membership hash of the live link set -- a faithful transcription
     of the host loop, exact by construction. The work is tiny (managed cells x small
     degree, off the hot Metropolis path), and it removes ALL the Python + the whole-
     graph PCIe copy the host path paid. The claims are appended to the device
     inventory with no host round-trip (``FPPLinks.create_links_bulk_device``).

  2. **Substrate min-id link** (``substrate_link_create_kernel``): one thread per
     passive cell; if the cell has NO substrate link (derived from the inventory) and
     borders the Substrate, claim a link to its SMALLEST-id Substrate neighbor
     (deterministic stand-in for CC3D ``random.choice`` -- which substrate cell is not
     an observable). Naturally parallel (one independent link per cell).

  3. **Poisson keep-mask** (``poisson_keep_mask_kernel``): per inventory slot, if the
     slot is a link of the target kind (matched by its per-link ``lambda``) draw a
     keyed-Philox Bernoulli(1-exp(-rate)) and mark keep=0 on a hit; all other slots
     keep=1. Feeds ``FPPLinks.compact_with_keep_mask`` (Phase 6's device keep/compact)
     so the Poisson delete never round-trips a Python ``to_delete`` set. The draw is
     keyed by (mcs, stable-link-key, stream, base_seed) -- the link's UNORDERED packed
     ``{a,b}`` key, a stable per-link identity (so it is reproducible per key and
     invariant to inventory compaction), NOT CC3D's shared-RNG draw order (carry-
     forward sanctioned substitution; the gate checks the RATE + per-key repro).

All kernels are in this real ``.py`` file (this Warp build reads kernel source via
``inspect``), pass constant tables as flat device arrays, and use only integer/id
atomics + compaction -- no float atomics (no new nondeterminism).
"""

from __future__ import annotations

import numpy as np

import warp as wp

wp.init()


# Stream offsets so independent turnover processes draw from disjoint Philox keys.
# Match embryo/steppables.py so the device path keys identically to the host path.
STREAM_TISSUE = 1
STREAM_SUBLINK = 2
STREAM_LAMELLAE = 0


# ---------------------------------------------------------------------------
# Inventory-derived per-cell structures (built once per manager pass, on device).
# ---------------------------------------------------------------------------
@wp.kernel
def inventory_degree_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    m: wp.int32,
    degree: wp.array(dtype=wp.int32),       # (n1,) per-cell live link count (all kinds)
):
    """Per-cell degree over the WHOLE current inventory (all link kinds), counting
    both endpoints -- the cap budget base (CC3D ``len(get_fpp_links_by_cell(cell))``).
    One thread per inventory slot; tombstoned (a<0) slots skipped."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:
        return
    wp.atomic_add(degree, ai, 1)
    wp.atomic_add(degree, bi, 1)


@wp.kernel
def inventory_hash_fill_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    m: wp.int32,
    mult: wp.int64,
    cap: wp.int32,                           # hash capacity (a power of 2)
    table: wp.array(dtype=wp.int64),         # open-addressed set of packed lo*mult+hi keys (-1 empty)
):
    """Insert every live inventory link's unordered packed key into the open-addressed
    membership hash (linear probing). One thread per inventory slot; integer-CAS only
    (no float atomics). Pre-fills the set the serial relink kernel then queries +
    extends, so a single probe answers 'does (c,nb) already exist' over inventory AND
    this-pass claims."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:
        return
    lo = wp.int64(ai)
    hi = wp.int64(bi)
    if bi < ai:
        lo = wp.int64(bi)
        hi = wp.int64(ai)
    key = lo * mult + hi
    h = wp.int32(key & wp.int64(cap - 1))
    for _probe in range(cap):
        prev = wp.atomic_cas(table, h, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            return
        h = (h + 1) & (cap - 1)


@wp.func
def _hash_contains(table: wp.array(dtype=wp.int64), cap: wp.int32,
                   key: wp.int64) -> wp.int32:
    """Linear-probe lookup in the open-addressed membership set."""
    h = wp.int32(key & wp.int64(cap - 1))
    for _probe in range(cap):
        v = table[h]
        if v == wp.int64(-1):
            return wp.int32(0)
        if v == key:
            return wp.int32(1)
        h = (h + 1) & (cap - 1)
    return wp.int32(0)


@wp.func
def _hash_insert_serial(table: wp.array(dtype=wp.int64), cap: wp.int32,
                        key: wp.int64):
    """Insert into the membership set from the SINGLE serial relink thread (no CAS
    needed -- only one writer)."""
    h = wp.int32(key & wp.int64(cap - 1))
    for _probe in range(cap):
        v = table[h]
        if v == wp.int64(-1) or v == key:
            table[h] = key
            return
        h = (h + 1) & (cap - 1)


# ---------------------------------------------------------------------------
# Tissue cap-ordered relink -- the load-bearing kernel (serial over managed cells).
# ---------------------------------------------------------------------------
@wp.kernel
def tissue_relink_serial_kernel(
    managed: wp.array(dtype=wp.int32),       # managed cell ids, ASCENDING
    n_managed: wp.int32,
    cap: wp.int32,                           # per-cell degree cap (MaxNeighborNum [+1])
    substrate_type: wp.int32,
    cell_type: wp.array(dtype=wp.int32),
    indptr: wp.array(dtype=wp.int64),        # neighbor CSR (order-1) device handles
    indices: wp.array(dtype=wp.int32),
    live_degree: wp.array(dtype=wp.int32),   # (n1,) init = full-inventory degree; mutated
    table: wp.array(dtype=wp.int64),         # membership set pre-filled w/ inventory keys
    tab_cap: wp.int32,                       # membership hash capacity (power of 2)
    mult: wp.int64,
    out_a: wp.array(dtype=wp.int32),         # claimed links (this pass), appended
    out_b: wp.array(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),     # slot 0 = number of claims
):
    """SINGLE thread: walk managed cells ascending; for each, scan its neighbor CSR
    row ascending and claim a tissue link to every non-Substrate, not-yet-linked
    neighbor while the cell's live degree (all kinds) is below ``cap``. Exact
    transcription of the CC3D per-cell loop (intra-pass visibility via ``live_degree``
    + the membership set). Deterministic; claims appended in (cell, neighbor) order so
    the inventory layout is stable across identical replays."""
    t = wp.tid()
    if t != 0:                               # serial: only thread 0 does the greedy
        return
    cnt = wp.int32(0)
    for mi in range(n_managed):
        c = managed[mi]
        if live_degree[c] >= cap:
            continue
        lo = wp.int32(indptr[c])
        hi = wp.int32(indptr[c + 1])
        for k in range(lo, hi):
            if live_degree[c] >= cap:
                break
            nb = indices[k]
            if nb == 0:                      # Medium
                continue
            if cell_type[nb] == substrate_type:
                continue
            cl = wp.int64(c)
            nl = wp.int64(nb)
            klo = cl
            khi = nl
            if nb < c:
                klo = nl
                khi = cl
            key = klo * mult + khi
            if _hash_contains(table, tab_cap, key) == wp.int32(1):
                continue
            # claim the undirected link c-nb
            out_a[cnt] = c
            out_b[cnt] = nb
            cnt += 1
            live_degree[c] += 1
            live_degree[nb] += 1
            _hash_insert_serial(table, tab_cap, key)
    out_count[0] = cnt


# ---------------------------------------------------------------------------
# Substrate min-id link create (parallel: one thread per passive cell).
# ---------------------------------------------------------------------------
@wp.kernel
def substrate_has_link_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    m: wp.int32,
    slink_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    has_link: wp.array(dtype=wp.int32),      # (n1,) out: 1 if cell already has a substrate link
):
    """Mark each cell that owns a substrate link in the current inventory (a link
    whose lambda == SLinkLambda, touching a Substrate endpoint). One thread per
    inventory slot. The device-native replacement for CC3D's ``cell.dict['link']``
    bookkeeping (derive 'has substrate link' from the inventory, not a side SoA)."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:
        return
    dlam = lam[i] - slink_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        return
    # the non-substrate endpoint is the passive cell that "has" the link
    if cell_type[ai] == substrate_type:
        has_link[bi] = 1
    else:
        has_link[ai] = 1


@wp.kernel
def substrate_link_create_kernel(
    passive: wp.array(dtype=wp.int32),       # passive cell ids
    n_passive: wp.int32,
    has_link: wp.array(dtype=wp.int32),      # (n1,) 1 if cell already has a substrate link
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    indptr: wp.array(dtype=wp.int64),
    indices: wp.array(dtype=wp.int32),
    out_a: wp.array(dtype=wp.int32),         # passive endpoint (per passive slot, or -1)
    out_b: wp.array(dtype=wp.int32),         # chosen min-id substrate, or -1
):
    """One thread per passive cell: if it has no substrate link and borders the
    Substrate, choose its SMALLEST-id Substrate order-1 neighbor and emit the pair.
    Independent per cell (each passive cell makes at most one substrate link), so this
    is embarrassingly parallel -- no cross-cell budget. Deterministic min-id pick =
    the sanctioned reproducible stand-in for CC3D ``random.choice`` (the *which*
    substrate cell is not a measured observable)."""
    i = wp.tid()
    if i >= n_passive:
        return
    c = passive[i]
    out_a[i] = -1
    out_b[i] = -1
    if has_link[c] != 0:
        return
    lo = wp.int32(indptr[c])
    hi = wp.int32(indptr[c + 1])
    best = wp.int32(-1)
    for k in range(lo, hi):
        nb = indices[k]
        if nb == 0:
            continue
        if cell_type[nb] != substrate_type:
            continue
        if best < 0 or nb < best:
            best = nb
    if best >= 0:
        out_a[i] = c
        out_b[i] = best


# ---------------------------------------------------------------------------
# Poisson keep-mask (per inventory slot, keyed-Philox Bernoulli of one link kind).
# ---------------------------------------------------------------------------
@wp.kernel
def poisson_keep_mask_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    m: wp.int32,
    kind_lambda: wp.float32,                 # match links of this kind by their lambda
    prob: wp.float32,                        # 1 - exp(-rate)
    mcs: wp.int32,
    base_seed: wp.int32,
    stream: wp.int32,
    keep: wp.array(dtype=wp.int32),          # (m,) out: 1 survive, 0 delete this MCS
):
    """Per inventory slot: keep=1 unless the slot is a link of ``kind_lambda`` AND a
    keyed-Philox Bernoulli(prob) fires -> keep=0. Key = (mcs, packed-{a,b}, stream,
    base_seed): a STABLE per-link identity (unordered pair key) so the decision is
    reproducible per key and invariant to inventory compaction. One thread per slot;
    the survivors feed ``compact_with_keep_mask``."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:                      # tombstone -> drop
        keep[i] = 0
        return
    dlam = lam[i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:                          # not this kind -> always keep
        keep[i] = 1
        return
    lo = ai
    hi = bi
    if bi < ai:
        lo = bi
        hi = ai
    # stable scalar link key from the unordered pair, folded into the Philox key the
    # same shape as embryo/steppables._poisson_turnover_stream_kernel.
    link_key = lo * 131072 + hi
    seed = base_seed + mcs * 131072 + link_key * 16384 + stream * 257
    state = wp.rand_init(seed, lo + hi * 7919 + stream * 13)
    u = wp.randf(state)
    if u < prob:
        keep[i] = 0
    else:
        keep[i] = 1


@wp.kernel
def append_block_kernel(
    src_a: wp.array(dtype=wp.int32),         # claim arrays (device)
    src_b: wp.array(dtype=wp.int32),
    k: wp.int32,
    base: wp.int32,                          # write at inventory[base + i]
    lam_val: wp.float32, tgt_val: wp.float32, max_val: wp.float32,
    dst_a: wp.array(dtype=wp.int32), dst_b: wp.array(dtype=wp.int32),
    dst_lam: wp.array(dtype=wp.float32), dst_tgt: wp.array(dtype=wp.float32),
    dst_max: wp.array(dtype=wp.float32),
):
    """Copy ``k`` device-resident claims into the inventory block ``[base, base+k)``
    with broadcast (scalar) per-link params. One thread per claim -- the device-native
    block append used by ``create_links_bulk_device`` (no host round-trip of the
    pairs)."""
    i = wp.tid()
    if i >= k:
        return
    j = base + i
    dst_a[j] = src_a[i]
    dst_b[j] = src_b[i]
    dst_lam[j] = lam_val
    dst_tgt[j] = tgt_val
    dst_max[j] = max_val


@wp.kernel
def emit_flag_kernel(
    emit_a: wp.array(dtype=wp.int32),        # per-slot emitted pair endpoint (or -1)
    n: wp.int32,
    flag: wp.array(dtype=wp.int32),          # out: 1 if a pair was emitted, else 0
):
    """flag[i] = (emit_a[i] >= 0). One thread per slot. Drives the exclusive scan that
    compacts the substrate kernel's emit-or-nothing output into a dense claim block."""
    i = wp.tid()
    if i >= n:
        return
    if emit_a[i] >= 0:
        flag[i] = 1
    else:
        flag[i] = 0


@wp.kernel
def compact_pairs_kernel(
    in_a: wp.array(dtype=wp.int32),          # per-slot emitted pair (or -1)
    in_b: wp.array(dtype=wp.int32),
    n: wp.int32,
    pos: wp.array(dtype=wp.int32),           # exclusive prefix sum of (in_a>=0)
    out_a: wp.array(dtype=wp.int32),
    out_b: wp.array(dtype=wp.int32),
):
    """Stream the emitted (>=0) pairs from a per-slot array into a dense claim block
    using a precomputed exclusive-prefix position -- the substrate kernel emits one
    pair-or-nothing per passive cell; this compacts them for the block append. One
    thread per slot."""
    i = wp.tid()
    if i >= n:
        return
    if in_a[i] < 0:
        return
    j = pos[i]
    out_a[j] = in_a[i]
    out_b[j] = in_b[i]


@wp.kernel
def poisson_keep_mask_by_cell_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    m: wp.int32,
    kind_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    prob: wp.float32,
    mcs: wp.int32,
    base_seed: wp.int32,
    keep: wp.array(dtype=wp.int32),
):
    """Poisson keep-mask for a kind whose links are (managed-cell, Substrate) -- e.g.
    lamellipodia. Keyed by the NON-Substrate (managed) endpoint id with the SAME
    Philox formula as ``cohesotaxis.poisson_delete_decisions`` / ``poisson_turnover_
    kernel`` (seed = base_seed + mcs*131072 + cell*16384; rand_init(seed, cell)) so the
    device keep/compact reproduces the prior per-cell lamellipodia delete decisions
    exactly. One thread per inventory slot."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:
        keep[i] = 0
        return
    dlam = lam[i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        keep[i] = 1
        return
    # the managed (non-substrate) endpoint is the RNG key (CC3D keys per leader cell)
    cell = ai
    if cell_type[ai] == substrate_type:
        cell = bi
    seed = base_seed + mcs * 131072 + cell * 16384
    state = wp.rand_init(seed, cell)
    u = wp.randf(state)
    if u < prob:
        keep[i] = 0
    else:
        keep[i] = 1


@wp.kernel
def cell_has_kind_link_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    m: wp.int32,
    kind_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    has_link: wp.array(dtype=wp.int32),      # (n1,) out: 1 if the managed endpoint has a kind link
):
    """Mark each managed (non-Substrate) cell that owns a link of ``kind_lambda`` (the
    other endpoint Substrate) in the current inventory -- the device-native 'does this
    leader still have a lamellipodia link' query (replaces the host link_target SoA for
    the recreate 'need' set). One thread per inventory slot."""
    i = wp.tid()
    if i >= m:
        return
    ai = a[i]
    bi = b[i]
    if ai < 0 or bi < 0:
        return
    dlam = lam[i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        return
    if cell_type[ai] == substrate_type:
        has_link[bi] = 1
    else:
        has_link[ai] = 1


# ---------------------------------------------------------------------------
# Host orchestration helpers (the tiny per-pass setup; off the hot Metropolis path).
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    cap = 1
    while cap < n:
        cap <<= 1
    return cap


class _LinkScratch:
    """Per-engine reusable device scratch for the link kernels (degree / membership
    hash / claim buffers / keep mask), grown on demand. One instance per FPPLinks."""

    def __init__(self, device, n1):
        self.device = device
        self.n1 = int(n1)
        self.degree = wp.zeros(self.n1, dtype=wp.int32, device=device)
        self.has_link = wp.zeros(self.n1, dtype=wp.int32, device=device)
        self._table = wp.zeros(1, dtype=wp.int64, device=device)
        self._table_cap = 0
        self._out_a = wp.zeros(1, dtype=wp.int32, device=device)
        self._out_b = wp.zeros(1, dtype=wp.int32, device=device)
        self._out_cap = 0
        self._count = wp.zeros(1, dtype=wp.int32, device=device)
        self._keep = wp.zeros(1, dtype=wp.int32, device=device)
        self._keep_cap = 0
        # substrate-create scratch (per passive cell)
        self._subflag = wp.zeros(1, dtype=wp.int32, device=device)
        self._subpos = wp.zeros(1, dtype=wp.int32, device=device)
        self._sub_a = wp.zeros(1, dtype=wp.int32, device=device)
        self._sub_b = wp.zeros(1, dtype=wp.int32, device=device)
        self._sub_cap = 0

    def _grow_sub(self, n):
        if self._sub_cap < n:
            cap = max(8, int(n))
            self._subflag = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._subpos = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._sub_a = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._sub_b = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._sub_cap = cap

    def subflag(self, n):
        self._grow_sub(n)
        return self._subflag

    def subpos(self, n):
        self._grow_sub(n)
        return self._subpos

    def subclaims(self, n):
        self._grow_sub(n)
        return self._sub_a, self._sub_b

    def table(self, need_entries):
        cap = _next_pow2(max(8, int(need_entries) * 2))
        if self._table_cap < cap:
            self._table = wp.zeros(cap, dtype=wp.int64, device=self.device)
            self._table_cap = cap
        return self._table, cap

    def claims(self, k):
        if self._out_cap < k:
            self._out_a = wp.zeros(max(8, k), dtype=wp.int32, device=self.device)
            self._out_b = wp.zeros(max(8, k), dtype=wp.int32, device=self.device)
            self._out_cap = max(8, k)
        return self._out_a, self._out_b

    def keep(self, m):
        if self._keep_cap < m:
            self._keep = wp.zeros(max(8, m), dtype=wp.int32, device=self.device)
            self._keep_cap = max(8, m)
        return self._keep
