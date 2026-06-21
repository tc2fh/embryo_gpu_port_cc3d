"""Batched (replica-axis) device link kernels (Phase 8, Objective 2).

These collapse the per-replica ``for r in range(R)`` device-kernel loops of the
batched Embryo link dynamics into SINGLE batched launches over R replicas, reusing the
exact Phase-7 per-cell logic + keying along the replica axis. The inventory is a
per-replica device-authoritative block array (``a/b/lam/tgt/max`` of shape ``R*cap``
with a per-replica live count ``m[r]`` on device), so an entire batched relink /
substrate-create / Poisson-delete + compact pass runs with NO per-replica host sync.

Bit-exactness: each replica uses its own neighbor-CSR slice (the batched global CSR
row ``r*n1 + cid``), its own membership hash region, its own ``base_seed_r[r]``, and
the SAME stable-pair / per-leader Philox keys as the single-engine Phase-7 kernels --
so a batched pass equals R independent single passes bit-identically.

The tissue relink stays SERIAL within a replica (the load-bearing cap-truncation order)
but runs all R replicas' serial greedies in parallel (dim=R, one thread per replica).
Substrate create + Poisson keep-mask + degree + compact are naturally per-(replica,
slot/cell). All integer/id atomics + keyed Philox -- no float atomics.
"""

from __future__ import annotations

import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# Per-replica inventory degree (all kinds, both endpoints) over R*cap slots.
# ---------------------------------------------------------------------------
@wp.kernel
def degree_batched_kernel(
    a: wp.array(dtype=wp.int32),            # (R*cap,) per-replica blocks
    b: wp.array(dtype=wp.int32),
    R: wp.int32, cap: wp.int32, n1: wp.int32,
    m: wp.array(dtype=wp.int32),            # (R,) per-replica live count
    degree: wp.array(dtype=wp.int32),       # (R*n1,) per-replica per-cell degree
):
    """One thread per (replica, slot): bump degree[r*n1 + endpoint] for live links."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        return
    base = r * cap
    ai = a[base + i]
    bi = b[base + i]
    if ai < 0 or bi < 0:
        return
    wp.atomic_add(degree, r * n1 + ai, 1)
    wp.atomic_add(degree, r * n1 + bi, 1)


@wp.kernel
def hash_fill_batched_kernel(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    R: wp.int32, cap: wp.int32,
    m: wp.array(dtype=wp.int32),
    mult: wp.int64,
    tab_cap: wp.int32,                      # per-replica hash capacity (power of 2)
    table: wp.array(dtype=wp.int64),        # (R*tab_cap,) per-replica membership set
):
    """Insert every live link's unordered packed key into its replica's membership hash
    region (linear probing within ``[r*tab_cap, (r+1)*tab_cap)``). One thread per
    (replica, slot); integer CAS only."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        return
    ai = a[r * cap + i]
    bi = b[r * cap + i]
    if ai < 0 or bi < 0:
        return
    lo = wp.int64(ai)
    hi = wp.int64(bi)
    if bi < ai:
        lo = wp.int64(bi)
        hi = wp.int64(ai)
    key = lo * mult + hi
    tb = r * tab_cap
    h = wp.int32(key & wp.int64(tab_cap - 1))
    for _p in range(tab_cap):
        slot = tb + h
        prev = wp.atomic_cas(table, slot, wp.int64(-1), key)
        if prev == wp.int64(-1) or prev == key:
            return
        h = (h + 1) & (tab_cap - 1)


@wp.func
def _bhash_contains(table: wp.array(dtype=wp.int64), tb: wp.int32, tab_cap: wp.int32,
                    key: wp.int64) -> wp.int32:
    h = wp.int32(key & wp.int64(tab_cap - 1))
    for _p in range(tab_cap):
        v = table[tb + h]
        if v == wp.int64(-1):
            return wp.int32(0)
        if v == key:
            return wp.int32(1)
        h = (h + 1) & (tab_cap - 1)
    return wp.int32(0)


@wp.func
def _bhash_insert(table: wp.array(dtype=wp.int64), tb: wp.int32, tab_cap: wp.int32,
                  key: wp.int64):
    h = wp.int32(key & wp.int64(tab_cap - 1))
    for _p in range(tab_cap):
        v = table[tb + h]
        if v == wp.int64(-1) or v == key:
            table[tb + h] = key
            return
        h = (h + 1) & (tab_cap - 1)


# ---------------------------------------------------------------------------
# Batched tissue cap-ordered relink: dim=R, one SERIAL greedy per replica.
# ---------------------------------------------------------------------------
@wp.kernel
def tissue_relink_batched_kernel(
    R: wp.int32, cap: wp.int32, n1: wp.int32,
    managed: wp.array(dtype=wp.int32),      # managed cell ids, ASCENDING (shared)
    n_managed: wp.int32,
    capdeg: wp.int32,                       # per-cell degree cap (MaxNeighborNum[+1])
    substrate_type: wp.int32,
    cell_type: wp.array(dtype=wp.int32),    # shared
    indptr: wp.array(dtype=wp.int64),       # (R*n1+1,) batched global CSR row pointer
    indices: wp.array(dtype=wp.int32),      # (n_contacts,) global indices
    live_degree: wp.array(dtype=wp.int32),  # (R*n1,) init = full per-replica degree; mutated
    table: wp.array(dtype=wp.int64),        # (R*tab_cap,) membership set, pre-filled
    tab_cap: wp.int32,
    mult: wp.int64,
    a: wp.array(dtype=wp.int32),            # (R*cap,) inventory blocks (appended in place)
    b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    tgt: wp.array(dtype=wp.float32),
    mx: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.int32),            # (R,) per-replica live count (advanced)
    lam_val: wp.float32, tgt_val: wp.float32, max_val: wp.float32,
):
    """One thread per REPLICA: walk managed cells ascending; for each, scan its CSR row
    (replica r's global row r*n1+c) ascending and claim a tissue link to every
    non-Substrate, not-yet-linked neighbor while the cell's live degree < cap. Appends
    into replica r's inventory block at ``[r*cap + m[r]]`` and advances ``m[r]``. Exact
    transcription of the single serial relink, per replica -> bit-identical."""
    r = wp.tid()
    if r >= R:
        return
    deg_base = r * n1
    tb = r * tab_cap
    blk = r * cap
    cnt = m[r]
    for mi in range(n_managed):
        c = managed[mi]
        if live_degree[deg_base + c] >= capdeg:
            continue
        row = r * n1 + c
        lo = wp.int32(indptr[row])
        hi = wp.int32(indptr[row + 1])
        for k in range(lo, hi):
            if live_degree[deg_base + c] >= capdeg:
                break
            nb = indices[k]
            if nb == 0:
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
            if _bhash_contains(table, tb, tab_cap, key) == wp.int32(1):
                continue
            a[blk + cnt] = c
            b[blk + cnt] = nb
            lam[blk + cnt] = lam_val
            tgt[blk + cnt] = tgt_val
            mx[blk + cnt] = max_val
            cnt += 1
            live_degree[deg_base + c] += 1
            live_degree[deg_base + nb] += 1
            _bhash_insert(table, tb, tab_cap, key)
    m[r] = cnt


# ---------------------------------------------------------------------------
# Batched substrate min-id create: parallel emit per (replica, passive cell) then a
# per-replica serial append (dim=R) to keep the inventory block dense + deterministic.
# ---------------------------------------------------------------------------
@wp.kernel
def substrate_has_link_batched_kernel(
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    R: wp.int32, cap: wp.int32, n1: wp.int32,
    m: wp.array(dtype=wp.int32),
    slink_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    has_link: wp.array(dtype=wp.int32),     # (R*n1,) out: 1 if cell already has a sub link
):
    """One thread per (replica, slot): mark the non-substrate endpoint of each live
    substrate-kind link as having a substrate link."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        return
    ai = a[r * cap + i]
    bi = b[r * cap + i]
    if ai < 0 or bi < 0:
        return
    dlam = lam[r * cap + i] - slink_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        return
    if cell_type[ai] == substrate_type:
        has_link[r * n1 + bi] = 1
    else:
        has_link[r * n1 + ai] = 1


@wp.kernel
def substrate_emit_batched_kernel(
    R: wp.int32, n1: wp.int32,
    passive: wp.array(dtype=wp.int32), n_passive: wp.int32,
    has_link: wp.array(dtype=wp.int32),     # (R*n1,)
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    indptr: wp.array(dtype=wp.int64),       # (R*n1+1,) batched global CSR
    indices: wp.array(dtype=wp.int32),
    emit_a: wp.array(dtype=wp.int32),       # (R*n_passive,) passive endpoint or -1
    emit_b: wp.array(dtype=wp.int32),       # (R*n_passive,) min-id substrate or -1
):
    """One thread per (replica, passive cell): if it has no substrate link and borders
    the Substrate, emit a link to its SMALLEST-id Substrate neighbor (replica r's CSR
    row). Independent per cell -> parallel. Deterministic min-id pick."""
    gid = wp.tid()
    r = gid / n_passive
    j = gid % n_passive
    if r >= R:
        return
    c = passive[j]
    out = r * n_passive + j
    emit_a[out] = -1
    emit_b[out] = -1
    if has_link[r * n1 + c] != 0:
        return
    row = r * n1 + c
    lo = wp.int32(indptr[row])
    hi = wp.int32(indptr[row + 1])
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
        emit_a[out] = c
        emit_b[out] = best


@wp.kernel
def substrate_append_batched_kernel(
    R: wp.int32, cap: wp.int32, n_passive: wp.int32,
    emit_a: wp.array(dtype=wp.int32), emit_b: wp.array(dtype=wp.int32),
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32), tgt: wp.array(dtype=wp.float32),
    mx: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.int32),
    lam_val: wp.float32, tgt_val: wp.float32, max_val: wp.float32,
):
    """One thread per REPLICA: serially append replica r's emitted substrate pairs (in
    passive-cell order -> deterministic, dense) into its inventory block, advancing
    m[r]. Matches the single-engine emit order (passive ascending)."""
    r = wp.tid()
    if r >= R:
        return
    blk = r * cap
    eb = r * n_passive
    cnt = m[r]
    for j in range(n_passive):
        if emit_a[eb + j] < 0:
            continue
        a[blk + cnt] = emit_a[eb + j]
        b[blk + cnt] = emit_b[eb + j]
        lam[blk + cnt] = lam_val
        tgt[blk + cnt] = tgt_val
        mx[blk + cnt] = max_val
        cnt += 1
    m[r] = cnt


# ---------------------------------------------------------------------------
# Batched lamellipodia create: append per-replica selected (leader, target) pairs.
# ---------------------------------------------------------------------------
@wp.kernel
def lamellipodia_append_batched_kernel(
    R: wp.int32, cap: wp.int32, n1: wp.int32, n_lead: wp.int32,
    targets: wp.array(dtype=wp.int32),      # (R*n_lead,) chosen substrate id or -1
    lead_ids: wp.array(dtype=wp.int32),     # (n_lead,) leader cell id per dense slot
    lam: wp.array(dtype=wp.float32),        # inventory lam column (kind matching)
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    inv_lam: wp.array(dtype=wp.float32), inv_tgt: wp.array(dtype=wp.float32),
    inv_max: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.int32),
    has_link: wp.array(dtype=wp.int32),     # (R*n1,) 1 if leader already has a lam link
    only_need: wp.int32,
    lam_val: wp.float32, tgt_val: wp.float32, max_val: wp.float32,
):
    """One thread per REPLICA: serially append (leader -> target) lamellipodia links for
    leaders that selected a target (and, if only_need, lack one), in dense-slot order ->
    deterministic, matches the single create order. Advances m[r]."""
    r = wp.tid()
    if r >= R:
        return
    blk = r * cap
    cnt = m[r]
    for s in range(n_lead):
        tgt = targets[r * n_lead + s]
        if tgt < 0:
            continue
        cell = lead_ids[s]
        if only_need == 1 and has_link[r * n1 + cell] != 0:
            continue
        a[blk + cnt] = cell
        b[blk + cnt] = tgt
        inv_lam[blk + cnt] = lam_val
        inv_tgt[blk + cnt] = tgt_val
        inv_max[blk + cnt] = max_val
        cnt += 1
    m[r] = cnt


@wp.kernel
def lam_has_link_batched_kernel(
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    R: wp.int32, cap: wp.int32, n1: wp.int32,
    m: wp.array(dtype=wp.int32),
    kind_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32),
    substrate_type: wp.int32,
    has_link: wp.array(dtype=wp.int32),     # (R*n1,) out
):
    """Mark each managed (non-substrate) cell that owns a link of kind_lambda. One
    thread per (replica, slot)."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        return
    ai = a[r * cap + i]
    bi = b[r * cap + i]
    if ai < 0 or bi < 0:
        return
    dlam = lam[r * cap + i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        return
    if cell_type[ai] == substrate_type:
        has_link[r * n1 + bi] = 1
    else:
        has_link[r * n1 + ai] = 1


# ---------------------------------------------------------------------------
# Batched Poisson keep-mask + stable compaction (per replica).
# ---------------------------------------------------------------------------
@wp.kernel
def poisson_keep_mask_batched_kernel(
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    R: wp.int32, cap: wp.int32,
    m: wp.array(dtype=wp.int32),
    kind_lambda: wp.float32, prob: wp.array(dtype=wp.float32),   # (R,) per-replica rate
    mcs: wp.int32, base_seed_r: wp.array(dtype=wp.int32), stream: wp.int32,
    keep: wp.array(dtype=wp.int32),         # (R*cap,) 1 survive / 0 delete (live slots)
):
    """Per (replica, slot): keep=1 unless the slot is a link of kind_lambda AND a keyed
    Bernoulli(prob[r]) fires. Key = (mcs, packed-{a,b}, stream, base_seed_r[r]) -- the
    SAME stable-pair key as the single-engine Phase-7 kernel, per replica. ``prob`` is
    per-replica so a sweep varies the delete rate across the batch axis."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        keep[gid] = 0
        return
    ai = a[r * cap + i]
    bi = b[r * cap + i]
    if ai < 0 or bi < 0:
        keep[gid] = 0
        return
    dlam = lam[r * cap + i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        keep[gid] = 1
        return
    lo = ai
    hi = bi
    if bi < ai:
        lo = bi
        hi = ai
    link_key = lo * 131072 + hi
    seed = base_seed_r[r] + mcs * 131072 + link_key * 16384 + stream * 257
    state = wp.rand_init(seed, lo + hi * 7919 + stream * 13)
    u = wp.randf(state)
    if u < prob[r]:
        keep[gid] = 0
    else:
        keep[gid] = 1


@wp.kernel
def poisson_keep_mask_by_cell_batched_kernel(
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32),
    R: wp.int32, cap: wp.int32,
    m: wp.array(dtype=wp.int32),
    kind_lambda: wp.float32,
    cell_type: wp.array(dtype=wp.int32), substrate_type: wp.int32,
    prob: wp.array(dtype=wp.float32), mcs: wp.int32,        # (R,) per-replica rate
    base_seed_r: wp.array(dtype=wp.int32),
    keep: wp.array(dtype=wp.int32),
):
    """Per (replica, slot): lamellipodia keep-mask keyed by the non-substrate (leader)
    endpoint with the SAME per-cell Philox formula as the single-engine kernel, per
    replica's base seed. ``prob`` is per-replica (the LamellaeRate sweep axis)."""
    gid = wp.tid()
    r = gid / cap
    i = gid % cap
    if r >= R:
        return
    if i >= m[r]:
        keep[gid] = 0
        return
    ai = a[r * cap + i]
    bi = b[r * cap + i]
    if ai < 0 or bi < 0:
        keep[gid] = 0
        return
    dlam = lam[r * cap + i] - kind_lambda
    if dlam < 0.0:
        dlam = -dlam
    if dlam > 1.0e-3:
        keep[gid] = 1
        return
    cell = ai
    if cell_type[ai] == substrate_type:
        cell = bi
    seed = base_seed_r[r] + mcs * 131072 + cell * 16384
    state = wp.rand_init(seed, cell)
    u = wp.randf(state)
    if u < prob[r]:
        keep[gid] = 0
    else:
        keep[gid] = 1


@wp.kernel
def compact_batched_kernel(
    R: wp.int32, cap: wp.int32,
    keep: wp.array(dtype=wp.int32),         # (R*cap,) keep flags (live slots)
    m: wp.array(dtype=wp.int32),            # (R,) live count (updated)
    a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
    lam: wp.array(dtype=wp.float32), tgt: wp.array(dtype=wp.float32),
    mx: wp.array(dtype=wp.float32),
):
    """One thread per REPLICA: stable in-place compaction of replica r's block (a
    serial scan that writes survivors forward, preserving insertion order). Updates
    m[r]. Serial per replica keeps the survivor order deterministic (== the single
    engine's stable compaction)."""
    r = wp.tid()
    if r >= R:
        return
    blk = r * cap
    w = wp.int32(0)
    old = m[r]
    for i in range(old):
        if keep[blk + i] == 1:
            if w != i:
                a[blk + w] = a[blk + i]
                b[blk + w] = b[blk + i]
                lam[blk + w] = lam[blk + i]
                tgt[blk + w] = tgt[blk + i]
                mx[blk + w] = mx[blk + i]
            w += 1
    m[r] = w
