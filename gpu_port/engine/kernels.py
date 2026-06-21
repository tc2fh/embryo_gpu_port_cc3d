"""Warp GPU kernels for the CC3D-GPU production engine (Volume + Contact + trackers).

Kernels live in this real ``.py`` file because this Warp build reads kernel source
via ``inspect`` (no ``exec()``'d kernels -- Phase 1 carry-forward constraint).

Energy semantics mirror the vendored CC3D source (read-only reference):

* Volume  (``VolumePlugin::changeEnergyByCellType``):
      dE = lambda[t_new] * (1 + 2*(V_new - Vt_new))      [if new cell != Medium]
         + lambda[t_old] * (1 - 2*(V_old - Vt_old))      [if old cell != Medium]
  the incremental form of lambda*(V - Vt)^2.

* Contact (``ContactPlugin::changeEnergy``): over the NeighborOrder shell of the
  changing pixel,  dE = sum [ J(new,nCell) - J(old,nCell) ] with the term skipped
  when the neighbor *is* the cell on that side (``nCell != oldCell`` / ``!= newCell``).
  Medium = type 0. (The Embryo model gives every cell its own clusterId, so the
  ContactPlugin clusterId branch reduces to this plain id-inequality form.)

* Metropolis (``DefaultAcceptanceFunction``): accept if dE <= 0 else with prob
  exp(-dE / T).  k = 1, offset = 0.

Flip mechanic (mirrors ``Potts3D::metropolisFast``): one thread per same-color
target voxel; pick a random neighbor (the "source") whose cell id becomes the
candidate value for the target voxel -- i.e. the target pixel flips to the
source pixel's cell. Frozen types are never a source or target.

Trackers: volume via float ``atomic_add``; COM via **int64 fixed-point**
``atomic_add`` of integer pixel coordinates (bit-reproducible, no float
non-associativity). Per-cell volume is asserted == lattice voxel count exactly.
"""

from __future__ import annotations

import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# small device helpers
# ---------------------------------------------------------------------------
@wp.func
def in_bounds(x: wp.int32, y: wp.int32, z: wp.int32,
              Lx: wp.int32, Ly: wp.int32, Lz: wp.int32) -> wp.bool:
    return x >= 0 and y >= 0 and z >= 0 and x < Lx and y < Ly and z < Lz


@wp.func
def lin_idx(x: wp.int32, y: wp.int32, z: wp.int32,
            Lx: wp.int32, Ly: wp.int32) -> wp.int32:
    return (z * Ly + y) * Lx + x


@wp.func
def get_id(ids: wp.array(dtype=wp.int32),
           x: wp.int32, y: wp.int32, z: wp.int32,
           Lx: wp.int32, Ly: wp.int32, Lz: wp.int32) -> wp.int32:
    # out-of-bounds is Medium (id 0), matching non-periodic CC3D boundaries
    if not in_bounds(x, y, z, Lx, Ly, Lz):
        return wp.int32(0)
    return ids[lin_idx(x, y, z, Lx, Ly)]


# ---------------------------------------------------------------------------
# FocalPointPlasticity spring-energy delta (device func)
# ---------------------------------------------------------------------------
@wp.func
def fpp_delta_cell(
    cid: wp.int32,
    dvol: wp.float32,
    dx: wp.float32, dy: wp.float32, dz: wp.float32,
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    volume: wp.array(dtype=wp.float32),
    link_ptr: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    link_lambda: wp.array(dtype=wp.float32),
    link_target: wp.array(dtype=wp.float32),
) -> wp.float32:
    """dE_fpp for shifting cell ``cid`` by (dvol, dx, dy, dz) in its COM
    accumulators, over its active FPP links.

    Link length L = || COM_cid - COM_other ||_2, with COM read from the engine's
    int64 xsum/ysum/zsum / volume (the exact, reproducible single source of truth
    -- no separate COM tracker). Energy per link = lambda*(L - target)^2; the
    constant ``offset`` cancels in the before/after delta.

    Mirrors FocalPointPlasticityPlugin::potentialFunction + distInvariantCM
    (plain Euclidean for non-periodic BC). Medium (id 0) has no links.
    """
    if cid == 0:
        return wp.float32(0.0)
    v0 = volume[cid]
    if v0 <= 0.0:
        return wp.float32(0.0)
    v1 = v0 + dvol
    if v1 <= 0.0:
        return wp.float32(0.0)
    # COM before / after the proposed single-voxel shift (exact float64-of-int math
    # done in float32 here; identical functional form to the CPU reference)
    x0 = wp.float64(xsum[cid])
    y0 = wp.float64(ysum[cid])
    z0 = wp.float64(zsum[cid])
    cx0 = wp.float32(x0 / wp.float64(v0))
    cy0 = wp.float32(y0 / wp.float64(v0))
    cz0 = wp.float32(z0 / wp.float64(v0))
    cx1 = wp.float32((x0 + wp.float64(dx)) / wp.float64(v1))
    cy1 = wp.float32((y0 + wp.float64(dy)) / wp.float64(v1))
    cz1 = wp.float32((z0 + wp.float64(dz)) / wp.float64(v1))
    e = wp.float32(0.0)
    start = link_ptr[cid]
    end = link_ptr[cid + 1]
    for k in range(start, end):
        other = link_other[k]
        vo = volume[other]
        if vo <= 0.0:
            continue
        ox = wp.float32(wp.float64(xsum[other]) / wp.float64(vo))
        oy = wp.float32(wp.float64(ysum[other]) / wp.float64(vo))
        oz = wp.float32(wp.float64(zsum[other]) / wp.float64(vo))
        lam = link_lambda[k]
        tgt = link_target[k]
        b0 = cx0 - ox
        b1 = cy0 - oy
        b2 = cz0 - oz
        lbefore = wp.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
        a0 = cx1 - ox
        a1 = cy1 - oy
        a2 = cz1 - oz
        lafter = wp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        e += lam * ((lafter - tgt) * (lafter - tgt) - (lbefore - tgt) * (lbefore - tgt))
    return e


# ---------------------------------------------------------------------------
# Metropolis -- one checkerboard color per launch
# ---------------------------------------------------------------------------
@wp.kernel
def metropolis_color_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    target_volume: wp.array(dtype=wp.float32),     # per-cell
    lambda_volume: wp.array(dtype=wp.float32),     # per-cell
    contact: wp.array(dtype=wp.float32, ndim=2),   # (n_types,n_types)
    type_frozen: wp.array(dtype=wp.int32),         # (n_types,) 1 if frozen
    contact_off: wp.array(dtype=wp.int32),         # flat 3*Nc contact-shell offsets
    n_contact: wp.int32,
    flip_off: wp.array(dtype=wp.int32),            # flat 3*Nf flip-target offsets
    n_flip: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    color: wp.int32,
    mcs: wp.int32,
    base_seed: wp.int32,
    temperature: wp.float32,
    # --- FocalPointPlasticity (Phase 3). fpp_enabled==0 -> no-op (arrays may be
    #     length-1 dummies). Links are STATIC within a color sweep (rebuilt at the
    #     per-MCS steppable boundary), so reading the CSR here is race-free. ---
    fpp_enabled: wp.int32,
    link_ptr: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    link_lambda: wp.array(dtype=wp.float32),
    link_target: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    # ---- decode this color's sublattice and map tid -> lattice coords ----
    cx = color & 1
    cy = (color >> 1) & 1
    cz = (color >> 2) & 1
    nx = (Lx - cx + 1) / 2
    ny = (Ly - cy + 1) / 2
    nz = (Lz - cz + 1) / 2
    total = nx * ny * nz
    if tid >= total:
        return
    ix = tid % nx
    rem = tid / nx
    iy = rem % ny
    iz = rem / ny
    x = cx + 2 * ix
    y = cy + 2 * iy
    z = cz + 2 * iz

    target_idx = lin_idx(x, y, z, Lx, Ly)
    old_id = ids[target_idx]
    old_t = cell_type[old_id]

    # frozen target never flips (CC3D Potts3D: checkIfFrozen is gated on a
    # non-null cell -> Medium (id 0) is NEVER frozen and always participates).
    if old_id != 0 and type_frozen[old_t] == 1:
        return

    # ---- RNG: stateless Philox keyed by (mcs, color, base_seed) ----
    seed = base_seed + mcs * 131072 + color * 16384
    state = wp.rand_init(seed, target_idx)

    # ---- pick a source neighbor; its cell id is the candidate new_id ----
    k = wp.randi(state, 0, n_flip)
    sx = x + flip_off[3 * k + 0]
    sy = y + flip_off[3 * k + 1]
    sz = z + flip_off[3 * k + 2]
    if not in_bounds(sx, sy, sz, Lx, Ly, Lz):
        return
    new_id = ids[lin_idx(sx, sy, sz, Lx, Ly)]
    if new_id == old_id:
        return
    new_t = cell_type[new_id]
    # frozen source cannot spread (Medium id 0 always allowed, as above)
    if new_id != 0 and type_frozen[new_t] == 1:
        return

    # ---- Volume delta (incremental, VolumePlugin::changeEnergyByCellType) ----
    de = float(0.0)
    if new_id != 0:
        de += lambda_volume[new_id] * (1.0 + 2.0 * (volume[new_id] - target_volume[new_id]))
    if old_id != 0:
        de += lambda_volume[old_id] * (1.0 - 2.0 * (volume[old_id] - target_volume[old_id]))

    # ---- Contact delta over the contact shell (ContactPlugin::changeEnergy) ----
    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        nt = cell_type[ncell]
        if ncell != old_id:
            de -= contact[old_t, nt]
        if ncell != new_id:
            de += contact[new_t, nt]

    # ---- FPP spring delta at the SAME changePixel/newCell evaluation point ----
    # The voxel (x,y,z) leaves old_id and joins new_id (mirrors the apply below).
    # COM is read from xsum/ysum/zsum / volume (the exact single source of truth).
    if fpp_enabled != 0:
        fpx = wp.float32(x)
        fpy = wp.float32(y)
        fpz = wp.float32(z)
        de += fpp_delta_cell(
            new_id, 1.0, fpx, fpy, fpz,
            xsum, ysum, zsum, volume, link_ptr, link_other, link_lambda, link_target,
        )
        de += fpp_delta_cell(
            old_id, -1.0, -fpx, -fpy, -fpz,
            xsum, ysum, zsum, volume, link_ptr, link_other, link_lambda, link_target,
        )

    # ---- Metropolis acceptance (DefaultAcceptanceFunction) ----
    accept = False
    if de <= 0.0:
        accept = True
    else:
        if wp.randf(state) < wp.exp(-de / temperature):
            accept = True
    if not accept:
        return

    # ---- apply: lattice write + atomic SoA updates ----
    ids[target_idx] = new_id
    fx = wp.int64(x)
    fy = wp.int64(y)
    fz = wp.int64(z)
    if old_id != 0:
        wp.atomic_add(volume, old_id, -1.0)
        wp.atomic_add(xsum, old_id, -fx)
        wp.atomic_add(ysum, old_id, -fy)
        wp.atomic_add(zsum, old_id, -fz)
    if new_id != 0:
        wp.atomic_add(volume, new_id, 1.0)
        wp.atomic_add(xsum, new_id, fx)
        wp.atomic_add(ysum, new_id, fy)
        wp.atomic_add(zsum, new_id, fz)


# ---------------------------------------------------------------------------
# CUDA-Graph capture support (Phase 4 Pass B)
#
# A captured CUDA graph BAKES every *scalar* kernel argument in at capture time,
# so a graph recorded with a literal ``mcs`` would replay that SAME ``mcs`` (and
# thus the SAME Philox key ``base_seed + mcs*131072 + color*16384``) on every
# replay -- physically wrong (each MCS must draw a fresh RNG stream). To replay a
# SINGLE captured graph across many MCS, the step index must live in DEVICE memory
# and advance on-device between replays.
#
# ``metropolis_color_dev_mcs_kernel`` is a mechanical copy of
# ``metropolis_color_kernel`` differing in EXACTLY one line: ``mcs`` is read from
# ``mcs_dev[0]`` (a 1-element device array) instead of a baked-in scalar. Every
# other line -- the RNG key, the flip mechanic, the Volume/Contact/FPP deltas, the
# Metropolis test, and the int64 atomic COM apply -- is byte-identical, so a
# graph-captured run is BIT-EXACT to the eager ``metropolis_color_kernel`` run with
# the same seed (asserted by the Pass B graph==eager test, which guards this copy
# against any divergence). ``incr_mcs_kernel`` does the on-device ``mcs_dev[0] += 1``
# captured at the end of each MCS so replaying the graph N times walks mcs = m0,
# m0+1, ..., m0+N-1 exactly as the eager host loop does.
# ---------------------------------------------------------------------------
@wp.kernel
def incr_mcs_kernel(mcs_dev: wp.array(dtype=wp.int32)):
    # single-thread on-device increment of the per-MCS step counter; captured as
    # the last node of the per-MCS graph so each replay advances the RNG key.
    if wp.tid() == 0:
        mcs_dev[0] = mcs_dev[0] + 1


@wp.kernel
def metropolis_color_dev_mcs_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    target_volume: wp.array(dtype=wp.float32),     # per-cell
    lambda_volume: wp.array(dtype=wp.float32),     # per-cell
    contact: wp.array(dtype=wp.float32, ndim=2),   # (n_types,n_types)
    type_frozen: wp.array(dtype=wp.int32),         # (n_types,) 1 if frozen
    contact_off: wp.array(dtype=wp.int32),         # flat 3*Nc contact-shell offsets
    n_contact: wp.int32,
    flip_off: wp.array(dtype=wp.int32),            # flat 3*Nf flip-target offsets
    n_flip: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    color: wp.int32,
    mcs_dev: wp.array(dtype=wp.int32),             # <-- device-resident mcs (vs scalar)
    base_seed: wp.int32,
    temperature: wp.float32,
    fpp_enabled: wp.int32,
    link_ptr: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    link_lambda: wp.array(dtype=wp.float32),
    link_target: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    cx = color & 1
    cy = (color >> 1) & 1
    cz = (color >> 2) & 1
    nx = (Lx - cx + 1) / 2
    ny = (Ly - cy + 1) / 2
    nz = (Lz - cz + 1) / 2
    total = nx * ny * nz
    if tid >= total:
        return
    ix = tid % nx
    rem = tid / nx
    iy = rem % ny
    iz = rem / ny
    x = cx + 2 * ix
    y = cy + 2 * iy
    z = cz + 2 * iz

    target_idx = lin_idx(x, y, z, Lx, Ly)
    old_id = ids[target_idx]
    old_t = cell_type[old_id]

    if old_id != 0 and type_frozen[old_t] == 1:
        return

    # ---- RNG: stateless Philox keyed by (mcs, color, base_seed); mcs from device
    mcs = mcs_dev[0]
    seed = base_seed + mcs * 131072 + color * 16384
    state = wp.rand_init(seed, target_idx)

    k = wp.randi(state, 0, n_flip)
    sx = x + flip_off[3 * k + 0]
    sy = y + flip_off[3 * k + 1]
    sz = z + flip_off[3 * k + 2]
    if not in_bounds(sx, sy, sz, Lx, Ly, Lz):
        return
    new_id = ids[lin_idx(sx, sy, sz, Lx, Ly)]
    if new_id == old_id:
        return
    new_t = cell_type[new_id]
    if new_id != 0 and type_frozen[new_t] == 1:
        return

    de = float(0.0)
    if new_id != 0:
        de += lambda_volume[new_id] * (1.0 + 2.0 * (volume[new_id] - target_volume[new_id]))
    if old_id != 0:
        de += lambda_volume[old_id] * (1.0 - 2.0 * (volume[old_id] - target_volume[old_id]))

    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        nt = cell_type[ncell]
        if ncell != old_id:
            de -= contact[old_t, nt]
        if ncell != new_id:
            de += contact[new_t, nt]

    if fpp_enabled != 0:
        fpx = wp.float32(x)
        fpy = wp.float32(y)
        fpz = wp.float32(z)
        de += fpp_delta_cell(
            new_id, 1.0, fpx, fpy, fpz,
            xsum, ysum, zsum, volume, link_ptr, link_other, link_lambda, link_target,
        )
        de += fpp_delta_cell(
            old_id, -1.0, -fpx, -fpy, -fpz,
            xsum, ysum, zsum, volume, link_ptr, link_other, link_lambda, link_target,
        )

    accept = False
    if de <= 0.0:
        accept = True
    else:
        if wp.randf(state) < wp.exp(-de / temperature):
            accept = True
    if not accept:
        return

    ids[target_idx] = new_id
    fx = wp.int64(x)
    fy = wp.int64(y)
    fz = wp.int64(z)
    if old_id != 0:
        wp.atomic_add(volume, old_id, -1.0)
        wp.atomic_add(xsum, old_id, -fx)
        wp.atomic_add(ysum, old_id, -fy)
        wp.atomic_add(zsum, old_id, -fz)
    if new_id != 0:
        wp.atomic_add(volume, new_id, 1.0)
        wp.atomic_add(xsum, new_id, fx)
        wp.atomic_add(ysum, new_id, fy)
        wp.atomic_add(zsum, new_id, fz)


# ---------------------------------------------------------------------------
# BATCHED Metropolis (Phase 4 Pass A) -- R independent replicas, one launch.
#
# Layout (documented, coalescing-preserving): the replica axis is the SLOWEST
# (leading) dimension. The id-lattice is flat ``ids[r*nvox + lin_idx(x,y,z)]`` and
# every per-cell SoA array is ``arr[r*n1 + cid]``; within a fixed replica the voxel
# stride is exactly the single-engine stride, so per-color voxel reads/writes stay
# coalesced. One thread = (replica, color-voxel) pair: ``dim = R*color_threads``,
# ``r = tid/color_threads``, ``local = tid%color_threads``. Replicas never read or
# write outside their own [r*nvox, (r+1)*nvox) / [r*n1, (r+1)*n1) slices, so they
# are independent (no cross-replica interaction).
#
# Reproducibility: the Philox key folds the replica in via a PER-REPLICA base seed
# (``base_seed_r[r]``, host sets it to base_seed + r*stride) and is otherwise the
# single-engine key ``+ mcs*131072 + color*16384``; ``rand_init``'s second arg is
# the LOCAL voxel index (0..nvox-1), identical to the single engine. Hence batched
# replica r == a single GPUEngine run seeded base_seed + r*stride, BIT-EXACT. COM /
# volume use int64 / float atomics into the replica's own slice -- the int64 COM is
# bit-reproducible; volume is an exact voxel count (NO float atomics across the
# batch axis: each replica's atomics target a disjoint address range).
#
# Per-replica swept params: ``contact`` is flat ``[r*nt*nt + t1*nt + t2]``,
# ``target_volume``/``lambda_volume`` are per-replica-per-cell ``[r*n1 + cid]``,
# ``temperature`` is per-replica ``[r]``. FPP is single-replica only (Phase 4 Pass
# A defers batched FPP); ``fpp_enabled`` is 0 in the batched path.
# ---------------------------------------------------------------------------
@wp.func
def contact_rt(contact: wp.array(dtype=wp.float32),
               r: wp.int32, n_types: wp.int32,
               t1: wp.int32, t2: wp.int32) -> wp.float32:
    return contact[r * n_types * n_types + t1 * n_types + t2]


@wp.func
def get_id_b(ids: wp.array(dtype=wp.int32),
             vox_base: wp.int32,
             x: wp.int32, y: wp.int32, z: wp.int32,
             Lx: wp.int32, Ly: wp.int32, Lz: wp.int32) -> wp.int32:
    # out-of-bounds is Medium (id 0); in-bounds reads the replica's own slice
    if not in_bounds(x, y, z, Lx, Ly, Lz):
        return wp.int32(0)
    return ids[vox_base + lin_idx(x, y, z, Lx, Ly)]


@wp.kernel
def metropolis_color_batched_kernel(
    ids: wp.array(dtype=wp.int32),                 # (R*nvox,) flat, replica-major
    cell_type: wp.array(dtype=wp.int32),           # (n_types-indexed) per-cell type, shared
    cell_type_r: wp.int32,                         # 0 -> cell_type shared; 1 -> per-replica (R*n1)
    volume: wp.array(dtype=wp.float32),            # (R*n1,)
    xsum: wp.array(dtype=wp.int64),                # (R*n1,)
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    target_volume: wp.array(dtype=wp.float32),     # (R*n1,) per-replica-per-cell
    lambda_volume: wp.array(dtype=wp.float32),     # (R*n1,)
    contact: wp.array(dtype=wp.float32),           # (R*n_types*n_types,) flat per replica
    n_types: wp.int32,
    type_frozen: wp.array(dtype=wp.int32),         # (n_types,) shared across replicas
    contact_off: wp.array(dtype=wp.int32),
    n_contact: wp.int32,
    flip_off: wp.array(dtype=wp.int32),
    n_flip: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nvox: wp.int32,
    n1: wp.int32,
    R: wp.int32,
    color_threads: wp.int32,
    color: wp.int32,
    mcs: wp.int32,
    base_seed_r: wp.array(dtype=wp.int32),         # (R,) per-replica base seed
    temperature: wp.array(dtype=wp.float32),       # (R,) per-replica temperature
):
    gid = wp.tid()
    r = gid / color_threads
    tid = gid % color_threads
    if r >= R:
        return

    # ---- decode this color's sublattice and map tid -> lattice coords ----
    cx = color & 1
    cy = (color >> 1) & 1
    cz = (color >> 2) & 1
    nx = (Lx - cx + 1) / 2
    ny = (Ly - cy + 1) / 2
    nz = (Lz - cz + 1) / 2
    total = nx * ny * nz
    if tid >= total:
        return
    ix = tid % nx
    rem = tid / nx
    iy = rem % ny
    iz = rem / ny
    x = cx + 2 * ix
    y = cy + 2 * iy
    z = cz + 2 * iz

    # per-replica base offsets into the flat lattice / SoA
    vox_base = r * nvox
    cell_base = r * n1
    ct_base = cell_type_r * cell_base       # 0 if shared, r*n1 if per-replica

    local_idx = lin_idx(x, y, z, Lx, Ly)
    target_idx = vox_base + local_idx
    old_id = ids[target_idx]
    old_t = cell_type[ct_base + old_id]

    if old_id != 0 and type_frozen[old_t] == 1:
        return

    # ---- RNG: stateless Philox keyed by (mcs, color, per-replica base seed) ----
    # rand_init's 2nd arg is the LOCAL voxel index -> identical stream to a single
    # GPUEngine run seeded base_seed_r[r] (bit-exact batched==single).
    seed = base_seed_r[r] + mcs * 131072 + color * 16384
    state = wp.rand_init(seed, local_idx)

    # ---- pick a source neighbor; its cell id is the candidate new_id ----
    k = wp.randi(state, 0, n_flip)
    sx = x + flip_off[3 * k + 0]
    sy = y + flip_off[3 * k + 1]
    sz = z + flip_off[3 * k + 2]
    if not in_bounds(sx, sy, sz, Lx, Ly, Lz):
        return
    new_id = ids[vox_base + lin_idx(sx, sy, sz, Lx, Ly)]
    if new_id == old_id:
        return
    new_t = cell_type[ct_base + new_id]
    if new_id != 0 and type_frozen[new_t] == 1:
        return

    # ---- Volume delta (incremental, VolumePlugin::changeEnergyByCellType) ----
    de = float(0.0)
    if new_id != 0:
        de += lambda_volume[cell_base + new_id] * (
            1.0 + 2.0 * (volume[cell_base + new_id] - target_volume[cell_base + new_id])
        )
    if old_id != 0:
        de += lambda_volume[cell_base + old_id] * (
            1.0 - 2.0 * (volume[cell_base + old_id] - target_volume[cell_base + old_id])
        )

    # ---- Contact delta over the contact shell (ContactPlugin::changeEnergy) ----
    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id_b(ids, vox_base, nnx, nny, nnz, Lx, Ly, Lz)
        nt = cell_type[ct_base + ncell]
        if ncell != old_id:
            de -= contact_rt(contact, r, n_types, old_t, nt)
        if ncell != new_id:
            de += contact_rt(contact, r, n_types, new_t, nt)

    # ---- Metropolis acceptance (DefaultAcceptanceFunction) ----
    accept = False
    if de <= 0.0:
        accept = True
    else:
        if wp.randf(state) < wp.exp(-de / temperature[r]):
            accept = True
    if not accept:
        return

    # ---- apply: lattice write + atomic SoA updates (replica-local slice) ----
    ids[target_idx] = new_id
    fx = wp.int64(x)
    fy = wp.int64(y)
    fz = wp.int64(z)
    if old_id != 0:
        wp.atomic_add(volume, cell_base + old_id, -1.0)
        wp.atomic_add(xsum, cell_base + old_id, -fx)
        wp.atomic_add(ysum, cell_base + old_id, -fy)
        wp.atomic_add(zsum, cell_base + old_id, -fz)
    if new_id != 0:
        wp.atomic_add(volume, cell_base + new_id, 1.0)
        wp.atomic_add(xsum, cell_base + new_id, fx)
        wp.atomic_add(ysum, cell_base + new_id, fy)
        wp.atomic_add(zsum, cell_base + new_id, fz)


# ---------------------------------------------------------------------------
# BATCHED device-mcs Metropolis (Phase 4 Pass B): mechanical copy of
# ``metropolis_color_batched_kernel`` differing ONLY in reading ``mcs`` from
# ``mcs_dev[0]`` (device-resident) so one captured CUDA graph replays across many
# MCS with a correctly-advancing per-replica Philox key. Byte-identical otherwise
# (bit-exact to the eager batched kernel; the Pass B graph==eager test guards it).
# ---------------------------------------------------------------------------
@wp.kernel
def metropolis_color_batched_dev_mcs_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    cell_type_r: wp.int32,
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    target_volume: wp.array(dtype=wp.float32),
    lambda_volume: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.float32),
    n_types: wp.int32,
    type_frozen: wp.array(dtype=wp.int32),
    contact_off: wp.array(dtype=wp.int32),
    n_contact: wp.int32,
    flip_off: wp.array(dtype=wp.int32),
    n_flip: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nvox: wp.int32,
    n1: wp.int32,
    R: wp.int32,
    color_threads: wp.int32,
    color: wp.int32,
    mcs_dev: wp.array(dtype=wp.int32),             # <-- device-resident mcs (vs scalar)
    base_seed_r: wp.array(dtype=wp.int32),
    temperature: wp.array(dtype=wp.float32),
):
    gid = wp.tid()
    r = gid / color_threads
    tid = gid % color_threads
    if r >= R:
        return

    cx = color & 1
    cy = (color >> 1) & 1
    cz = (color >> 2) & 1
    nx = (Lx - cx + 1) / 2
    ny = (Ly - cy + 1) / 2
    nz = (Lz - cz + 1) / 2
    total = nx * ny * nz
    if tid >= total:
        return
    ix = tid % nx
    rem = tid / nx
    iy = rem % ny
    iz = rem / ny
    x = cx + 2 * ix
    y = cy + 2 * iy
    z = cz + 2 * iz

    vox_base = r * nvox
    cell_base = r * n1
    ct_base = cell_type_r * cell_base

    local_idx = lin_idx(x, y, z, Lx, Ly)
    target_idx = vox_base + local_idx
    old_id = ids[target_idx]
    old_t = cell_type[ct_base + old_id]

    if old_id != 0 and type_frozen[old_t] == 1:
        return

    mcs = mcs_dev[0]
    seed = base_seed_r[r] + mcs * 131072 + color * 16384
    state = wp.rand_init(seed, local_idx)

    k = wp.randi(state, 0, n_flip)
    sx = x + flip_off[3 * k + 0]
    sy = y + flip_off[3 * k + 1]
    sz = z + flip_off[3 * k + 2]
    if not in_bounds(sx, sy, sz, Lx, Ly, Lz):
        return
    new_id = ids[vox_base + lin_idx(sx, sy, sz, Lx, Ly)]
    if new_id == old_id:
        return
    new_t = cell_type[ct_base + new_id]
    if new_id != 0 and type_frozen[new_t] == 1:
        return

    de = float(0.0)
    if new_id != 0:
        de += lambda_volume[cell_base + new_id] * (
            1.0 + 2.0 * (volume[cell_base + new_id] - target_volume[cell_base + new_id])
        )
    if old_id != 0:
        de += lambda_volume[cell_base + old_id] * (
            1.0 - 2.0 * (volume[cell_base + old_id] - target_volume[cell_base + old_id])
        )

    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id_b(ids, vox_base, nnx, nny, nnz, Lx, Ly, Lz)
        nt = cell_type[ct_base + ncell]
        if ncell != old_id:
            de -= contact_rt(contact, r, n_types, old_t, nt)
        if ncell != new_id:
            de += contact_rt(contact, r, n_types, new_t, nt)

    accept = False
    if de <= 0.0:
        accept = True
    else:
        if wp.randf(state) < wp.exp(-de / temperature[r]):
            accept = True
    if not accept:
        return

    ids[target_idx] = new_id
    fx = wp.int64(x)
    fy = wp.int64(y)
    fz = wp.int64(z)
    if old_id != 0:
        wp.atomic_add(volume, cell_base + old_id, -1.0)
        wp.atomic_add(xsum, cell_base + old_id, -fx)
        wp.atomic_add(ysum, cell_base + old_id, -fy)
        wp.atomic_add(zsum, cell_base + old_id, -fz)
    if new_id != 0:
        wp.atomic_add(volume, cell_base + new_id, 1.0)
        wp.atomic_add(xsum, cell_base + new_id, fx)
        wp.atomic_add(ysum, cell_base + new_id, fy)
        wp.atomic_add(zsum, cell_base + new_id, fz)


@wp.kernel
def recompute_volume_com_batched_kernel(
    ids: wp.array(dtype=wp.int32),                 # (R*nvox,)
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nvox: wp.int32,
    n1: wp.int32,
    volume: wp.array(dtype=wp.float32),            # (R*n1,)
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
):
    """Batched recompute of per-cell volume + int64 COM sums from the id-lattice
    (one thread per (replica,voxel)). Arrays must be zeroed first. Used to init and
    to assert the incremental atomics never drift, per replica."""
    g = wp.tid()
    r = g / nvox
    i = g % nvox
    cid = ids[g]
    if cid == 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    cell_base = r * n1
    wp.atomic_add(volume, cell_base + cid, 1.0)
    wp.atomic_add(xsum, cell_base + cid, wp.int64(x))
    wp.atomic_add(ysum, cell_base + cid, wp.int64(y))
    wp.atomic_add(zsum, cell_base + cid, wp.int64(z))


@wp.kernel
def total_contact_energy_batched_kernel(
    ids: wp.array(dtype=wp.int32),                 # (R*nvox,)
    cell_type: wp.array(dtype=wp.int32),
    cell_type_r: wp.int32,
    contact: wp.array(dtype=wp.float32),           # (R*n_types*n_types,)
    n_types: wp.int32,
    contact_off: wp.array(dtype=wp.int32),
    n_contact: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nvox: wp.int32,
    n1: wp.int32,
    contact_accum: wp.array(dtype=wp.float32),     # (R,) atomically summed per replica
):
    """Per-replica contact energy = 0.5 * sum over voxels of sum over shell
    J(t_self,t_nbr) for differing-cell neighbors. One thread per (replica,voxel)."""
    g = wp.tid()
    r = g / nvox
    i = g % nvox
    vox_base = r * nvox
    cell_base = r * n1
    ct_base = cell_type_r * cell_base
    self_id = ids[g]
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    self_t = cell_type[ct_base + self_id]
    e = float(0.0)
    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id_b(ids, vox_base, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            nt = cell_type[ct_base + ncell]
            e += contact_rt(contact, r, n_types, self_t, nt)
    wp.atomic_add(contact_accum, r, 0.5 * e)


@wp.kernel
def surface_batched_kernel(
    ids: wp.array(dtype=wp.int32),                 # (R*nvox,)
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nvox: wp.int32,
    n1: wp.int32,
    surf_off: wp.array(dtype=wp.int32),
    n_surf: wp.int32,
    surface: wp.array(dtype=wp.int32),             # (R*n1,) per-cell surface area
):
    """Per-replica per-cell surface area = count of (voxel, shell-neighbor) pairs
    where the neighbor belongs to a different cell. One thread per (replica,voxel)."""
    g = wp.tid()
    r = g / nvox
    i = g % nvox
    vox_base = r * nvox
    cell_base = r * n1
    self_id = ids[g]
    if self_id == 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    s = wp.int32(0)
    for n in range(n_surf):
        nnx = x + surf_off[3 * n + 0]
        nny = y + surf_off[3 * n + 1]
        nnz = z + surf_off[3 * n + 2]
        ncell = get_id_b(ids, vox_base, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            s += 1
    wp.atomic_add(surface, cell_base + self_id, s)


# ---------------------------------------------------------------------------
# Tracker / observable kernels
# ---------------------------------------------------------------------------
@wp.kernel
def recompute_volume_com_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
):
    """Recompute per-cell volume + integer COM sums from the id-lattice (one
    thread per voxel). Used to (a) initialize and (b) assert the incremental
    atomics never drift. Arrays must be zeroed before launch."""
    i = wp.tid()
    cid = ids[i]
    if cid == 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    wp.atomic_add(volume, cid, 1.0)
    wp.atomic_add(xsum, cid, wp.int64(x))
    wp.atomic_add(ysum, cid, wp.int64(y))
    wp.atomic_add(zsum, cid, wp.int64(z))


@wp.kernel
def total_energy_kernel(
    ids: wp.array(dtype=wp.int32),
    cell_type: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float32),
    target_volume: wp.array(dtype=wp.float32),
    lambda_volume: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.float32, ndim=2),
    contact_off: wp.array(dtype=wp.int32),
    n_contact: wp.int32,
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    contact_accum: wp.array(dtype=wp.float32),     # length 1, atomically summed
):
    """Contact energy = 0.5 * sum over voxels of sum over shell J(t_self,t_nbr)
    for neighbors of a *different* cell. (Volume energy is added on host from the
    per-cell volumes -- it is a cheap exact reduction.) One thread per voxel."""
    i = wp.tid()
    self_id = ids[i]
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    self_t = cell_type[self_id]
    e = float(0.0)
    for n in range(n_contact):
        nnx = x + contact_off[3 * n + 0]
        nny = y + contact_off[3 * n + 1]
        nnz = z + contact_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            nt = cell_type[ncell]
            e += contact[self_t, nt]
    # each unordered boundary pair counted twice -> 0.5 factor
    wp.atomic_add(contact_accum, 0, 0.5 * e)


@wp.kernel
def surface_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    surf_off: wp.array(dtype=wp.int32),
    n_surf: wp.int32,
    surface: wp.array(dtype=wp.int32),             # per-cell surface area
):
    """Per-cell surface area = count of (voxel, shell-neighbor) pairs where the
    neighbor belongs to a different cell. One thread per voxel; atomic per cell.
    Matches the common-surface notion used by BoundaryPixelTracker/Contact."""
    i = wp.tid()
    self_id = ids[i]
    if self_id == 0:
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    s = wp.int32(0)
    for n in range(n_surf):
        nnx = x + surf_off[3 * n + 0]
        nny = y + surf_off[3 * n + 1]
        nnz = z + surf_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            s += 1
    wp.atomic_add(surface, self_id, s)


# ---------------------------------------------------------------------------
# Boundary-pixel + neighbor-contact (CSR) kernels -- recomputed once per MCS
# ---------------------------------------------------------------------------
@wp.kernel
def boundary_pixel_flag_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nbr_off: wp.array(dtype=wp.int32),
    n_nbr: wp.int32,
    is_boundary: wp.array(dtype=wp.int32),         # per-voxel 0/1
    boundary_count: wp.array(dtype=wp.int32),      # per-cell count of boundary voxels
):
    """A voxel is a boundary pixel iff at least one of its NeighborOrder-shell
    neighbors belongs to a different cell (matches BoundaryPixelTracker /
    NeighborTracker::isBoundaryPixel). One thread per voxel."""
    i = wp.tid()
    self_id = ids[i]
    if self_id == 0:
        is_boundary[i] = 0
        return
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    flag = wp.int32(0)
    for n in range(n_nbr):
        nnx = x + nbr_off[3 * n + 0]
        nny = y + nbr_off[3 * n + 1]
        nnz = z + nbr_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            flag = 1
    is_boundary[i] = flag
    if flag == 1:
        wp.atomic_add(boundary_count, self_id, 1)


@wp.kernel
def neighbor_contact_count_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nbr_off: wp.array(dtype=wp.int32),
    n_nbr: wp.int32,
    n_cells_p1: wp.int32,
    pair_counts: wp.array(dtype=wp.int32),         # dense (n_cells+1)^2, atomically summed
):
    """LEGACY dense-matrix neighbor-contact counter (Phase 2). Counts directed
    (self -> neighbor) face-contacts into a dense (n_cells+1)^2 matrix. Superseded
    in Phase 3 by ``neighbor_contact_hash_kernel`` (+ ``GPUEngine.neighbor_contact_csr``),
    an O(#contacts) hashed build that scales to 63k-cell Embryo (the dense matrix
    is ~16 GB there). Kept for reference / small-scale cross-checks only. Includes
    self->Medium (id 0). One thread per voxel."""
    i = wp.tid()
    self_id = ids[i]
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    for n in range(n_nbr):
        nnx = x + nbr_off[3 * n + 0]
        nny = y + nbr_off[3 * n + 1]
        nnz = z + nbr_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            wp.atomic_add(pair_counts, self_id * n_cells_p1 + ncell, 1)


# ---------------------------------------------------------------------------
# FocalPointPlasticity dynamic link CSR: flag/count (delete) + atomic-append
# (create) -- the proven Phase 1 pattern (gpu_port/phase1/cpm_gpu.py), here with
# PER-LINK lambda/target/max (CC3D new_fpp_link(cell_a, cell_b, lam, tgt, max)).
# ---------------------------------------------------------------------------
@wp.kernel
def fpp_count_active_links_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    pair_max: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.int64),
    ysum: wp.array(dtype=wp.int64),
    zsum: wp.array(dtype=wp.int64),
    volume: wp.array(dtype=wp.float32),
    keep_flag: wp.array(dtype=wp.int32),
    degree: wp.array(dtype=wp.int32),
):
    """Flag links to keep (COM length <= the link's own max_length) and count the
    per-cell degree (atomic). A dead cell (volume 0) drops its links. One thread
    per undirected link."""
    i = wp.tid()
    a = pair_a[i]
    b = pair_b[i]
    if a < 0 or b < 0:                 # tombstoned (deleted) link slot
        keep_flag[i] = 0
        return
    va = volume[a]
    vb = volume[b]
    if va <= 0.0 or vb <= 0.0:
        keep_flag[i] = 0
        return
    ax = wp.float64(xsum[a]) / wp.float64(va)
    ay = wp.float64(ysum[a]) / wp.float64(va)
    az = wp.float64(zsum[a]) / wp.float64(va)
    bx = wp.float64(xsum[b]) / wp.float64(vb)
    by = wp.float64(ysum[b]) / wp.float64(vb)
    bz = wp.float64(zsum[b]) / wp.float64(vb)
    dx = wp.float32(ax - bx)
    dy = wp.float32(ay - by)
    dz = wp.float32(az - bz)
    d = wp.sqrt(dx * dx + dy * dy + dz * dz)
    if d <= pair_max[i]:
        keep_flag[i] = 1
        wp.atomic_add(degree, a, 1)
        wp.atomic_add(degree, b, 1)
    else:
        keep_flag[i] = 0


@wp.kernel
def fpp_fill_csr_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    pair_lambda: wp.array(dtype=wp.float32),
    pair_target: wp.array(dtype=wp.float32),
    keep_flag: wp.array(dtype=wp.int32),
    link_ptr: wp.array(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    link_lambda: wp.array(dtype=wp.float32),
    link_target: wp.array(dtype=wp.float32),
):
    """Atomic-append each kept undirected link into BOTH endpoints' CSR ranges,
    carrying the per-link lambda/target so the energy kernel reads them directly.
    One thread per undirected link."""
    i = wp.tid()
    if keep_flag[i] == 0:
        return
    a = pair_a[i]
    b = pair_b[i]
    lam = pair_lambda[i]
    tgt = pair_target[i]
    pa = wp.atomic_add(cursor, a, 1)
    slot_a = link_ptr[a] + pa
    link_other[slot_a] = b
    link_lambda[slot_a] = lam
    link_target[slot_a] = tgt
    pb = wp.atomic_add(cursor, b, 1)
    slot_b = link_ptr[b] + pb
    link_other[slot_b] = a
    link_lambda[slot_b] = lam
    link_target[slot_b] = tgt


# ---------------------------------------------------------------------------
# Scalable neighbor-contact CSR: open-addressing device hash over directed
# (self,neighbor) pairs -> O(#contacts) memory instead of the dense (n+1)^2
# matrix (the Phase 2 scale blocker; ~16 GB at 63k cells). Exact: counts are
# exact, the host compacts the occupied slots and sorts by source to a CSR.
# ---------------------------------------------------------------------------
@wp.kernel
def neighbor_contact_hash_kernel(
    ids: wp.array(dtype=wp.int32),
    Lx: wp.int32, Ly: wp.int32, Lz: wp.int32,
    nbr_off: wp.array(dtype=wp.int32),
    n_nbr: wp.int32,
    n_cells_p1: wp.int64,
    cap: wp.int32,
    ht_key: wp.array(dtype=wp.int64),      # packed key src*n1+dst, -1 = empty
    ht_count: wp.array(dtype=wp.int32),    # contact count for that key
):
    """For each voxel, for each shell-neighbor of a different cell, insert the
    packed directed key ``self_id*n1 + ncell`` into the hash table (CAS into an
    empty slot, linear probing) and atomically bump its count. One thread per
    voxel. ``cap`` MUST be a power of two and > number of distinct directed pairs."""
    i = wp.tid()
    self_id = ids[i]
    x = i % Lx
    rem = i / Lx
    y = rem % Ly
    z = rem / Ly
    for n in range(n_nbr):
        nnx = x + nbr_off[3 * n + 0]
        nny = y + nbr_off[3 * n + 1]
        nnz = z + nbr_off[3 * n + 2]
        ncell = get_id(ids, nnx, nny, nnz, Lx, Ly, Lz)
        if ncell != self_id:
            key = wp.int64(self_id) * n_cells_p1 + wp.int64(ncell)
            # hash (Knuth multiplicative) into [0, cap); cap is a power of two
            h = wp.int32((key * wp.int64(2654435761)) & wp.int64(cap - 1))
            for _p in range(cap):
                slot = (h + _p) & (cap - 1)
                prev = wp.atomic_cas(ht_key, slot, wp.int64(-1), key)
                if prev == wp.int64(-1) or prev == key:
                    wp.atomic_add(ht_count, slot, 1)
                    break


# ---------------------------------------------------------------------------
# On-device neighbor-contact CSR compaction (post-Phase-4 perf): turn the hash
# table (``neighbor_contact_hash_kernel`` output) into a compact CSR sorted
# ascending within each source row WITHOUT copying the cap-sized table to host.
# Mirrors the FPP link-CSR build: count per-source degree -> host cumsum ->
# atomic-cursor scatter -> per-row sort. ``cap`` threads for count/scatter (one
# per hash slot); ``n1`` threads for the per-row sort (one per source cell).
# ---------------------------------------------------------------------------
@wp.kernel
def neighbor_csr_count_kernel(
    ht_key: wp.array(dtype=wp.int64),      # packed key src*n1+dst, -1 = empty
    cap: wp.int32,
    n_cells_p1: wp.int64,
    row_counts: wp.array(dtype=wp.int32),  # per-source-row occupied-slot count
):
    """One thread per hash slot: for each occupied slot, bump its source row's
    count (the per-cell out-degree of the directed contact graph)."""
    i = wp.tid()
    if i >= cap:
        return
    key = ht_key[i]
    if key < wp.int64(0):
        return
    src = wp.int32(key / n_cells_p1)
    wp.atomic_add(row_counts, src, 1)


@wp.kernel
def neighbor_csr_compact_kernel(
    ht_key: wp.array(dtype=wp.int64),
    ht_count: wp.array(dtype=wp.int32),
    cap: wp.int32,
    cursor: wp.array(dtype=wp.int32),      # ONE global append counter (len>=1), zeroed
    out_keys: wp.array(dtype=wp.int64),    # packed key src*n1+dst (n_contacts)
    out_data: wp.array(dtype=wp.int32),    # contact counts (n_contacts)
):
    """One thread per hash slot: stream each occupied (packed-key, count) into dense
    arrays via a single global append cursor. Output order is arbitrary; a following
    GLOBAL radix sort on the packed key (src*n1+dst) restores both src-major grouping
    and dst-ascending within-row order in one pass -- far cheaper than a segmented
    sort over n1 tiny, highly skewed segments (the Medium row holds ~every surface
    cell while most rows hold a handful)."""
    i = wp.tid()
    if i >= cap:
        return
    key = ht_key[i]
    if key < wp.int64(0):
        return
    pos = wp.atomic_add(cursor, 0, 1)
    out_keys[pos] = key
    out_data[pos] = ht_count[i]


@wp.kernel
def neighbor_csr_extract_dst_kernel(
    keys: wp.array(dtype=wp.int64),        # radix-sorted packed keys src*n1+dst
    n: wp.int32,
    n_cells_p1: wp.int64,
    out_dst: wp.array(dtype=wp.int32),     # dst id per contact (CSR indices)
):
    """One thread per contact: dst = key % n1, as int32. Lets the host copy back a
    compact int32 indices array instead of the int64 keys (half the PCIe volume, no
    host-side modulo over the whole double-buffer -- copyback was the CSR hotspot)."""
    i = wp.tid()
    if i >= n:
        return
    out_dst[i] = wp.int32(keys[i] % n_cells_p1)


# Within-row ascending order is obtained by a single GLOBAL wp.utils.radix_sort_pairs
# on the int64 packed key src*n1+dst (see GPUEngine._compact_csr_device): ascending
# key == ascending (src, dst) since dst < n1, so one O(#contacts) sort yields the CSR
# layout directly. This replaced an earlier segmented_sort_pairs over the CSR rows,
# which CUB serviced poorly given ~n1 tiny, highly skewed segments (~7 ms -> <1 ms).
# A per-row comparison-sort kernel was tried first and removed: O(k^2) per row, and
# the one giant Medium row serialized onto a single thread dominated the whole build.
