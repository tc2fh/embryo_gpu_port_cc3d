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
    """Count directed (self -> neighbor) face-contacts into a dense matrix; this
    is the common-surface-area between cell pairs (NeighborTracker). Includes
    self->Medium (id 0). One thread per voxel. The dense matrix is small for the
    test models; the host compresses the non-zero rows to CSR. (For full Embryo
    scale this becomes a hashed/segmented build -- noted in findings.)"""
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
