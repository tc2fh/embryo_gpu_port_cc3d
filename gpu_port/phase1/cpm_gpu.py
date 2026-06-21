"""GPU (NVIDIA Warp) prototype of the synthetic CPM + FPP model.

Throwaway feasibility prototype for Phase 1. Mirrors the CPU reference
(``cpm_cpu.py``) and the CC3D energy semantics, but runs GPU-resident:

* int32 id-lattice ``ids`` (0 = Medium) is the single source of truth.
* Per-cell Structure-of-Arrays: ``ctype``, ``volume``, ``target_volume``,
  ``lambda_volume`` and COM accumulators ``xsum/ysum/zsum`` (COM = sum/volume).
* 8-color checkerboard Metropolis sweep. For a Moore (NeighborOrder<=3)
  neighborhood, color = (x&1) + 2*(y&1) + 4*(z&1) makes all same-color sites
  mutually non-interacting, so they flip in parallel safely (this is the GPU
  analogue of CC3D's OpenMP subgrid checkerboard, Potts3D.cpp).
* Per-thread counter-based RNG via ``wp.rand_init(seed, offset)`` (Philox),
  keyed by (mcs, color, global_seed) as seed and the linear pixel index as
  offset -> reproducible, stateless streams.
* Energy = Volume + Contact + FPP, identical formulas to the CPU reference.
* On accept: atomic volume/COM updates (``wp.atomic_add``).
* FPP links live in a per-cell CSR (``link_ptr``, ``link_other``) rebuilt and
  pushed each MCS: atomic-append create, flag + compaction delete (links whose
  COM length exceeds ``fpp_max_length`` are dropped).

Statistical (not bit-identical) equivalence to the CPU reference is the gate.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from model import (
    MOORE_OFFSETS,
    VONNEUMANN_OFFSETS,
    ModelConfig,
    ModelState,
)

wp.init()

# Neighbor offset tables are passed into kernels as flat int32 device arrays
# (this Warp build has no fixed-size integer matrix constant type), indexed as
# off[3*n + axis]. MOORE has 26 rows (NeighborOrder<=3 shell), VN has 6 rows.


@wp.func
def _in_bounds(x: wp.int32, y: wp.int32, z: wp.int32, L: wp.int32) -> wp.bool:
    return (
        x >= 0 and y >= 0 and z >= 0 and x < L and y < L and z < L
    )


@wp.func
def _idx(x: wp.int32, y: wp.int32, z: wp.int32, L: wp.int32) -> wp.int32:
    return (z * L + y) * L + x


@wp.func
def _get_id(ids: wp.array(dtype=wp.int32), x: wp.int32, y: wp.int32, z: wp.int32, L: wp.int32) -> wp.int32:
    if not _in_bounds(x, y, z, L):
        return wp.int32(0)  # out-of-bounds = Medium
    return ids[_idx(x, y, z, L)]


@wp.func
def _fpp_delta_cell(
    cid: wp.int32,
    dvol: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    xsum: wp.array(dtype=wp.float32),
    ysum: wp.array(dtype=wp.float32),
    zsum: wp.array(dtype=wp.float32),
    volume: wp.array(dtype=wp.float32),
    link_ptr: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    lam: wp.float32,
    tgt: wp.float32,
) -> wp.float32:
    """dE_fpp for shifting cell `cid` by (dvol, dx, dy, dz) in its accumulators."""
    if cid == 0:
        return wp.float32(0.0)
    v0 = volume[cid]
    if v0 <= 0.0:
        return wp.float32(0.0)
    v1 = v0 + dvol
    if v1 <= 0.0:
        return wp.float32(0.0)
    cx0 = xsum[cid] / v0
    cy0 = ysum[cid] / v0
    cz0 = zsum[cid] / v0
    cx1 = (xsum[cid] + dx) / v1
    cy1 = (ysum[cid] + dy) / v1
    cz1 = (zsum[cid] + dz) / v1
    e = wp.float32(0.0)
    start = link_ptr[cid]
    end = link_ptr[cid + 1]
    for k in range(start, end):
        other = link_other[k]
        vo = volume[other]
        if vo <= 0.0:
            continue
        ox = xsum[other] / vo
        oy = ysum[other] / vo
        oz = zsum[other] / vo
        lb0 = cx0 - ox
        lb1 = cy0 - oy
        lb2 = cz0 - oz
        lbefore = wp.sqrt(lb0 * lb0 + lb1 * lb1 + lb2 * lb2)
        la0 = cx1 - ox
        la1 = cy1 - oy
        la2 = cz1 - oz
        lafter = wp.sqrt(la0 * la0 + la1 * la1 + la2 * la2)
        e += lam * ((lafter - tgt) * (lafter - tgt) - (lbefore - tgt) * (lbefore - tgt))
    return e


@wp.kernel
def metropolis_color_kernel(
    ids: wp.array(dtype=wp.int32),
    ctype: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float32),
    xsum: wp.array(dtype=wp.float32),
    ysum: wp.array(dtype=wp.float32),
    zsum: wp.array(dtype=wp.float32),
    target_volume: wp.array(dtype=wp.float32),
    lambda_volume: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.float32, ndim=2),
    link_ptr: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
    moore: wp.array(dtype=wp.int32),
    vn: wp.array(dtype=wp.int32),
    L: wp.int32,
    color: wp.int32,
    mcs: wp.int32,
    base_seed: wp.int32,
    temperature: wp.float32,
    fpp_lambda: wp.float32,
    fpp_target: wp.float32,
):
    """Process one checkerboard color. One thread per same-color voxel.

    The 8 colors of a 2x2x2 tiling guarantee that any two distinct same-color
    voxels are NOT within each other's 26-neighborhood, so reads of neighbor ids
    are stable within the launch and accepted flips never collide on the lattice.
    """
    tid = wp.tid()
    # decode color sublattice coordinate
    cx = color & 1
    cy = (color >> 1) & 1
    cz = (color >> 2) & 1
    # how many sites of this color along each axis
    nx = (L - cx + 1) / 2
    ny = (L - cy + 1) / 2
    nz = (L - cz + 1) / 2
    total = nx * ny * nz
    if tid >= total:
        return
    # map flat tid -> (ix,iy,iz) in the color sublattice, then to lattice coords
    ix = tid % nx
    rem = tid / nx
    iy = rem % ny
    iz = rem / ny
    x = cx + 2 * ix
    y = cy + 2 * iy
    z = cz + 2 * iz

    old_id = ids[_idx(x, y, z, L)]

    # RNG: stateless Philox keyed by (mcs, color, base_seed); offset = linear idx
    seed = base_seed + mcs * 131072 + color * 16384
    state = wp.rand_init(seed, _idx(x, y, z, L))

    # pick a face neighbor to copy FROM
    k = wp.randi(state, 0, 6)
    dx = vn[3 * k + 0]
    dy = vn[3 * k + 1]
    dz = vn[3 * k + 2]
    sx = x + dx
    sy = y + dy
    sz = z + dz
    if not _in_bounds(sx, sy, sz, L):
        return
    new_id = ids[_idx(sx, sy, sz, L)]
    if new_id == old_id:
        return

    new_t = ctype[new_id]
    old_t = ctype[old_id]

    # --- Volume delta (incremental form, matches VolumePlugin) ---
    de = wp.float32(0.0)
    if new_id != 0:
        de += lambda_volume[new_id] * (1.0 + 2.0 * (volume[new_id] - target_volume[new_id]))
    if old_id != 0:
        de += lambda_volume[old_id] * (1.0 - 2.0 * (volume[old_id] - target_volume[old_id]))

    # --- Contact delta over the Moore shell ---
    for n in range(26):
        nnx = x + moore[3 * n + 0]
        nny = y + moore[3 * n + 1]
        nnz = z + moore[3 * n + 2]
        ncell = _get_id(ids, nnx, nny, nnz, L)
        nt = ctype[ncell]
        if ncell != old_id:
            de -= contact[old_t, nt]
        if ncell != new_id:
            de += contact[new_t, nt]

    # --- FPP delta (COM is single source of truth) ---
    fx = wp.float32(x)
    fy = wp.float32(y)
    fz = wp.float32(z)
    de += _fpp_delta_cell(
        new_id, 1.0, fx, fy, fz,
        xsum, ysum, zsum, volume, link_ptr, link_other, fpp_lambda, fpp_target,
    )
    de += _fpp_delta_cell(
        old_id, -1.0, -fx, -fy, -fz,
        xsum, ysum, zsum, volume, link_ptr, link_other, fpp_lambda, fpp_target,
    )

    # --- Metropolis acceptance (Boltzmann; matches DefaultAcceptanceFunction) ---
    accept = False
    if de <= 0.0:
        accept = True
    else:
        if wp.randf(state) < wp.exp(-de / temperature):
            accept = True

    if not accept:
        return

    # --- apply: lattice write + atomic SoA updates ---
    ids[_idx(x, y, z, L)] = new_id
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


# ----------------------------------------------------------------------------
# Dynamic CSR link list: atomic-append build + flag/compaction delete.
# ----------------------------------------------------------------------------
@wp.func
def _com_dist_xyz(ax: wp.float32, ay: wp.float32, az: wp.float32,
                  bx: wp.float32, by: wp.float32, bz: wp.float32) -> wp.float32:
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def count_active_links_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    xsum: wp.array(dtype=wp.float32),
    ysum: wp.array(dtype=wp.float32),
    zsum: wp.array(dtype=wp.float32),
    volume: wp.array(dtype=wp.float32),
    max_length: wp.float32,
    keep_flag: wp.array(dtype=wp.int32),
    degree: wp.array(dtype=wp.int32),
):
    """Flag links to keep (COM length <= max) and count per-cell degree (atomic)."""
    i = wp.tid()
    a = pair_a[i]
    b = pair_b[i]
    va = volume[a]
    vb = volume[b]
    if va <= 0.0 or vb <= 0.0:
        keep_flag[i] = 0
        return
    ax = xsum[a] / va
    ay = ysum[a] / va
    az = zsum[a] / va
    bx = xsum[b] / vb
    by = ysum[b] / vb
    bz = zsum[b] / vb
    d = _com_dist_xyz(ax, ay, az, bx, by, bz)
    if d <= max_length:
        keep_flag[i] = 1
        wp.atomic_add(degree, a, 1)
        wp.atomic_add(degree, b, 1)
    else:
        keep_flag[i] = 0


@wp.kernel
def fill_csr_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    keep_flag: wp.array(dtype=wp.int32),
    link_ptr: wp.array(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),
    link_other: wp.array(dtype=wp.int32),
):
    """Atomic-append each kept (undirected) link into both endpoints' CSR ranges."""
    i = wp.tid()
    if keep_flag[i] == 0:
        return
    a = pair_a[i]
    b = pair_b[i]
    # append b to a's range
    pa = wp.atomic_add(cursor, a, 1)
    link_other[link_ptr[a] + pa] = b
    # append a to b's range
    pb = wp.atomic_add(cursor, b, 1)
    link_other[link_ptr[b] + pb] = a


class GPUEngine:
    def __init__(self, state: ModelState, device: str = "cuda:0"):
        self.cfg = state.cfg
        self.L = state.cfg.L
        self.device = device
        n = self.cfg.n_cells
        self.n_cells = n

        ids_flat = np.ascontiguousarray(state.ids.reshape(-1), dtype=np.int32)
        self.ids = wp.array(ids_flat, dtype=wp.int32, device=device)
        self.ctype = wp.array(state.types.astype(np.int32), dtype=wp.int32, device=device)
        self.volume = wp.array(state.volume.astype(np.float32), dtype=wp.float32, device=device)
        self.xsum = wp.array(state.xsum.astype(np.float32), dtype=wp.float32, device=device)
        self.ysum = wp.array(state.ysum.astype(np.float32), dtype=wp.float32, device=device)
        self.zsum = wp.array(state.zsum.astype(np.float32), dtype=wp.float32, device=device)
        self.target_volume = wp.array(state.target_volume.astype(np.float32), dtype=wp.float32, device=device)
        self.lambda_volume = wp.array(state.lambda_volume.astype(np.float32), dtype=wp.float32, device=device)
        self.contact = wp.array(self.cfg.contact_matrix().astype(np.float32), dtype=wp.float32, device=device)

        # neighbor offset tables as flat int32 device arrays (off[3*n + axis])
        self.moore = wp.array(MOORE_OFFSETS.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.vn = wp.array(VONNEUMANN_OFFSETS.flatten().astype(np.int32), dtype=wp.int32, device=device)

        # link topology (full); active CSR rebuilt each MCS
        self.pair_a = wp.array(state.link_pairs[:, 0].astype(np.int32), dtype=wp.int32, device=device)
        self.pair_b = wp.array(state.link_pairs[:, 1].astype(np.int32), dtype=wp.int32, device=device)
        self.n_pairs = int(state.link_pairs.shape[0])

        # CSR buffers sized to max possible (2 directed entries per pair)
        self.link_ptr = wp.zeros(n + 2, dtype=wp.int32, device=device)
        self.link_other = wp.zeros(max(1, 2 * self.n_pairs), dtype=wp.int32, device=device)
        self._keep = wp.zeros(max(1, self.n_pairs), dtype=wp.int32, device=device)
        self._degree = wp.zeros(n + 2, dtype=wp.int32, device=device)
        self._cursor = wp.zeros(n + 2, dtype=wp.int32, device=device)

        self.base_seed = int(self.cfg.seed)
        self.T = float(self.cfg.temperature)
        self.colors = list(range(8))
        self._max_color_threads = self.L * self.L * self.L  # safe upper bound

        self.rebuild_links()

    def rebuild_links(self):
        """Rebuild the per-cell CSR link list (delete-by-flag, atomic-append)."""
        if self.n_pairs == 0:
            self.link_ptr.zero_()
            return
        self._degree.zero_()
        self._cursor.zero_()
        wp.launch(
            count_active_links_kernel,
            dim=self.n_pairs,
            inputs=[
                self.pair_a, self.pair_b,
                self.xsum, self.ysum, self.zsum, self.volume,
                wp.float32(self.cfg.fpp_max_length),
                self._keep, self._degree,
            ],
            device=self.device,
        )
        # exclusive prefix sum of degree -> link_ptr (done on host for simplicity;
        # n_cells is tiny, this is not on the hot per-flip path)
        deg = self._degree.numpy()
        ptr = np.zeros(self.n_cells + 2, dtype=np.int32)
        ptr[1:] = np.cumsum(deg[: self.n_cells + 1])
        self.link_ptr = wp.array(ptr, dtype=wp.int32, device=self.device)
        self._cursor.zero_()
        wp.launch(
            fill_csr_kernel,
            dim=self.n_pairs,
            inputs=[
                self.pair_a, self.pair_b, self._keep,
                self.link_ptr, self._cursor, self.link_other,
            ],
            device=self.device,
        )

    def step_mcs(self, mcs: int):
        self.rebuild_links()
        for color in self.colors:
            wp.launch(
                metropolis_color_kernel,
                dim=self._max_color_threads,
                inputs=[
                    self.ids, self.ctype, self.volume,
                    self.xsum, self.ysum, self.zsum,
                    self.target_volume, self.lambda_volume, self.contact,
                    self.link_ptr, self.link_other,
                    self.moore, self.vn,
                    wp.int32(self.L), wp.int32(color), wp.int32(mcs),
                    wp.int32(self.base_seed), wp.float32(self.T),
                    wp.float32(self.cfg.fpp_lambda), wp.float32(self.cfg.fpp_target_length),
                ],
                device=self.device,
            )

    def run(self, n_mcs: int, mcs_offset: int = 0):
        for m in range(n_mcs):
            self.step_mcs(mcs_offset + m)
        wp.synchronize()

    # -- observables (pulled back to host as NumPy) --------------------------
    def _coms(self):
        vol = self.volume.numpy()
        xs = self.xsum.numpy()
        ys = self.ysum.numpy()
        zs = self.zsum.numpy()
        v = np.where(vol > 0, vol, 1.0)
        return xs / v, ys / v, zs / v, vol

    def active_link_lengths(self) -> np.ndarray:
        """Link lengths of the currently active (kept) links, COM-based."""
        cx, cy, cz, _ = self._coms()
        keep = self._keep.numpy().astype(bool)
        a = self.pair_a.numpy()[keep]
        b = self.pair_b.numpy()[keep]
        dx = cx[a] - cx[b]
        dy = cy[a] - cy[b]
        dz = cz[a] - cz[b]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def volumes(self) -> np.ndarray:
        return self.volume.numpy()[1:].copy()

    def coms(self) -> np.ndarray:
        cx, cy, cz, _ = self._coms()
        return np.stack([cx, cy, cz], axis=1)[1:]


def run_gpu(cfg: ModelConfig, n_mcs: int, state: ModelState | None = None, device: str = "cuda:0") -> GPUEngine:
    from model import build_state

    if state is None:
        state = build_state(cfg)
    eng = GPUEngine(state, device=device)
    eng.run(n_mcs)
    return eng
