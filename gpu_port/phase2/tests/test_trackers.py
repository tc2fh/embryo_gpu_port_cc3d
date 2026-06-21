"""Tracker gate: on-GPU boundary-pixel flags + per-cell boundary counts, and the
neighbor-contact (common-surface-area) CSR, validated against an independent CPU
recompute on the SAME lattice (these are deterministic functions of the lattice,
so they must match EXACTLY -- not just statistically).

Semantics mirror the vendored CC3D trackers (read-only reference):
* BoundaryPixelTracker / NeighborTracker::isBoundaryPixel -- a voxel is a boundary
  pixel iff >=1 of its NeighborOrder-shell neighbors belongs to a different cell.
  Embryo uses NeighborOrder=1 (6 face neighbors).
* NeighborTracker common-surface-area -- count of face-adjacent (cell A, cell B)
  voxel pairs across the A-B boundary (here as a directed dense matrix -> CSR;
  includes contact with Medium, id 0).
"""

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine
from engine.config import neighbor_offsets


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cpu_boundary_and_pairs(ids, n_cells, order):
    """Reference: per-voxel boundary flag, per-cell boundary count, and dense
    directed neighbor-contact matrix, computed plainly in NumPy."""
    Lz, Ly, Lx = ids.shape
    off = neighbor_offsets(order)
    is_boundary = np.zeros_like(ids)
    n1 = n_cells + 1
    pairs = np.zeros((n1, n1), dtype=np.int64)
    zz, yy, xx = np.nonzero(np.ones_like(ids))
    self_id = ids[zz, yy, xx]
    for dx, dy, dz in off:
        nx, ny, nz = xx + dx, yy + dy, zz + dz
        valid = (nx >= 0) & (ny >= 0) & (nz >= 0) & (nx < Lx) & (ny < Ly) & (nz < Lz)
        ncell = np.zeros_like(self_id)
        ncell[valid] = ids[nz[valid], ny[valid], nx[valid]]
        diff = ncell != self_id
        # boundary flag for non-medium voxels with >=1 differing neighbor
        bnd = diff & (self_id != 0)
        is_boundary[zz[bnd], yy[bnd], xx[bnd]] = 1
        # directed contact counts (all voxels, incl medium self)
        np.add.at(pairs, (self_id[diff], ncell[diff]), 1)
    boundary_count = np.zeros(n1, dtype=np.int64)
    bz, by, bx = np.nonzero(is_boundary)
    np.add.at(boundary_count, ids[bz, by, bx], 1)
    return is_boundary, boundary_count, pairs


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_boundary_pixel_and_neighbor_csr_match_cpu():
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(10)
    eng.assert_volume_partition()

    trk = eng.recompute_trackers()
    ids = eng.get_ids()
    ref_isb, ref_bc, ref_pairs = _cpu_boundary_and_pairs(ids, eng.n_cells, cfg.tracker_neighbor_order)

    # boundary flags + counts must match exactly
    assert np.array_equal(trk["is_boundary"], ref_isb), "boundary-pixel field mismatch"
    assert np.array_equal(trk["boundary_count"], ref_bc), "per-cell boundary count mismatch"

    # neighbor CSR must reproduce the dense directed contact matrix exactly
    indptr, indices, data = trk["neighbor_csr"]
    n1 = eng.n_cells + 1
    dense = np.zeros((n1, n1), dtype=np.int64)
    for cid in range(n1):
        for k in range(indptr[cid], indptr[cid + 1]):
            dense[cid, indices[k]] = data[k]
    assert np.array_equal(dense, ref_pairs), "neighbor-contact CSR mismatch vs CPU"

    # the contact matrix is symmetric in undirected common-surface-area between
    # real cells (A->B count == B->A count)
    real = dense[1:, 1:]
    assert np.array_equal(real, real.T), "common-surface matrix not symmetric"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_surface_area_matches_cpu_recompute():
    """Per-cell surface area (GPU kernel) == CPU recompute on the same lattice."""
    from engine import CPUReference

    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=8, tracker_neighbor_order=1)
    st = build_grid_state(cfg, cells_per_axis=3)
    eng = GPUEngine(st)
    eng.run(10)
    ids = eng.get_ids()
    # CPU surface recompute from the SAME post-run lattice
    ref = CPUReference(st)
    ref.ids = ids
    ref.volume = np.bincount(ids.reshape(-1), minlength=eng.n_cells + 1).astype(np.float64)
    gpu_surf = eng.surface_areas(order=1)
    cpu_surf = ref.surface_areas(order=1)
    assert np.array_equal(gpu_surf, cpu_surf), "GPU surface area != CPU recompute"
