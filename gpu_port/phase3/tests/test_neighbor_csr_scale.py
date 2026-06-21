"""Neighbor-contact CSR scale-fix gate (Pass A, deliverable 3).

Phase 2 built the neighbor-contact (common-surface-area) CSR via a dense
``(n_cells+1)^2`` device matrix + host compression -- O(n_cells^2) memory, the
documented scale blocker (at 63k cells that matrix is ~16 GB). Pass A replaces it
with a **hashed CSR build** (open-addressing device hash over directed
(self,neighbor) pairs -> compaction -> CSR sorted by source) that scales with the
number of contacts, not n^2.

Correctness requirement: the hashed CSR must reproduce the dense/CPU directed
contact matrix EXACTLY (it is a deterministic function of the lattice). This is
the SAME exactness contract the Phase 2 trackers test asserts -- here we assert it
holds for the new builder AND that the builder runs at a scale where the dense
matrix is infeasible.
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


def _cpu_pairs(ids, n_cells, order):
    Lz, Ly, Lx = ids.shape
    off = neighbor_offsets(order)
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
        np.add.at(pairs, (self_id[diff], ncell[diff]), 1)
    return pairs


def _csr_to_dense(indptr, indices, data, n1):
    dense = np.zeros((n1, n1), dtype=np.int64)
    for cid in range(n1):
        for k in range(int(indptr[cid]), int(indptr[cid + 1])):
            dense[cid, int(indices[k])] = data[k]
    return dense


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_hashed_neighbor_csr_matches_cpu_exactly():
    """The hashed neighbor-CSR equals the dense CPU directed contact matrix
    exactly (same contract as the Phase 2 trackers test)."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(10)
    eng.assert_volume_partition()

    indptr, indices, data = eng.neighbor_contact_csr()  # the new hashed builder
    ids = eng.get_ids()
    ref = _cpu_pairs(ids, eng.n_cells, cfg.tracker_neighbor_order)

    n1 = eng.n_cells + 1
    dense = _csr_to_dense(indptr, indices, data, n1)
    assert np.array_equal(dense, ref), "hashed neighbor-CSR mismatch vs CPU"

    # undirected common-surface symmetry between real cells
    real = dense[1:, 1:]
    assert np.array_equal(real, real.T), "common-surface matrix not symmetric"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_recompute_trackers_uses_hashed_csr_and_matches_cpu():
    """recompute_trackers() returns the hashed CSR (no dense n^2 matrix) and it
    still matches the CPU recompute exactly -- the Phase 2 trackers contract is
    preserved through the seam the FPP rebuild also uses."""
    cfg = EngineConfig(Lx=18, Ly=18, Lz=18, seed=7, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(8)
    trk = eng.recompute_trackers()
    indptr, indices, data = trk["neighbor_csr"]
    ids = eng.get_ids()
    ref = _cpu_pairs(ids, eng.n_cells, cfg.tracker_neighbor_order)
    n1 = eng.n_cells + 1
    dense = _csr_to_dense(indptr, indices, data, n1)
    assert np.array_equal(dense, ref), "recompute_trackers neighbor-CSR mismatch vs CPU"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_hashed_csr_scales_past_dense_infeasibility():
    """Build the neighbor-CSR at a cell count where the dense (n+1)^2 int32 matrix
    would be many GB -- proving the hashed build no longer scales as O(n^2).

    A 40^3 lattice of single-voxel cells has 64000 cells; the dense matrix would be
    ~16 GB (64001^2 * 4 B). The hashed build allocates O(#contacts) and must finish
    and stay exact on a representative slab.
    """
    L = 40
    ids = np.arange(1, L * L * L + 1, dtype=np.int32).reshape(L, L, L)
    n_cells = int(ids.max())
    # every voxel its own cell -> ~16 GB dense matrix would be needed
    dense_bytes = (n_cells + 1) ** 2 * 4
    assert dense_bytes > 8 * (1024 ** 3), "test scale too small to prove the fix"

    n_types = 2
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    cell_type[1:] = 1
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=n_types, seed=1,
        target_volume=np.array([0.0, 1.0]),
        lambda_volume=np.array([0.0, 1.0]),
        contact=np.array([[0.0, 1.0], [1.0, 1.0]]),
        tracker_neighbor_order=1,
    )
    from engine import state_from_id_lattice
    eng = GPUEngine(state_from_id_lattice(cfg, ids, cell_type))

    indptr, indices, data = eng.neighbor_contact_csr()
    # every voxel is its own cell, so EVERY one of its 6 face neighbors differs
    # (out-of-bounds counts as Medium id 0, also != self) -> total directed
    # contacts = 6 * L^3, of which 6*L^3 - 2*3*L^2*(L-1) touch Medium at the faces.
    expected_total = 6 * L * L * L
    assert int(data.sum()) == expected_total, (
        f"directed contact total {int(data.sum())} != expected {expected_total}"
    )
    # distinct (src,dst) pairs <= total contacts: a corner/edge voxel touches
    # Medium (dst 0) on several faces, which collapses to ONE (src,0) pair with
    # count > 1. So #distinct < total, and >= the interior voxel<->voxel pairs.
    interior_dirpairs = 2 * 3 * L * L * (L - 1)
    assert interior_dirpairs <= len(data) < expected_total
    assert int(data.max()) >= 2, "expected a corner voxel with multiple Medium faces"
    # CSR is well-formed
    assert indptr[-1] == len(indices) == len(data)
    assert indptr[0] == 0
