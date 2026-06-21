"""On-device neighbor-contact CSR build gate (post-Phase-4 perf work).

Phase 3 built the neighbor-contact CSR by copying the WHOLE device hash table to the
host and compacting it there (boolean mask -> np.lexsort -> np.bincount). At full
Embryo scale (100^3, 63k cells) that host copy is ~200 MB/MCS. This replaces the host
compaction with an on-device build -- count occupied slots per source -> host cumsum
of the tiny per-row degree -> atomic-cursor scatter into the CSR -> per-row ascending
sort -- mirroring the FPP link-CSR pattern, transferring only the compact O(#contacts)
result.

Correctness contract: the device build must be BYTE-IDENTICAL to the kept host path
(same dtype, same ascending-within-row order) -- a stronger anchor than the existing
dense-reconstruction-vs-CPU test -- and must still reproduce the CPU directed contact
matrix exactly.
"""

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine, state_from_id_lattice
from engine.config import neighbor_offsets


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cpu_pairs(ids, n_cells, order):
    """Dense directed neighbor-contact matrix computed plainly in NumPy (the same
    reference the Phase 2/3 CSR tests use)."""
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


def _assert_wellformed(indptr, indices, data, n1):
    assert int(indptr[0]) == 0
    assert int(indptr[-1]) == len(indices) == len(data)
    assert indptr.shape[0] == n1 + 1
    assert np.all(np.diff(indptr) >= 0), "indptr not monotonic non-decreasing"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_csr_matches_host_exactly():
    """The on-device CSR build is byte-identical to the kept host compaction path
    (same indptr/indices/data, same ascending-within-row order, same dtype)."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(10)

    hp, hi, hd = eng.neighbor_contact_csr(method="host")
    dp, di, dd = eng.neighbor_contact_csr(method="device")

    assert np.array_equal(hp, dp), "indptr differs host vs device"
    assert np.array_equal(hi, di), "indices differ host vs device"
    assert np.array_equal(hd, dd), "data differ host vs device"
    assert (dp.dtype, di.dtype, dd.dtype) == (hp.dtype, hi.dtype, hd.dtype), "dtype contract broken"
    _assert_wellformed(dp, di, dd, eng.n_cells + 1)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_csr_matches_cpu_reference():
    """Device build reproduces the dense CPU directed contact matrix exactly."""
    cfg = EngineConfig(Lx=18, Ly=18, Lz=18, seed=7, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(8)
    indptr, indices, data = eng.neighbor_contact_csr(method="device")
    ids = eng.get_ids()
    ref = _cpu_pairs(ids, eng.n_cells, cfg.tracker_neighbor_order)
    n1 = eng.n_cells + 1
    dense = _csr_to_dense(indptr, indices, data, n1)
    assert np.array_equal(dense, ref), "device neighbor-CSR mismatch vs CPU"
    # undirected common-surface symmetry between real cells
    real = dense[1:, 1:]
    assert np.array_equal(real, real.T), "common-surface matrix not symmetric"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_csr_rows_sorted_ascending():
    """Every source row's indices are strictly ascending -- matches the host
    lexsort order, so tissue-link selection under the per-cell cap is unchanged
    (each (src,dst) is one hash slot, so a dst appears at most once per row)."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(10)
    indptr, indices, _ = eng.neighbor_contact_csr(method="device")
    for c in range(eng.n_cells + 1):
        lo, hi = int(indptr[c]), int(indptr[c + 1])
        row = indices[lo:hi]
        assert np.all(np.diff(row) > 0), f"row {c} not strictly ascending: {row}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_csr_scales_past_dense_infeasibility():
    """Build the neighbor-CSR via the device path at 40^3 = 64000 single-voxel
    cells (the dense (n+1)^2 matrix would be ~16 GB). Mirrors the Phase-3 scale
    test, asserting the same totals + well-formedness through the on-device build."""
    L = 40
    ids = np.arange(1, L * L * L + 1, dtype=np.int32).reshape(L, L, L)
    n_cells = int(ids.max())
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    cell_type[1:] = 1
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=2, seed=1,
        target_volume=np.array([0.0, 1.0]),
        lambda_volume=np.array([0.0, 1.0]),
        contact=np.array([[0.0, 1.0], [1.0, 1.0]]),
        tracker_neighbor_order=1,
    )
    eng = GPUEngine(state_from_id_lattice(cfg, ids, cell_type))
    indptr, indices, data = eng.neighbor_contact_csr(method="device")
    expected_total = 6 * L * L * L  # every face neighbor differs (incl. OOB Medium)
    assert int(data.sum()) == expected_total, (
        f"directed contact total {int(data.sum())} != expected {expected_total}")
    _assert_wellformed(indptr, indices, data, n_cells + 1)
    # spot-check ascending order on a few rows at this scale
    for c in (1, n_cells // 2, n_cells):
        lo, hi = int(indptr[c]), int(indptr[c + 1])
        row = indices[lo:hi]
        assert np.all(np.diff(row) > 0), f"row {c} not strictly ascending"


def _giant_medium_row_state(L):
    """A 2-plane lattice: z=0 all Medium, z=1 every voxel its own 1-voxel cell.
    Each z=1 cell sits above a Medium voxel, so the Medium row (src=0) of the CSR
    has L*L distinct dst entries -- a deliberately HUGE single row, while every
    cell row stays tiny. This is the skew that broke a per-row O(k^2) sort (the
    Medium contact row in the real Embryo); a single giant row must still build
    fast and correctly."""
    ids = np.zeros((2, L, L), dtype=np.int32)
    ids[1] = np.arange(1, L * L + 1, dtype=np.int32).reshape(L, L)
    n_cells = int(ids.max())
    cell_type = np.zeros(n_cells + 1, dtype=np.int32)
    cell_type[1:] = 1
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=2, n_types=2, seed=1,
        target_volume=np.array([0.0, 1.0]),
        lambda_volume=np.array([0.0, 1.0]),
        contact=np.array([[0.0, 1.0], [1.0, 1.0]]),
        tracker_neighbor_order=1,
    )
    return state_from_id_lattice(cfg, ids, cell_type), n_cells


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_csr_handles_giant_row_fast_and_exact():
    """A single huge CSR row (Medium touching ~40k cells) must build correctly AND
    quickly. A per-row O(k^2) sort would spend many seconds on this one row in a
    single thread; the segmented build stays well under a generous bound and the
    output is still byte-identical to the host path on the giant row."""
    import time
    L = 200  # Medium row = 40000 entries
    state, n_cells = _giant_medium_row_state(L)
    eng = GPUEngine(state)
    eng.neighbor_contact_csr(order=1, method="device")  # warm (compile/alloc)

    t0 = time.perf_counter()
    dp, di, dd = eng.neighbor_contact_csr(order=1, method="device")
    dt = time.perf_counter() - t0

    hp, hi, hd = eng.neighbor_contact_csr(order=1, method="host")
    assert int(np.diff(dp).max()) >= L * L, "expected a Medium row of ~L*L entries"
    assert np.array_equal(dp, hp) and np.array_equal(di, hi) and np.array_equal(dd, hd), \
        "device CSR mismatch vs host on the giant-row scene"
    assert dt < 2.0, f"device CSR too slow on a giant row ({dt:.2f}s) -- O(k^2) sort regression?"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_is_default_method():
    """The default (no method=) build equals the explicit device build -- i.e. the
    on-device path is the production default."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=3, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(6)
    a = eng.neighbor_contact_csr()
    b = eng.neighbor_contact_csr(method="device")
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
