"""Phase 6 deliverable 1 -- device exclusive prefix-sum (CSR row pointer) gate.

The three per-MCS CSR builds (FPP ``link_ptr``, neighbor-CSR ``indptr``, batched
per-replica ``link_ptr``) used the host idiom ``ptr[1:] = np.cumsum(degree[:n])``.
Phase 6 replaces them with on-device scans. The hard requirement: the device scan
is **byte-identical** to that ``np.cumsum`` -- same values, same dtype -- on random
degree vectors, both single-segment and segmented (per-replica reset). Integer add
is associative, so a sequential accumulate (and an inclusive ``array_scan`` into the
shifted slice) equals NumPy bit-for-bit; this test pins that.
"""

import numpy as np
import pytest

import warp as wp

from engine import scan as S
from engine import kernels as K


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _ref_ptr(deg, n):
    """The exact host idiom being replaced: ptr[0]=0, ptr[1:n+1]=cumsum(deg[:n])."""
    ptr = np.zeros(n + 1, dtype=deg.dtype)
    ptr[1:] = np.cumsum(deg[:n])
    return ptr


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 256, 4097])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_single_scan_i32_byte_identical(n, seed):
    """int32 single-segment scan (the FPP link_ptr site) == np.cumsum, exactly."""
    dev = "cuda:0"
    rng = np.random.default_rng(seed)
    # degree array is length n+1 (only [0:n] read), like the engine's _degree
    deg = rng.integers(0, 13, size=n + 1).astype(np.int32)
    d = wp.array(deg, dtype=wp.int32, device=dev)
    out = wp.zeros(n + 1, dtype=wp.int32, device=dev)
    S.exclusive_scan_to_ptr_i32(d, n, out, dev)
    wp.synchronize()
    got = out.numpy()
    ref = _ref_ptr(deg, n)
    assert got.dtype == ref.dtype == np.int32
    assert np.array_equal(got, ref), f"n={n}: {got} != {ref}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
@pytest.mark.parametrize("n", [0, 1, 5, 17, 256, 4097])
@pytest.mark.parametrize("seed", [0, 2, 9])
def test_single_scan_i64_byte_identical(n, seed):
    """int64 single-segment scan (the neighbor-CSR indptr site) == np.cumsum int64."""
    dev = "cuda:0"
    rng = np.random.default_rng(seed + 100)
    deg = rng.integers(0, 13, size=n + 1).astype(np.int32)
    d = wp.array(deg, dtype=wp.int32, device=dev)
    out = wp.zeros(n + 1, dtype=wp.int64, device=dev)
    S.exclusive_scan_to_ptr_i64(d, n, out, dev)
    wp.synchronize()
    got = out.numpy()
    ref = np.zeros(n + 1, dtype=np.int64)
    ref[1:] = np.cumsum(deg[:n].astype(np.int64))
    assert got.dtype == ref.dtype == np.int64
    assert np.array_equal(got, ref), f"n={n}: {got} != {ref}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_single_scan_i64_large_values_exact():
    """The int64-output path stays exact for large cumulative sums (here ~2.0e9, well
    past 2^30 -- so the result genuinely needs the int64 output dtype, not a small
    int). The cumulative stays under 2^31 because n_contacts is memory-bounded far
    below that (2*n_contacts int64 must fit GPU RAM), matching the real CSR build."""
    dev = "cuda:0"
    n = 2000
    deg = np.full(n + 1, 1_000_000, dtype=np.int32)  # cumulative 2.0e9 (< 2^31, > 2^30)
    d = wp.array(deg, dtype=wp.int32, device=dev)
    out = wp.zeros(n + 1, dtype=wp.int64, device=dev)
    S.exclusive_scan_to_ptr_i64(d, n, out, dev)
    wp.synchronize()
    ref = np.zeros(n + 1, dtype=np.int64)
    ref[1:] = np.cumsum(deg[:n].astype(np.int64))
    assert int(out.numpy()[-1]) == int(ref[-1]) == 1_000_000 * n
    assert out.numpy().dtype == np.int64
    assert np.array_equal(out.numpy(), ref)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
@pytest.mark.parametrize("R", [1, 2, 4, 8])
@pytest.mark.parametrize("n1", [1, 3, 13, 257])
@pytest.mark.parametrize("seed", [0, 4])
def test_segmented_scan_byte_identical(R, n1, seed):
    """int32 per-replica (segmented) scan (the batched link_ptr site) ==
    np.cumsum(deg[:, :n1], axis=1) shifted into [:,1:], exactly -- the per-replica
    reset must be honored so replica r's pointer starts at 0."""
    dev = "cuda:0"
    rng = np.random.default_rng(seed + 200)
    deg = rng.integers(0, 11, size=(R, n1 + 1)).astype(np.int32)
    d = wp.array(deg.reshape(-1), dtype=wp.int32, device=dev)
    out = wp.zeros(R * (n1 + 1), dtype=wp.int32, device=dev)
    S.segmented_exclusive_scan_to_ptr_i32(d, R, n1, out, dev)
    wp.synchronize()
    got = out.numpy().reshape(R, n1 + 1)
    ref = np.zeros((R, n1 + 1), dtype=np.int32)
    ref[:, 1:] = np.cumsum(deg[:, :n1], axis=1)
    assert got.dtype == ref.dtype == np.int32
    assert np.array_equal(got, ref), f"R={R} n1={n1}"
    # every replica row resets at 0
    assert np.all(got[:, 0] == 0)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_segmented_equals_independent_single_scans():
    """The segmented scan equals running the single-segment scan independently on
    each replica's degree slice -- i.e. segments are truly independent."""
    dev = "cuda:0"
    R, n1 = 5, 23
    rng = np.random.default_rng(321)
    deg = rng.integers(0, 9, size=(R, n1 + 1)).astype(np.int32)
    d = wp.array(deg.reshape(-1), dtype=wp.int32, device=dev)
    out = wp.zeros(R * (n1 + 1), dtype=wp.int32, device=dev)
    S.segmented_exclusive_scan_to_ptr_i32(d, R, n1, out, dev)
    wp.synchronize()
    seg = out.numpy().reshape(R, n1 + 1)
    for r in range(R):
        dd = wp.array(deg[r], dtype=wp.int32, device=dev)
        o = wp.zeros(n1 + 1, dtype=wp.int32, device=dev)
        S.exclusive_scan_to_ptr_i32(dd, n1, o, dev)
        wp.synchronize()
        assert np.array_equal(seg[r], o.numpy()), f"replica {r} segment != single scan"
