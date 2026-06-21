"""Phase 6 deliverable 2 -- neighbor-contact CSR device handles.

``neighbor_contact_csr`` still returns its host arrays unchanged (so ``embryo``
consumers are untouched), but now ALSO publishes the resident device arrays
(``neighbor_csr_indptr_dev`` / ``neighbor_csr_indices_dev`` / ``neighbor_csr_data_dev``)
-- the seam Phase 7 reads to drop the host copyback. This gate asserts those device
handles are ``array_equal`` to the host return (same values, after the natural int64
vs int32 storage), single-engine and through ``recompute_trackers``; and that the
batched per-replica CSR (now built with the same device scan) stays byte-identical
to the single engine per replica.
"""

import numpy as np
import pytest

from engine import (
    EngineConfig, build_grid_state, GPUEngine,
    BatchedGPUEngine, build_batched_grid_state,
)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_handles_match_host_return():
    """The published device handles equal the host return exactly (indptr int64,
    indices/data the same values), for the production device build."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(10)

    indptr, indices, data = eng.neighbor_contact_csr(method="device")

    # handles exist and alias the resident device arrays
    assert eng.neighbor_csr_indptr_dev is not None
    assert eng.neighbor_csr_indices_dev is not None
    assert eng.neighbor_csr_data_dev is not None
    assert eng.neighbor_csr_n_contacts == int(indptr[-1]) == len(indices)

    dp = eng.neighbor_csr_indptr_dev.numpy()
    di = eng.neighbor_csr_indices_dev.numpy()
    dd = eng.neighbor_csr_data_dev.numpy()

    assert dp.dtype == np.int64
    assert np.array_equal(dp, indptr), "device indptr != host indptr"
    assert np.array_equal(di.astype(np.int64), indices), "device indices != host indices"
    assert np.array_equal(dd.astype(np.int64), data), "device data != host data"
    # handle lengths == n_contacts (not the 2x radix double-buffer)
    assert di.shape[0] == dd.shape[0] == int(indptr[-1])


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_handles_refresh_each_build():
    """A fresh build republishes handles consistent with its own host return (the
    handles track the latest CSR, not a stale one)."""
    cfg = EngineConfig(Lx=18, Ly=18, Lz=18, seed=7, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(4)
    p0, i0, d0 = eng.neighbor_contact_csr(method="device")
    h0 = eng.neighbor_csr_indptr_dev.numpy().copy()
    assert np.array_equal(h0, p0)
    eng.run(6)  # change the lattice -> different contacts
    p1, i1, d1 = eng.neighbor_contact_csr(method="device")
    h1i = eng.neighbor_csr_indices_dev.numpy()
    assert np.array_equal(eng.neighbor_csr_indptr_dev.numpy(), p1)
    assert np.array_equal(h1i.astype(np.int64), i1)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_device_handles_via_recompute_trackers():
    """Going through the per-MCS ``recompute_trackers`` seam (what the model calls)
    leaves device handles consistent with the returned host CSR."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=3, tracker_neighbor_order=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    eng.run(5)
    tr = eng.recompute_trackers()
    indptr, indices, data = tr["neighbor_csr"]
    assert np.array_equal(eng.neighbor_csr_indptr_dev.numpy(), indptr)
    assert np.array_equal(eng.neighbor_csr_indices_dev.numpy().astype(np.int64), indices)
    assert np.array_equal(eng.neighbor_csr_data_dev.numpy().astype(np.int64), data)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_single_cell_handles_consistent():
    """A single cell filling the lattice contacts only Medium (id 0) at the OOB
    faces -> a tiny 2-row CSR (Medium row + cell row). The device handles stay
    consistent with the host return on this minimal scene (exercises a degenerate
    CSR through the handle-publishing path)."""
    L = 8
    ids = np.ones((L, L, L), dtype=np.int32)  # one cell; only OOB->Medium contacts
    cell_type = np.array([0, 1], dtype=np.int32)
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=2, seed=1,
        target_volume=np.array([0.0, float(L ** 3)]),
        lambda_volume=np.array([0.0, 0.0]),
        contact=np.array([[0.0, 1.0], [1.0, 0.0]]),
        tracker_neighbor_order=1,
    )
    from engine import state_from_id_lattice
    eng = GPUEngine(state_from_id_lattice(cfg, ids, cell_type))
    indptr, indices, data = eng.neighbor_contact_csr(method="device")
    assert eng.neighbor_csr_n_contacts == int(indptr[-1]) == len(indices)
    assert np.array_equal(eng.neighbor_csr_indptr_dev.numpy(), indptr)
    assert np.array_equal(eng.neighbor_csr_indices_dev.numpy().astype(np.int64), indices)
    assert np.array_equal(eng.neighbor_csr_data_dev.numpy().astype(np.int64), data)
    # the cell (id 1) touches OOB Medium (id 0) on all 6 faces of every voxel; Medium
    # is purely out-of-bounds (no Medium voxel looks back), so the single non-empty
    # row is cell(1)->Medium(0) with the full OOB face count.
    src = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    cell_to_med = data[(src == 1) & (indices == 0)]
    assert int(cell_to_med.sum()) == 6 * L * L
    assert int(data.sum()) == 6 * L * L  # no other contacts


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_empty_csr_handles_wellformed():
    """The empty branch (no contacts at all -- an all-Medium lattice, n_cells=0)
    publishes well-formed empty handles: indptr all-zero, zero-length indices/data."""
    from engine import state_from_id_lattice
    L = 6
    ids = np.zeros((L, L, L), dtype=np.int32)       # all Medium -> no inter-id contacts
    cell_type = np.array([0], dtype=np.int32)
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=1, seed=1,
        target_volume=np.array([0.0]), lambda_volume=np.array([0.0]),
        contact=np.array([[0.0]]), tracker_neighbor_order=1,
    )
    eng = GPUEngine(state_from_id_lattice(cfg, ids, cell_type))
    indptr, indices, data = eng.neighbor_contact_csr(method="device")
    assert int(indptr[-1]) == 0 and len(indices) == 0 and len(data) == 0
    assert eng.neighbor_csr_n_contacts == 0
    assert bool((eng.neighbor_csr_indptr_dev.numpy() == 0).all())
    assert np.array_equal(eng.neighbor_csr_indptr_dev.numpy(), indptr)
    assert eng.neighbor_csr_indices_dev.shape[0] == 0


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_csr_byte_identical_to_single_after_device_scan():
    """The batched per-replica neighbor-CSR (now built with the Phase-6 device scan)
    stays byte-identical to the single ``GPUEngine`` on each replica's lattice -- the
    batched scan site reproduces the host cumsum it replaced."""
    R = 4
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=11, tracker_neighbor_order=1)
    beng = BatchedGPUEngine(build_batched_grid_state(cfg, R=R, cells_per_axis=3))
    beng.run(6)

    per_replica = beng.neighbor_contact_csr()
    assert len(per_replica) == R

    ids_all = beng.get_ids()  # (R, Lz, Ly, Lx)
    for r in range(R):
        from engine import state_from_id_lattice
        ct = np.zeros(beng.n_cells + 1, dtype=np.int32)
        ct[1:] = 1
        seng = GPUEngine(state_from_id_lattice(cfg, ids_all[r], ct))
        sp, si, sd = seng.neighbor_contact_csr(method="device")
        bp, bi, bd = per_replica[r]
        assert np.array_equal(bp, sp), f"replica {r} indptr != single"
        assert np.array_equal(bi, si), f"replica {r} indices != single"
        assert np.array_equal(bd, sd), f"replica {r} data != single"
        assert bp.dtype == sp.dtype == np.int64
