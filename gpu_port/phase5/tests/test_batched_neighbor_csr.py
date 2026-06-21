"""Batched per-replica neighbor-contact CSR (Tier 2a) gate.

``BatchedGPUEngine.neighbor_contact_csr`` returns one (indptr, indices, data) per
replica. Each must be BYTE-IDENTICAL to the single ``GPUEngine.neighbor_contact_csr``
run on that replica's own (divergent) lattice -- the per-replica building block every
batched Embryo steppable consumes. One batched hash launch fills R independent hash
regions; each is compacted with the same on-device pipeline as the single engine.
"""

import numpy as np
import pytest

from engine import (
    EngineConfig, build_grid_state, GPUEngine, state_from_id_lattice,
    BatchedGPUEngine, build_batched_grid_state,
)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_neighbor_csr_matches_single_per_replica():
    cpa = 3
    cfg = EngineConfig(Lx=15, Ly=15, Lz=15, seed=11)
    R = 4
    n_mcs = 20
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    beng.run(n_mcs)                       # let replicas diverge (distinct seeds)

    cell_type = beng.cell_type.numpy()
    ids_R = beng.get_ids()                # (R, Lz, Ly, Lx)
    for order in (1, 2):
        bcsr = beng.neighbor_contact_csr(order=order)
        assert len(bcsr) == R
        for r in range(R):
            seng = GPUEngine(state_from_id_lattice(cfg, ids_R[r], cell_type))
            sip, sidx, sdat = seng.neighbor_contact_csr(order=order, method="device")
            bip, bidx, bdat = bcsr[r]
            assert np.array_equal(bip, sip), f"order{order} r{r} indptr"
            assert np.array_equal(bidx, sidx), f"order{order} r{r} indices"
            assert np.array_equal(bdat, sdat), f"order{order} r{r} data"
            assert bidx.dtype == sidx.dtype and bdat.dtype == sdat.dtype


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_neighbor_csr_identical_replicas_agree():
    """Before any divergence (MCS 0, identical ICs) every replica's CSR is identical."""
    cpa = 3
    cfg = EngineConfig(Lx=12, Ly=12, Lz=12, seed=1)
    R = 3
    beng = BatchedGPUEngine(build_batched_grid_state(cfg, R, cells_per_axis=cpa))
    bcsr = beng.neighbor_contact_csr(order=1)
    for r in range(1, R):
        assert np.array_equal(bcsr[r][0], bcsr[0][0])
        assert np.array_equal(bcsr[r][1], bcsr[0][1])
        assert np.array_equal(bcsr[r][2], bcsr[0][2])
