"""Batched intercalation (Tier 2c) gate.

``BatchedTissueLinkSteppable`` runs the TISSUE-link Poisson turnover + neighbor
recreate for R replicas, varying the per-replica intercalation rate (TissueRate).
On a STATIC lattice the topology dynamics are deterministic (keyed Philox + the
deterministic neighbor adjacency), so each replica must match -- as a SET of
undirected links -- a single ``TissueLinkSteppable`` run with that replica's seed AND
rate. This pins both the per-replica seed plumbing and the per-replica rate plumbing.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from engine import (
    EngineConfig, build_grid_state, GPUEngine, FPPLinks,
    BatchedGPUEngine, build_batched_grid_state, BatchedFPPLinks,
)
from embryo.params import DEFAULT
from embryo.steppables import TissueLinkSteppable
from embryo.batched_steppables import BatchedTissueLinkSteppable


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_intercalation_matches_single_per_replica_static_lattice():
    """Per-replica tissue topology == a single TissueLinkSteppable on that replica's
    seed + rate, over several MCS on a static lattice (no engine sweep)."""
    cpa = 3
    L = 15
    base = EngineConfig(Lx=L, Ly=L, Lz=L, seed=4)
    R = 3
    n_mcs = 8
    probs = [0.0, 0.3, 0.7]                       # per-replica delete probability
    stride = BatchedGPUEngine.REPLICA_SEED_STRIDE
    P = DEFAULT

    beng = BatchedGPUEngine(build_batched_grid_state(base, R, cells_per_axis=cpa))
    bfpp = BatchedFPPLinks(beng, target_length_default=P.tissue_target,
                           lambda_default=P.tissue_lambda, max_length_default=P.tissue_max)
    bstep = BatchedTissueLinkSteppable(beng, bfpp, cell_types=(1,), params=P,
                                       link_cap_offset=1, substrate_type=4,
                                       delete_prob=probs)
    beng.attach_fpp(bfpp)
    bstep.start()
    for m in range(n_mcs):
        bstep.step(m)                            # NO beng.run() -> static lattice

    for r in range(R):
        seed_r = int(base.seed) + r * stride
        scfg = EngineConfig(Lx=L, Ly=L, Lz=L, seed=seed_r)
        # single params whose tissue_delete_prob == probs[r]
        rate_r = -math.log(1.0 - probs[r]) if probs[r] > 0 else 0.0
        Pr = replace(P, tissue_rate=rate_r)
        assert abs(Pr.tissue_delete_prob - probs[r]) < 1e-12
        seng = GPUEngine(build_grid_state(scfg, cpa))
        sfpp = FPPLinks(seng, target_length_default=P.tissue_target,
                        lambda_default=P.tissue_lambda, max_length_default=P.tissue_max)
        seng.attach_fpp(sfpp)
        sstep = TissueLinkSteppable(seng, sfpp, cell_types=(1,), params=Pr,
                                    link_cap_offset=1, substrate_type=4)
        sstep.shared_csr = seng.neighbor_contact_csr(order=1)
        sstep.start()
        for m in range(n_mcs):
            sstep.shared_csr = seng.neighbor_contact_csr(order=1)
            sstep.step(m)
        assert bstep._tissue[r] == sstep._tissue, (
            f"replica {r}: {len(bstep._tissue[r])} links != single {len(sstep._tissue)}")


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_intercalation_runs_with_sweep_and_keeps_partition():
    """End-to-end smoke: batched CPM sweep (with FPP energy) + intercalation per MCS
    over R replicas with different rates; the volume partition stays exact and the
    replicas diverge (independent trajectories)."""
    cpa = 3
    L = 14
    base = EngineConfig(Lx=L, Ly=L, Lz=L, seed=2)
    R = 3
    n_mcs = 15
    P = DEFAULT
    beng = BatchedGPUEngine(build_batched_grid_state(base, R, cells_per_axis=cpa))
    bfpp = BatchedFPPLinks(beng, target_length_default=P.tissue_target,
                           lambda_default=P.tissue_lambda, max_length_default=P.tissue_max)
    bstep = BatchedTissueLinkSteppable(beng, bfpp, cell_types=(1,), params=P,
                                       link_cap_offset=1, substrate_type=4,
                                       delete_prob=[0.0, 0.2, 0.5])
    beng.attach_fpp(bfpp)
    bstep.start()
    for m in range(n_mcs):
        beng.step_mcs(m)        # batched sweep reads the per-replica tissue link CSR
        bstep.step(m)           # then intercalation updates the per-replica topology
    beng.assert_volume_partition()
    xs, _, _ = beng.com_sums()
    assert not np.array_equal(xs[0], xs[1])     # replicas diverge
