"""Batched substrate links + combined Embryo driver (Tier 2d) gate.

Scene: a substrate floor (type 4, frozen) at z=0 with a layer of passive cells
(type 2) directly above, so each passive cell is face-adjacent to one substrate cell.

(1) On a static lattice the batched substrate-link dynamics (create to smallest-id
    substrate neighbor + Poisson SubLinkRate delete) must match a single
    PassiveSubstrateSteppable on the matching per-replica seed AND rate.
(2) The combined BatchedEmbryoModel (tissue + passive_substrate) runs end-to-end with
    the batched sweep, keeps the volume partition exact, and builds a non-empty FPP
    inventory.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from engine import (
    EngineConfig, GPUEngine, FPPLinks, state_from_id_lattice,
    BatchedGPUEngine, BatchedState, BatchedFPPLinks,
)
from embryo.params import DEFAULT
from embryo.steppables import PassiveSubstrateSteppable
from embryo.batched_steppables import BatchedPassiveSubstrateSteppable, BatchedEmbryoModel


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _sub_scene(seed, Lx=6, Ly=6, Lz=4):
    ids = np.zeros((Lz, Ly, Lx), dtype=np.int32)
    ct = [0]
    cid = 0
    for y in range(Ly):                       # z=0 substrate floor (type 4)
        for x in range(Lx):
            cid += 1
            ids[0, y, x] = cid
            ct.append(4)
    for y in range(Ly):                       # z=1 passive layer (type 2)
        for x in range(Lx):
            cid += 1
            ids[1, y, x] = cid
            ct.append(2)
    ct = np.array(ct, dtype=np.int32)
    cfg = EngineConfig(
        Lx=Lx, Ly=Ly, Lz=Lz, n_types=5, seed=seed,
        target_volume=np.array([0., 1., 1., 1., 1.]),
        lambda_volume=np.array([0., 2., 2., 2., 2.]),
        contact=np.zeros((5, 5)), frozen=np.array([4], dtype=np.int32),
        contact_neighbor_order=1, tracker_neighbor_order=1)
    return cfg, ids, ct


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_substrate_matches_single_per_replica():
    base, ids, ct = _sub_scene(seed=4)
    R = 3
    n_mcs = 6
    probs = [0.0, 0.4, 0.8]
    stride = BatchedGPUEngine.REPLICA_SEED_STRIDE
    P = DEFAULT

    bstate = BatchedState(base, np.repeat(ids[None], R, axis=0), ct)
    beng = BatchedGPUEngine(bstate)
    bfpp = BatchedFPPLinks(beng, target_length_default=P.slink_target,
                           lambda_default=P.slink_lambda, max_length_default=P.slink_max)
    bsub = BatchedPassiveSubstrateSteppable(beng, bfpp, passive_type=2, substrate_type=4,
                                            params=P, delete_prob=probs)
    beng.attach_fpp(bfpp)
    bsub.start()
    for m in range(n_mcs):
        bsub.step(m)                          # static lattice (no beng.run)

    passive = np.nonzero(ct == 2)[0]
    for r in range(R):
        seed_r = int(base.seed) + r * stride
        scfg = replace(base, seed=seed_r) if hasattr(base, "seed") else base
        scfg = EngineConfig(
            Lx=base.Lx, Ly=base.Ly, Lz=base.Lz, n_types=5, seed=seed_r,
            target_volume=base.target_volume, lambda_volume=base.lambda_volume,
            contact=base.contact, frozen=base.frozen,
            contact_neighbor_order=1, tracker_neighbor_order=1)
        rate_r = -math.log(1.0 - probs[r]) if probs[r] > 0 else 0.0
        Pr = replace(P, sl_cycling_rate=rate_r / P.timestep)
        assert abs(Pr.sub_link_delete_prob - probs[r]) < 1e-12
        seng = GPUEngine(state_from_id_lattice(scfg, ids, ct))
        sfpp = FPPLinks(seng, target_length_default=P.slink_target,
                        lambda_default=P.slink_lambda, max_length_default=P.slink_max)
        seng.attach_fpp(sfpp)
        ssub = PassiveSubstrateSteppable(seng, sfpp, passive_type=2, substrate_type=4, params=Pr)
        ssub.shared_csr = seng.neighbor_contact_csr(order=1)
        ssub.start()
        for m in range(n_mcs):
            ssub.shared_csr = seng.neighbor_contact_csr(order=1)
            ssub.step(m)
        s_sl = ssub.cell_dict.get("sub_link")
        assert np.array_equal(bsub._sl[r][passive], s_sl[passive]), f"replica {r} sub_link != single"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_embryo_model_runs_end_to_end():
    base, ids, ct = _sub_scene(seed=7)
    R = 3
    n_mcs = 12
    bstate = BatchedState(base, np.repeat(ids[None], R, axis=0), ct)
    beng = BatchedGPUEngine(bstate)
    model = BatchedEmbryoModel(beng, params=DEFAULT,
                               enable=("tissue", "passive_substrate"),
                               tissue_delete_prob=[0.0, 0.1, 0.3],
                               sub_delete_prob=[0.0, 0.2, 0.5])
    model.start()
    model.run(n_mcs)
    beng.assert_volume_partition()            # batched sweep + FPP stays consistent
    assert bool((model.links._a >= 0).any())  # a non-empty FPP inventory was built
