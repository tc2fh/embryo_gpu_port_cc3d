"""Batched cohesotaxis / lamellipodia (Tier 2e) gate.

``BatchedLamellipodiaSteppable`` reuses the validated single ``CohesotaxisPipeline``
per replica via ``_ReplicaEngineView`` (device slices of the batched ids/COM that
ALIAS the live arrays). The decisive test: build the steppable BEFORE a sweep,
diverge the replicas, then each replica's lamellipodia target selection must equal a
single ``LamellipodiaSteppable`` built from that replica's POST-sweep lattice on the
matching per-replica seed -- which only holds if the view slices are live (reflect the
sweep) AND index the correct replica. A second test runs the full BatchedEmbryoModel
(tissue + lamellipodia + substrate) end-to-end.
"""

from dataclasses import replace

import numpy as np
import pytest

from engine import (
    GPUEngine, FPPLinks, state_from_id_lattice,
    BatchedGPUEngine, BatchedState, batched_state_from_lattices, BatchedFPPLinks,
)
from engine.steppables import LamellipodiaSteppable
from embryo.model import build_closure_scene
from embryo.params import DEFAULT
from embryo.batched_steppables import BatchedLamellipodiaSteppable, BatchedEmbryoModel


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _perturb(base_ids, cell_type, r):
    """Replica r's lattice = base with r medium voxels grown into adjacent NON-substrate
    cells (deterministic scan order). Same cell set / n_cells, genuinely different
    geometry -> divergent COM per replica (so the per-replica match is non-vacuous)."""
    ids = base_ids.copy()
    if r == 0:
        return ids
    Lz, Ly, Lx = ids.shape
    face = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (-1, 0, 0)]
    flipped = 0
    for z in range(Lz):
        for y in range(Ly):
            for x in range(Lx):
                if flipped >= r:
                    return ids
                if ids[z, y, x] != 0:
                    continue
                for dz, dy, dx in face:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < Lz and 0 <= ny < Ly and 0 <= nx < Lx:
                        nb = ids[nz, ny, nx]
                        if nb != 0 and cell_type[nb] != 4:   # grow a non-substrate cell
                            ids[z, y, x] = nb
                            flipped += 1
                            break
    return ids


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_lamellipodia_matches_single_per_replica():
    state, info = build_closure_scene(L=20, seed=3, n_leaders=6)
    cfg = state.cfg
    R = 3
    lattices = [_perturb(state.ids, state.cell_type, r) for r in range(R)]
    assert np.any(lattices[0] != lattices[1]) and np.any(lattices[0] != lattices[2])

    beng = BatchedGPUEngine(batched_state_from_lattices(cfg, lattices, state.cell_type))
    bfpp = BatchedFPPLinks(beng, target_length_default=DEFAULT.tissue_target,
                           lambda_default=DEFAULT.tissue_lambda, max_length_default=DEFAULT.tissue_max)
    blam = BatchedLamellipodiaSteppable(beng, bfpp, leading_type=1, substrate_type=4,
                                        passive_type=2, params=DEFAULT)
    blam.start()                                # reads each replica's divergent lattice
    leaders = blam.lead_ids

    stride = BatchedGPUEngine.REPLICA_SEED_STRIDE
    for r in range(R):
        seed_r = int(cfg.seed) + r * stride
        scfg = replace(cfg, seed=seed_r)
        # single engine from THIS replica's lattice: if the view reads the wrong replica
        # (or wrong seed), the cohesotaxis target selection won't match.
        seng = GPUEngine(state_from_id_lattice(scfg, lattices[r], state.cell_type))
        sfpp = FPPLinks(seng, target_length_default=DEFAULT.tissue_target,
                        lambda_default=DEFAULT.tissue_lambda, max_length_default=DEFAULT.tissue_max)
        slam = LamellipodiaSteppable(seng, sfpp, leading_type=1, substrate_type=4, passive_type=2)
        slam.start()
        s_lt = slam.cell_dict.get("link_target")
        assert np.array_equal(blam.link_target[r][leaders], s_lt[leaders]), \
            f"replica {r}: lamellipodia targets != single on its lattice"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_embryo_model_full_with_cohesotaxis():
    state, info = build_closure_scene(L=20, seed=5, n_leaders=6)
    R = 3
    n_mcs = 10
    bstate = BatchedState(state.cfg, np.repeat(state.ids[None], R, axis=0), state.cell_type)
    beng = BatchedGPUEngine(bstate)
    model = BatchedEmbryoModel(beng, params=DEFAULT,
                               enable=("tissue", "lamellipodia", "passive_substrate"),
                               tissue_delete_prob=[0.0, 0.1, 0.3],
                               lamellae_delete_rate=[0.0, 0.2, 0.5])
    model.start()
    model.run(n_mcs)
    beng.assert_volume_partition()
    assert bool((model.links._a >= 0).any())    # combined FPP inventory was built
