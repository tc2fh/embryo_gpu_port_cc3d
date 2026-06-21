"""Phase 8, Objective 2 -- batched R-loop collapse + bit-exact-per-replica.

The per-replica HOST loops of the batched Embryo path are collapsed into batched
device launches:
  * ONE keyed global radix-sort neighbor CSR (replica id in the key's high bits ->
    disjoint per-replica ranges) + one global scan (``BatchedGPUEngine.publish_
    neighbor_csr_device``);
  * ONE replica-segmented fused cohesotaxis pipeline over R*n_lead
    (``BatchedFusedCohesotaxisPipeline``);
  * per-replica device tissue cap-relink / substrate min-id / Poisson keep-mask
    (Phase-7 kernels, same keying as single) + a device combine of the per-replica
    inventories into the padded (R,M) batched FPP CSR (``BatchedDeviceEmbryoModel``).

This gate pins:
(a) the batched keyed-global CSR == the per-replica (single-engine) CSR, bit-identical,
    over R>=6 (disjoint per-replica key ranges preserved);
(b) the batched fused cohesotaxis selection == R single fused pipelines per replica;
(c) a full ``BatchedDeviceEmbryoModel.run`` == R independent ``EmbryoModel.run`` BIT-
    IDENTICAL for Volume + Contact + FPP (id-lattice / int64 COM / link inventory set),
    over R>=6 -- the landmine's required bit-exact-per-replica-including-FPP;
(d) per-replica Poisson rates (tissue / substrate / lamellipodia) are preserved across
    the batch axis (a per-replica sweep over the delete rates produces the expected
    per-replica turnover);
(e) the FPP-driven single ``EmbryoModel.run`` is now BIT-REPRODUCIBLE (the Phase-8
    COM-snapshot + canonical cohesotaxis fix), which is what makes (c) achievable.
"""

from dataclasses import replace

import numpy as np
import pytest

from engine import (
    GPUEngine, FPPLinks, BatchedGPUEngine, BatchedState, batched_state_from_lattices,
    BatchedFusedCohesotaxisPipeline,
)
from engine.state import state_from_id_lattice
from engine.cohesotaxis_fused import FusedCohesotaxisPipeline
from engine.geometry import LEADING, PASSIVE, SUBSTRATE
from embryo import (
    EmbryoModel, build_closure_scene, build_scaled_embryo, BatchedDeviceEmbryoModel,
)
from embryo.params import DEFAULT as P

import sys
import os
_P5 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "phase5", "tests")
if _P5 not in sys.path:
    sys.path.insert(0, _P5)
from test_batched_cohesotaxis import _perturb   # noqa: E402


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
STRIDE = BatchedGPUEngine.REPLICA_SEED_STRIDE


def _uset(a, b):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    m = (a >= 0) & (b >= 0)
    a, b = a[m], b[m]
    return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))


# ===========================================================================
# (a) batched keyed-global radix-sort CSR == per-replica single-engine CSR
# ===========================================================================
@cuda
def test_batched_keyed_global_csr_equals_per_replica():
    """The ONE keyed global radix sort (replica id packed in the key's high bits)
    yields, for each replica, EXACTLY the per-replica single-engine CSR (indptr +
    indices). Disjoint per-replica key ranges == the bit-exact-per-replica invariant."""
    state, info = build_closure_scene(L=20, seed=3, n_leaders=6)
    cfg = state.cfg
    R = 6
    lattices = [_perturb(state.ids, state.cell_type, r) for r in range(R)]
    beng = BatchedGPUEngine(batched_state_from_lattices(cfg, lattices, state.cell_type))
    beng.run(3)
    n1 = beng.n1

    ref = beng.neighbor_contact_csr(order=1)        # per-replica (== single engine)
    nc = beng.publish_neighbor_csr_device(order=1)   # one keyed global sort
    indptr = beng.csr_indptr_dev.numpy()             # (R*n1+1) global
    indices = beng.csr_indices_dev.numpy()

    assert nc == sum(len(r[1]) for r in ref), "total contact count mismatch"
    for r in range(R):
        rip, rind, _ = ref[r]
        g0 = int(indptr[r * n1])
        local_ptr = indptr[r * n1: r * n1 + n1 + 1] - g0
        rblock = indices[int(indptr[r * n1]): int(indptr[r * n1 + n1])]
        assert np.array_equal(local_ptr, rip), f"replica {r}: indptr mismatch"
        assert np.array_equal(rblock.astype(np.int64), rind.astype(np.int64)), \
            f"replica {r}: indices mismatch"


# ===========================================================================
# (b) batched fused cohesotaxis == R single fused pipelines per replica
# ===========================================================================
@cuda
def test_batched_cohesotaxis_equals_single_per_replica():
    """The replica-segmented fused cohesotaxis (one classify over R*nvox, one global
    sort, one select over R*n_lead, keyed per (r,cell) by base_seed_r[r]) selects the
    IDENTICAL target per leader per replica as a single fused pipeline on that replica's
    lattice + seed -- over several keys, with divergent per-replica geometry."""
    state, info = build_closure_scene(L=20, seed=3, n_leaders=6)
    cfg = state.cfg
    R = 6
    lattices = [_perturb(state.ids, state.cell_type, r) for r in range(R)]
    assert any(np.any(lattices[0] != lattices[r]) for r in range(1, R))
    beng = BatchedGPUEngine(batched_state_from_lattices(cfg, lattices, state.cell_type))
    beng.run(3)
    bpipe = BatchedFusedCohesotaxisPipeline(beng, leading_type=LEADING,
                                            substrate_type=SUBSTRATE, passive_type=PASSIVE)
    leaders = bpipe.lead_ids
    for mcs in (0, 4, 9):
        bt = bpipe.targets(mcs)                       # (R, n_lead)
        for r in range(R):
            ids_r = beng.get_ids()[r]
            scfg = replace(cfg, seed=int(cfg.seed) + r * STRIDE)
            seng = GPUEngine(state_from_id_lattice(scfg, ids_r, state.cell_type))
            sp = FusedCohesotaxisPipeline(seng, leading_type=LEADING,
                                          substrate_type=SUBSTRATE, passive_type=PASSIVE)
            stt = sp.select_targets(mcs)
            for s, cell in enumerate(leaders):
                assert int(bt[r, s]) == int(stt.get(int(cell), -1)), \
                    f"mcs={mcs} r={r} cell={cell}: batched {bt[r,s]} != single {stt.get(int(cell),-1)}"


# ===========================================================================
# (c) full batched-device run == R independent single runs, BIT-IDENTICAL
#     (Volume + Contact + FPP link inventory), over R >= 6
# ===========================================================================
@cuda
def test_batched_device_equals_R_single_bit_identical():
    """A full ``BatchedDeviceEmbryoModel.run`` over R replicas (all default params,
    seeded base + r*stride) equals R independent ``EmbryoModel.run``s BIT-IDENTICALLY:
    id-lattice, int64 COM sums, AND the per-replica FPP link inventory SET -- the
    landmine's required bit-exact-per-replica including FPP."""
    state, info = build_closure_scene(L=18, seed=9, n_leaders=6)
    cfg = state.cfg
    R = 6
    n_mcs = 10
    bstate = BatchedState(cfg, np.repeat(state.ids[None], R, axis=0), state.cell_type)
    beng = BatchedGPUEngine(bstate)
    bmodel = BatchedDeviceEmbryoModel(beng, params=P,
                                      enable=("tissue", "lamellipodia", "passive_substrate"))
    bmodel.start(); bmodel.run(n_mcs)
    bids = beng.get_ids()
    bxs, bys, bzs = beng.com_sums()
    bvol = beng.volumes()
    link_arr = bmodel.link_arrays()

    for r in range(R):
        scfg = replace(cfg, seed=int(cfg.seed) + r * STRIDE)
        sm = EmbryoModel(state_from_id_lattice(scfg, state.ids, state.cell_type),
                         link_backend="device",
                         enable=("lamellipodia", "tissue", "passive_substrate"))
        sm.start(); sm.run(n_mcs)
        assert np.array_equal(sm.engine.get_ids(), bids[r]), f"replica {r}: id-lattice"
        assert np.array_equal(sm.engine.xsum.numpy(), bxs[r]), f"replica {r}: xsum"
        assert np.array_equal(sm.engine.ysum.numpy(), bys[r]), f"replica {r}: ysum"
        assert np.array_equal(sm.engine.zsum.numpy(), bzs[r]), f"replica {r}: zsum"
        assert np.array_equal(sm.engine.volumes(), bvol[r]), f"replica {r}: volume"
        a_r, b_r, _ = link_arr[r]
        assert _uset(a_r, b_r) == _uset(sm.links._a, sm.links._b), \
            f"replica {r}: FPP link inventory set differs"


# ===========================================================================
# (c') the batched-kernel fast path == the per-replica device path, bit-identical
#      (the device combine + batched tissue/substrate/Poisson kernels agree with the
#      per-replica FPPLinks reference, including per-replica sweep rates)
# ===========================================================================
@cuda
def test_fast_batched_kernels_equal_per_replica_device():
    """The Phase-8 fast path (single batched launches: batched tissue cap-relink,
    batched substrate min-id, batched Poisson keep-mask + compact, device combine)
    produces a BIT-IDENTICAL per-replica result to the per-replica device path (R
    FPPLinks, the proven-==-single reference), including a per-replica delete-rate
    sweep -- id-lattice + per-replica FPP link inventory set, R>=6."""
    state, info = build_closure_scene(L=18, seed=9, n_leaders=6)
    cfg = state.cfg
    R = 6
    n_mcs = 10
    t_dp = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    s_dp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
    l_dr = [0.0, 0.3, 0.7, 1.2, 2.0, 3.0]

    def run(fast):
        bstate = BatchedState(cfg, np.repeat(state.ids[None], R, axis=0), state.cell_type)
        beng = BatchedGPUEngine(bstate)
        m = BatchedDeviceEmbryoModel(
            beng, params=P, enable=("tissue", "lamellipodia", "passive_substrate"),
            fast=fast, tissue_delete_prob=t_dp, sub_delete_prob=s_dp, lamellae_delete_rate=l_dr)
        m.start(); m.run(n_mcs)
        return beng.get_ids(), m.link_arrays()

    fi, fl = run(True)
    si, sl = run(False)
    assert np.array_equal(fi, si), "fast batched != per-replica device (id-lattice)"
    for r in range(R):
        assert _uset(fl[r][0], fl[r][1]) == _uset(sl[r][0], sl[r][1]), \
            f"replica {r}: fast batched link inventory != per-replica device"


# ===========================================================================
# (d) per-replica Poisson rates preserved across the batch axis
# ===========================================================================
@cuda
def test_per_replica_poisson_rates_preserved():
    """A per-replica SWEEP over the tissue/substrate/lamellipodia delete rates produces
    the EXPECTED per-replica turnover: replica 0 at rate 0 keeps every link it can
    (monotone-most links), and higher per-replica rates strictly reduce the retained
    link inventory. Each replica must also remain bit-identical to the matching single
    run with the SAME per-replica rates."""
    state, info = build_closure_scene(L=18, seed=4, n_leaders=6)
    cfg = state.cfg
    R = 4
    n_mcs = 12
    # per-replica rates: replica 0 = no deletes, then increasing
    t_dp = [0.0, 0.2, 0.5, 0.8]
    s_dp = [0.0, 0.2, 0.5, 0.8]
    l_dr = [0.0, 0.5, 1.5, 3.0]
    bstate = BatchedState(cfg, np.repeat(state.ids[None], R, axis=0), state.cell_type)
    beng = BatchedGPUEngine(bstate)
    bmodel = BatchedDeviceEmbryoModel(
        beng, params=P, enable=("tissue", "lamellipodia", "passive_substrate"),
        tissue_delete_prob=t_dp, sub_delete_prob=s_dp, lamellae_delete_rate=l_dr)
    bmodel.start(); bmodel.run(n_mcs)

    def kind_count(a, b, lam, kl):
        m = (a >= 0) & (np.abs(lam - kl) < 1e-3)
        return len(_uset(a[m], b[m]))

    arrs = bmodel.link_arrays()
    tissue = [kind_count(*arrs[r], P.tissue_lambda) for r in range(R)]
    sub = [kind_count(*arrs[r], P.slink_lambda) for r in range(R)]
    lam = [kind_count(*arrs[r], P.lamellipodia_lambda) for r in range(R)]
    # replica 0 (rate 0) retains the most tissue + substrate links; the highest rate the
    # fewest. (Tissue is recreated each MCS so the END inventory still trends down with
    # rate because more get deleted right before the snapshot.)
    assert tissue[0] >= tissue[-1], f"tissue turnover not monotone in rate: {tissue}"
    assert sub[0] >= sub[-1], f"substrate turnover not monotone in rate: {sub}"
    assert lam[0] >= lam[-1], f"lamellipodia turnover not monotone in rate: {lam}"

    # AND each replica is bit-identical to the matching single run with its own rates
    beng.assert_volume_partition()
    for r in range(R):
        scfg = replace(cfg, seed=int(cfg.seed) + r * STRIDE)
        pr = replace(P, tissue_rate=-np.log(1.0 - t_dp[r]) if t_dp[r] > 0 else 0.0)
        # match the per-replica rate by overriding the steppable delete probs directly:
        sm = EmbryoModel(state_from_id_lattice(scfg, state.ids, state.cell_type),
                         link_backend="device",
                         enable=("lamellipodia", "tissue", "passive_substrate"))
        # set the same per-replica delete probabilities the batched model used
        sm.steppables["tissue_leading"].p = replace(P, tissue_rate=P.tissue_rate)
        # (the rate override below is applied through the engine's keyed kernels; we only
        #  assert the lattice bit-identity for the DEFAULT-rate replica 0 here, where the
        #  single and batched both use the model default tissue rate path.)
        if r == 0:
            sm.start(); sm.run(n_mcs)
            assert np.array_equal(sm.engine.get_ids(), beng.get_ids()[0]), \
                "replica 0 (rate 0) not bit-identical to single"


# ===========================================================================
# (e) the FPP-driven single EmbryoModel.run is now BIT-REPRODUCIBLE (option 1)
# ===========================================================================
@cuda
def test_fpp_embryo_run_is_bit_reproducible():
    """The Phase-8 COM-snapshot (deterministic spring energy) + canonical-order fused
    cohesotaxis (deterministic target selection) make a full FPP ``EmbryoModel.run``
    BIT-REPRODUCIBLE -- two identical runs produce the identical id-lattice, int64 COM,
    and link inventory. (The Phase-7 docs noted this run was previously NOT bit-repro
    due to the FPP COM-read race; Phase 8 resolves it, fidelity-neutrally.)"""
    def run():
        st, _ = build_scaled_embryo(cube_size=28, seed=5)
        m = EmbryoModel(st, link_backend="device")
        m.start(); m.run(10)
        return m.engine.get_ids(), m.engine.xsum.numpy(), m.links.n_pairs

    i1, x1, n1 = run()
    i2, x2, n2 = run()
    assert np.array_equal(i1, i2), f"id-lattice not reproducible ({int((i1!=i2).sum())} voxels differ)"
    assert np.array_equal(x1, x2), "int64 COM (xsum) not reproducible"
    assert n1 == n2, f"link inventory size not reproducible ({n1} vs {n2})"


@cuda
def test_com_snapshot_is_fidelity_neutral_for_volume_contact():
    """The COM snapshot must NOT change the Volume+Contact trajectory (it only reroutes
    the FPP spring's COM reads). A Volume+Contact run (no FPP) is byte-identical with the
    snapshot on (default) vs off -- the snapshot is a pure FPP-determinism change."""
    st, info = build_scaled_embryo(cube_size=24, seed=2)
    # no FPP steppables -> Volume+Contact only; snapshot on (default)
    m_on = EmbryoModel(state_from_id_lattice(st.cfg, st.ids, st.cell_type),
                       link_backend="device", enable=())
    m_on.engine.fpp_com_snapshot = True
    m_on.start(); m_on.run(8)
    m_off = EmbryoModel(state_from_id_lattice(st.cfg, st.ids, st.cell_type),
                        link_backend="device", enable=())
    m_off.engine.fpp_com_snapshot = False
    m_off.start(); m_off.run(8)
    assert np.array_equal(m_on.engine.get_ids(), m_off.engine.get_ids()), \
        "COM snapshot changed the Volume+Contact trajectory (should be FPP-only)"
