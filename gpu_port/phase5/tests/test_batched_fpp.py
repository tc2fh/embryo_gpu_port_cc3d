"""Batched FocalPointPlasticity (Tier 1) gate.

``BatchedFPPLinks`` + the FPP path in ``metropolis_color_batched_kernel`` give the
batched engine a PER-REPLICA link CSR so a sweep can vary the FPP spring params
(lambda / target / max length) across replicas on a shared link network.

What is and isn't bit-exact (the honest contract):
  * The link-CSR BUILD is deterministic as a per-cell adjacency SET (only the
    within-row append ORDER is racy), so a replica's kept neighbor sets are asserted
    EQUAL to a single ``FPPLinks`` built with that replica's params.
  * A replica with ``lambda == 0`` adds exactly 0.0 to every energy delta, so its
    trajectory is BIT-EXACT (int64 COM sums) to a single ``GPUEngine`` (no FPP) on
    the matching per-replica seed -- this pins the per-replica plumbing + gating.
  * FPP runs with lambda > 0 are statistically faithful but NOT bit-reproducible
    (the documented Phase-3/4 COM-read race + atomic-append ordering); we assert FPP
    changes the trajectory and that replicas with different lambda differ, not
    bit-equality between two independent FPP runs.
"""

import numpy as np
import pytest

from engine import (
    EngineConfig, build_grid_state, GPUEngine, FPPLinks, grid_graph_links,
    BatchedGPUEngine, build_batched_grid_state, BatchedFPPLinks,
)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cfg(seed, L=12):
    return EngineConfig(Lx=L, Ly=L, Lz=L, seed=seed)


def _adj_single(ptr, other, n1):
    return {c: set(int(x) for x in other[int(ptr[c]):int(ptr[c + 1])]) for c in range(1, n1)}


def _adj_batched(ptr_flat, other_flat, n1, R, pay_stride):
    ptr = ptr_flat.reshape(R, n1 + 1)
    out = []
    for r in range(R):
        base = r * pay_stride
        d = {}
        for c in range(1, n1):
            lo, hi = int(ptr[r, c]), int(ptr[r, c + 1])
            d[c] = set(int(x) for x in other_flat[base + lo:base + hi])
        out.append(d)
    return out


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_fpp_build_matches_single_per_replica():
    """Each replica's link CSR (adjacency sets) equals a single FPPLinks built with
    that replica's max_length, on the identical (pristine grid) COM."""
    cpa = 3
    cfg = _cfg(seed=1)
    R = 3
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    pairs = grid_graph_links(cpa)
    n1 = beng.n1
    # per-replica max: replica 0 culls all (max < cell spacing ~4), 1 & 2 keep all
    maxes = np.array([3.0, 6.0, 100.0], dtype=np.float32)
    bfpp = BatchedFPPLinks(beng, target_length_default=4.0, lambda_default=2.0)
    bfpp.set_topology(pairs, maxlens=np.repeat(maxes[:, None], pairs.shape[0], axis=1))
    beng.attach_fpp(bfpp)  # rebuild at MCS 0 (pristine COM)

    badj = _adj_batched(bfpp.link_ptr.numpy(), bfpp.link_other.numpy(), n1, R, bfpp.link_pay_stride)
    for r in range(R):
        seng = GPUEngine(build_grid_state(cfg, cpa))
        sfpp = FPPLinks(seng, target_length_default=4.0, lambda_default=2.0,
                        max_length_default=float(maxes[r]))
        sfpp.set_topology(pairs)
        seng.attach_fpp(sfpp)   # rebuild on the same pristine COM
        sadj = _adj_single(sfpp.link_ptr.numpy(), sfpp.link_other.numpy(), n1)
        assert badj[r] == sadj, f"replica {r} adjacency != single build"
    # replica 0 culled everything, replicas 1/2 kept all undirected links
    na = bfpp.num_active()
    assert na[0] == 0
    assert na[1] == pairs.shape[0] and na[2] == pairs.shape[0]


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_fpp_lambda_zero_is_bit_exact_to_no_fpp():
    """A replica with lambda==0 adds exactly 0 to every delta, so it is bit-exact
    (int64 COM sums) to a single GPUEngine with NO FPP on the matching seed. Pins the
    per-replica indexing + the fpp gating."""
    cpa = 3
    cfg = _cfg(seed=7)
    R = 3
    n_mcs = 25
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    pairs = grid_graph_links(cpa)
    # replica 0 lambda 0 (no-op); replicas 1,2 strong springs (just to exercise them)
    lam = np.zeros((R, pairs.shape[0]), dtype=np.float32)
    lam[1] = 5.0
    lam[2] = 9.0
    bfpp = BatchedFPPLinks(beng, target_length_default=4.0, max_length_default=100.0)
    bfpp.set_topology(pairs, lambdas=lam)
    beng.attach_fpp(bfpp)
    beng.run(n_mcs)
    xs, ys, zs = beng.com_sums()

    # single engine, no FPP, seed of replica 0 = base + 0*stride = cfg.seed
    seng = GPUEngine(build_grid_state(cfg, cpa))
    seng.run(n_mcs)
    assert np.array_equal(xs[0], seng.xsum.numpy())
    assert np.array_equal(ys[0], seng.ysum.numpy())
    assert np.array_equal(zs[0], seng.zsum.numpy())


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_fpp_changes_trajectory_and_is_per_replica():
    """FPP with lambda>0 must perturb the trajectory vs no-FPP, and two replicas with
    different lambda must diverge from each other (params are genuinely per-replica)."""
    cpa = 3
    cfg = _cfg(seed=3)
    R = 2
    n_mcs = 40
    pairs = grid_graph_links(cpa)
    # short target so the springs pull cells together -> measurable effect
    lam = np.array([[0.0] * pairs.shape[0], [12.0] * pairs.shape[0]], dtype=np.float32)
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    bfpp = BatchedFPPLinks(beng, target_length_default=1.0, max_length_default=100.0)
    bfpp.set_topology(pairs, lambdas=lam)
    beng.attach_fpp(bfpp)
    beng.run(n_mcs)
    xs, ys, zs = beng.com_sums()
    # replica 0 (lambda 0) and replica 1 (strong springs) must differ
    assert not (np.array_equal(xs[0], xs[1]) and np.array_equal(ys[0], ys[1])
                and np.array_equal(zs[0], zs[1]))
    # and the partition stays exact under FPP for every replica
    beng.assert_volume_partition()


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_fpp_per_replica_topology_builds_correctly():
    """set_per_replica_pairs: each replica carries a DIFFERENT link set (padded with
    -1 tombstones). Each replica's CSR must match a single FPPLinks on that replica's
    own pairs, and num_active reflects each replica's link count."""
    cpa = 3
    cfg = _cfg(seed=2)
    R = 3
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    full = grid_graph_links(cpa)
    subsets = [full[:5], full[:10], full]      # different lengths -> tombstone padding
    bfpp = BatchedFPPLinks(beng, target_length_default=4.0, lambda_default=2.0,
                           max_length_default=100.0)
    bfpp.set_per_replica_pairs(subsets)
    beng.attach_fpp(bfpp)                       # rebuild on pristine COM

    badj = _adj_batched(bfpp.link_ptr.numpy(), bfpp.link_other.numpy(),
                        beng.n1, R, bfpp.link_pay_stride)
    for r in range(R):
        seng = GPUEngine(build_grid_state(cfg, cpa))
        sfpp = FPPLinks(seng, target_length_default=4.0, lambda_default=2.0,
                        max_length_default=100.0)
        sfpp.set_topology(subsets[r])
        seng.attach_fpp(sfpp)
        sadj = _adj_single(sfpp.link_ptr.numpy(), sfpp.link_other.numpy(), beng.n1)
        assert badj[r] == sadj, f"replica {r} per-replica adjacency != single"
    na = bfpp.num_active()
    assert na[0] == 5 and na[1] == 10 and na[2] == full.shape[0]


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_batched_fpp_disabled_when_no_topology():
    """has_links() False -> the engine runs the no-FPP path; bit-exact to a plain
    batched run (the new kernel args are a clean no-op when fpp_enabled=0)."""
    cpa = 3
    cfg = _cfg(seed=5)
    R = 2
    n_mcs = 15
    bstate = build_batched_grid_state(cfg, R, cells_per_axis=cpa)
    beng = BatchedGPUEngine(bstate)
    bfpp = BatchedFPPLinks(beng)   # no set_topology -> empty
    beng.attach_fpp(bfpp)
    assert not bfpp.has_links()
    beng.run(n_mcs)
    xs, ys, zs = beng.com_sums()

    beng2 = BatchedGPUEngine(build_batched_grid_state(cfg, R, cells_per_axis=cpa))
    beng2.run(n_mcs)
    xs2, ys2, zs2 = beng2.com_sums()
    assert np.array_equal(xs, xs2) and np.array_equal(ys, ys2) and np.array_equal(zs, zs2)
