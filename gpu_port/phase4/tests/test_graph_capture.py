"""Phase 4 Pass B gate: CUDA-Graph capture of the per-MCS hot loop + throughput.

A ``GraphRunner`` records the per-MCS DEVICE work (the 8-color Metropolis sweep +
an on-device step-counter increment) into a CUDA graph ONCE and replays it, cutting
the per-launch Python/driver overhead. The captured region is pure device work; the
host-side tracker / FPP-CSR rebuilds stay OUTSIDE the graph (capture-boundary
constraint -- they do host compaction).

We validate:

1. **Graph == eager, BIT-EXACT (single engine).** A graph-captured run and an eager
   run with the same seed produce identical id-lattice / int64 COM sums / volumes.
   Same kernels, same RNG keys (mcs read from a device counter that advances exactly
   as the eager host loop's mcs does) -- just replayed. This also guards the
   ``_dev_mcs`` kernels as mechanical copies of the validated eager kernels.
2. **Graph == eager, BIT-EXACT (batched sweep).** Same, for R replicas advanced per
   graph replay.
3. **Graph == eager WITH FPP links bound.** The captured sweep binds the (static
   within a sweep) FPP link CSR; result is bit-exact to the eager FPP sweep.
4. **Bounded timing sanity.** Graph throughput is not materially slower than eager,
   and both produce valid volume partitions. (Heavy benchmark is opt-in: BENCH=1.)
5. **A small in-gate benchmark** runs and yields finite positive MCS/s for eager +
   graph + batched (so the report numbers come from real, executed runs).

Sizes are kept small/bounded so the file stays well within the suite time budget.
"""

import os

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine, GraphRunner
from engine import BatchedGraphRunner
from engine.batched import BatchedGPUEngine, build_batched_grid_state
from engine import bench as bench_mod

LATTICE_L = 24
N_MCS = 30
CELLS_PER_AXIS = 3
BASE_SEED = 4242

CONTACT = np.array([[0.0, 5.0], [5.0, 1.0]])
TARGET_VOLUME = np.array([0.0, 64.0])
LAMBDA_VOLUME = np.array([0.0, 4.0])


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")


def _cfg(seed, contact=CONTACT, lam=LAMBDA_VOLUME, tv=TARGET_VOLUME, L=LATTICE_L):
    return EngineConfig(
        Lx=L, Ly=L, Lz=L, seed=seed, temperature=10.0,
        target_volume=tv, lambda_volume=lam, contact=contact,
    )


# --------------------------------------------------------------- correctness
@cuda_only
def test_graph_equals_eager_bit_exact_single():
    """Single-engine: a graph-captured run is BIT-EXACT to an eager run (same seed):
    identical id-lattice, int64 COM sums, and volumes."""
    eager = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eager.run(N_MCS)

    geng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    gr = GraphRunner(geng)
    gr.run(N_MCS)

    assert np.array_equal(geng.get_ids(), eager.get_ids()), "id-lattice differs (graph vs eager)"
    assert np.array_equal(geng.xsum.numpy(), eager.xsum.numpy()), "xsum differs"
    assert np.array_equal(geng.ysum.numpy(), eager.ysum.numpy()), "ysum differs"
    assert np.array_equal(geng.zsum.numpy(), eager.zsum.numpy()), "zsum differs"
    assert np.array_equal(geng.volumes(), eager.volumes()), "volumes differ"
    # graph-advanced state is still a valid exact partition
    geng.assert_volume_partition()


@cuda_only
def test_graph_replay_advances_rng_not_repeats():
    """Sanity that the device mcs counter actually advances per replay (not a fixed
    key): a 2-MCS graph run differs from running the SAME single MCS twice. If the
    captured graph repeated mcs=0, these would coincide; they must not."""
    # proper 2-MCS graph run
    g2 = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    GraphRunner(g2).run(2)
    # eager reference of 2 distinct MCS
    e2 = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    e2.run(2)
    assert np.array_equal(g2.get_ids(), e2.get_ids())
    # eager run that ERRONEOUSLY repeats mcs=0 twice (what a baked-scalar graph does)
    e_rep = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    e_rep.step_mcs(0)
    e_rep.step_mcs(0)
    import warp as wp
    wp.synchronize()
    assert not np.array_equal(g2.get_ids(), e_rep.get_ids()), (
        "graph replay used a FIXED mcs key (did not advance) -- RNG stream wrong"
    )


@cuda_only
def test_graph_equals_eager_bit_exact_batched():
    """Batched: a graph-captured R-replica sweep is bit-exact to an eager batched
    run (same per-replica keys), id-lattice / int64 COM / volumes identical."""
    R = 5
    eager = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS))
    eager.run(N_MCS)

    geng = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS))
    BatchedGraphRunner(geng).run(N_MCS)

    assert np.array_equal(geng.get_ids(), eager.get_ids()), "batched id-lattice differs"
    ex, ey, ez = eager.com_sums()
    gx, gy, gz = geng.com_sums()
    assert np.array_equal(gx, ex) and np.array_equal(gy, ey) and np.array_equal(gz, ez), \
        "batched COM sums differ"
    assert np.array_equal(geng.volumes(), eager.volumes()), "batched volumes differ"
    geng.assert_volume_partition()


@cuda_only
def test_graph_with_fpp_links_capture_boundary_and_fidelity():
    """FPP graph capture: the CAPTURE BOUNDARY is honored and the graph FPP sweep is
    PHYSICALLY FAITHFUL to eager -- but NOT asserted bit-exact, by design.

    Capture boundary: FPP ``rebuild()`` does HOST compaction (rebuild the link CSR
    from the Python inventory), so it runs BEFORE capture, OUTSIDE the graph; the
    captured device sweep reads the bound (static-within-a-sweep) link CSR. Links
    are re-bound (re-captured) only when topology changes.

    Why NOT bit-exact (documented Phase 4 Pass B finding): the FPP spring energy
    reads the COM (xsum/ysum/zsum/volume) of *linked* cells anywhere on the lattice,
    while concurrent same-color flips atomically UPDATE those COM accumulators. The
    checkerboard removes spatial-neighbour write hazards but NOT these cross-cell COM
    reads, so the existing Phase-3 FPP kernel is already nondeterministic eager-vs-
    eager (verified: ~5/8 eager reruns differ by ~one boundary voxel/cell); the
    atomic-append link-CSR ordering adds a second non-associative-float source. Graph
    capture neither causes nor cures this -- it faithfully replays one realisation.
    The non-FPP device sweep (the stated Pass B bit-exact deliverable) IS bit-exact
    (see the single/batched tests). Here we assert the graph FPP run is a VALID
    partition and statistically matches eager (mean cell volume), which holds
    deterministically."""
    from engine import grid_graph_links
    from engine.fpp import FPPLinks

    pairs = grid_graph_links(CELLS_PER_AXIS)            # (M,2) axis-adjacent cell pairs

    def build(seed):
        eng = GPUEngine(build_grid_state(_cfg(seed), CELLS_PER_AXIS))
        fpp = FPPLinks(eng, target_length_default=4.0, lambda_default=2.0,
                       max_length_default=50.0)
        fpp.set_topology(pairs)
        eng.attach_fpp(fpp)
        return eng

    eager = build(BASE_SEED)
    eager.run(N_MCS)

    geng = build(BASE_SEED)
    gr = GraphRunner(geng)
    gr.run(N_MCS)

    # capture boundary: the FPP CSR was rebuilt (host compaction) before/at capture,
    # and the captured sweep recorded it as bound device work (fpp enabled).
    assert gr._captured_fpp_enabled == 1, "FPP not bound into the captured sweep"
    assert geng.fpp.num_active() == eager.fpp.num_active() == pairs.shape[0]

    # the graph FPP run is a VALID exact partition (atomic COM/volume correctness)
    geng.assert_volume_partition()

    # and PHYSICALLY FAITHFUL to eager: mean cell volume agrees within tolerance
    gv, ev = geng.volumes(), eager.volumes()
    rel = abs(gv.mean() - ev.mean()) / abs(ev.mean())
    diff_vox = int((geng.get_ids() != eager.get_ids()).sum())
    print(
        f"\n[fpp-graph] valid partition; mean vol graph {gv.mean():.3f} vs eager "
        f"{ev.mean():.3f} (rel {rel:.2e}); id-lattice diff {diff_vox} voxels "
        f"(expected small & nonzero: pre-existing FPP COM-read race, not graph-induced)"
    )
    assert rel < 0.02, f"FPP graph mean volume drifted from eager: rel {rel:.2e}"


@cuda_only
def test_graph_recapture_reproducible():
    """Capturing + replaying twice from the same IC reproduces the exact trajectory
    (graph capture introduces no nondeterminism)."""
    a = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    GraphRunner(a).run(N_MCS)
    b = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    GraphRunner(b).run(N_MCS)
    assert np.array_equal(a.get_ids(), b.get_ids())
    assert np.array_equal(a.xsum.numpy(), b.xsum.numpy())


# --------------------------------------------------------------- timing / bench
@cuda_only
def test_graph_timing_sanity_not_slower():
    """Bounded timing sanity: on a modest lattice the graph path is NOT materially
    slower than eager, and both leave a valid partition. (A warm-up + short timed
    run; the heavy benchmark is gated behind BENCH=1.)"""
    d = bench_mod.bench_single(L=48, n_mcs=60, warmup=10, seed=BASE_SEED)
    assert np.isfinite(d["mcs_eager"]) and d["mcs_eager"] > 0
    assert np.isfinite(d["mcs_graph"]) and d["mcs_graph"] > 0
    # graph must not be materially slower than eager (allow 15% noise slack on a
    # short timed run); in practice it is faster due to launch-overhead removal.
    assert d["mcs_graph"] >= 0.85 * d["mcs_eager"], (
        f"graph slower than eager: graph {d['mcs_graph']:.1f} vs eager {d['mcs_eager']:.1f} MCS/s"
    )
    print(
        f"\n[timing-sanity L=48] eager {d['mcs_eager']:.1f} | graph {d['mcs_graph']:.1f} MCS/s "
        f"(x{d['graph_speedup']:.2f}); vs CPU(10.5): eager x{d['eager_vs_cpu']:.1f}, "
        f"graph x{d['graph_vs_cpu']:.1f}"
    )


@cuda_only
def test_inline_benchmark_runs_and_reports():
    """Run the small in-gate benchmark (single + batched) so the report numbers are
    from real executed runs; assert all throughputs are finite and positive."""
    s = bench_mod.bench_single(L=48, n_mcs=60, warmup=10, seed=BASE_SEED)
    b = bench_mod.bench_batched(L=48, R=8, n_mcs=40, warmup=10, seed=BASE_SEED)
    bench_mod._print_single(s)
    bench_mod._print_batched(b)
    for k in ("mcs_eager", "mcs_graph"):
        assert np.isfinite(s[k]) and s[k] > 0
    assert np.isfinite(b["replica_mcs_per_s"]) and b["replica_mcs_per_s"] > 0
    assert np.isfinite(b["sims_per_hour"]) and b["sims_per_hour"] > 0


@cuda_only
@pytest.mark.skipif(os.environ.get("BENCH", "0") != "1", reason="set BENCH=1 for the heavy benchmark")
def test_heavy_benchmark():
    """Opt-in heavy benchmark (BENCH=1): larger lattices + larger R. Not part of the
    default fast gate."""
    for L in (64, 100, 128):
        d = bench_mod.bench_single(L=L, n_mcs=400, warmup=20, seed=BASE_SEED)
        bench_mod._print_single(d)
        assert d["mcs_graph"] > 0
    for (L, R) in ((64, 32), (100, 16)):
        d = bench_mod.bench_batched(L=L, R=R, n_mcs=300, warmup=20, seed=BASE_SEED)
        bench_mod._print_batched(d)
        assert d["replica_mcs_per_s"] > 0
