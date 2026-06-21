"""Throughput benchmark -- Phase 4 Pass B.

Measures MCS/s for the GPU CPM engine in three configurations and reports speedup
vs the Phase-0 tuned multicore CPU baseline (~10.5 MCS/s, 32 OpenMP threads):

* GPU eager  : the per-MCS Python loop (8 kernel launches/MCS).
* GPU graph  : a single CUDA-graph replay of the per-MCS device hot loop.
* GPU batched (graph): R sweep replicas advanced per graph replay -> sims/hour.

The timed region is the inner per-MCS loop ONLY (host-side tracker rebuilds /
observable read-back are excluded -- they are the SAME work in eager and graph
paths and live outside the capture boundary; this isolates the launch-overhead
win the graph delivers). A warm-up run precedes every timed run so module
compilation / first-touch allocation are never timed.

Importable: ``bench_single`` / ``bench_batched`` return dicts of numbers (used by
the bounded in-gate sanity test). As a script (``pixi run python -m engine.bench``
or with ``BENCH=1`` for the heavy config) it prints a report.
"""

from __future__ import annotations

import os
import time

import numpy as np

import warp as wp

from .config import EngineConfig
from .state import build_grid_state
from .batched import build_batched_grid_state, BatchedGPUEngine
from .engine import GPUEngine
from .graph import GraphRunner, BatchedGraphRunner

wp.init()

CPU_BASELINE_MCS_PER_S = 10.5   # Phase 0 tuned multicore CPU (PROGRESS.md)


def _make_cfg(L: int, seed: int = 12345) -> EngineConfig:
    return EngineConfig(
        Lx=L, Ly=L, Lz=L, seed=seed, temperature=10.0,
        target_volume=np.array([0.0, 64.0]),
        lambda_volume=np.array([0.0, 2.0]),
        contact=np.array([[0.0, 16.0], [16.0, 4.0]]),
    )


def _cells_per_axis(L: int, block_target: int = 4) -> int:
    # roughly one cell per (block_target*~1.5)^3 box; keep cells well-separated
    return max(2, L // (block_target + 2))


def _time_loop(step_fn, n_mcs: int) -> float:
    """Time ``n_mcs`` calls of ``step_fn`` (device work), returning MCS/s. Assumes
    the device is already warm; synchronises once at the end."""
    wp.synchronize()
    t0 = time.perf_counter()
    step_fn(n_mcs)
    wp.synchronize()
    dt = time.perf_counter() - t0
    return n_mcs / dt if dt > 0 else float("inf")


def bench_single(L: int = 64, n_mcs: int = 200, warmup: int = 10, seed: int = 12345) -> dict:
    """Eager vs graph-captured MCS/s for a single ``GPUEngine`` on an ``L^3`` lattice."""
    cfg = _make_cfg(L, seed)
    cpa = _cells_per_axis(L)

    # --- eager ---
    eng_e = GPUEngine(build_grid_state(cfg, cpa))
    eng_e.run(warmup)                                   # warm-up (module load)
    mcs_eager = _time_loop(lambda n: eng_e.run(n, mcs_offset=warmup), n_mcs)

    # --- graph ---
    eng_g = GPUEngine(build_grid_state(cfg, cpa))
    gr = GraphRunner(eng_g)
    gr.run(warmup)                                      # captures + warms up
    mcs_graph = _time_loop(lambda n: gr.run(n, mcs_offset=warmup), n_mcs)

    return {
        "L": L, "n_cells": eng_e.n_cells, "n_mcs": n_mcs,
        "mcs_eager": mcs_eager, "mcs_graph": mcs_graph,
        "graph_speedup": mcs_graph / mcs_eager if mcs_eager > 0 else float("nan"),
        "eager_vs_cpu": mcs_eager / CPU_BASELINE_MCS_PER_S,
        "graph_vs_cpu": mcs_graph / CPU_BASELINE_MCS_PER_S,
    }


def bench_batched(L: int = 64, R: int = 16, n_mcs: int = 200, warmup: int = 10,
                  seed: int = 12345) -> dict:
    """Graph-captured batched sweep: R replicas advanced per graph replay.

    Reports aggregate MCS/s (R*n_mcs / wall), and sims/hour for a sweep of
    ``sweep_mcs`` MCS per replica (the throughput a parameter sweep would see)."""
    cfg = _make_cfg(L, seed)
    cpa = _cells_per_axis(L)
    bs = build_batched_grid_state(cfg, R, cpa)
    beng = BatchedGPUEngine(bs)
    gr = BatchedGraphRunner(beng)
    gr.run(warmup)                                      # capture + warm-up
    wp.synchronize()
    t0 = time.perf_counter()
    gr.run(n_mcs, mcs_offset=warmup)
    wp.synchronize()
    dt = time.perf_counter() - t0
    replica_mcs_per_s = (R * n_mcs) / dt if dt > 0 else float("inf")
    sweep_mcs = 10000
    sims_per_hour = (replica_mcs_per_s / sweep_mcs) * 3600.0
    return {
        "L": L, "R": R, "n_cells": beng.n_cells, "n_mcs": n_mcs,
        "replica_mcs_per_s": replica_mcs_per_s,
        "wall_mcs_per_s": n_mcs / dt if dt > 0 else float("inf"),
        "sweep_mcs": sweep_mcs, "sims_per_hour": sims_per_hour,
        "replica_vs_cpu": replica_mcs_per_s / CPU_BASELINE_MCS_PER_S,
    }


def _print_single(d: dict):
    print(
        f"[single  L={d['L']}^3 cells={d['n_cells']} mcs={d['n_mcs']}] "
        f"eager {d['mcs_eager']:8.1f} MCS/s | graph {d['mcs_graph']:8.1f} MCS/s | "
        f"graph speedup x{d['graph_speedup']:.2f} | "
        f"vs CPU(10.5): eager x{d['eager_vs_cpu']:.1f}, graph x{d['graph_vs_cpu']:.1f}"
    )


def _print_batched(d: dict):
    print(
        f"[batched L={d['L']}^3 R={d['R']} cells={d['n_cells']} mcs={d['n_mcs']}] "
        f"aggregate {d['replica_mcs_per_s']:9.1f} replica-MCS/s "
        f"(wall {d['wall_mcs_per_s']:.1f} MCS/s) | "
        f"vs CPU(10.5) x{d['replica_vs_cpu']:.1f} | "
        f"~{d['sims_per_hour']:.0f} sims/hour @ {d['sweep_mcs']} MCS/sim"
    )


def main():
    heavy = os.environ.get("BENCH", "0") == "1"
    print("=== Phase 4 Pass B throughput benchmark (RTX 5090) ===")
    if heavy:
        for L in (64, 100, 128):
            _print_single(bench_single(L=L, n_mcs=500, warmup=20))
        for (L, R) in ((64, 16), (64, 32), (100, 16)):
            _print_batched(bench_batched(L=L, R=R, n_mcs=300, warmup=20))
    else:
        _print_single(bench_single(L=64, n_mcs=200, warmup=10))
        _print_batched(bench_batched(L=64, R=16, n_mcs=150, warmup=10))
        print("(set BENCH=1 for the larger multi-lattice / multi-R sweep)")


if __name__ == "__main__":
    main()
