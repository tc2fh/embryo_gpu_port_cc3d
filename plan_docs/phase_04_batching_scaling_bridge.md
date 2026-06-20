# Phase 4 — Batching (sweeps), bigger lattices, Player/interactive bridge

> Status: pending Phase 3. Source: `plan_docs/CC3D-GPU-port-plan.md` -> "Phase 4" and "GPU
> utilization & scaling". Refine from Phase 3's handoff delta. This phase is ongoing/iterative.

## Objective

Add the batch (leading) dimension for parameter sweeps — the dominant utilization win — reusing the
`ifPythonCall`/`RunNumber` injection hooks; wrap the MCS in a CUDA Graph (dynamic link work behind
`wp.capture_if`); scale to larger lattices (multi-GPU halo exchange if needed); bridge to
cc3d-player5 / a lightweight torch-fed viewer for interactivity.

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/bridge/`
- `gpu_port/phase4/`

## Exit gate

Tests under `gpu_port/phase4/tests/`: batched runs produce per-replica results matching single-run
distributions; sweep throughput (sims/hour) measured vs experiment-level scale-out; a larger-lattice
run validated; benchmark MCS/s vs the tuned multicore CPU baseline (Phase 0: ~10.5 MCS/s).
