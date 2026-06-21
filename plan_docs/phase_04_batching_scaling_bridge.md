# Phase 4 — Batching (sweeps), bigger lattices, Player/interactive bridge

> Status: Phase 3 complete; spec refined 2026-06-20. Source: `plan_docs/CC3D-GPU-port-plan.md` -> "Phase 4" and "GPU
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

## Carry-forward from Phase 3 (verdict handoff — `.phaserun/verdict_phase3.json`)

- **The engine + Embryo model are complete and validated — do NOT re-port physics.** `gpu_port/engine/` has the
  GPU-resident core (state, 8-color Volume+Contact Metropolis, int64 fixed-point COM, hashed neighbor-CSR that
  scales to 63k cells), `fpp.py` (device link CSR), `cohesotaxis.py` (on-device lamellipodia pipeline), and the
  steppable API; `gpu_port/embryo/` has `EmbryoModel` (full driver) + tissue/substrate/closure steppables. Phase 4
  wraps these for scale.
- **Reproducibility primitives to reuse for the batch axis:** Philox RNG is keyed by `(mcs, color, seed)` (and
  `(mcs, cell, seed)` for steppables). Fold the replica index INTO the key so each batch replica is an independent,
  reproducible stream. COM is int64 fixed-point (bit-exact); keep per-replica reductions reproducible the same way —
  do NOT introduce float atomics across the batch axis.
- **Production stays at `contact_neighbor_order=3`** (race-safe under the 8-color checkerboard). The 27-color order-4
  SWEEP remains future work (order-4 contact ENERGY is already validated GPU-reproducible). Batch/scale on the
  order-3 production config.
- **Injection hooks:** reuse the model's `ifPythonCall` / `RunNumber` seam to vary per-replica parameters for the
  sweep — this is the CC3D-side mechanism the batch dimension mirrors.
- **Outstanding (NOT blocking Phase 4):** the long-horizon offline CC3D closure/intercalation ensemble
  (`CC3D_OFFLINE=1`) has not been run; it is a fidelity backstop independent of scaling work.

## Delivery discipline (large/iterative phase — sequential, tested passes)

Like Phase 3, the orchestrator drives Phase 4 as tested increments, each leaving `pixi run python -m pytest -q
gpu_port` green (all prior 49 included):
1. **Batch dimension (the dominant utilization win):** add a replica/batch axis to the engine state + kernels so N
   parameter-sweep replicas advance concurrently on-device (replica index folded into the Philox key; per-replica
   params via the `ifPythonCall`/`RunNumber` seam). Validate per-replica results match the single-run distributions
   (a batched run == N independent single runs, statistically).
2. **CUDA-Graph MCS capture + throughput:** wrap the per-MCS loop in a CUDA Graph (dynamic per-MCS link
   create/delete behind `wp.capture_if`) to cut launch/Python overhead; benchmark MCS/s (and sims/hour for sweeps)
   vs the Phase-0 tuned multicore CPU baseline (~10.5 MCS/s).
3. **Larger-lattice validation + bridge:** validate a larger single-GPU lattice run (RTX 5090 / 32 GB); bridge to
   cc3d-player5 or a lightweight torch-fed viewer for interactivity. **Multi-GPU halo exchange is DEFERRED to future
   work unless a target lattice provably exceeds single-GPU memory** — note it, don't build it speculatively.

Each pass: green tests, report DONE vs DEFERRED. Escalate only on a real problem (incorrect / statistically
unfaithful per-replica results, a throughput regression vs CPU, infeasibility, or a gate flag).

## Exit gate

Tests under `gpu_port/phase4/tests/`: batched runs produce per-replica results matching single-run
distributions; sweep throughput (sims/hour) measured vs experiment-level scale-out; a larger-lattice
run validated; benchmark MCS/s vs the tuned multicore CPU baseline (Phase 0: ~10.5 MCS/s).
