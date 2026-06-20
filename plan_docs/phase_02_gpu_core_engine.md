# Phase 2 — GPU-native core: Volume + Contact + trackers + GPU steppable model

> Status: pending Phase 1 GO. Source: `plan_docs/CC3D-GPU-port-plan.md` -> "Phase 2",
> "Architecture", "GPU-resident steppables". Refine from Phase 1's handoff delta before starting.

## Objective

Build the GPU-resident production engine: 8-color checkerboard + tiled shared-memory halos, the
GPU-resident state (int32 id-lattice + per-cell SoA + contact matrix), on-GPU trackers (volume/COM
via atomics; boundary-pixel + neighbor CSR once per MCS), and the **GPU steppable API** (Python
`@wp.kernel` logic over SoA `cell.dict`-equivalents). Port the Embryo `start()` geometry
(`create_hollow_sphere`, `mesendoderm_sphere_circumference`) and the **non-FPP** steppable logic as
on-device kernels. FPP and cohesotaxis are deferred to Phase 3.

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/phase2/`
- `pixi.toml`
- `pixi.lock`

## Exit gate

Tests under `gpu_port/phase2/tests/`: Volume+Contact statistics (energy, volume, surface, COM
distributions) match CPU CC3D within Monte-Carlo tolerance over an ensemble; geometry constructors
reproduce the Embryo initial condition; tests skip cleanly without CUDA.
