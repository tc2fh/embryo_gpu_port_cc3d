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

The Phase 1 prototype at `gpu_port/phase1/` is **read-only reference** — do not modify it (out of
scope), but reuse its proven patterns. Build the production engine fresh under `gpu_port/engine/`.

## Carry-forward from Phase 1 (verdict handoff)

- **Warp is not in `pixi.lock`** — ensure it is installed (`pixi run pip install warp-lang`); do not
  `pixi add` (re-solve risk).
- This Warp build has **no `wp.mat`/`wp.matrix` const type** — pass constant tables (neighbor offsets,
  contact/fpp matrices) as flat `int32`/`float32` device arrays. Kernels must live in **real `.py`
  files** (Warp reads source via `inspect`; no `exec()`'d kernels).
- The **8-color** checkerboard is correct for NeighborOrder<=3. The Embryo model is NeighborOrder=4
  (needs 27 colors or the plan's sanctioned order<=3 restriction). **Decide explicitly this phase**:
  default to order<=3 / 8-color and validate equivalence vs the CPU order-4 run; keep 27-color
  reachable if the order-4 physics proves necessary.
- GPU float `atomic_add` for COM/volume is **non-bit-reproducible** (thread-order, non-associative)
  though statistically equivalent. Per-cell volume vs lattice-count must still be exact (assert it).
  Where reproducibility matters, use **int64 fixed-point** accumulation or per-MCS COM recompute.

## Delivery discipline (this phase is large — deliver a tested slice)

Phase 2 is multi-week scope. Prioritize a **coherent, tested vertical slice** over a broad half-build:
finish the engine core first (GPU-resident state + 8-color Metropolis + Volume+Contact + on-GPU
volume/COM trackers + statistical validation vs the CPU reference) with **green tests**, then add the
boundary-pixel/neighbor CSR, the GPU steppable API skeleton, and the Embryo non-FPP geometry as far
as fits. The gate command `pixi run python -m pytest -q gpu_port` (which also re-runs Phase 1's tests)
MUST be green at your stopping point. Report precisely what is done vs deferred.

## Exit gate

Tests under `gpu_port/phase2/tests/`: Volume+Contact statistics (energy, volume, surface, COM
distributions) match CPU CC3D within Monte-Carlo tolerance over an ensemble; geometry constructors
reproduce the Embryo initial condition; tests skip cleanly without CUDA.
