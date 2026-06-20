# Phase 0 findings — profile + tuned CPU baseline (Embryo model)

**Date:** 2026-06-20 · **Machine:** 32 logical CPUs, NVIDIA GeForce RTX 5090 (Blackwell), Windows 11
**Model:** `Embryo_Model_dev/Embryo` — 100³, 3D, NeighborOrder=4, ~500 cells, Volume + Contact + FPP
**Harness:** [`profile_embryo.py`](profile_embryo.py) (copies project, caps Steps, captures CC3D's
built-in engine-vs-steppable timing). Runs are warm steady-state (150–200 MCS), `SaveSegmentation` off.

## Result: where wall-clock goes

| config | engine ms/MCS (C++ CPM+FPP+trackers) | steppables ms/MCS (Python FPP links) | MCS/s (loop) |
|---|---|---|---|
| procs=1 (baseline)      | 127.7  (76%) | 36.6 (22%) | **5.95** |
| procs=8 (OpenMP)        | 55.8         | 52.8       | 8.90 |
| procs=16                | 33.5         | 60.5       | 10.12 |
| procs=32                | 30.5  (29%)  | 59.6 (57%) | **10.52** |
| boundarywalker (p1)     | *invalid — 0 attempts* | — | *excluded* |

Steppable breakdown (procs=1): `PassiveSteppable` ≫ `LeadingEdgeSteppable` ≫ `EmbryoSteppable≈0`.
(Passive dominates because it manages link turnover over the growing passive-cell population.)

## Conclusions (the Phase 0 go/no-go gate)

1. **GO for the GPU port — and it must target BOTH the CPM sweep and the steppables.**
   - The C++ CPM Metropolis sweep (incl. FPP energy + trackers) is the largest single cost (~76% at
     1 core) → the plan's primary GPU target is correct.
   - The Python steppables (FPP link management) are the second cost (~22% at 1 core) and **do not
     parallelize**. At 32 threads they become the *dominant* cost (~57%).

2. **The CPU is Amdahl-capped at ≈16 MCS/s.** 32 OpenMP threads give only **1.77×** overall (5.95 →
   10.52 MCS/s) even though the engine scales ~4.2×, because the serial Python steppables (~60 ms/MCS)
   form a hard ceiling (1000/60 ≈ 16.7 MCS/s) no matter how many cores. **To beat ~16 MCS/s you must
   move the link-management steppables off the serial Python path — exactly the plan's on-GPU
   steppables thrust.** Accelerating only the sweep is not enough (at p32 the engine is already
   smaller than the steppables).

3. **The bar the GPU must beat ≈ 10.5 MCS/s** (tuned 32-thread CPU), with a theoretical CPU ceiling of
   ~16 MCS/s. Full 4000-MCS run today: ~11 min (1 core) → ~6.3 min (32 cores).

4. **`metropolisBoundaryWalker` is not a usable CPU lever here.** It sets
   `numberOfAttempts = boundaryPixelSet.size()` (Potts3D.cpp:1084); that set is unpopulated in this
   model's configuration, so the run did **0 pixel-copy attempts** (engine 0.02 ms/MCS) — the CPM
   never evolved. Would require extra wiring to enable; excluded from the baseline.

5. **Context for GPU expectations** (consistent with the plan): a single 100³/~500-cell run is the
   weakest case for the GPU; the wins are ensemble batching (sweeps) and bigger lattices. But even for
   a single run, removing the serial-steppable ceiling is the key, which only the GPU-resident
   steppable model delivers.

## Reproduce

```
pixi run python gpu_port/phase0/profile_embryo.py --steps 200 --procs 1  --tag baseline_p1
pixi run python gpu_port/phase0/profile_embryo.py --steps 200 --procs 32 --tag tuned_p32
```
Results land in `gpu_port/phase0/_runs/<tag>/result.json`.

## Environment fix required to run at all

The pixi env was broken: `import ctypes` (and thus `torch`, `imageio`, the simulation) failed with
`DLL load failed while importing _ctypes`. Root cause: defaults-channel Python 3.12's `_ctypes.pyd`
loads **`ffi.dll`**, but conda-forge `libffi` ships it as **`ffi-8.dll`** (`channel-priority =
"disabled"` mixed channels). Fix applied: alias `DLLs/ffi.dll` → `Library/bin/ffi-8.dll` (ABI-identical
libffi). Re-runnable via [`../fix_env_ffi.py`](../fix_env_ffi.py). Durable fix = make Python + libffi
channel-consistent (see that script's header).
