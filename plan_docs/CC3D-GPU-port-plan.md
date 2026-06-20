# Plan: GPU-resident Cellular Potts engine for the Embryo model (CC3D-GPU)

## Context

`Embryo_Model_dev` is a 3D *Xenopus* gastrulation (mesendoderm mantle closure) model built on
CompuCell3D (CC3D). The goal is to run the **entire simulation on the GPU** while keeping CC3D's
modeling objects and the ease of **custom Python scripting**, by the path of least resistance.

User intent (from clarifying questions):
- **All four payoffs wanted**: faster single runs, high-throughput parameter sweeps, bigger lattices,
  real-time interactivity.
- **NVIDIA-only** target (env already pulls `torch 2.12.1+cu130`).
- Scope = **core engine + extensible** (cover this model now; allow new energy terms/plugins later).
- Willing to invest in a **substantial rebuild** for full GPU residency.

Honest framing (validated by the research workflow + adversarial review): **no GPU CPM reproduces all
of CC3D**, and the one thing this model depends on most — FocalPointPlasticity (FPP) spring links
driven by arbitrary per-MCS Python — has **zero GPU precedent in any framework or paper**. So the plan
is not "port CC3D." It is: build a new GPU-native engine for the *bounded* feature set this model
needs, but **de-risk the single hard unknown (GPU FPP) before committing**, and keep an honest
CPU/scale-out baseline as the comparator.

### Build-context fact that shapes everything

The CC3D that **runs** is the **installed conda binary** (`compucell3d >=4.7.0,<4.8`, win-64, Python
3.12, in the pixi env) — **not** the in-repo `CompuCell3D/` source tree (which reports Version 4.3.2
in `Embryo.xml`). Consequence:
- Any approach that **modifies CC3D's C++** (Potts3D, watchers, SWIG bindings) requires building CC3D
  from source on Windows with the full SWIG/conda toolchain and shadowing the binary package — a
  major, recurring cost.
- A **separate GPU-native engine** that does *not* touch CC3D's C++ **avoids that build entirely**.
  This is a decisive point in favor of the recommended approach, and the in-repo source becomes a
  *read-only reference for porting semantics*, not a build target.

## What the Embryo model actually needs (the scope that matters)

From `Embryo.xml` + `EmbryoSteppables.py`:
- **Lattice**: 100×100×100, 3D, `NeighborOrder=4`, Temperature 10, 4000 MCS, non-periodic.
  `NumberOfProcessors=1` (runs serial today — a tuning lever, see baseline below).
- **Energy terms in use**: `Volume`, `Contact` (5 types), **`FocalPointPlasticity` (Local links)**.
  **No diffusion/PDE fields, no chemotaxis, no secretion** → CC3D's existing OpenCL PDE solvers
  (`core/.../PDESolvers/OpenCL/`) are **irrelevant to this model**.
- **Trackers in use**: `CenterOfMass`, `BoundaryPixelTracker`, `NeighborTracker`, `CellType`, + FPP
  link inventory.
- **Scale is small**: a few hundred non-substrate cells (rings of ~40–60 cells each from
  `mesendoderm_sphere_circumference`) + a hollow-sphere substrate shell; only a few dozen `Leading`
  cells. `ifCohesotaxis=0` in the current config — but the user wants the **`ifCohesotaxis=1`** path
  (boundary-pixel classification + `PixelDist` + `SigWeights` weighted selection in
  `create_lamellipodia_link`) GPU-accelerated too, so it is **in scope** (design in "GPU-resident
  steppables" below).
- **FPP-dominated dynamics**: Python steppables (`LeadingEdgeSteppable`, `PassiveSteppable`,
  `EmbryoSteppable`) create/delete spring links each MCS via biological rules (lamellipodia links,
  Poisson turnover, substrate/tissue links). These run at **cell granularity, Poisson-gated** —
  order **10³–10⁴ lattice accesses/MCS**, *not* millions. Full-lattice `every_pixel()` scans happen
  only at I/O checkpoints (`SaveSegmentation`, every 500 MCS).

The GPU engine must therefore implement exactly **Volume + Contact + FPP** energies, the
**COM/Volume/Boundary/Neighbor** trackers, and a **per-MCS Python-steppable boundary** — a tractable,
bounded surface, not all of CC3D.

## Key engine facts (from direct source reading)

- `Potts3D::metropolisFast` (`Potts3D.cpp:660-917`): per MCS does `Nx·Ny·Nz·flip2DimRatio` (~10⁶)
  attempts; each picks a random pixel + neighbor, calls `energyCalculator->changeEnergy` (sum over
  registered energy functions, `Potts3D.cpp:474`), Boltzmann-accepts, then `cellFieldG->set(...)`
  cascades incremental updates to every watchable tracker.
- **CC3D already parallelizes the CPM** via OpenMP over **subgrid sections** (`#pragma omp parallel` +
  `pUtils->getPottsSection`, `Potts3D.cpp:737-917`) — the same checkerboard idea a GPU uses. So GPU
  parallel updates are an *extension of an approximation CC3D users already accept in multicore mode*,
  not a novel correctness sin.
- `Potts3D` also offers **`metropolisBoundaryWalker`** (`Potts3D.cpp:~1007`, single-thread) that only
  attempts flips at cell boundaries — for ~500 compact cells the boundary set is a small fraction of
  10⁶, slashing attempt count. A relevant CPU optimization and a model for boundary-restricted GPU
  sampling. (NOTE: Phase 0 found it does nothing in this model as-is — `numberOfAttempts =
  boundaryPixelSet.size()` and that set is unpopulated → 0 attempts. Would need extra wiring.)
- **FPP energy** = `offset + λ·(L − L_target)²` per link (`FocalPointPlasticityPlugin.cpp:258`), with
  per-type-pair λ/target/max arrays. **Caveat**: FPP `changeEnergy` reads per-cell COM
  (`xCM/yCM/zCM`, `precalculateCentroid`) **and can mutate the link inventory mid-evaluation**
  (`tryAddingNewJunction`). So COM and link state are entangled with the inner loop — the GPU port
  must define a single source of truth for centroids and links.
- The authoritative field is `WatchableField3D<CellG*>` (a field of **heap pointers**, host RAM in the
  binary build). There is **no contiguous device-side cell-ID buffer to "zero-copy"** out of CC3D; a
  GPU engine maintains its own dense `int32` ID lattice as the source of truth (zero-copy then applies
  *within* the new engine, e.g. to `torch`, not to CC3D's field).

## Recommended approach

Build a new **GPU-native, GPU-resident CPM engine ("CC3D-GPU")** in **NVIDIA Warp** (leading
candidate, see spike) exposing a **CC3D-style Python steppable API** — but **gate the build on two
cheap up-front phases** that test the one assumption that can sink the project.

This aligns with the user's choices: a separate engine matches "substantial rebuild" + "core engine +
extensible," fully GPU-resident state serves *bigger lattices* and *single-run speed*, a batched
lattice dimension serves *sweeps*, and `torch` interop serves *interactivity*. It also **sidesteps the
CC3D-from-source/SWIG build** entirely.

### Engine substrate (validate in the spike, don't pre-commit)

**NVIDIA Warp** is the leading candidate, and the research strengthened the case for *this* stack:
Python-authored `@wp.kernel` GPU code (preserves "easy custom scripting" + "extensible energy terms
in Python"); `wp.array` GPU residency; `wp.atomic_add`/`atomic_cas` for per-cell reductions and
dynamic link append; **counter-based RNG (Philox) keyed by `(id, mcs)`** for reproducible,
stateless per-element streams; **CUDA-graph replay of the whole MCS** via `wp.capture_if`/
`wp.capture_while` (Warp ≥1.8 → CUDA 12.4+ conditional graph nodes) to kill per-MCS launch overhead;
and **zero-copy `torch` interop** (`wp.from_torch`/`wp.to_torch`) matching the cu130 env. Taichi is
the alternate (more "biologist-writable" surface, native dynamic SNodes, but weaker RNG-seed control
and no conditional-graph capture; zero-copy only via `ti.types.ndarray()` kernel args). **None of
Warp/CuPy/Taichi is in `pixi.lock`** — Phase 1 must confirm win-64 + Python 3.12 + CUDA-13 wheels and
interop with `torch 2.12.1+cu130` before locking the choice.

### Architecture (target engine)

GPU-resident state (per simulation, batchable along a leading dim):
- `cell_ids`: int32 lattice `[B,Nz,Ny,Nx]` (0 = Medium) — the source of truth.
- Per-cell SoA arrays: `type`, `volume`, `target_volume`, `lambda_volume`, `com_x/y/z`, `frozen`.
- `contact_energy[type][type]`, `fpp_params[type][type]` (λ, target, max).
- **FPP link list**: `(cell_a, cell_b, λ, target, max)` arrays + per-cell CSR index, with an atomic
  free-list for dynamic create/destroy; **rebuilt/pushed from Python each MCS**.

Kernels:
- **Checkerboard Metropolis** (mirrors CC3D subgrid coloring; multi-color sized for `NeighborOrder=4`),
  per-thread RNG, optional boundary-restricted sampling. `ΔE = ΔVolume + ΔContact + ΔFPP`; atomic
  volume/COM updates on accept.
- **Composable energy-term registry**: Volume/Contact/FPP as device functions; new terms added as
  Python-authored Warp functions → "extensible." FPP uses each cell's current (possibly 1-color-step
  stale) COM — same class of approximation as CC3D's parallel mode.
- **Tracker maintenance on GPU**: volume/COM via atomics; boundary pixels + neighbor contacts
  recomputed once per MCS for the Python layer.

### GPU-resident steppables (running your per-MCS Python on the GPU)

The user wants the steppable logic itself — including the `ifCohesotaxis=1` lamellipodia path — to run
on the GPU solver, not be synced to the CPU. This is feasible: most per-MCS *numeric* steppable logic
JIT-compiles to device kernels. The design has three layers:

- **GPU-authored steppable kernels.** Hot per-cell/per-pixel logic is written as Python `@wp.kernel`
  functions that compile to CUDA and operate directly on the GPU-resident arrays. `cell.dict`-style
  per-cell scratch state becomes registered **Struct-of-Arrays** (e.g. `link_id[cell]`,
  `link_time[cell]`, `substrate_link_counter[cell]`), not Python dicts. Stochastic rules (Poisson
  link turnover, weighted choice) use on-device Philox keyed by `(cell_id, mcs)` → reproducible.
- **Dynamic FPP links on device.** Links live in a per-cell **CSR** (`link_ptr`, `link_other`,
  `link_lambda/target/max`, `link_kind`). Create = **atomic tail-pointer append** into a pre-sized
  buffer; delete = **flag + segmented stream-compaction** once per MCS; degree cap (`MaxNeighborNum`)
  = per-cell prefix-sum + threshold over a deterministically ordered candidate list. The FPP energy
  kernel reads this CSR directly, so links never round-trip to the host.
- **The `ifCohesotaxis=1` pipeline** (`create_lamellipodia_link`) becomes ~12 back-to-back kernels on
  one stream with no host sync (full design in the research dossier):
  (1) 18-neighbor **stencil classification** of boundary pixels (next-to-medium/substrate/follower),
  offsets in `__constant__` memory; (2) **segmented compaction** into per-cell Free/Adhesion CSR
  lists; (3) per-cell **all-pairs `PixelDist` reduction** (warp-per-cell, shared-memory adhesion
  tiles); (4) **Gumbel-max weighted selection** that reproduces `SigWeights`+`rng.choice` *without an
  explicit sort*; (5) fixed **Manhattan-shell `nth_order_neighbors`** gather + warp-argmax by `zCOM`.
  Steps 3–5 fuse into one per-cell shared-memory block kernel — the biggest win for the cohesotaxis
  path. Keep the current shell form of `nth_order_neighbors`, not the recursive `_old` BFS variant.

**Host-vs-device boundary (kept explicit):** stays on host — file/TIFF I/O (`SaveSegmentation`,
async-copied every 500 MCS), plotting, arbitrary Python libs, buffer capacity re-allocation, and any
*aggregate scalar* read into a Python `if` (e.g. closure-area stop test) — note such a readback breaks
CUDA-graph replay for that step, so gate those checks to coarse intervals. Everything per-cell/per-link
and per-pixel runs on device. Users still write `start/step(mcs)/finish` Python; the hot loops are
kernels instead of per-object SWIG calls.

## GPU utilization & scaling techniques (how to "subdivide the lattice" and eke out performance)

Two **nested decompositions** give "many GPU cores per patch":
- **Color decomposition (correctness):** interleave voxels into colors so all voxels of one color are
  mutually non-interacting and flip simultaneously. Colors needed scale with interaction radius:
  6-neighbor → **2 colors**; 26-neighbor (Moore, `NeighborOrder≤3`) → **8 colors**; **the model's
  `NeighborOrder=4` reaches distance-2 axial neighbors → needs a 3×3×3 = 27-color scheme.**
  *Design decision*: 27 colors cuts per-sweep parallelism to ~1/27 (worse for the small lattice). The
  research recommends a default of **restricting the GPU CPM path to `NeighborOrder≤3` (8 colors)** —
  most CPM biology (incl. adhesion-driven sorting/closure) is insensitive to the order-4 shell —
  while **validating equivalence against the existing `NeighborOrder=4` CPU run**, and keeping a
  27-color mode available if the order-4 physics proves scientifically necessary.
- **Spatial tiling (locality/occupancy):** within a color-sweep, cut the lattice into 3D tiles
  (e.g. 8³); **one thread-block per tile** loads the tile + a halo of width = interaction radius into
  **shared memory**, and all threads in the block cooperatively process the tile's voxels (single-ns
  shared reads vs ~400 ns global). This is literally "multiple GPU cores per patch."

Utilization levers, **ranked by payoff for the small 100³/~500-cell case**:
1. **Ensemble/batched simulations (dominant, ~10–50×)** — pack many independent runs into one launch
   (extra leading dim). The *only* way to fill all SMs at 100³; perfectly serves the sweep goal.
2. **Launch-overhead amortization** — at 100³ each color-sweep is microseconds, so ~5–10 µs/launch
   overhead dominates; fuse an MCS into few launches and use **CUDA Graphs** (with dynamic link work
   behind `wp.capture_if`).
3. **Boundary-restricted sampling + stream compaction** — only cell-surface voxels matter; compact
   active sites into a dense array so threads aren't wasted on the ~10⁶ bulk interior (mirrors CC3D's
   `metropolisBoundaryWalker`).
4. **Structural**: SoA + coalesced ID-lattice layout, warp-divergence reduction (bin work by cell
   type/energy path), mixed-precision energy accumulation.
5. **Multi-GPU domain decomposition with halo exchange** (NVLink/NCCL) — for the *bigger-lattice*
   goal only.

Correctness note (Sego et al. 2023): large active subsections switched frequently reproduce serial
waiting-time statistics; small/infrequently-switched sections distort kinetics — so for small lattices
prefer **fine per-voxel coloring**, and validate dynamics, not just statics.

## Phased roadmap (de-risk the binding constraint first)

**Phase 0 — Profile + tune the CPU baseline (hard go/no-go) — ✅ DONE** (see
`gpu_port/phase0/PHASE0_FINDINGS.md`, harness `gpu_port/phase0/profile_embryo.py`). Result: **GO**.
At 1 core the C++ CPM sweep (incl. FPP) is ~76% of wall-clock and the Python FPP-link steppables ~22%;
the steppables **do not parallelize**, so 32 OpenMP threads give only **1.77×** overall (5.95→10.5
MCS/s) and the CPU is **Amdahl-capped at ~16 MCS/s**. The GPU port must therefore move **both** the
sweep and the steppables on-device — accelerating only the sweep hits the serial-steppable wall (at 32
threads the steppables already dominate the engine). `metropolisBoundaryWalker` is not a usable lever
here (empty `boundaryPixelSet` → 0 attempts → invalid run). Env had to be repaired first (libffi
`ffi.dll` vs `ffi-8.dll`; `gpu_port/fix_env_ffi.py`). GPU verified available: RTX 5090, torch
2.12.1+cu130.

**Phase 1 — GPU-FPP feasibility spike + toolchain validation (the binding constraint) (3–4 wk).**
Throwaway prototype: a GPU checkerboard CPM with Volume + Contact + **FPP spring energy + a dynamic
link list** on a small synthetic model, in Warp, in this pixi/win-64/cu130 env. **Gate**: (i) the
toolchain installs and interops with `torch`; (ii) GPU FPP runs and reproduces a CPU CC3D FPP demo's
link-length and cell-sorting **distributions** within Monte-Carlo noise. **If this fails, stop and
pivot** to the fallback below — do not build the full engine.

**Phase 2 — GPU-native core: Volume + Contact + trackers + GPU steppable model (8–12 wk).** Build the
GPU-resident engine (8-color checkerboard + tiled shared-memory halos), the **GPU steppable API**
(Python `@wp.kernel` logic over SoA `cell.dict`-equivalents), and the boundary-pixel/neighbor CSR
machinery. Port the Embryo's `start()` geometry (`create_hollow_sphere`,
`mesendoderm_sphere_circumference`) and the non-FPP steppable logic as on-device kernels. Validate
Volume+Contact statistics vs CPU CC3D.

**Phase 3 — GPU FPP + on-device cohesotaxis + full Embryo port (10–14 wk).** Fold the spike's FPP
(device-authoritative link CSR, atomic-append/compaction) into the engine; implement the
`ifCohesotaxis=1` pipeline (stencil classify → compaction → `PixelDist` → Gumbel-max select →
Manhattan-shell argmax) and the Poisson turnover as on-device kernels; port `EmbryoSteppables.py` with
minimal edits. Validate the **full model** (closure free-area-vs-time, intercalation) vs CPU CC3D
within thermal noise over ensembles.

**Phase 4 — Batching (sweeps), bigger lattices, Player/interactive bridge (ongoing).** Add the batch
dimension for parameter sweeps (reuse the `ifPythonCall`/`RunNumber` injection hooks; this is the
dominant utilization win), wrap the MCS in a CUDA Graph, scale to larger lattices (and multi-GPU halo
exchange if needed), and bridge to cc3d-player5 / a lightweight `torch`-fed viewer for interactivity.

## Honest performance expectations

- The 600×–3500× figures in the GPU-CPM literature are for **large saturated lattices**. At 100³ with
  ~500 cells a single run may see only **single-to-low-double-digit×**: the lattice can't fill the
  SMs, and host-side aggregate readbacks (e.g. the closure-area stop test) break CUDA-graph replay
  unless gated to coarse intervals.
- The GPU's real wins for this user are **(a) sweeps** — ensemble batching keeps the GPU saturated
  even for small lattices and is the dominant 10–50× lever — and **(b) bigger lattices** (occupancy
  improves). A single 100³ run is the *weakest* case for GPU; calibrate expectations accordingly.
- Phase 0 measured the CPU bar to beat: ~10.5 MCS/s (32-thread), ~16 MCS/s theoretical CPU ceiling.

## Biggest risk + fallback

- **Risk**: GPU FocalPointPlasticity (dynamic per-cell link graph + allocation inside the sweep +
  parallel correctness, with **no prior art**) proves infeasible or statistically unfaithful — which
  blocks the full "entire sim on GPU" goal. Phase 1 exists specifically to surface this in month 1.
- **Fallback** (always-available, correctness-safe): keep the **Metropolis sweep + FPP on CPU
  (multicore OpenMP, already present)** and bank GPU/throughput wins from **(1)** a GPU-resident field
  + batched/array-style steppables (Strategy C) and **(2) experiment-level scale-out** — run many
  independent CC3D instances across cores/cluster for sweeps (zero correctness risk, dominates on
  effort for parameter studies). This preserves every feature and the Python idiom and is strictly
  better than the status quo, without any novel parallel-CPM research.

## Critical files

- Semantics to mirror (read-only reference): `CompuCell3D/.../Potts3D/Potts3D.cpp` (metropolis,
  `changeEnergy`, boundary walker), `.../Potts3D/EnergyFunctionCalculator.cpp`, `.../plugins/Volume`,
  `.../plugins/Contact/ContactPlugin.cpp`, `.../plugins/FocalPointPlasticity/*`,
  `.../plugins/{CenterOfMass,BoundaryPixelTracker,NeighborTracker}`.
- Build/interop reference (not reused for compute): `core/.../PDESolvers/OpenCL/*`, `core/ViennaCL/`.
- Python API to emulate: `cc3d/core/PySteppables.py`; SWIG bridge under `core/pyinterface/`.
- Model to port/validate: `Embryo_Model_dev/Embryo/Simulation/{Embryo.xml,Embryo.py,EmbryoSteppables.py}`.
- Env to extend: `pixi.toml` / `pixi.lock` (add the GPU framework once Phase 1 confirms wheels).

## Verification

- **All gates are statistical, not bit-identical** (CPU↔GPU RNG streams differ): compare
  distributions of energy, cell volume/surface/COM, link length, and for the full model the closure
  free-area-vs-time curve, over ensembles, within Monte-Carlo tolerances.
- **Performance**: benchmark MCS/s vs the *tuned multicore CPU baseline* (not serial) at 100³ and
  larger; measure sweep throughput (sims/hour) vs experiment-level scale-out.
- **Regression**: keep small CC3D golden tests (Volume+Contact, and an FPP demo) for each phase.

---

## Phase 0 results summary (measured 2026-06-20)

Machine: 32 logical CPUs, NVIDIA GeForce RTX 5090 (Blackwell), Windows 11. Model: 100³, ~500 cells.
Full detail and reproduction in `gpu_port/phase0/PHASE0_FINDINGS.md`.

| config | C++ engine (CPM+FPP) | Python steppables (FPP links) | MCS/s (loop) |
|---|---|---|---|
| procs=1 (baseline) | 127.7 ms/MCS (76%) | 36.6 ms/MCS (22%) | 5.95 |
| procs=8 (OpenMP)   | 55.8 ms/MCS        | 52.8 ms/MCS       | 8.90 |
| procs=16           | 33.5 ms/MCS        | 60.5 ms/MCS       | 10.12 |
| procs=32           | 30.5 ms/MCS (29%)  | 59.6 ms/MCS (57%) | 10.52 |

Engine OpenMP scaling per MCS: 1→8 = 2.29×, 1→16 = 3.82×, 1→32 = 4.19× (sublinear, memory-bound).
Overall only 1.77× from 32 cores because the serial Python steppables are the Amdahl ceiling.
