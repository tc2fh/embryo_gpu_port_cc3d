# Phase 2 findings — GPU-native core engine (Volume + Contact + trackers + steppable API)

Status: **DONE — gate green.** `pixi run python -m pytest -q gpu_port` → `21 passed in ~38s`
(7 Phase 1 + 14 Phase 2), exit 0, on RTX 5090 / Warp 1.14.0 / torch 2.12.1+cu130.

This phase built the production GPU-resident engine under `gpu_port/engine/` (fresh, not a
copy of the Phase 1 throwaway), with green tests + run artifacts under `gpu_port/phase2/`.
FocalPointPlasticity and the cohesotaxis pipeline are **explicitly deferred to Phase 3**.

---

## 1. Engine architecture

GPU-resident state (single source of truth on device, `gpu_port/engine/`):

- **`ids`** — flat `int32` id-lattice `[Lz*Ly*Lx]` (0 = Medium). The authoritative field.
- **Per-cell SoA** (`engine.py`): `cell_type` (int32), `volume` (float32),
  `xsum/ysum/zsum` (**int64** integer coordinate sums), plus per-cell `target_volume` /
  `lambda_volume` derived from the per-type config (CC3D `VolumeEnergyParameters` are
  by-cell-type).
- **`contact`** — dense `(n_types, n_types)` float32 symmetric adhesion matrix.
- **`type_frozen`** — `(n_types,)` int32 mask (Substrate etc.; Medium is never frozen).
- **Constant neighbor tables** — flat `int32` device arrays `off[3*n + axis]` for the
  contact shell, the flip-target shell, and the tracker shell (no `wp.mat` const type on
  this Warp build → flat arrays, per Phase 1 carry-forward).

Module layout:

| file | role |
|---|---|
| `config.py` | `EngineConfig`; distance-sorted neighbor-order shells (matches CC3D `BoundaryStrategy`) |
| `state.py` | host id-lattice → SoA builder; `build_grid_state` for tests |
| `kernels.py` | all `@wp.kernel`/`@wp.func` (real `.py` file — Warp reads source via `inspect`) |
| `engine.py` | `GPUEngine`: 8-color sweep, trackers, observables, partition asserts |
| `cpu_reference.py` | NumPy Volume+Contact reference (statistical ground truth) |
| `geometry.py` | Embryo `create_hollow_sphere` + `mesendoderm_sphere_circumference` + `build_embryo_start` |
| `steppables.py` | GPU steppable API skeleton + worked non-FPP example |

**Energy (mirrors vendored CC3D, verified by source reading):**
- Volume = `λ·(1 + 2(V_new−Vt)) + λ·(1 − 2(V_old−Vt))` incremental form
  (`VolumePlugin::changeEnergyByCellType`).
- Contact = `Σ_shell [ J(new,nbr) − J(old,nbr) ]`, term skipped when the neighbor *is* the
  cell on that side (`ContactPlugin::changeEnergy`; the Embryo gives every cell its own
  clusterId so the clusterId branch reduces to plain id-inequality).
- Metropolis = accept if ΔE≤0 else prob `exp(−ΔE/T)` (`DefaultAcceptanceFunction`, k=1).
- Flip mechanic mirrors `Potts3D::metropolisFast`: pick a source pixel's cell, propose it
  into an adjacent target pixel of a different cell; frozen cells never source or accept
  (Medium, a null cell in CC3D, always participates).

**Trackers on GPU:**
- volume + COM via atomics on accept; COM uses **int64 fixed-point** sums of integer pixel
  coordinates → bit-exact and reproducible (resolves the carry-forward note that float
  `atomic_add` COM is non-associative). `assert_volume_partition()` checks per-cell volume
  SoA == lattice voxel count **exactly**, and that the live int64 COM sums equal a fresh
  full-lattice recompute (atomics never drift).
- boundary-pixel flags + per-cell boundary counts, and the neighbor-contact
  (common-surface-area) **CSR**, recomputed once per MCS (`recompute_trackers()`), at
  NeighborOrder=1 (the Embryo `BoundaryPixelTracker`/`NeighborTracker` order).

---

## 2. Color-scheme decision (+ validation)

**Decision: default to the 8-color (2×2×2) checkerboard, restricting the GPU sweep's
flip-target connectivity to NeighborOrder ≤ 3 (Moore-26).** Enforced in `EngineConfig`
(`flip_neighbor_order > 3` raises). The contact-energy neighbor order is configured
independently (`contact_neighbor_order`, default 3) and may be ≤ the coloring order.

Why: with the 2×2×2 parity coloring, two distinct same-color voxels differ by an even
offset on every axis, so they are never within each other's Moore neighborhood and never
face-adjacent. Therefore, within one color launch, neighbor reads are stable and two
accepted flips never write the same voxel — the GPU analogue of CC3D's OpenMP subgrid
checkerboard (`Potts3D.cpp`). This is correct iff the flip-target reach is ≤ order 3.
The Embryo XML uses **NeighborOrder=4** (reaches distance-2 axial voxels), which would need
a 3×3×3 = 27-color scheme; per the plan we default to order≤3 / 8 colors (most CPM
adhesion-driven biology is insensitive to the order-4 shell) and keep 27-color reachable.

Validation:
- **Exact partition invariant** holds after every run (`test_volume_partition_exact`,
  `assert_volume_partition()`), which is the direct proof that the 8-color scheme has no
  write races and the atomics are correct.
- **Statistical equivalence** vs an order≤3 CPU reference (below) is within the
  seed-to-seed noise floor.

> The CPU comparator here is an **order≤3 NumPy reference**, not a full order-4 CPU CC3D
> run (a 100³/4000-MCS CC3D run is far outside a <3 min gate). The order-3-vs-order-4
> equivalence is taken from the plan's research recommendation; a one-off order-4 CC3D
> ensemble comparison is left as a Phase 3 validation item.

---

## 3. Statistical results (metric / tolerance / observed)

Model: 24³ lattice, 27 cells on a grid, 40 MCS, stable **adhesive** regime
(J_medium_cell=5, J_cell_cell=1, target_volume=64, λ_volume=4) so cells reach a finite
quasi-stationary equilibrium. Pooled ensembles: CPU seeds {1,2,3}, GPU seeds {1..6}.
Tolerances calibrated in `_runs/_calibrate.py` at a few× the measured CPU-vs-CPU noise
floor (vol ~0.15%, surf ~1.2%, E ~0.2%).

| observable | metric | tolerance | observed (GPU vs CPU) |
|---|---|---|---|
| cell volume   | pooled-mean rel diff | < 0.05 | **0.0013** |
| cell volume   | KS statistic D        | < 0.20 | **0.099** (p=0.66) |
| surface area  | pooled-mean rel diff | < 0.08 | **0.0085** |
| total energy  | pooled-mean rel diff | < 0.03 | **0.0025** |
| COM \|r−center\| | pooled-mean rel diff | < 0.05 | **0.0068** |

All four observables (energy, volume, surface, COM) agree at or near the seed-to-seed
noise floor; the volume distributions are statistically indistinguishable (KS p≈0.66).

**Geometry** is validated far more strongly than statistically: the ported `start()` builds
an id-lattice that is **identical voxel-for-voxel and id-for-id** to running the actual
`EmbryoSteppables.py` geometry functions (`test_embryo_geometry_matches_reference_voxel_exact`;
also reproduced standalone in `_runs/_geom_vs_reference.py`). Full-scale counts: 60 Leading,
618 Passive, 62333 single-voxel Substrate shell cells (n_cells = 63011).

---

## 4. Tests (the gate)

`gpu_port/phase2/tests/`, discoverable by `pixi run python -m pytest -q gpu_port`:

- `test_engine_core.py` — Volume+Contact statistical equivalence (energy/volume/surface/COM);
  exact volume-partition + COM-drift invariant; CPU-only build/relaxation sanity.
- `test_trackers.py` — boundary-pixel field + per-cell counts and the neighbor-contact CSR
  match an independent CPU recompute **exactly**; common-surface symmetry; surface-area
  kernel vs CPU recompute.
- `test_geometry.py` — hollow-sphere shell shape, ring block count/size, full-`start()`
  per-type counts, voxel-exact match to the reference functions, state partition exactness.
- `test_steppable_api.py` — `CellDict` (cell.dict-equivalent SoA) round-trip; the non-FPP
  `FloorFreeAreaSteppable` end-to-end vs a NumPy recompute; the manager advancing the engine.

GPU tests `pytest.skip(...)` cleanly without CUDA (each guarded by `_cuda_available()`);
CUDA is present here so they run + pass. Phase 1's 7 tests stay green. Total ~38 s.

---

## 5. DONE vs DEFERRED

**DONE (this phase):**
- GPU-resident state (int32 id-lattice + per-cell SoA + contact matrix).
- 8-color checkerboard Metropolis, NeighborOrder≤3, Philox RNG keyed by (mcs, color, seed);
  energy = Volume + Contact; frozen-type support (Substrate).
- On-GPU volume + COM trackers via atomics; exact int64 fixed-point COM; exact
  partition/drift asserts.
- Boundary-pixel + neighbor-contact CSR, recomputed once per MCS (NeighborOrder=1).
- Total-energy + per-cell surface-area observables on GPU.
- GPU steppable API skeleton (SoA `cell.dict`-equivalent registry, base class, manager) +
  one worked **non-FPP** example end-to-end (floor-free-area, the Embryo
  `SubstrateSteppable` metric).
- Embryo non-FPP `start()` geometry (`create_hollow_sphere`,
  `mesendoderm_sphere_circumference`) as host-init kernels feeding the GPU engine,
  **voxel-exact** vs the reference.
- Statistical validation of Volume+Contact vs a CPU reference (energy, volume, surface, COM).

**DEFERRED — later Phase 2 pass or Phase 3 (honest):**
- **FocalPointPlasticity** (device link CSR, atomic-append/compaction, spring energy) and
  **cohesotaxis** (stencil-classify → compaction → PixelDist → Gumbel-max → Manhattan-shell
  argmax). Explicitly Phase 3. The Phase 1 throwaway proved the FPP CSR pattern; this engine
  leaves the seam (per-cell SoA + per-MCS CSR rebuild + `CellDict`) for it.
- **Order-4 (27-color) sweep mode** — not implemented; the order≤3 / 8-color default is the
  sanctioned path. An order-4 CPU-CC3D ensemble comparison to confirm the physics is a
  Phase 3 validation item.
- **CUDA-graph MCS capture** and the **batch dimension** (sweeps) — Phase 4.
- **Boundary-restricted sampling / stream compaction** — the sweep currently launches over
  the full color sublattice (fine at test scale; a utilization lever for Phase 4).
- **Neighbor-CSR at full Embryo scale** — the current CSR builds a dense (n_cells+1)²
  contact matrix on device then compresses on host. That is fine for the test models but is
  O(n_cells²) memory; at 63k cells it must become a hashed/segmented build (noted in
  `kernels.py`). The boundary-pixel and volume/COM trackers already scale fine.
- **Steppable kernels authored in pure Python by the user** — the API shape exists; the
  ergonomic "write `step(mcs)` as a `@wp.kernel`" sugar is minimal.

---

## 6. New constraints / surprises discovered

1. **Medium must never be frozen.** A first cut treated type 0 as frozen (mirroring a naive
   reading), which froze *all* dynamics (a cell that must grow stayed put). CC3D gates
   `checkIfFrozen` on a **non-null** cell, so Medium (a null `CellG*`) always participates.
   Fixed in the kernel, the CPU reference, and `EngineConfig.frozen_mask` (index 0 forced 0).
   This is a real semantic constraint for any frozen-type model (the Embryo Substrate).
2. **Equilibrium vs runaway for the stat gate.** With J_medium_cell ≥ J_cell_cell and weak
   volume constraint, isolated cells dissolve to zero — the volume/surface distributions are
   then a non-stationary transient and GPU/CPU diverge in *kinetics* (energy still agrees).
   The gate uses a **stable adhesive** regime (cell-cell adhesion + strong λ) so the compared
   distributions are quasi-stationary. (Documented in the test.)
3. **Discrete surface area + KS.** Surface area is integer-valued; KS has step artifacts so
   seed-to-seed KS D for surface (~0.26) exceeds GPU-vs-CPU (~0.09). The gate uses
   relative-mean for surface and reserves KS for the (finer) volume distribution.
4. **Warp `-c` string kernels fail** (confirmed again): `inspect` can't read source from an
   `exec`'d string. Kernels live in `kernels.py`/`steppables.py`; probes live in real files
   under `_runs/`. (Phase 1 carry-forward, re-verified.)
5. **`pixi.toml`/`pixi.lock` left untouched** — Warp stays pip-installed and out of the lock
   exactly as the carry-forward requires (avoids a re-solve). scipy 1.18.0 is available
   transitively; nothing else needed locking.

---

## 7. What Phase 3 must carry forward

- **The frozen-Medium rule** and the flip mechanic direction (target flips to source's cell)
  are now the engine contract — FPP ΔE must be added at the same `changePixel`/`newCell`
  evaluation point.
- **COM is the single source of truth** for any FPP link length (int64 sums / volume),
  already exact and reproducible — FPP can read it directly.
- **The CSR rebuild seam** (`recompute_trackers`, per-MCS) and the **`CellDict` SoA registry**
  are where dynamic FPP links + `cell.dict` link inventory plug in (atomic-append create,
  flag+compaction delete — the Phase 1 prototype pattern).
- **Neighbor-CSR must move off the dense (n_cells+1)² matrix** before full-Embryo scale.
- **8-color validity ceiling is order 3.** If the order-4 physics proves necessary, add the
  27-color mode (the offset tables and per-color decode already generalize; only the color
  count and parity decode change).
- An **order-4 CPU-CC3D ensemble** comparison (closure/intercalation observables) is the
  outstanding physics-fidelity check the statistical gate could not run in-budget.
