# Phase 1 findings — GPU-FPP feasibility spike + toolchain validation

Status: **GO**. Decision date 2026-06-20. Machine: NVIDIA RTX 5090 (Blackwell,
sm_120), Windows 11, pixi env (win-64 / Python 3.12 / CUDA driver 13.1 /
torch 2.12.1+cu130).

This was a throwaway prototype whose only job was to de-risk the one binding
unknown — **can a GPU-resident checkerboard CPM with Volume + Contact +
FocalPointPlasticity (dynamic spring-link list) be built in this environment, and
does it reproduce a CPU reference's statistics within Monte-Carlo noise?** The
answer is yes on both counts.

---

## 1. What was built (all under `gpu_port/phase1/`)

| file | role |
|---|---|
| `model.py` | shared synthetic model: 32³ int32 id-lattice (0=Medium), 27 cells on a 3×3×3 grid, per-cell SoA (type, volume, target_volume, lambda, COM accumulators), a 54-edge spring-link topology (cell grid graph), and the CC3D-faithful parameters. Single source of truth for both engines. |
| `cpm_cpu.py` | CPU **reference** (NumPy). Plain single-site Metropolis. The ground truth. |
| `cpm_gpu.py` | GPU **prototype** (NVIDIA Warp). 8-color checkerboard Metropolis, Philox RNG, atomic volume/COM updates, dynamic per-cell **CSR** link list (atomic-append build + flag/compaction delete), rebuilt each MCS. |
| `tests/test_toolchain.py` | toolchain + zero-copy interop gate. |
| `tests/test_fpp_statistical_equivalence.py` | GPU-vs-CPU distribution gate (+ lattice/SoA invariants). |
| `conftest.py` | puts `phase1/` on `sys.path` so the gate command imports the modules. |
| `_runs/` | scratch probes (gitignored): smoke, convergence trace, ensemble tolerance study. |

### Energy semantics (mirrored read-only from vendored CC3D)
- **Volume** — incremental form `λ_V·(1 + 2·(V_new − V_t)) + λ_V·(1 − 2·(V_old − V_t))`
  (`VolumePlugin::changeEnergyByCellType`, VolumePlugin.cpp:189-194).
- **Contact** — `Σ_{Moore shell} [ J(new,nCell) − J(old,nCell) ]`, skipping the
  neighbor equal to the cell itself; Medium = id 0 (`ContactPlugin::changeEnergy`,
  ContactPlugin.cpp:125-161). NeighborOrder ≤ 3 ⇒ 26-neighbor shell ⇒ **8 colors**.
- **FPP** — per link `offset + λ·(L − L_target)²`, `L = ‖COM_a − COM_b‖₂`
  (`potentialFunction` FocalPointPlasticityPlugin.cpp:258-260; `distInvariantCM`
  → plain Euclidean for non-periodic BC, NumericalUtils.cpp:238-242). offset
  cancels in ΔL deltas. **COM is the single source of truth** for link length.
- **Metropolis** — Boltzmann: accept if ΔE ≤ 0 else with prob `exp(−ΔE/T)`, T=10
  (`DefaultAcceptanceFunction`, k=1, offset=0).

### GPU design points proven
- int32 id-lattice as source of truth; per-cell SoA in `wp.array`.
- 8-color checkerboard: `color = (x&1) + 2(y&1) + 4(z&1)`; one thread per
  same-color voxel — the GPU analogue of CC3D's OpenMP subgrid checkerboard.
- Per-thread **Philox** RNG via `wp.rand_init(seed, offset)`, keyed by
  `(mcs, color, base_seed)` (seed) and the linear voxel index (offset).
- `wp.atomic_add` volume/COM updates on accept.
- Dynamic FPP **CSR** (`link_ptr`, `link_other`) rebuilt each MCS: a flag+degree
  count kernel (drops links whose COM length exceeds `fpp_max_length` = delete),
  host exclusive-prefix-sum for `link_ptr` (n_cells is tiny, off the hot path),
  then an **atomic-append** fill kernel (create). The FPP energy kernel reads this
  CSR directly — links never round-trip to host during the sweep.

---

## 2. Toolchain result

- **Framework:** NVIDIA **Warp 1.14.0** (`warp_lang-1.14.0-py3-none-win_amd64.whl`).
- **Install method:** `pixi run pip install warp-lang` (NOT `pixi add`, per the
  env-fragility lesson). It pip-installs cleanly into `.pixi/envs/default`. Note:
  pip-installed packages are **not** written to `pixi.lock`, so `pixi.toml`/
  `pixi.lock` are unchanged by this phase. **To reproduce the env, re-run
  `pixi run pip install warp-lang` after `pixi install`.** (Deferred adding it to
  `[pypi-dependencies]` to avoid a conda re-solve that could perturb the fragile
  torch/CC3D/libffi setup; revisit in Phase 2 once the env is confirmed stable.)
- **Import + device:** Warp initializes, reports CUDA Toolkit 12.9 / Driver 13.1,
  and detects `"cuda:0" : "NVIDIA GeForce RTX 5090" (32 GiB, sm_120, mempool
  enabled)`. JIT compiles kernels for sm_120 (Blackwell) without issue.
- **Kernel on GPU:** trivial `@wp.kernel` runs and returns correct results
  (`test_warp_kernel_runs_on_gpu`).
- **torch zero-copy interop:** `wp.from_torch` aliases the same device pointer
  (`wa.ptr == t.data_ptr()`); a Warp kernel mutating that buffer is visible in the
  torch tensor; `wp.to_torch` round-trips back to the same pointer
  (`test_torch_warp_zero_copy_roundtrip`). **Zero-copy interop confirmed.**

### Toolchain gotchas (for Phase 2)
- Warp **cannot compile `@wp.kernel` defined in an `exec()`'d string** — kernels
  must live in real `.py` files (it reads source via `inspect`). Keep all kernels
  in modules.
- This Warp build has **no `wp.mat` / `wp.matrix`** fixed-size integer-matrix
  constant type. Neighbor-offset tables are passed as flat int32 device arrays
  (`off[3*n + axis]`) instead of `wp.constant(wp.mat(...))`.

---

## 3. Statistical comparison (the gate)

Both engines run the identical 32³ / 27-cell / 54-link model for **25 MCS** (past
the ~15-20 MCS relaxation horizon measured in `_runs/_relax_trace.py`), then pool
an ensemble of independent seeds. We compare the **distributions** of FPP link
length and cell volume. Bit-identical is NOT expected (independent RNG streams).

**Metric + tolerance (documented in the test):**
- link-length pooled-mean relative diff `< 0.06`
- volume pooled-mean relative diff `< 0.12`
- link-length distribution overlap: Kolmogorov–Smirnov `D < 0.20`

**Justification — measured Monte-Carlo noise floor** (`_runs/_ensemble.py`,
4 seeds/side, 40 MCS): the **CPU-vs-CPU** seed-to-seed relative-mean spread is
~3-4% (link) / ~3% (volume). The CPU-vs-GPU spread is *at or below* that floor.
The gate sits a few× above the floor — tight enough to catch a real semantics bug
(any order-of-magnitude or systematic shift fails), loose enough not to flake.

**Observed (gate run, CPU seeds {1,2,3} × GPU seeds {1..6}, 25 MCS):**

| observable | CPU mean ± std | GPU mean ± std | rel. mean diff | KS |
|---|---|---|---|---|
| FPP link length | 10.000 ± 0.253 | 10.172 ± 1.533 | **1.7 %** | D=0.061, p=0.78 |
| cell volume     | 30.296 ± 4.454 | 29.809 ± 5.886 | **1.6 %** | — |

The link-length KS p=0.78 means the two distributions are statistically
indistinguishable. Both engines relax cells from the seeded volume (64) to the
same equilibrium band (~30) and hold springs at ~target. (In a longer 4-seed/40-MCS
run the agreement was equally good: link rel 3.8 % with KS p=0.84, volume rel
1.4 % with KS p=1.0 — vs a CPU-vs-CPU link floor of rel 4.1 %.)

**Invariants also asserted:** GPU id-lattice stays a valid partition and the
per-cell volume SoA matches lattice voxel counts exactly after 25 MCS
(`test_gpu_conserves_lattice_partition`) — confirming the atomic accumulators are
correct.

### Performance (incidental, not a Phase 1 gate)
Even at this tiny 32³ lattice with **no batching**, the GPU ran ~**60-67× faster
per MCS** than the CPU reference (≈2.7 ms/MCS vs ≈180 ms/MCS, `_runs/_converge.py`).
The CPU reference is intentionally a clear NumPy single-site loop, not optimized —
so this is not a production speedup number, just evidence the GPU path is viable.

### Known property: GPU runs are NOT bit-reproducible across identical seeds
The Philox stream is deterministic per `(voxel, mcs, color, seed)`, but **float
`atomic_add` accumulation of COM/volume within a color sweep is non-associative
and thread-order-dependent**. When several accepted flips in one color touch the
same cell's COM sum, the resulting COM (hence subsequent ΔE / acceptance) varies
run-to-run. So identical-seed runs give statistically-equivalent but not
bit-identical lattices (different seeds correctly diverge). This is consistent
with the plan's "all gates are statistical, not bit-identical" stance and with the
sanctioned "1-color-step-stale COM" approximation. The per-cell volume SoA still
matches lattice counts exactly, so this is an ordering effect, not an accumulator
bug. **Phase 2 note:** if stronger determinism is ever needed, accumulate COM/
volume in int64 fixed-point (associative) or recompute COM from the lattice once
per MCS instead of via in-sweep atomics.

---

## 4. Decision: **GO**

Evidence:
1. **Toolchain works** — Warp 1.14.0 installs via pip into the pixi env, JIT-
   compiles for the RTX 5090 (sm_120), runs kernels on GPU, and zero-copy
   interops with torch 2.12.1+cu130 (the binding env question — answered yes).
2. **GPU FPP is buildable** — a dynamic per-cell CSR spring-link list with
   atomic-append create / flag+compaction delete, rebuilt each MCS, with FPP
   energy read from COM as the single source of truth, compiles and runs on
   device. This is the feature with *zero prior GPU art*; it works at prototype
   scale.
3. **GPU FPP is statistically faithful** — link-length and volume distributions
   match the CPU reference within (indeed below) the measured Monte-Carlo noise
   floor; the primary FPP observable's distributions are statistically
   indistinguishable (KS p≈0.78).

No condition for PIVOT (toolchain failure or statistical infidelity) was met.
Proceed to **Phase 2** (GPU-native core engine: Volume + Contact + trackers +
GPU steppable API), folding this spike's FPP CSR machinery in at Phase 3.

### Hand-off notes for the next phase
- Reinstall Warp with `pixi run pip install warp-lang` (it is not in `pixi.lock`).
- Keep all `@wp.kernel`s in real modules; pass small constant tables as device
  arrays (no `wp.mat` in this build).
- The 8-color scheme is correct for NeighborOrder ≤ 3; the Embryo model uses
  NeighborOrder = 4 (needs 27 colors or the sanctioned order≤3 restriction —
  decide in Phase 2/3 per the plan's color-decomposition section).
- For reproducibility-sensitive validation, consider int64 fixed-point COM
  accumulation (see "Known property" above).
- Gate command: `pixi run python -m pytest -q gpu_port` (currently 7 passed,
  ~21 s).
