# Phase 1 — GPU-FPP feasibility spike + toolchain validation

> Status: NEXT. Gate type: statistical + toolchain. Source plan section:
> `plan_docs/CC3D-GPU-port-plan.md` -> "Phase 1" and "Engine substrate".

## Objective

De-risk the single binding unknown before committing to the full engine: prove that a GPU-resident
checkerboard Cellular Potts step with **Volume + Contact + FocalPointPlasticity (dynamic spring-link
list)** can be built in this environment (win-64 / Python 3.12 / CUDA-13 / torch 2.12.1+cu130) and
that its **statistical behavior matches a CPU reference within Monte-Carlo noise**. This is a
**throwaway prototype**, not the production engine — favor clarity over architecture.

## Scope

This phase may only create or modify files under:
- `gpu_port/phase1/`
- `pixi.toml`
- `pixi.lock`

Do NOT touch the vendored CC3D trees, the Embryo model, or any other path. Put ALL prototype code,
the CPU reference, and tests under `gpu_port/phase1/`.

## Tasks

1. **Toolchain install + interop check.** Install the GPU framework (NVIDIA Warp is the lead
   candidate — `pixi run pip install warp-lang` into the pixi env; do NOT run `pixi add` / trigger a
   re-solve, per the env-fragility lesson in PROGRESS.md). Confirm: import works, a trivial
   `@wp.kernel` runs on the RTX 5090, and zero-copy interop with `torch`
   (`wp.from_torch`/`wp.to_torch`) round-trips a CUDA tensor.
2. **Minimal GPU CPM.** A small synthetic 3D lattice (e.g. 32^3, a few dozen cells) with an int32
   id-lattice source of truth, per-cell SoA (type, volume, target_volume, lambda, COM), an 8-color
   checkerboard Metropolis sweep (NeighborOrder<=3), per-thread Philox RNG keyed by (id, mcs),
   energy = Volume + Contact, atomic volume/COM updates on accept.
3. **FPP spring energy + dynamic link list.** Per-link `offset + lambda*(L - L_target)^2` over a
   per-cell CSR link list with atomic-append create / flag+compaction delete, rebuilt/pushed each MCS.
   Read COM as the single source of truth for link length.
4. **CPU reference + statistical gate.** A small CPU implementation (NumPy, or a tiny CC3D FPP demo)
   of the same synthetic model; compare **distributions** (link length, cell volume, sorting/COM)
   GPU vs CPU over an ensemble, within a stated Monte-Carlo tolerance. Bit-identical is NOT expected.

## Exit gate (what `pytest gpu_port` must assert)

Tests under `gpu_port/phase1/tests/` must:
- `test_toolchain`: framework imports, a kernel runs on GPU, torch zero-copy interop round-trips.
- `test_fpp_statistical_equivalence`: GPU vs CPU-reference distributions agree within a stated
  tolerance (document the metric + tolerance in the test).
- Skip cleanly (pytest skip, not fail) with a clear message if no CUDA GPU is present, so the gate is
  meaningful only where a GPU exists.

## Decision this phase produces

GO (Warp confirmed; proceed to Phase 2 full engine) or PIVOT (toolchain or GPU-FPP statistically
unfaithful -> fall back to the CPU-FPP + GPU-resident-field + experiment scale-out path in the plan's
"Biggest risk + fallback" section). Record the decision + evidence in the handoff delta.
