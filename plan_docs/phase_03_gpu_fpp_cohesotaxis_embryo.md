# Phase 3 — GPU FPP + on-device cohesotaxis + full Embryo port

> Status: pending Phase 2. Source: `plan_docs/CC3D-GPU-port-plan.md` -> "Phase 3" and "GPU-resident
> steppables" (the ~12-kernel ifCohesotaxis pipeline). Refine from Phase 2's handoff delta.

## Objective

Fold the Phase 1 spike's FPP (device-authoritative link CSR, atomic-append/compaction) into the
engine; implement the `ifCohesotaxis=1` pipeline (stencil classify -> segmented compaction ->
all-pairs `PixelDist` reduction -> Gumbel-max weighted select -> Manhattan-shell argmax) and Poisson
link turnover as on-device kernels; port `EmbryoSteppables.py` with minimal edits.

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/embryo/`
- `gpu_port/phase3/`

## Exit gate

Tests under `gpu_port/phase3/tests/`: full Embryo model — closure free-area-vs-time and
intercalation statistics match CPU CC3D within thermal noise over ensembles; link-length
distributions match; cohesotaxis path validated against the CPU `create_lamellipodia_link` behavior.
