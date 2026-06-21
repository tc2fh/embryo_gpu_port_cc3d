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

You EXTEND `gpu_port/engine/` in place (it is in scope). Do NOT modify `gpu_port/phase1/` or
`gpu_port/phase2/` (out of scope; read-only reference).

## Carry-forward from Phase 2 (verdict handoff)

- **FPP deltaE enters at the existing seam:** add the spring term at the same changePixel/newCell
  evaluation point in `engine/kernels.py` (`metropolis_color_kernel`). Do NOT restructure the kernel
  first. The frozen-Medium rule (id 0 always participates) is engine contract — preserve it.
- **COM is the exact, reproducible single source of truth for link length:** read the engine int64
  `xsum/ysum/zsum / volume` directly; no separate COM tracker, no float drift.
- **FPP-link plug-in points:** the `CellDict` SoA registry (`engine/steppables.py`) + the per-MCS
  `recompute_trackers()` seam (`engine/engine.py`). Reuse the Phase 1 device link-CSR pattern
  (atomic-append create, flag+compaction delete) proven in `gpu_port/phase1/cpm_gpu.py`. Links are
  static within a color-sweep; create/delete happens at the per-MCS steppable boundary, not inside
  the Metropolis inner loop.
- **Scale blocker (fix early):** the neighbor-contact CSR is a dense (n_cells+1)^2 device matrix
  (O(n^2) memory) — it will not fit at 63k-cell Embryo scale. Move it to a hashed/segmented CSR build
  before any full-scale Embryo run.
- **Outstanding fidelity check:** the order-4 CPU-CC3D ensemble comparison (closure free-area,
  intercalation) was out of Phase 2 budget — Phase 3 must run it for the full Embryo validation.

## Delivery discipline (largest phase — sequential, tested passes)

Phase 3 is too large for one pass; the orchestrator drives it as tested increments, each leaving
`pixi run python -m pytest -q gpu_port` green (all prior phases included):
1. **FPP integration:** device link-CSR + spring energy folded in at the seam; the neighbor-CSR scale
   fix; validate FPP link-length/sorting statistics vs the Phase 1 / CPU reference AND keep
   Volume+Contact green.
2. **Cohesotaxis:** the ifCohesotaxis=1 pipeline (stencil classify -> segmented compaction ->
   PixelDist reduction -> Gumbel-max weighted select -> Manhattan-shell argmax) + Poisson link
   turnover, as on-device kernels.
3. **Full Embryo port + validation:** port `EmbryoSteppables.py` (under `gpu_port/embryo/`) with
   minimal edits; validate closure free-area-vs-time + intercalation vs CPU CC3D over ensembles.

Each pass: green tests, report DONE vs DEFERRED. Escalate only on a real problem (incorrect /
statistically unfaithful results, infeasibility, or a gate flag) — not merely because later passes
remain.

## Exit gate

Tests under `gpu_port/phase3/tests/`: full Embryo model — closure free-area-vs-time and
intercalation statistics match CPU CC3D within thermal noise over ensembles; link-length
distributions match; cohesotaxis path validated against the CPU `create_lamellipodia_link` behavior.
