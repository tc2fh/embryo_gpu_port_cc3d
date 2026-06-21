# Phase 6 — Device-resident scans, neighbor-CSR residency, device-authoritative link topology

> Status: defined 2026-06-21. Source: `plan_docs/CC3D-GPU-host-to-device-plan.md` (candidates #1–5).
> Foundation phase: fidelity-neutral, low risk, unblocks Phases 7–8. Profiling basis:
> `gpu_port/engine/bench_csr.py::_run_instrumented` — per-MCS ≈ sweep 0.9 / csr 2.3 / steppables 4.9 ms.

## Objective

Move the per-MCS host arithmetic and link storage onto the GPU, **without changing any host-facing
API or any observable**, so all prior tests stay byte-green and Phases 7–8 have device primitives to
consume. Three deliverables:

1. **Device exclusive-scan** replaces the host `np.cumsum` + `wp.array` reallocation in
   `fpp.py:263-265` (link_ptr, every `step_mcs`), `engine.py:329-331` (neighbor-CSR indptr), and
   `batched_fpp.py:190-193` (per-replica link_ptr, segmented with a per-replica reset).
2. **Neighbor-contact CSR stays resident on the GPU.** `neighbor_contact_csr` keeps returning its
   current host arrays (so `embryo/` consumers are untouched this phase), but additionally exposes
   **device handles** (`indptr`, `indices` as `wp.array`) — the seam Phase 7 reads to drop the host
   copyback.
3. **The FPP link inventory becomes device-authoritative.** `FPPLinks` holds `_a/_b/_lam/_tgt/_max`
   as `wp.array`; create/delete become device **append (atomic) / tombstone / compact** kernels;
   `create_links_bulk`/`delete_links_bulk` keep their signatures (host pair-list in → device append),
   `rebuild()` is fully on-device. Add a device **keep/compact** primitive driven by a device decision
   mask (the Poisson seam Phase 7 wires in).

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/phase6/`

## Constraints and fidelity invariants

- **No observable changes.** Volume+Contact runs stay **bit-reproducible** (int64 COM untouched);
  FPP runs stay statistically faithful. Device scans must produce **byte-identical** integer prefixes
  to the `np.cumsum` they replace.
- **Link set is order-independent.** create/delete are set operations on `{a,b}` pairs; the device
  inventory must be set-equal to the host reference after any sequence of edits. Inventory iteration
  order used by `rebuild()` must stay stable so downstream CSR offsets are deterministic.
- **Keep the host APIs working.** `neighbor_contact_csr` host return, `create_links_bulk`,
  `delete_links_bulk`, `active_link_lengths`, and `rebuild()` keep their current signatures and
  return values; this phase changes their *implementation/storage*, not their contract. All 100+
  existing tests must pass unchanged.
- Warp build limits (from prior phases): kernels live in real `.py` files; no `wp.mat` const type;
  device scan via a hand-written kernel or `wp.utils` scan, not a host roundtrip.

## Delivery discipline (tested passes, each green)

1. **Device scans.** Add the exclusive-scan kernel(s); use them in the three cumsum sites. Test:
   device scan == `np.cumsum` exactly on random degree vectors (single + segmented/batched); engine
   run bit-exact vs pre-phase for Volume+Contact; FPP `rebuild()` produces identical `link_ptr`.
2. **CSR device handles.** Expose `indptr`/`indices` device arrays from `neighbor_contact_csr`
   (additive). Test: device handles `array_equal` the existing host return, single + batched.
3. **Device link topology.** Move `FPPLinks` storage to `wp.array`; implement append/tombstone/
   compact + the decision-mask keep/compact primitive; back `create_links_bulk`/`delete_links_bulk`/
   `rebuild()` with them. Test: device inventory set-equal to a NumPy reference after randomized
   create/delete sequences; `active_link_lengths` unchanged; a short full-Embryo FPP run matches the
   prior validated link inventory/length statistics.

## Exit gate

Tests under `gpu_port/phase6/tests/`: (a) device scan == `np.cumsum` exactly, single + batched;
(b) CSR device handles == host return exactly; (c) device link CRUD == NumPy-reference inventory
exactly over randomized edit sequences, with stable rebuild order; (d) Volume+Contact engine runs
**bit-identical** to pre-phase, FPP runs statistically unchanged (link-length KS within the Phase-3
gate); (e) **all prior tests green and byte-unchanged** (host APIs preserved). Report the measured
`sweep_ms`/`csr_ms` change (expect `sweep_ms` down from killed reallocs; `csr_ms` largely unchanged
until Phase 7 drops the copyback).
