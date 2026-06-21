# Phase 7 — Device link steppables (tissue / substrate / Poisson) + drop the CSR copyback

> Status: defined 2026-06-21. Depends on Phase 6 (device CSR handles + device-authoritative link
> inventory). Source: `plan_docs/CC3D-GPU-host-to-device-plan.md` (candidates #4, #6–10). This is
> where most of `steppable_ms` (~4.9 ms) and `csr_ms` (~2.3 ms) is recovered. Medium risk.

## Objective

Port the per-cell **host** link-management loops to device kernels that consume Phase 6's device CSR
handles and device link inventory, and **stop copying the neighbor-CSR to the host every MCS**.
Concretely:

1. **Tissue links** (`embryo/steppables.py` `TissueLinkSteppable`, runs twice — leading + passive):
   replace `_grouped_csr_neighbors`/`neighbor_adjacency` (host adjacency dict), `_current_link_map`
   (Python `zip` set-building), and the nested create-under-cap loop with a **device per-cell kernel**
   that scans each cell's CSR row, filters by type, and claims up to the degree cap **in CSR-row
   order**, appending to the device inventory.
2. **Substrate links** (`PassiveSubstrateSteppable`): device kernel over passive cells that finds the
   **smallest-id** substrate neighbor with no existing link and emits the pair; Poisson delete on
   device.
3. **Poisson compaction** (tissue / substrate / lamellipodia delete): the Bernoulli decisions are
   already drawn on device — consume them with Phase 6's device keep/compact primitive instead of
   `.numpy()` → Python `to_delete` set loops.
4. **Drop the host CSR copyback**: switch the steppable consumers to the device handles so
   `_inject_shared_csr` no longer round-trips the whole contact graph to NumPy each MCS.

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/embryo/`
- `gpu_port/phase7/`

## Constraints and fidelity invariants

- **Tissue cap-truncation order is load-bearing.** Iterate each cell's neighbors in **CSR-row
  (neighbor-list) order** so the per-cell degree cap (`MaxNeighborNum`) keeps the *same* links the
  CPU reference (`cpu_reference.py`) / `EmbryoSteppables.py` `get_cell_neighbor_data_list` order
  keeps. The exit test asserts the resulting link **set is identical**, not just the count.
- **Substrate min-id rule + `SubLinkRate`** preserved (the *which* substrate cell is not observable;
  existence/rate is). **Tissue `TissueRate`** and **lamellipodia `LamellaeRate`** Poisson rates
  preserved with their keyed-Philox streams (`1-exp(-rate)`); migrations keep rate + stream keys, not
  CC3D's shared-RNG draw order.
- **No new float-atomic nondeterminism.** Link claims use the Phase-1/6 atomic-append + compaction
  pattern (integer/id ops), not float atomics.
- Validate against **both** references the port already uses: the NumPy `cpu_reference.py` (exact
  per-MCS link set) and a short **real CC3D** full-Embryo cross-check (link inventory counts), as in
  Phase 3 Pass C.

## Delivery discipline (tested passes, each green)

1. **Tissue links on device.** Port the adjacency + cap-ordered relink to a kernel; keep the host
   path available behind a flag for the differential test. Test: device link set after `start()` and
   after each `step(mcs)` **equals** the host-reference set exactly on the real scaled-Embryo
   geometry (both leading and passive managers); degree caps respected.
2. **Substrate links + Poisson compaction on device.** Port substrate create/delete and wire device
   keep/compact for all three link kinds. Test: substrate link set == reference (min-id rule);
   turnover fractions match `1-exp(-rate)` within the Phase-3 tolerance; reproducible per key.
3. **Drop the copyback.** Route consumers to device CSR handles; remove the per-MCS `.numpy()` of the
   contact graph. Test: full `EmbryoModel.run` matches the prior validated full-Embryo link inventory
   (tissue/substrate/lamellipodia counts + link-length mean/KS) within Phase-3 tolerances; re-measure
   and report `csr_ms` + `steppable_ms`.

## Exit gate

Tests under `gpu_port/phase7/tests/`: (a) device tissue/substrate link **set** == host/CPU reference
exactly on real Embryo geometry (cap order preserved); (b) Poisson turnover rates match
`1-exp(-rate)` per kind, reproducible per key; (c) full-Embryo link inventory + link-length
mean/median/KS within Phase-3 tolerances vs the prior validated numbers and a short real-CC3D
cross-check; (d) the per-MCS host CSR copyback is gone (assert no full-graph `.numpy()` on the hot
path); (e) all prior tests green. Report the new per-MCS breakdown (target: `csr_ms` + `steppable_ms`
materially reduced).
