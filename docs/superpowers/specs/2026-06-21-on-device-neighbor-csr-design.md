# On-device neighbor-contact CSR build — design

Date: 2026-06-21
Status: approved (user approved Approach 1 + "preserve ascending order"; authorized straight-through
implementation, validation, and benchmarking).

## Problem

`GPUEngine.neighbor_contact_csr()` (`gpu_port/engine/engine.py`) is the per-MCS neighbor relation the
Embryo steppables consume (one shared build per MCS via `EmbryoModel._inject_shared_csr`). Today it:

1. runs a device open-addressing **hash kernel** (`neighbor_contact_hash_kernel`) that aggregates the
   directed `(src,dst)` face-contact counts into a table of `cap` slots, where `cap` = next power of two
   ≥ 2× an *upper bound* on distinct pairs (`min(nvox*n_off, n1*n1)`), then
2. **copies the entire `cap`-sized table to the host** (`ht_key` int64 + `ht_count` int32) and compacts
   it there: boolean-mask occupied slots → decode `src,dst` → `np.lexsort((dst,src))` → `np.bincount`
   → `np.cumsum` → CSR `(indptr, indices, data)`.

At the full Embryo scale (100³, 63 011 cells, order-1) `cap` ≈ 2²⁴ ≈ 16.7 M slots ⇒ ~**200 MB**
device→host copy **every MCS**, plus a host `lexsort` over the occupied entries. That copy + host sort is
the dominant host cost the prior profiling attributed to the ~114 ms/MCS "neighbor-CSR + steppable"
step. The compact result is only O(#contacts) (~hundreds of thousands of entries → a few MB).

## Goal

Build the compact, **ascending-within-row** CSR on the GPU so only the O(#contacts) result crosses
PCIe and the host `lexsort`/`bincount` disappear — with **byte-identical output** to the current host
path (zero behavior change; all existing CC3D validation gates pass unchanged). Then measure and report
the speedup: the CSR build in isolation **and** the end-to-end full-Embryo MCS/s with a per-MCS
breakdown (CSR build vs the steppable host loops), so the real wall-clock impact and the next
bottleneck are both visible.

## Approach (Approach 1 — mirror the FPP CSR build)

Keep the existing hash kernel unchanged (it already aggregates per-pair counts on-device). Replace only
the host compaction with three additive Warp kernels, reusing the exact shape already proven for the FPP
link CSR (`fpp_count_active_links_kernel` + host cumsum + `fpp_fill_csr_kernel`, `engine/fpp.py:178-221`):

1. **`neighbor_csr_count_kernel`** (`cap` threads): for each occupied slot, decode `src = key // n1`,
   `wp.atomic_add(row_counts, src, 1)`.
2. **host cumsum** of `row_counts` (n1 int32 ≈ 252 KB transfer) → `indptr` (int64, n1+1);
   `n_contacts = indptr[-1]`; upload `indptr` as an int32 device array. (Identical to the FPP step; n1 is
   tiny and off the hot per-flip path.)
3. **`neighbor_csr_scatter_kernel`** (`cap` threads): for each occupied slot, decode `src,dst`;
   `pos = indptr[src] + wp.atomic_add(cursor, src, 1)`; write `indices[pos]=dst`, `data[pos]=count`.
4. **per-row ascending sort** via `wp.utils.segmented_sort_pairs(indices, data, n_contacts, indptr)`
   — a segmented radix sort over the CSR rows (keys = `dst` int32, values = `data`, segments =
   `[indptr[i], indptr[i+1])`). O(#contacts).

   > **Implementation note (pitfall found in benchmarking).** This was first written as a per-row
   > insertion-sort kernel (one thread per row), on the assumption rows are short. They are not: the
   > **Medium row (src=0) touches ~every surface cell** — 9 039 entries at 50³, ~40 000 at 100³ — while
   > every other row is ≤ ~50. A single thread doing O(k²) on that one giant row dominated everything
   > (357 ms at 50³; **34 s** at 100³ — ~290× *slower* than host). Replaced with the segmented radix
   > sort, which is O(#contacts) and indifferent to row-size skew. Regression-guarded by
   > `test_device_csr_handles_giant_row_fast_and_exact` (a 40 000-entry Medium row, generous 2 s bound).

The atomic-cursor scatter is order-nondeterministic, but the per-row sort makes the final
`(indptr, indices, data)` deterministic and ascending-by-`dst` within each `src` row — exactly what the
host `lexsort((dst,src))` produces. Result is byte-identical to the host path.

### Engine change

`neighbor_contact_csr(self, order=None, method="device")`:
- shared: build `cap`, (re)alloc `_ht_key/_ht_count`, launch the hash kernel (unchanged).
- `method="host"`: the existing compaction (kept verbatim, renamed into a private helper) — the
  reference path the device path is asserted equal to, and a CPU/no-fancy fallback.
- `method="device"`: kernels 1–4 above. Returns `(indptr, indices, data)` as host numpy **int64**
  (device arrays are int32; cast on return to preserve the current API contract). Empty topology /
  `n_contacts == 0` ⇒ well-formed empty CSR (`indptr` all zeros, empty `indices/data`).
- Lazy scratch on the engine: `_csr_row_counts` (n1+1 int32), `_csr_cursor` (n1+1 int32), and
  `_csr_indices`/`_csr_data` device buffers cached with a high-water mark (grown when `n_contacts`
  exceeds capacity), sliced `[:n_contacts]` on transfer.

### End-to-end toggle

`EmbryoModel.__init__(..., csr_method="device")`; `_inject_shared_csr` calls
`neighbor_contact_csr(order=1, method=self.csr_method)`. Lets the benchmark A/B host vs device
end-to-end with one switch. Default `"device"` makes the on-device build the production path.

## Correctness / edge cases

- **Exactness anchor:** device output == host output **exactly** (same dtype, same ascending order),
  asserted directly — stronger than only the existing dense-reconstruction-vs-CPU test.
- Existing gates rebuild a dense matrix from the CSR (order-independent) and compare to a NumPy CPU
  reference; preserving ascending order keeps every current test (`test_trackers.py`,
  `test_neighbor_csr_scale.py`, the Embryo validation gates) green unchanged.
- `int32` suffices for `indices` (`dst < n1 < 2³¹`) and `data` (per-pair count ≪ 2³¹); `indptr` int64.
- Hash `cap` sizing and the hash kernel are untouched — no change to memory ceiling or scale behavior.
- Independent of the known FPP bit-reproducibility item (this is the contact CSR, not the FPP energy).

## Testing (`gpu_port/phase5/tests/`, new conftest mirroring phase3/4)

`test_neighbor_csr_device.py` (CUDA-skipif, same as the existing CSR tests):
1. **device == host exactly**: `indptr/indices/data` array_equal across both methods on a small lattice
   after a few MCS.
2. **device == CPU reference**: dense reconstruction equals `_cpu_pairs` (the existing contract) via the
   device path.
3. **ascending within-row order**: every row's `indices[lo:hi]` is strictly increasing.
4. **scale**: 40³ = 64 000 single-voxel cells (the existing dense-infeasible scale) builds via the device
   path; `data.sum()`/well-formedness assertions match the existing scale test.
5. **empty / no-contact** guard: degenerate input returns a well-formed empty CSR.

Then the full suite (`pixi run python -m pytest -q gpu_port`) must stay green (76 passed / 3 sanctioned
skips baseline).

## Benchmark — "see the speedup" (`gpu_port/engine/bench_csr.py`)

- **CSR microbenchmark**: at order-1 over a representative scale (and the full 100³ Embryo geometry),
  warm then time N calls of `neighbor_contact_csr(method="host")` vs `method="device")`; report ms/call,
  speedup, `n_contacts`, `cap`, and the host-copy size avoided.
- **End-to-end**: full `EmbryoModel` (cube_size=100), run K MCS with `csr_method` = host vs device;
  report MCS/s and a per-MCS host-time breakdown (CSR build vs steppable `.step()` loops) so the Amdahl
  reality and the next bottleneck (e.g. the steppables' `np.append`-per-link host loops) are explicit.

## Scope

In: `engine/kernels.py` (+3 kernels, additive), `engine/engine.py` (`neighbor_contact_csr` device path +
lazy scratch), `embryo/model.py` (`csr_method` passthrough), new `engine/bench_csr.py`, new
`phase5/tests/` + conftest.

Out: porting the steppable neighbor *consumption* on-device (would remove the remaining host transfer
but is a much larger rewrite of `TissueLinkSteppable`/`PassiveSubstrateSteppable`); FPP/batched paths;
the int64 voxel-index widening. The hash kernel itself is unchanged.

## Risks

- If the steppable host loops dominate the 114 ms (plausible: `FPPLinks.create_link` uses
  `np.append` per call ⇒ O(M²)), the end-to-end win will be smaller than the CSR-microbench win. The
  benchmark's breakdown is designed to surface this honestly rather than over-claim.
