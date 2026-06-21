# Phase 8 — Cohesotaxis pipeline fusion + batched R-loop collapse

> Status: defined 2026-06-21. Depends on Phases 6–7 (device CSR + device link inventory + device
> steppables). Source: `plan_docs/CC3D-GPU-host-to-device-plan.md` (candidates #11–16). Highest
> fidelity risk and the biggest **sweep-throughput** win. The batched path's per-MCS host cost is
> O(R), so this is the phase that makes large-R sweeps GPU-bound instead of host-bound.

## Objective

1. **Fuse the single-engine cohesotaxis pipeline.** Collapse the 5 separately-launched stages
   (`cohesotaxis.py:508-624`: classify → fill → pixeldist → gumbel_select → manhattan_argmax) plus
   their host prefix-sums, per-slot `SigWeights` table, and inter-stage `.numpy()` glue into **one
   persistent-buffer device pipeline** — only the final `{cell→target}` leaves the device.
2. **Collapse the batched per-replica host loops** in `batched_steppables.py` so a sweep of R
   replicas runs as single batched launches:
   - **Batched CSR compaction** (`batched.py:273-325`): replace `for r in range(R)` per-replica
     compact + `radix_sort` + syncs with **one keyed global radix sort** (replica id packed into the
     high bits of the int64 `(src,dst)` key, so replicas sort into disjoint ranges) + one segmented
     scan.
   - **Batched tissue/substrate** (`batched_steppables.py:166-183,248-278`): one batched kernel over
     R·cells (reusing Phase 7's per-cell kernels along the replica axis).
   - **Device combine** (`batched_steppables.py:412-430`): assemble the padded `(R,M)` link arrays
     with one device gather instead of per-replica Python emit/concat/upload.
   - **Batched cohesotaxis** (`batched_steppables.py:344-365`): replace R independent pipelines with
     one **replica-segmented** pipeline over R·n_lead, per-`(r,cell)` keys.

## Scope

This phase may only create or modify files under:
- `gpu_port/engine/`
- `gpu_port/embryo/`
- `gpu_port/phase8/`

## Constraints and fidelity invariants

- **Cohesotaxis selection must be bit-exact per key.** The Gumbel-max over the `SigWeights` PDF (the
  exact equivalent of CC3D `rng.choice(p=w)`) and the Manhattan-shell argmax-by-zCOM tie-break must
  reproduce the **same selected target** per `(mcs, cell, seed)`. Fusion must not reorder free-pixel
  ranks (`free_ptr` segments, ascending cumulative-distance, id tiebreak) or change the RNG key
  layout. The fused pipeline is tested **==** the prior staged pipeline on identical keys.
- **Bit-exact-per-replica preserved.** A batched run must equal R independent single runs
  **bit-identically** (id-lattice, int64 COM, volumes, link inventory): disjoint per-replica atomic
  ranges, per-replica `base_seed_r`/stream keys, no float atomics across the batch axis. The keyed
  radix sort must keep each replica's contacts in a disjoint key range.
- **Rates preserved** across the batch axis (tissue/substrate/lamellipodia Poisson with per-replica
  keyed Philox).

## Delivery discipline (tested passes, each green)

1. **Cohesotaxis fusion (single engine).** Build the persistent-buffer device pipeline; precompute
   the `SigWeights` table on device. Test: fused pipeline selects the **identical** lamellipodia
   target as the current staged pipeline for every leader over many keys (bit-exact), on the toy
   scene and the real scaled-Embryo shell; no inter-stage host readback on the hot path.
2. **Batched CSR + tissue/substrate + combine.** Keyed global radix sort + segmented scan; batched
   link kernels along R; device combine. Test: batched per-replica CSR/link inventory **==** the
   single-engine result for each replica, bit-identical, over R=6; `count`/timing sanity.
3. **Batched cohesotaxis + end-to-end.** Replica-segmented cohesotaxis pipeline. Test: a full batched
   `BatchedEmbryoModel.run` over R replicas == R independent `EmbryoModel.run` **bit-identically**
   (Volume+Contact) and statistically-identical (FPP/cohesotaxis), per-replica; benchmark
   replica-MCS/s + sims/hour at R=8/32/64 vs the current O(R)-host batched path.

## Exit gate

Tests under `gpu_port/phase8/tests/`: (a) fused cohesotaxis selects identical targets to the staged
pipeline per `(mcs,cell,seed)` (bit-exact), no host glue on the hot path; (b) batched path ==
R independent single runs **bit-identical** for Volume+Contact+FPP (id-lattice/COM/volumes/link
inventory) and statistically-identical for cohesotaxis, over R≥6; (c) per-replica Poisson rates
preserved; (d) measured batched throughput (replica-MCS/s, sims/hour) at R=8/32/64 reported and
improved vs the pre-phase O(R)-host path; (e) all prior tests green.
