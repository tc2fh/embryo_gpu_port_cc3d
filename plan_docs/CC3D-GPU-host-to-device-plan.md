# Plan: host→device optimization of the GPU Embryo port (Phases 6–8)

## What this is

The CC3D-GPU port (Phases 0–4, plus an ad-hoc Phase 5 batched-FPP effort) is functionally complete:
a GPU-resident CPM engine (8-color Volume+Contact, int64 fixed-point COM, hashed neighbor-CSR), FPP +
on-device cohesotaxis, the full `EmbryoModel`, a parameter-sweep batch axis, and CUDA-graph MCS
capture at **815–1834× the tuned multicore CPU baseline**. The Metropolis sweep is already cheap.

This plan defines the **next optimization frontier: the work that still runs on the host (CPU) every
MCS** and moves it onto the GPU, while staying faithful to the original CompuCell3D model in
`Embryo_Model_dev/Embryo/Simulation/EmbryoSteppables.py`. It is the deliverable for the request
"define what in `gpu_port` currently runs on CPU that could be moved to GPU." It is organized as three
workflow-executable phases (`plan_docs/phase_06..08_*.md`) so the phase-orchestration workflow
(`.claude/workflow.md`) can drive them.

> **Phase numbering note.** `plan_docs/` has specs for phases 01–04. "Phase 5" (batched FPP for
> parameter sweeps — per-replica link CSR, batched intercalation/substrate/cohesotaxis, the combined
> `BatchedEmbryoModel`) was delivered ad-hoc in commits `5b12948`→`aba4c54` **without a formal spec**;
> these new phases continue from that state and are numbered 06–08 to avoid renumbering history.

## The profiling finding that drives everything

Measured per-MCS wall time on the **full Embryo (100³, 63 011 cells)**, from the repo's own
instrumented benchmark `gpu_port/engine/bench_csr.py::_run_instrumented`:

| segment | what | ~time/MCS | where the time goes |
|---|---|---|---|
| `sweep_ms`  | `engine.step_mcs` — FPP rebuild + 8 color kernel launches | **~0.9 ms** | already GPU-resident (even cheaper under graph capture) |
| `csr_ms`    | `model._inject_shared_csr` — order-1 neighbor CSR build | **~2.3 ms** | built on GPU but **copied to host NumPy every MCS** + a host cumsum |
| `steppable_ms` | all steppables' `.step(mcs)` — link create/delete/relink | **~4.9 ms** | **host Python/NumPy loops** over cells, links, and the CSR |

**The GPU sweep is no longer the bottleneck; the host-side CSR roundtrip and the Python steppable
link-management loops are** (~7.2 of ~8.1 ms). Under the CUDA graph the sweep shrinks further, so the
host fraction is even larger. For parameter sweeps the batched path is worse: every per-MCS host cost
is **O(R)** (a `for r in range(R)` loop), so as replicas scale 8→32→64 the GPU sits idle waiting on
host orchestration.

## Root-cause architecture (one seam explains all of it)

`engine.neighbor_contact_csr` builds the contact graph **on the GPU but returns it to the host as
NumPy** `(indptr, indices, data)` (`engine.py:221-303`). Every steppable then runs its link
create/delete logic in **host Python/NumPy** over those arrays, mutating the **host-resident**
`FPPLinks._a/_b` link inventory (`fpp.py:159-191`), and the engine re-uploads + rebuilds the device
link-CSR once per MCS. So the fix is one architectural move with downstream consequences:

1. **Keep the neighbor-contact CSR resident on the GPU** (stop the per-MCS device→host copy).
2. **Make the GPU the authoritative home of the FPP link inventory** (stop mutating host NumPy).
3. Then **the per-cell relink loops, Poisson compaction, and cohesotaxis become device kernels**, and
   **the batched R-loops collapse into single batched launches**.

## Consolidated CPU→GPU candidate inventory

Merged + deduped from the three subsystem audits (single engine / steppables / batched). Risk is
fidelity risk vs the original Embryo model. "Phase" maps each candidate to its spec below.

| # | candidate | location (file:line) | host work today | GPU approach | risk | phase |
|---|---|---|---|---|---|---|
| 1 | fpp-rebuild scan | `fpp.py:263-265` | `np.cumsum` degree → `link_ptr` + `wp.array` realloc **every** `step_mcs` | device exclusive-scan in place | low | 6 |
| 2 | csr-indptr scan | `engine.py:329-331` | `np.cumsum` degrees → `indptr` | device exclusive-scan | low | 6 |
| 3 | batched ptr scan | `batched_fpp.py:190-193` | host `cumsum(axis=1)` → per-replica `link_ptr` | device segmented scan (per-replica reset) | low | 6 |
| 4 | CSR residency | `engine.py:221-303`, `model.py:259-266` | whole contact graph `.numpy()` every MCS to feed host adjacency | keep `indptr/indices` on device; expose handles; drop copyback once consumers are device-side | low (byte-equal) | 6→7 |
| 5 | link topology storage | `fpp.py:159-191` | `np.concatenate`/`np.isin` O(M) host passes on create/delete, ~6×/MCS | device append/tombstone/compact kernels; device-authoritative inventory | low–med | 6 |
| 6 | tissue adjacency | `embryo/steppables.py:87-152` | `_grouped_csr_neighbors` host adjacency dict, ×3/MCS | device per-cell CSR-row scan + type filter | low–med | 7 |
| 7 | tissue link-map | `embryo/steppables.py:192-210` | Python `zip` loop building per-cell partner `set()`s | device per-cell degree + membership test | med | 7 |
| 8 | tissue relink | `embryo/steppables.py:218-274` | nested Python create-under-cap loop (×2: leading+passive) | device per-cell **CSR-row-ordered** claim kernel | **med–high** | 7 |
| 9 | substrate link | `embryo/steppables.py:306-368` | Python loop: min-id substrate pick + Poisson delete | device kernel over passive cells | low–med | 7 |
| 10 | Poisson compaction | `embryo/steppables.py:237-247,326-368`, `engine/steppables.py:209-239` | `.numpy()` device decisions → Python `to_delete` set loops | device keep/compact (decisions already on device) | low | 7 |
| 11 | cohesotaxis fusion | `cohesotaxis.py:508-624` | 5 separately-launched kernels + host prefix-sums + `.numpy()` glue (~5 syncs) | fuse into one persistent-buffer device pipeline | med–high | 8 |
| 12 | sig-weights table | `cohesotaxis.py:599-609` | per-slot `SigWeights`+`np.log` rebuilt host-side each relink | device weight table, indexed by rank | low | 8 |
| 13 | batched CSR compact | `batched.py:273-325` | `for r in range(R)`: per-replica compact + `radix_sort` + ~3 sync + 3 `.numpy()` | one keyed global radix sort (replica id in high key bits) + segmented scan | low | 8 |
| 14 | batched combine | `batched_steppables.py:412-430` | per-replica Python emit + `concat` + pad/upload, O(R·links) | device gather into the padded `(R,M)` arrays | low–med | 8 |
| 15 | batched tissue/sub | `batched_steppables.py:166-183,248-278` | `for r in range(R)`: Poisson + create/delete Python loops | batched kernels over R·cells | med | 8 |
| 16 | batched cohesotaxis | `batched_steppables.py:344-365` | `for r in range(R)`: a full 5-stage pipeline **per replica** (largest serial cost at high R) | one replica-segmented pipeline over R·n_lead, per-`(r,cell)` keys | **high** | 8 |

Explicitly **left on host** (negligible / not on the Embryo hot path, documented so they aren't
mistaken for omissions): `ClosureSteppable` (already a device kernel; only a 1-int observable
readback), the external observable read APIs (`get_ids/volumes/coms`, queried, not per-MCS),
`recompute_trackers` (the Phase-2 generic path; `EmbryoModel` doesn't call it), and `__init__`-time
geometry builds.

## Fidelity constraints (faithfulness to `Embryo_Model_dev`)

These are the invariants every migration above must preserve; they are the reason this is a careful
port, not a rewrite. Each phase's exit gate tests the relevant ones.

1. **Poisson turnover (tissue / substrate / lamellae).** Preserve the rate `1-exp(-rate)` and the
   keyed-Philox stream per `(mcs, item, stream, seed)`. Decisions are *already* drawn on device; a
   migration must keep the rate and stream keys, but need **not** reproduce CC3D's single shared
   `np.random` draw order (the sanctioned per-key substitution, validated within 5–6σ in Phase 3).
2. **Tissue link cap truncation order (the one genuinely order-sensitive rule).** Neighbors must be
   iterated in **CSR-row (neighbor-list) order** so the per-cell degree cap (`MaxNeighborNum`)
   truncates to the *same* links CC3D's `get_cell_neighbor_data_list` order would keep.
3. **Substrate link partner.** CC3D uses `random.choice`; the port uses deterministic **smallest-id**
   (sanctioned — *which* 1-voxel substrate cell is not an observable, only link existence/rate is).
   Preserve the min-id rule + the `SubLinkRate` Poisson rate.
4. **Cohesotaxis selection.** Gumbel-max over the `SigWeights` PDF (the exact equivalent of
   `rng.choice(p=w)`) + Manhattan-shell argmax-by-zCOM tie-break must stay **bit-exact per
   `(mcs, cell, seed)` key**. This is the most intricate behavior; fusion/batching must not reorder
   free-pixel ranks or change the RNG key layout.
5. **Contact CSR + COM are exact.** Device scans must produce **byte-identical** integer prefixes;
   COM stays int64 fixed-point. Volume+Contact runs remain **bit-reproducible**.
6. **Bit-exact-per-replica (batched).** Each replica must remain `==` an independent single run:
   disjoint per-replica atomic ranges, per-replica seed/stream keys, no float atomics across the
   batch axis.

> Open reproducibility item carried from Phase 4 (orthogonal, not required here): FPP runs are
> statistically faithful but not bit-reproducible because the spring term reads linked cells' COM
> while concurrent same-color flips update those accumulators. The sanctioned fix (snapshot/
> double-buffer COM at the start of each color sweep) can be folded into Phase 6's link-topology work
> if bit-exact FPP is wanted, but it is not a prerequisite for any phase here.

## The three phases (dependency-ordered, risk-ascending)

Each leaves `pixi run python -m pytest -q gpu_port` green and is a rollback point, exactly like
Phases 3–4. The ordering is forced by the architecture: storage/CSR residency first, then the
consumers that read them, then the highest-risk selection logic and the batched collapse.

- **Phase 6 — device-resident scans + neighbor-CSR residency + device-authoritative link topology**
  (`gpu_port/engine/` only). Fidelity-neutral foundation: replace the host cumsums with device scans
  (#1–3), expose device CSR handles (#4), and move the FPP link inventory onto the device with
  append/tombstone/compact kernels (#5), **keeping the existing host-facing APIs working** so all
  prior tests stay green and `embryo/` is untouched. Low risk; unblocks 7 and 8.

- **Phase 7 — device link steppables** (`gpu_port/engine/` + `gpu_port/embryo/`). Port the per-cell
  host relink loops to device kernels consuming Phase 6's device CSR + inventory (#6–10): tissue
  links (CSR-row cap order), substrate links (min-id), and device-side Poisson compaction; then
  **drop the per-MCS host CSR copyback**. This is where most of `steppable_ms` + `csr_ms` is
  recovered. Medium risk (cap-order fidelity).

- **Phase 8 — cohesotaxis fusion + batched R-loop collapse** (`gpu_port/engine/` + `gpu_port/embryo/`).
  Fuse the single-engine 5-stage cohesotaxis pipeline (#11–12), then collapse the batched per-replica
  host loops (#13–16) so a sweep of R replicas runs as single batched launches. Highest risk (Gumbel/
  Manhattan exactness + bit-exact-per-replica) and the biggest sweep-throughput win.

## How to run this

This is a normal phase run: say **"start the phase run"** / **"run the next phase"** and the
orchestrator drives `phase_06` → `phase_07` → `phase_08` per `.claude/workflow.md` (executor +
summarizer subagents, deterministic gate, commit per clean pass). Each phase's "Exit gate" section
names the fidelity tests that must pass before advancing.
