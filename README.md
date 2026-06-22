# embryo_gpu_port_cc3d

A GPU port of the **CompuCell3D "Embryo" model** (3-D *Xenopus* gastrulation, a Cellular Potts /
Glazier–Graner–Hogeweg model with FocalPointPlasticity links) to **[NVIDIA Warp](https://github.com/NVIDIA/warp)**.
The entire per–Monte-Carlo-Step (MCS) hot path — the CPM Metropolis sweep **and** the link-management
steppables (tissue / substrate intercalation, lamellipodia, on-device cohesotaxis) — runs on the GPU,
keeping CC3D-style Python scripting at the top.

> **Why.** Profiling the CPU model (see [`gpu_port/phase0/PHASE0_FINDINGS.md`](gpu_port/phase0/PHASE0_FINDINGS.md))
> showed the Python FPP-link steppables do **not** parallelize: at 32 OpenMP threads they dominate the wall
> clock and overall throughput is Amdahl-capped at ~10.5 MCS/s. To go faster, *both* the sweep and the
> steppables had to move on-device.

## Performance (RTX 5090, 100³ lattice, 63,011 cells)

- **~125 MCS/s** single-run at 100³ — **≈815–1834×** the tuned multicore CPU baseline (~10.5 MCS/s),
  depending on scene/lattice.
- **Batched parameter sweeps** run R replicas in one GPU launch: at R=64 the device-collapsed path is
  **≈17.8×** the per-replica host-loop path (≈5.3× / 12.7× / 17.8× at R = 8 / 32 / 64).
- **Bit-reproducible** `EmbryoModel.run` (deterministic FPP via a pre-sweep COM/volume snapshot).
- Scales to a single-GPU **1290³** lattice.

All GPU results are validated **statistically against CPU CompuCell3D** (volume / energy / surface / COM,
FPP link-length, cohesotaxis target selection, full-Embryo link inventory).

## What's modeled

8-color checkerboard CPM with **Volume + Contact + FocalPointPlasticity** energy, int64 fixed-point
center-of-mass, a hashed neighbor-contact CSR graph, device-authoritative FPP link inventory, tissue /
substrate / lamellipodia link dynamics with Poisson turnover, and an on-device **cohesotaxis** pipeline
(Gumbel-max selection over a SigWeights PDF + Manhattan-shell argmax tie-break).

## Requirements

- An **NVIDIA GPU with CUDA** (developed/tested on an RTX 5090, `sm_120`). Warp ships its own CUDA
  runtime, so you only need a recent NVIDIA driver — no separate CUDA Toolkit install.
- **Windows x64** — the pinned environment (`pixi.toml`) targets `win-64`. The Python/Warp code itself is
  largely OS-agnostic, but the locked env is Windows-only for now.
- **[pixi](https://pixi.sh)** to materialize the conda + pip environment.

## Setup

```bash
pixi install
# Warp is the core dependency but is not yet declared in pixi.toml / pixi.lock — install it into the env:
pixi run pip install warp-lang==1.14.0
```

Sanity check:

```bash
pixi run python -c "import warp; warp.init(); print('Warp', warp.__version__)"
```

> **Note:** `compucell3d` (the CPU reference + the VTK that the 3-D viewer reuses) *is* in `pixi.toml`,
> so `pixi install` provides it. Only Warp needs the extra `pip install` step above (see
> [Known gaps](#known-gaps--follow-ups)).

## Run the test suite

```bash
pixi run python -m pytest -q gpu_port
```

Expect **218 passed, 4 skipped**. The skips are opt-in heavy checks; enable them with environment flags:

```bash
# real-CC3D long-horizon ensemble cross-check:
CC3D_OFFLINE=1 pixi run python -m pytest -q gpu_port
# throughput / scaling benchmarks:
BENCH=1 pixi run python -m pytest -q gpu_port
```

## Minimal usage

The packages live under `gpu_port/` and are imported as `engine` / `embryo` / `bridge`, so put
`gpu_port` on the import path (the test suite does this via per-directory `conftest.py`).

```python
# run from the repo root with:  PYTHONPATH=gpu_port pixi run python your_script.py
from embryo import EmbryoModel, build_scaled_embryo

state, info = build_scaled_embryo(cube_size=100, seed=1)   # 63,011 cells (60 leading / 618 passive / substrate)
model = EmbryoModel(state).start()
model.run(1000)                                            # 1000 MCS on the GPU
print("active FPP links:", model.num_active_links())
```

`build_closure_scene(L=...)` gives a smaller, faster wound-closure scene used by the in-gate validation.

## Live viewers

A VTK 3-D viewer (closest to CC3D Player's 3-D cell view) and a 2-D viewer, both fed straight from GPU
device state each frame. They need `gpu_port` on `PYTHONPATH`:

```powershell
# PowerShell (Windows):
$env:PYTHONPATH = "gpu_port"; pixi run python -m bridge.live_viewer3d --scene embryo --size 100
```
```bash
# bash:
PYTHONPATH=gpu_port pixi run python -m bridge.live_viewer3d --scene embryo --size 100
```

Flags: `--scene {closure,embryo}`, `--size`, `--mcs-per-frame`, `--interval-ms`, `--max-mcs`,
`--no-substrate`, `--temperature`, `--seed`. Controls: **Play / Pause / Step**, substrate-shell toggle,
drag to rotate / scroll to zoom. The 2-D viewer is `python -m bridge.live_viewer` (same `PYTHONPATH`).

> The viewer's status bar shows MCS / cell count / active links — it does **not** print a throughput
> number. For measured MCS/s and a per-MCS breakdown, use the instrumented benchmark in
> [`gpu_port/engine/bench_csr.py`](gpu_port/engine/bench_csr.py).

## Parameter sweeps (batched)

`BatchedDeviceEmbryoModel` (in `embryo/batched_device.py`) runs R replicas with per-replica parameters
(e.g. `TissueRate` / `SubLinkRate` / `LamellaeRate` + spring params) in one GPU run, bit-exact per
replica. See [`gpu_port/phase8/tests/test_batched_collapse.py`](gpu_port/phase8/tests/test_batched_collapse.py)
for end-to-end usage.

## Repository layout

| Path | Contents |
|---|---|
| `gpu_port/engine/` | Core GPU CPM engine: `engine.py`, `kernels.py`, `fpp.py`, `cohesotaxis*.py`, `scan.py`, `link_kernels.py`, `graph.py` (CUDA-graph capture), `batched*.py`, `geometry.py`, `state.py`, `cpu_reference.py`, `bench*.py` |
| `gpu_port/embryo/` | The Embryo driver + steppables: `model.py` (`EmbryoModel`), `steppables.py`, `params.py`, `batched_device.py` |
| `gpu_port/bridge/` | Visualization: `live_viewer3d.py` (VTK 3-D), `live_viewer.py` (2-D), `viewer.py` (device→host view) |
| `gpu_port/phase0…8/` | Phased development, each with its test suite (`tests/`) and some with `PHASE*_FINDINGS.md` |
| `plan_docs/` | Per-phase design / plan documents |
| `pixi.toml`, `pixi.lock` | Environment definition |

## Validation & reproducibility

Each phase is gated against the NumPy `cpu_reference.py` (exact per-MCS link sets) and, for the full
Embryo, a real-CompuCell3D cross-check. Volume+Contact runs are bit-identical; FPP/cohesotaxis are
validated statistically (link-length mean/median/KS, turnover rates, target-selection identity).
`EmbryoModel.run` is bit-reproducible (toggle the determinism fix with `fpp_com_snapshot`).

## Project history

Built in phases 0–8 (toolchain spike → GPU core → FPP/cohesotaxis/Embryo → batching/scaling/viewer →
host→device optimization). The narrative lives in [`PROGRESS.md`](PROGRESS.md) and [`plan_docs/`](plan_docs/).
(The `.claude/` directory holds an internal phase-orchestration workflow used during development.)

## Known gaps / follow-ups

- **Warp is not declared in `pixi.toml`** (hence the extra `pip install warp-lang==1.14.0`). Declaring it
  with `pixi add --pypi warp-lang==1.14.0` would make `pixi install` self-contained.
- `BatchedDeviceFPP.build_csr_and_attach` still does R serial device→device copies (O(R) launches at
  large R); a batched-gather kernel would remove it.
- int64 voxel indexing to exceed the 1290³ int32-index ceiling.
- The long-horizon offline CC3D ensemble (`CC3D_OFFLINE=1`) is defined but not routinely run.

## License

No license has been set yet, so **default copyright (all rights reserved)** applies. If you would like to
use, extend, or build on this work, please contact the authors (GlazierFoxLab; `tc2fh@virginia.edu`).

## Acknowledgements

Built on **[CompuCell3D](https://compucell3d.org/)** (the Embryo model and the CPM/FPP reference
implementation) and **[NVIDIA Warp](https://github.com/NVIDIA/warp)**.
