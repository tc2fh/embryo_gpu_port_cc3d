"""
Phase 0 profiling harness for the Embryo CC3D model (GPU-port plan, Phase 0).

Runs the Embryo model headless for a capped number of MCS and reports the wall-clock
split between:
  - the C++ engine  (CPM Metropolis sweep + FocalPointPlasticity energy + trackers) -> `compiled_code_run_time`
  - the Python steppables (per registered steppable class)                          -> SteppableRegistry profiler
  - everything else / I-O / overhead                                                 -> remainder of the loop

It does NOT touch the user's original model: it copies the project to a working dir and
patches only the copy (Steps, NumberOfProcessors, optional MetropolisAlgorithm, SaveSegmentation).

Usage (always via pixi so the env DLLs resolve):
  pixi run python gpu_port/phase0/profile_embryo.py --steps 200 --procs 1 --tag baseline_p1
  pixi run python gpu_port/phase0/profile_embryo.py --steps 200 --procs 8 --tag tuned_p8
  pixi run python gpu_port/phase0/profile_embryo.py --steps 200 --procs 1 --boundary-walker --tag bw_p1
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "Embryo_Model_dev" / "Embryo"


def patch_xml(xml_path: Path, steps: int, procs: int, boundary_walker: bool) -> None:
    text = xml_path.read_text(encoding="utf-8")
    text = re.sub(r"<Steps>\s*\d+\s*</Steps>", f"<Steps>{steps}</Steps>", text)
    text = re.sub(
        r"<NumberOfProcessors>\s*\d+\s*</NumberOfProcessors>",
        f"<NumberOfProcessors>{procs}</NumberOfProcessors>",
        text,
    )
    if boundary_walker:
        # Insert a MetropolisAlgorithm element inside <Potts> if not already present.
        if "MetropolisAlgorithm" not in text:
            text = text.replace(
                "<Steps>", "<MetropolisAlgorithm>Boundarywalker</MetropolisAlgorithm>\n      <Steps>", 1
            )
    xml_path.write_text(text, encoding="utf-8")


def patch_steppables(py_path: Path, save_seg: bool) -> None:
    if save_seg:
        return
    text = py_path.read_text(encoding="utf-8")
    text = re.sub(r"^SaveSegmentation\s*=\s*\d+", "SaveSegmentation = 0", text, flags=re.MULTILINE)
    py_path.write_text(text, encoding="utf-8")


def prepare_project(src: Path, work: Path, steps: int, procs: int, boundary_walker: bool, save_seg: bool) -> Path:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work)
    sim_dir = work / "Simulation"
    patch_xml(sim_dir / "Embryo.xml", steps, procs, boundary_walker)
    patch_steppables(sim_dir / "EmbryoSteppables.py", save_seg)
    return work / "Embryo.cc3d"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 Embryo profiler")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--boundary-walker", action="store_true")
    ap.add_argument("--save-seg", action="store_true", help="keep SaveSegmentation on (default off)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    args = ap.parse_args()

    out_root = Path(__file__).resolve().parent / "_runs" / args.tag
    work = out_root / "project"
    output_dir = out_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cc3d_file = prepare_project(
        Path(args.src), work, args.steps, args.procs, args.boundary_walker, args.save_seg
    )

    # Import after env is up; monkeypatch the profiling-report sink to capture metrics.
    import cc3d.CompuCellSetup.simulation_setup as ss
    from cc3d import run_script

    captured = {}
    orig = ss.generate_profiling_report

    def capture(**kw):
        captured.update(kw)
        return orig(**kw)

    ss.generate_profiling_report = capture

    ns = argparse.Namespace(
        input=str(cc3d_file),
        output_file_core_name="Step",
        current_dir=None,
        output_dir=str(output_dir),
        output_frequency=0,
        screenshot_output_frequency=0,
        restart_snapshot_frequency=0,
        restart_multiple_snapshots=False,
        parameter_scan_iteration="",
        execute_step_at_mcs_0=False,
        log_level="",
        log_to_file=False,
    )

    wall_begin = time.time()
    run_script.main(ns)
    wall_total_ms = (time.time() - wall_begin) * 1000.0

    engine_ms = float(captured.get("compiled_code_run_time", 0.0))
    total_loop_ms = float(captured.get("total_run_time", 0.0))
    report = captured.get("py_steppable_profiler_report", []) or []

    per_steppable = {}
    for entry in report:
        name, _hash, rt = entry[0], entry[1], float(entry[2])
        per_steppable[name] = per_steppable.get(name, 0.0) + rt
    steppable_ms = sum(per_steppable.values())
    other_ms = max(total_loop_ms - engine_ms - steppable_ms, 0.0)

    steps = args.steps
    result = {
        "tag": args.tag,
        "steps": steps,
        "procs": args.procs,
        "boundary_walker": args.boundary_walker,
        "save_seg": args.save_seg,
        "wall_total_ms": wall_total_ms,
        "loop_total_ms": total_loop_ms,
        "engine_ms": engine_ms,
        "steppables_ms": steppable_ms,
        "other_ms": other_ms,
        "per_steppable_ms": per_steppable,
        "engine_ms_per_step": engine_ms / steps if steps else None,
        "steppables_ms_per_step": steppable_ms / steps if steps else None,
        "mcs_per_s_engine_plus_steppables": (
            steps / ((engine_ms + steppable_ms) / 1000.0) if (engine_ms + steppable_ms) else None
        ),
        "mcs_per_s_loop": steps / (total_loop_ms / 1000.0) if total_loop_ms else None,
    }

    (out_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def pct(x):
        return f"{100.0 * x / total_loop_ms:5.1f}%" if total_loop_ms else "  n/a"

    print("\n" + "=" * 64)
    print(f"PHASE 0 PROFILE  tag={args.tag}  steps={steps}  procs={args.procs}"
          f"  boundary_walker={args.boundary_walker}")
    print("=" * 64)
    print(f"  loop total      : {total_loop_ms:10.1f} ms")
    print(f"  C++ engine      : {engine_ms:10.1f} ms  {pct(engine_ms)}   "
          f"({result['engine_ms_per_step']:.2f} ms/MCS)")
    print(f"  Py steppables   : {steppable_ms:10.1f} ms  {pct(steppable_ms)}   "
          f"({result['steppables_ms_per_step']:.2f} ms/MCS)")
    for name, ms in sorted(per_steppable.items(), key=lambda kv: -kv[1]):
        print(f"      - {name:28s}: {ms:10.1f} ms  {pct(ms)}")
    print(f"  other / I-O     : {other_ms:10.1f} ms  {pct(other_ms)}")
    print(f"  MCS/s (engine+stp): {result['mcs_per_s_engine_plus_steppables']:.2f}")
    print(f"  MCS/s (loop)      : {result['mcs_per_s_loop']:.2f}")
    print(f"  result.json     : {out_root / 'result.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
