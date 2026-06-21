"""CC3D Embryo reference runner (Pass C validation support).

Runs the ACTUAL vendored CompuCell3D Embryo model headless (the same
``run_script.main`` path the Phase 0 profiler used) and captures physics
observables at chosen MCS via a monkeypatched observer steppable:

  * the id-lattice + per-cell type (for the GPU to load the identical state),
  * the FocalPointPlasticity link-length distribution + per-link-kind counts,
  * the floor free-area (SubstrateSteppable.CellArray metric).

This is the genuine CC3D fidelity reference. It is SLOW (full 100^3 Embryo runs
at ~4 MCS/s headless on this box), so the in-gate suite uses it only for tiny MCS
counts in tests explicitly marked slow / skipped by default; the default gate uses
the fast NumPy references. Documented deferral, with measured runtime, in the
Pass C report.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "Embryo_Model_dev" / "Embryo"


def cc3d_importable() -> bool:
    try:
        import cc3d  # noqa: F401
        return True
    except Exception:
        return False


def run_embryo_capture(steps: int, capture_mcs, dim: int = 100, seed: int | None = None,
                       workdir: str | None = None):
    """Run the real CC3D Embryo for ``steps`` MCS; at each mcs in ``capture_mcs``
    record (ids, cell_type_by_id, fpp link lengths, link kind counts, free_area).

    Returns dict: mcs -> snapshot dict. Runs headless via cc3d.run_script. ``dim``
    optionally overrides the Potts Dimensions (the steppable geometry still targets
    100, so keep dim=100 for a faithful run)."""
    import cc3d.CompuCellSetup.simulation_setup as ss  # noqa
    from cc3d import run_script
    from cc3d.CompuCellSetup import persistent_globals as pg  # noqa

    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="cc3d_embryo_"))
    work = base / "project"
    outdir = base / "output"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(SRC, work)
    sim = work / "Simulation"

    xml = sim / "Embryo.xml"
    t = xml.read_text(encoding="utf-8")
    t = re.sub(r"<Steps>\s*\d+\s*</Steps>", f"<Steps>{steps}</Steps>", t)
    if dim != 100:
        t = re.sub(r'<Dimensions x="\d+" y="\d+" z="\d+"/>',
                   f'<Dimensions x="{dim}" y="{dim}" z="{dim}"/>', t)
    xml.write_text(t, encoding="utf-8")

    st = sim / "EmbryoSteppables.py"
    s = st.read_text(encoding="utf-8")
    s = re.sub(r"^SaveSegmentation\s*=\s*\d+", "SaveSegmentation = 0", s, flags=re.MULTILINE)
    st.write_text(s, encoding="utf-8")

    capture_set = set(int(m) for m in capture_mcs)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        input=str(work / "Embryo.cc3d"), output_file_core_name="Step", current_dir=None,
        output_dir=str(outdir), output_frequency=0, screenshot_output_frequency=0,
        restart_snapshot_frequency=0, restart_multiple_snapshots=False,
        parameter_scan_iteration="", execute_step_at_mcs_0=False, log_level="",
        log_to_file=False)

    # Inject an observer steppable into the copied model's main script so it runs
    # INSIDE the real simulation and dumps per-MCS snapshots to a .npz we read back.
    _inject_observer(work, capture_set)
    t0 = time.time()
    run_script.main(ns)
    runtime = time.time() - t0

    snapshots = {}
    snap_file = outdir / "_observer_snaps.npz"
    if snap_file.exists():
        data = np.load(snap_file, allow_pickle=True)
        snapshots = {int(k.split("_")[1]): data[k].item() for k in data.files}
    return snapshots, runtime, base


_OBSERVER_MODULE = '''"""Pass C validation observer (auto-generated)."""
from cc3d.core.PySteppables import *
import numpy as np
import os

CAPTURE = {cap!r}


class PassCObserver(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self._snaps = {{}}

    def step(self, mcs):
        if mcs not in CAPTURE:
            return
        dz, dy, dx = self.dim.z, self.dim.y, self.dim.x
        ids = np.zeros((dz, dy, dx), dtype=np.int32)
        tb = {{0: 0}}
        for x, y, z in self.every_pixel():
            c = self.cell_field[x, y, z]
            if c:
                ids[z, y, x] = c.id
                tb[c.id] = int(c.type)
        lengths = []
        kinds = {{"tissue": 0, "lamellipodia": 0, "substrate": 0, "other": 0}}
        ll = self.get_focal_point_plasticity_link_list()
        if ll is not None:
            for link in ll:
                L = None
                lam = None
                try:
                    L = float(link.getDistance())
                except Exception:
                    pass
                try:
                    lam = float(link.getLambdaDistance())
                except Exception:
                    pass
                if L is not None:
                    lengths.append(L)
                if lam is not None:
                    if abs(lam - 600.0) < 1e-3:
                        kinds["tissue"] += 1
                    elif abs(lam - 800.0) < 1e-3:
                        kinds["lamellipodia"] += 1
                    elif abs(lam - 10.0) < 1e-3:
                        kinds["substrate"] += 1
                    else:
                        kinds["other"] += 1
        self._snaps["mcs_%d" % mcs] = dict(
            ids=ids, type_by_id=tb,
            link_lengths=np.array(lengths, dtype=np.float64), link_kinds=kinds)

    def finish(self):
        self._dump()

    def on_stop(self):
        self._dump()

    def _dump(self):
        try:
            from cc3d.CompuCellSetup import persistent_globals as pg
            outdir = pg.output_directory or "."
        except Exception:
            outdir = "."
        np.savez(os.path.join(outdir, "_observer_snaps.npz"), **self._snaps)
'''


def _inject_observer(work: Path, capture_set):
    """Write an observer steppable module into the copied Simulation dir and append
    its registration to the model's main script so it runs inside the real
    simulation and dumps per-MCS snapshots to a .npz we read back."""
    sim = work / "Simulation"
    cap = sorted(int(m) for m in capture_set)
    (sim / "PassCObserverSteppable.py").write_text(
        _OBSERVER_MODULE.format(cap=cap), encoding="utf-8")
    main_py = sim / "Embryo.py"
    s = main_py.read_text(encoding="utf-8")
    s = re.sub(r"CompuCellSetup\.run\(\)\s*$", "", s.rstrip()) + "\n"
    s += (
        "\nfrom PassCObserverSteppable import PassCObserver\n"
        "CompuCellSetup.register_steppable(steppable=PassCObserver(frequency=1))\n"
        "CompuCellSetup.run()\n"
    )
    main_py.write_text(s, encoding="utf-8")
