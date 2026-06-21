"""Run the real CC3D Embryo capture in an ISOLATED subprocess and dump the snapshots
to a .npz (Phase 7 helper).

The default gate already runs the vendored CC3D Embryo once IN-PROCESS in
``phase3/tests/test_embryo_cc3d.py``. Running CC3D's ``run_script.main`` a SECOND time
in the same Python process (for the Phase-7 device-backend cross-check) triggers a
native access violation -- CC3D's global CUDA/SWIG state is not cleanly
re-initializable within one process. Invoking this script as a subprocess gives the
second CC3D run its own fresh process (clean CUDA context + teardown), so both the
Phase-3 and Phase-7 CC3D cross-checks coexist in one ``pytest`` run.

Usage (the Phase-7 test calls this via ``subprocess``):
    python _cc3d_capture_subprocess.py <out_npz> <steps> <mcs0,mcs1,...> <dim>

Writes ``<out_npz>`` with one object array per captured mcs (key ``mcs_<n>``) holding
the snapshot dict, plus a scalar ``runtime``. Exits non-zero on any error.
"""

import os
import sys

import numpy as np


def main():
    out_npz = sys.argv[1]
    steps = int(sys.argv[2])
    capture = [int(x) for x in sys.argv[3].split(",") if x != ""]
    dim = int(sys.argv[4])

    # import the existing CC3D reference runner from phase3/tests (same dir-on-path
    # pattern the phase3 tests use; no out-of-scope edit -- just an import).
    here = os.path.dirname(os.path.abspath(__file__))
    phase3_tests = os.path.normpath(os.path.join(here, "..", "..", "phase3", "tests"))
    if phase3_tests not in sys.path:
        sys.path.insert(0, phase3_tests)
    import cc3d_embryo_ref as REF

    snaps, runtime, base = REF.run_embryo_capture(steps=steps, capture_mcs=capture, dim=dim)
    try:
        payload = {f"mcs_{int(m)}": np.array(snaps[int(m)], dtype=object) for m in capture}
        payload["runtime"] = np.array(runtime, dtype=np.float64)
        np.savez(out_npz, **payload)
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
