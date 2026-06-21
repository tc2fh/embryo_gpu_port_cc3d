"""Phase 7 real-CC3D cross-check for the DEVICE link-backend full Embryo.

Mirrors the Phase-3 Pass C convention (``phase3/tests/test_embryo_cc3d.py``): run the
ACTUAL vendored CompuCell3D Embryo headless a few MCS at full 100^3 scale and compare
its FocalPointPlasticity link inventory + link-length distribution to the GPU
``EmbryoModel`` -- here with ``link_backend="device"`` (the Phase-7 device link
steppables + dropped CSR copyback). It reuses the existing ``cc3d_embryo_ref`` runner
(no new CC3D dependency) and the existing opt-in skip convention:

  * ``@needs_cc3d`` -- skipif cc3d is not importable, so the default gate does NOT
    require CC3D installed (the same conditional skip Phase 3 established; NOT a new
    always-on skip).

CC3D is run in an ISOLATED SUBPROCESS (``_cc3d_capture_subprocess.py``): the default
gate already runs CC3D once IN-PROCESS in ``phase3/tests/test_embryo_cc3d.py``, and a
SECOND in-process CC3D ``run_script.main`` triggers a native access violation (CC3D's
global CUDA/SWIG state is not cleanly re-initializable in one process). The subprocess
gives the Phase-7 CC3D run a fresh process so both cross-checks coexist in one pytest
run -- keeping this check IN the gate (no extra skip) per the deliverable-3 brief.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

# import path for the CC3D-importability probe (the runner itself runs in a subprocess)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3_TESTS = os.path.normpath(os.path.join(_HERE, "..", "..", "phase3", "tests"))
if _PHASE3_TESTS not in sys.path:
    sys.path.insert(0, _PHASE3_TESTS)

from embryo import EmbryoModel
from embryo.model import build_scaled_embryo
from embryo.params import DEFAULT as P

import cc3d_embryo_ref as REF


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
needs_cc3d = pytest.mark.skipif(not REF.cc3d_importable(), reason="cc3d not importable")

# NOTE: the longer-horizon offline CC3D closure ensemble is INTENTIONALLY not
# duplicated here -- the existing ``phase3/tests/test_embryo_cc3d.py::
# test_embryo_closure_and_intercalation_vs_cc3d_offline`` already covers it (one of the
# 3 sanctioned default skips), and the link physics it checks is backend-agnostic (the
# device link steppables reproduce the same link sets exactly, verified in
# test_device_link_steppables.py). Adding a second offline skip would push the
# sanctioned skip count past 3 for no new coverage.


def _capture_cc3d_subprocess(steps, capture, dim=100):
    """Run the CC3D Embryo capture in a fresh subprocess; return {mcs: snapshot}."""
    script = os.path.join(_HERE, "_cc3d_capture_subprocess.py")
    with tempfile.TemporaryDirectory(prefix="cc3d_p7_") as td:
        out = os.path.join(td, "snaps.npz")
        cmd = [sys.executable, script, out, str(steps),
               ",".join(str(m) for m in capture), str(dim)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(
                f"CC3D subprocess failed (rc={r.returncode}).\nSTDERR tail:\n"
                + "\n".join(r.stderr.splitlines()[-15:]))
        # fully materialize + CLOSE the NpzFile before the TemporaryDirectory cleanup
        # (Windows refuses to delete a still-open file handle).
        with np.load(out, allow_pickle=True) as data:
            snaps = {int(k.split("_")[1]): data[k].item()
                     for k in data.files if k.startswith("mcs_")}
            runtime = float(data["runtime"]) if "runtime" in data.files else float("nan")
    return snaps, runtime


def _link_kinds(links):
    lam = links._lam
    return dict(
        tissue=int(np.sum(np.abs(lam - P.tissue_lambda) < 1e-3)),
        lamellipodia=int(np.sum(np.abs(lam - P.lamellipodia_lambda) < 1e-3)),
        substrate=int(np.sum(np.abs(lam - P.slink_lambda) < 1e-3)),
    )


@cuda
@needs_cc3d
def test_device_embryo_link_inventory_matches_cc3d():
    """Run the REAL CC3D Embryo (full 100^3, isolated subprocess) a few MCS and compare
    its FPP link inventory + link-length distribution to the device-backend GPU
    EmbryoModel at the SAME MCS. The device link steppables (tissue cap-ordered relink /
    substrate min-id / Poisson keep-compact) + the dropped copyback must still track
    CC3D's tissue / lamellipodia / substrate creation + the resulting link-length
    distribution. A few MCS at full scale keeps this ~10s."""
    from scipy import stats
    capture = [0, 3]
    snaps, runtime = _capture_cc3d_subprocess(steps=4, capture=capture, dim=100)

    st, info = build_scaled_embryo(cube_size=100, seed=12345)
    m = EmbryoModel(st, link_backend="device")
    m.start()
    m.links.rebuild()
    gpu0 = _link_kinds(m.links)
    gpu0_len = m.links.active_link_lengths()
    m.run(3)
    gpu3 = _link_kinds(m.links)
    gpu3_total = m.links.num_active()
    gpu3_len = m.links.active_link_lengths()
    m.engine.assert_volume_partition()

    cc0, cc3 = snaps[0], snaps[3]
    cc0k, cc3k = cc0["link_kinds"], cc3["link_kinds"]
    cc0_len, cc3_len = cc0["link_lengths"], cc3["link_lengths"]

    msg = (
        f"\nCC3D runtime {runtime:.1f}s (device backend, subprocess)"
        f"\n mcs0: CC3D tissue={cc0k['tissue']} lam={cc0k['lamellipodia']} "
        f"sub={cc0k['substrate']} tot={len(cc0_len)} mean={cc0_len.mean():.3f}"
        f"\n       GPU  tissue={gpu0['tissue']} lam={gpu0['lamellipodia']} "
        f"sub={gpu0['substrate']} mean={gpu0_len.mean() if len(gpu0_len) else 0:.3f}"
        f"\n mcs3: CC3D tissue={cc3k['tissue']} lam={cc3k['lamellipodia']} "
        f"sub={cc3k['substrate']} tot={len(cc3_len)} mean={cc3_len.mean():.3f}"
        f"\n       GPU  tissue={gpu3['tissue']} lam={gpu3['lamellipodia']} "
        f"sub={gpu3['substrate']} tot={gpu3_total} mean={gpu3_len.mean() if len(gpu3_len) else 0:.3f}"
    )
    print(msg)

    # Tissue links: created from the order-1 neighbor relation == CC3D NeighborTracker;
    # the device cap-ordered relink reproduces the host set exactly, so at start they
    # match CC3D as closely as the host port did (a few %).
    assert abs(gpu0["tissue"] - cc0k["tissue"]) <= 0.05 * cc0k["tissue"] + 5, (
        f"initial tissue-link count off vs CC3D.{msg}")
    # Lamellipodia: one per leader that found a target; O(#leaders=60), stochastic.
    assert abs(gpu3["lamellipodia"] - cc3k["lamellipodia"]) <= 25, (
        f"lamellipodia-link count off vs CC3D.{msg}")
    # Substrate links grow over the first MCS in BOTH; same order of magnitude by mcs3.
    assert gpu3["substrate"] >= cc3k["substrate"] * 0.5, (
        f"substrate-link count far below CC3D.{msg}")
    # Total active link count within ~10% at mcs3.
    assert abs(gpu3_total - len(cc3_len)) <= 0.10 * len(cc3_len) + 20, (
        f"total link count off vs CC3D.{msg}")
    # Link-length DISTRIBUTION matches (KS) and the MEAN within a few %.
    ks0 = stats.ks_2samp(gpu0_len, cc0_len)
    mean_rel3 = abs(gpu3_len.mean() - cc3_len.mean()) / abs(cc3_len.mean())
    assert ks0.statistic < 0.25, f"initial link-length distribution differs (KS).{msg}"
    assert mean_rel3 < 0.06, f"link-length mean off vs CC3D at mcs3.{msg}"
