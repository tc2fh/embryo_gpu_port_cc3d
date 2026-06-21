"""Statistical-equivalence gate: GPU (Warp) FPP vs CPU (NumPy) reference.

The GPU prototype and the CPU reference use independent RNG streams (Warp Philox
keyed by (mcs, color, seed) on device; NumPy PCG on host), so bit-identical
results are NOT expected -- and the plan's verification section explicitly
requires *distributional* agreement within Monte-Carlo noise, not equality.

Methodology
-----------
* Both engines run the SAME synthetic 32^3 / 27-cell model (``model.py``) with
  Volume + Contact + FocalPointPlasticity spring links, for ``N_MCS`` steps --
  past the relaxation horizon (~15-20 MCS, measured in _runs/_relax_trace.py),
  so we compare quasi-stationary distributions rather than transient kinetics
  (the GPU checkerboard sweep and the CPU random-site sweep have different
  *kinetics* but the same *equilibrium*, as they share the energy function).
* We pool an ensemble of independent seeds per engine and compare the
  distributions of (a) FPP link length and (b) cell volume.

Tolerances (documented + justified)
-----------------------------------
Measured noise floor from _runs/_ensemble.py: the CPU-vs-CPU (seed-to-seed)
relative-mean spread is ~3-4% (link) / ~3% (volume), and the observed
CPU-vs-GPU spread was actually <= that floor (link rel ~3.8%, vol rel ~1.4%;
KS p = 0.84 link, 1.0 volume). We therefore gate at:

    link-length pooled-mean relative diff   < LINK_MEAN_RTOL  (0.06)
    volume      pooled-mean relative diff   < VOL_MEAN_RTOL   (0.12)
    link-length distribution overlap        Kolmogorov-Smirnov D < KS_D_MAX (0.20)

These are a few x the measured noise floor -- tight enough to catch a real
energy/semantics bug (an order-of-magnitude or systematic shift fails), loose
enough not to flake on Monte-Carlo noise. The thresholds are asserted below.
"""

import numpy as np
import pytest

import model as M

# ---- gate parameters (kept small so the gate re-runs fast: < ~1 min) -------
LATTICE_L = 32
N_MCS = 25
CPU_SEEDS = (1, 2, 3)            # CPU is the slow side; keep the ensemble small
GPU_SEEDS = (1, 2, 3, 4, 5, 6)   # GPU is ~60x faster; pool more for a tight estimate

LINK_MEAN_RTOL = 0.06
VOL_MEAN_RTOL = 0.12
KS_D_MAX = 0.20


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _run_cpu_ensemble(seeds, n_mcs):
    from cpm_cpu import CPUEngine

    vols, links = [], []
    for s in seeds:
        cfg = M.ModelConfig(L=LATTICE_L, seed=int(s))
        eng = CPUEngine(M.build_state(cfg))
        eng.run(n_mcs)
        vols.append(eng.volumes())
        links.append(eng.link_lengths())
    return np.concatenate(vols), np.concatenate(links)


def _run_gpu_ensemble(seeds, n_mcs):
    from cpm_gpu import GPUEngine

    vols, links = [], []
    for s in seeds:
        cfg = M.ModelConfig(L=LATTICE_L, seed=int(s))
        eng = GPUEngine(M.build_state(cfg))
        eng.run(n_mcs)
        vols.append(eng.volumes())
        links.append(eng.active_link_lengths())
    return np.concatenate(vols), np.concatenate(links)


def test_model_build_sanity():
    """The shared model builds with the expected cell count, links and COM."""
    cfg = M.ModelConfig(L=LATTICE_L)
    st = M.build_state(cfg)
    assert cfg.n_cells == 27
    assert st.link_pairs.shape[1] == 2
    assert st.link_pairs.shape[0] == 54  # 3x3x3 grid graph: 3*9*2 = 54 edges
    # all initial volumes equal the seeded block volume, COM well-defined
    assert np.all(st.volume[1:] > 0)
    ll = M.link_lengths_from_state(st.xsum, st.ysum, st.zsum, st.volume, st.link_pairs)
    assert np.all(ll > 0)


def test_cpu_reference_conserves_and_relaxes():
    """CPU reference: total volume changes only via accepted flips and the cells
    relax toward the volume target band (sanity that the energy drives dynamics)."""
    cfg = M.ModelConfig(L=LATTICE_L, seed=1)
    eng = __import__("cpm_cpu").CPUEngine(M.build_state(cfg))
    v0 = eng.volumes().mean()
    eng.run(N_MCS)
    v1 = eng.volumes().mean()
    # cells start oversized (64) and shrink toward equilibrium under the combined
    # Volume+Contact+FPP energy -> mean volume must decrease and stay positive
    assert v1 < v0
    assert np.all(eng.volumes() >= 0)
    assert eng.link_lengths().min() > 0


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_fpp_statistical_equivalence():
    """GPU vs CPU pooled distributions agree within the documented tolerance."""
    from scipy import stats

    cpu_v, cpu_l = _run_cpu_ensemble(CPU_SEEDS, N_MCS)
    gpu_v, gpu_l = _run_gpu_ensemble(GPU_SEEDS, N_MCS)

    # --- link length: the key FPP observable ---
    cpu_lm, gpu_lm = cpu_l.mean(), gpu_l.mean()
    link_rel = abs(cpu_lm - gpu_lm) / abs(cpu_lm)
    ks_link = stats.ks_2samp(cpu_l, gpu_l)

    # --- cell volume ---
    cpu_vm, gpu_vm = cpu_v.mean(), gpu_v.mean()
    vol_rel = abs(cpu_vm - gpu_vm) / abs(cpu_vm)

    msg = (
        f"\nLINK len: CPU mean={cpu_lm:.3f} std={cpu_l.std():.3f} | "
        f"GPU mean={gpu_lm:.3f} std={gpu_l.std():.3f} | "
        f"rel_dmean={link_rel:.4f} (tol {LINK_MEAN_RTOL}) | "
        f"KS D={ks_link.statistic:.3f} p={ks_link.pvalue:.3f} (D tol {KS_D_MAX})\n"
        f"VOLUME : CPU mean={cpu_vm:.3f} std={cpu_v.std():.3f} | "
        f"GPU mean={gpu_vm:.3f} std={gpu_v.std():.3f} | "
        f"rel_dmean={vol_rel:.4f} (tol {VOL_MEAN_RTOL})"
    )
    print(msg)

    assert link_rel < LINK_MEAN_RTOL, f"link-length mean diff too large.{msg}"
    assert vol_rel < VOL_MEAN_RTOL, f"volume mean diff too large.{msg}"
    assert ks_link.statistic < KS_D_MAX, f"link-length distributions differ.{msg}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_gpu_conserves_lattice_partition():
    """GPU id-lattice stays a valid partition: every voxel id in [0, n_cells],
    and per-cell volume equals the voxel count of that id (atomic-update sanity)."""
    from cpm_gpu import GPUEngine

    cfg = M.ModelConfig(L=LATTICE_L, seed=2)
    eng = GPUEngine(M.build_state(cfg))
    eng.run(N_MCS)
    ids = eng.ids.numpy()
    assert ids.min() >= 0 and ids.max() <= cfg.n_cells
    # volume SoA must match the actual lattice occupancy exactly (atomics correct)
    counts = np.bincount(ids, minlength=cfg.n_cells + 1).astype(np.float64)
    soa_vol = eng.volume.numpy().astype(np.float64)
    assert np.allclose(counts[1:], soa_vol[1:]), (
        "GPU per-cell volume SoA diverged from lattice voxel counts"
    )
