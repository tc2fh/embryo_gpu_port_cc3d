"""Engine-core gate: Volume + Contact statistical equivalence (GPU vs CPU) and the
exact per-cell volume == lattice-partition invariant.

The GPU engine (8-color checkerboard Philox sweep) and the CPU reference (random
single-site PCG sweep) share the SAME energy function but have different RNG
streams and different *kinetics*, so the plan requires **distributional** agreement
within Monte-Carlo noise -- not bit-identical results.

Methodology
-----------
* Both run the same small (24^3, 27-cell) grid model with Volume + Contact in a
  STABLE adhesive regime: cell-cell adhesion (J_cc=1 < J_mc=5) keeps cells
  aggregated at a finite equilibrium volume; a strong volume constraint (lambda=4)
  holds them near target. This gives a quasi-stationary distribution to compare
  (rather than a runaway transient).
* We pool an ensemble of independent seeds per engine and compare the pooled
  distributions of cell volume, surface area, total energy, and COM radial
  displacement.

Tolerances (calibrated in _runs/_calibrate.py; a few x the measured noise floor)
--------------------------------------------------------------------------------
Measured seed-to-seed (CPU-vs-CPU) noise floor: vol ~0.15%, surf ~1.2%, E ~0.2%.
Observed CPU-vs-GPU spread was at that floor (vol 0.20%, surf 1.5%, E 0.15%; KS
p ~ 0.5-0.8). We gate at:

    volume  pooled-mean rel diff < VOL_MEAN_RTOL  (0.05)
    surface pooled-mean rel diff < SURF_MEAN_RTOL (0.08)
    energy  pooled-mean rel diff < E_MEAN_RTOL    (0.03)
    COM     pooled-mean rel diff < COM_MEAN_RTOL  (0.05)
    volume  distribution KS D     < KS_D_MAX       (0.20)

Tight enough to catch a real energy/semantics bug (an order-of-magnitude or
systematic shift fails), loose enough not to flake on Monte-Carlo noise.
"""

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine, CPUReference

# ---- gate parameters (kept small so the gate runs fast) --------------------
LATTICE_L = 24
N_MCS = 40
CELLS_PER_AXIS = 3
CPU_SEEDS = (1, 2, 3)
GPU_SEEDS = (1, 2, 3, 4, 5, 6)

CONTACT = np.array([[0.0, 5.0], [5.0, 1.0]])
TARGET_VOLUME = np.array([0.0, 64.0])
LAMBDA_VOLUME = np.array([0.0, 4.0])

VOL_MEAN_RTOL = 0.05
SURF_MEAN_RTOL = 0.08
E_MEAN_RTOL = 0.03
COM_MEAN_RTOL = 0.05
KS_D_MAX = 0.20

CENTER = np.array([LATTICE_L / 2.0] * 3)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cfg(seed):
    return EngineConfig(
        Lx=LATTICE_L, Ly=LATTICE_L, Lz=LATTICE_L, seed=seed, temperature=10.0,
        target_volume=TARGET_VOLUME, lambda_volume=LAMBDA_VOLUME, contact=CONTACT,
    )


def _com_radial(coms):
    return np.linalg.norm(coms - CENTER, axis=1)


def _run_cpu(seeds):
    V, S, E, R = [], [], [], []
    for s in seeds:
        e = CPUReference(build_grid_state(_cfg(int(s)), CELLS_PER_AXIS))
        e.run(N_MCS)
        V.append(e.volumes()); S.append(e.surface_areas())
        E.append(e.total_energy()); R.append(_com_radial(e.coms()))
    return np.concatenate(V), np.concatenate(S), np.array(E), np.concatenate(R)


def _run_gpu(seeds):
    V, S, E, R = [], [], [], []
    for s in seeds:
        e = GPUEngine(build_grid_state(_cfg(int(s)), CELLS_PER_AXIS))
        e.run(N_MCS)
        e.assert_volume_partition()  # exact partition invariant after the run
        V.append(e.volumes()); S.append(e.surface_areas())
        E.append(e.total_energy()); R.append(_com_radial(e.coms()))
    return np.concatenate(V), np.concatenate(S), np.array(E), np.concatenate(R)


# --------------------------------------------------------------- CPU-only tests
def test_config_and_state_build():
    cfg = _cfg(1)
    st = build_grid_state(cfg, CELLS_PER_AXIS)
    assert st.n_cells == CELLS_PER_AXIS ** 3 == 27
    # exact volume == voxel count from the lattice
    counts = np.bincount(st.ids.reshape(-1), minlength=st.n_cells + 1)
    assert np.array_equal(counts[1:], st.volume[1:].astype(np.int64))
    # COM well-defined and inside the lattice
    coms = st.coms()[1:]
    assert np.all(coms >= 0) and np.all(coms < LATTICE_L)


def test_cpu_reference_relaxes_and_conserves():
    """CPU reference: cells relax toward a finite equilibrium and stay a valid
    nonnegative partition (sanity that the energy drives dynamics)."""
    e = CPUReference(build_grid_state(_cfg(1), CELLS_PER_AXIS))
    v0 = e.volumes().mean()
    e.run(N_MCS)
    v1 = e.volumes().mean()
    assert v1 < v0                      # oversized-vs-equilibrium cells shrink a bit
    assert e.volumes().min() >= 0
    # total volume conservation: sum of cell volumes == occupied voxel count
    occupied = int(np.count_nonzero(e.ids))
    assert int(e.volumes().sum()) == occupied


# ------------------------------------------------------------------- GPU tests
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_volume_partition_exact():
    """The single most important GPU invariant: per-cell volume SoA == lattice
    voxel count EXACTLY, and the int64 COM sums never drift from a fresh recompute
    (atomic-update correctness, even with the parallel checkerboard)."""
    eng = GPUEngine(build_grid_state(_cfg(2), CELLS_PER_AXIS))
    eng.run(N_MCS)
    assert eng.assert_volume_partition()
    ids = eng.get_ids()
    assert ids.min() >= 0 and ids.max() <= eng.n_cells


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_volume_contact_statistical_equivalence():
    """GPU vs CPU pooled distributions of volume, surface, energy and COM agree
    within the documented Monte-Carlo tolerance."""
    from scipy import stats

    cpu_v, cpu_s, cpu_e, cpu_r = _run_cpu(CPU_SEEDS)
    gpu_v, gpu_s, gpu_e, gpu_r = _run_gpu(GPU_SEEDS)

    vol_rel = abs(cpu_v.mean() - gpu_v.mean()) / abs(cpu_v.mean())
    surf_rel = abs(cpu_s.mean() - gpu_s.mean()) / abs(cpu_s.mean())
    e_rel = abs(cpu_e.mean() - gpu_e.mean()) / abs(cpu_e.mean())
    com_rel = abs(cpu_r.mean() - gpu_r.mean()) / abs(cpu_r.mean())
    ks_vol = stats.ks_2samp(cpu_v, gpu_v)

    msg = (
        f"\nVOLUME : CPU {cpu_v.mean():.3f}+/-{cpu_v.std():.3f} | "
        f"GPU {gpu_v.mean():.3f}+/-{gpu_v.std():.3f} | rel {vol_rel:.4f} (tol {VOL_MEAN_RTOL}) | "
        f"KS D {ks_vol.statistic:.3f} p {ks_vol.pvalue:.3f} (D tol {KS_D_MAX})\n"
        f"SURFACE: CPU {cpu_s.mean():.3f} | GPU {gpu_s.mean():.3f} | rel {surf_rel:.4f} (tol {SURF_MEAN_RTOL})\n"
        f"ENERGY : CPU {cpu_e.mean():.1f} | GPU {gpu_e.mean():.1f} | rel {e_rel:.4f} (tol {E_MEAN_RTOL})\n"
        f"COM|r| : CPU {cpu_r.mean():.3f} | GPU {gpu_r.mean():.3f} | rel {com_rel:.4f} (tol {COM_MEAN_RTOL})"
    )
    print(msg)

    assert vol_rel < VOL_MEAN_RTOL, f"volume mean diff too large.{msg}"
    assert surf_rel < SURF_MEAN_RTOL, f"surface mean diff too large.{msg}"
    assert e_rel < E_MEAN_RTOL, f"energy mean diff too large.{msg}"
    assert com_rel < COM_MEAN_RTOL, f"COM mean diff too large.{msg}"
    assert ks_vol.statistic < KS_D_MAX, f"volume distributions differ.{msg}"
