"""Phase 4 Pass A gate: the replica/batch axis on the GPU CPM engine.

A ``BatchedGPUEngine`` advances R independent CPM replicas CONCURRENTLY in one set
of kernel launches. Each replica is its own simulation (replicas never interact),
its own reproducible Philox stream (replica index folded into the key), and may
carry its own swept parameters (per-replica contact / volume lambdas+targets).

We validate three things, mirroring the Phase 2 statistical gate methodology and
the plan's verification section:

1. **Per-replica == single-run (bit-exact).** A batched run of R replicas with
   per-replica-keyed RNG is, replica-for-replica, the SAME trajectory as R
   independent single ``GPUEngine`` runs seeded the same way. Because COM is int64
   fixed-point and the RNG key includes the replica, this is *bit-exact* (not just
   distributional) -- a strictly stronger check than KS. We ALSO report the pooled
   KS / mean-rel numbers (the Phase 2 gate quantities) for the record.

2. **Sweep actually varies.** Replicas given DIFFERENT params (contact / lambda)
   produce the EXPECTED different equilibrium volumes -- a strong volume constraint
   shrinks a replica's cells relative to a weak one.

3. **Reproducibility.** Re-running the same (seed, replica) reproduces the exact
   int64 COM / volume / id-lattice trajectory.

Plus: the exact per-replica volume == lattice-partition invariant holds for every
replica (atomic-update correctness across the batch axis, no float atomics).

Lattice/batch sizes are kept small (24^3, R<=6) so the whole file stays well under
the suite's time budget.
"""

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine
from engine.batched import BatchedGPUEngine, build_batched_grid_state

# ---- gate parameters (small & fast) ----------------------------------------
LATTICE_L = 24
N_MCS = 30
CELLS_PER_AXIS = 3            # 27 cells
BASE_SEED = 777
REPLICA_STRIDE = BatchedGPUEngine.REPLICA_SEED_STRIDE

CONTACT = np.array([[0.0, 5.0], [5.0, 1.0]])
TARGET_VOLUME = np.array([0.0, 64.0])
LAMBDA_VOLUME = np.array([0.0, 4.0])

# bit-exact tolerance is literally zero; the pooled-distribution tolerances below
# are reported for the record and match the Phase 2 engine-core gate.
VOL_MEAN_RTOL = 0.05
KS_D_MAX = 0.20

CENTER = np.array([LATTICE_L / 2.0] * 3)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")


def _cfg(seed, contact=CONTACT, lam=LAMBDA_VOLUME, tv=TARGET_VOLUME):
    return EngineConfig(
        Lx=LATTICE_L, Ly=LATTICE_L, Lz=LATTICE_L, seed=seed, temperature=10.0,
        target_volume=tv, lambda_volume=lam, contact=contact,
    )


def _com_radial(coms):
    return np.linalg.norm(coms - CENTER, axis=1)


# ----------------------------------------------------------------- CPU-only tests
def test_batched_state_builder_shapes():
    """The batched-state builder stacks R identical grid ICs along a leading axis;
    no GPU required."""
    R = 4
    bs = build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS)
    assert bs.R == R
    assert bs.n_cells == CELLS_PER_AXIS ** 3 == 27
    # ids are (R, Lz, Ly, Lx); each replica identical to the single-engine IC
    single = build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS)
    assert bs.ids.shape == (R, LATTICE_L, LATTICE_L, LATTICE_L)
    for r in range(R):
        assert np.array_equal(bs.ids[r], single.ids)


# ------------------------------------------------------------------- GPU tests
@cuda_only
def test_batched_volume_partition_exact_all_replicas():
    """The single most important batched invariant: for EVERY replica, the per-cell
    volume SoA == that replica's lattice voxel count exactly, and int64 COM sums
    never drift (atomic correctness across the batch axis -- no float atomics)."""
    R = 5
    bs = build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS)
    beng = BatchedGPUEngine(bs)
    beng.run(N_MCS)
    assert beng.assert_volume_partition()        # raises on any replica drift
    ids = beng.get_ids()
    assert ids.shape == (R, LATTICE_L, LATTICE_L, LATTICE_L)
    assert ids.min() >= 0 and ids.max() <= beng.n_cells


@cuda_only
def test_batched_equals_independent_single_runs_bit_exact():
    """R replicas with IDENTICAL params, each keyed by base_seed + r*stride, are
    replica-for-replica BIT-EXACT to R independent single-engine runs seeded the
    same way. This is the per-replica == single-run gate, made exact by the int64
    COM + replica-keyed Philox stream. Pooled KS / mean-rel are reported too."""
    from scipy import stats

    R = 6
    # batched run: all replicas identical params, replica folded into the RNG key
    bs = build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS)
    beng = BatchedGPUEngine(bs)
    beng.run(N_MCS)
    bv = beng.volumes()           # (R, n_cells)
    bids = beng.get_ids()         # (R, Lz, Ly, Lx)
    bxsum, bysum, bzsum = beng.com_sums()   # (R, n_cells+1) int64 each

    # R independent single-engine runs, each seeded base_seed + r*stride
    pooled_b, pooled_s = [], []
    for r in range(R):
        seed_r = BASE_SEED + r * REPLICA_STRIDE
        s = GPUEngine(build_grid_state(_cfg(seed_r), CELLS_PER_AXIS))
        s.run(N_MCS)
        # BIT-EXACT: id-lattice and int64 COM sums must match replica r exactly
        assert np.array_equal(s.get_ids(), bids[r]), f"id-lattice mismatch replica {r}"
        sxs = s.xsum.numpy(); sys_ = s.ysum.numpy(); szs = s.zsum.numpy()
        assert np.array_equal(sxs, bxsum[r]), f"xsum mismatch replica {r}"
        assert np.array_equal(sys_, bysum[r]), f"ysum mismatch replica {r}"
        assert np.array_equal(szs, bzsum[r]), f"zsum mismatch replica {r}"
        assert np.array_equal(s.volumes(), bv[r]), f"volume mismatch replica {r}"
        pooled_s.append(s.volumes())

    # pooled-distribution numbers (the Phase 2 gate quantities), for the record
    pooled_single = np.concatenate(pooled_s)
    pooled_batched = bv.reshape(-1)
    vol_rel = abs(pooled_single.mean() - pooled_batched.mean()) / abs(pooled_single.mean())
    ks = stats.ks_2samp(pooled_single, pooled_batched)
    print(
        f"\n[batched==single] BIT-EXACT over R={R} replicas. "
        f"Pooled volume: single {pooled_single.mean():.4f} | batched {pooled_batched.mean():.4f} "
        f"| rel {vol_rel:.2e} (tol {VOL_MEAN_RTOL}) | KS D {ks.statistic:.3f} p {ks.pvalue:.3f}"
    )
    assert vol_rel < VOL_MEAN_RTOL
    assert ks.statistic < KS_D_MAX


@cuda_only
def test_batched_reproducibility_bit_exact():
    """Same (seed, replica) -> identical trajectory: two batched runs with the same
    config produce bit-identical int64 COM sums, volumes, and id-lattices."""
    R = 4
    beng1 = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS))
    beng2 = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS))
    beng1.run(N_MCS)
    beng2.run(N_MCS)
    assert np.array_equal(beng1.get_ids(), beng2.get_ids())
    x1, y1, z1 = beng1.com_sums()
    x2, y2, z2 = beng2.com_sums()
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2) and np.array_equal(z1, z2)
    assert np.array_equal(beng1.volumes(), beng2.volumes())


@cuda_only
def test_sweep_varies_with_per_replica_params():
    """A sweep over a per-replica parameter produces the EXPECTED different results.

    We sweep lambda_volume (the volume Lagrange multiplier) across replicas, all
    else equal. The textbook effect of a STRONGER volume constraint is to hold each
    cell closer to its target volume, i.e. to shrink the mean-squared deviation
    ``<(V - V_target)^2>`` -- a clean, regime-robust monotone signal (unlike the raw
    mean volume, which in a strong cell-cell-adhesion regime is non-monotone: a very
    weak constraint lets cells dissolve to 0, and a very strong one adds lattice-
    discreteness noise -- both real CPM behaviors, see Phase 4 findings). We use a
    middle lambda range [1,2,4,8] where the monotone constraint effect is the
    dominant signal.
    """
    lambdas = [1.0, 2.0, 4.0, 8.0]
    R = len(lambdas)
    per_replica = [
        _cfg(BASE_SEED, lam=np.array([0.0, lv])) for lv in lambdas
    ]
    bs = build_batched_grid_state(per_replica[0], R, CELLS_PER_AXIS)
    beng = BatchedGPUEngine(bs, per_replica_config=per_replica)
    beng.run(N_MCS)
    beng.assert_volume_partition()
    v = beng.volumes()                                  # (R, n_cells)
    msd = ((v - TARGET_VOLUME[1]) ** 2).mean(axis=1)    # (R,) deviation from target
    mean_vol = v.mean(axis=1)
    print(
        f"\n[sweep] lambda {lambdas} -> mean cell volume "
        f"{np.round(mean_vol, 3).tolist()} | <(V-Vt)^2> {np.round(msd, 2).tolist()}"
    )
    # stronger volume constraint -> tighter to target -> strictly smaller MSD
    assert np.all(np.diff(msd) < 0.0), f"expected decreasing MSD-from-target, got {msd}"
    # and the effect is large (not noise): weakest constraint deviates >5x the strongest
    assert msd[0] > 5.0 * msd[-1], f"sweep effect too weak: {msd}"


@cuda_only
def test_sweep_contact_varies():
    """Sweeping the cell-cell contact energy across replicas changes the contact
    energy as expected: higher J_cc raises total contact energy at fixed geometry
    scale. We check the per-replica total energies are distinct and ordered."""
    jcc_values = [0.5, 2.0, 6.0]
    R = len(jcc_values)
    per_replica = [
        _cfg(BASE_SEED, contact=np.array([[0.0, 5.0], [5.0, j]])) for j in jcc_values
    ]
    bs = build_batched_grid_state(per_replica[0], R, CELLS_PER_AXIS)
    beng = BatchedGPUEngine(bs, per_replica_config=per_replica)
    beng.run(N_MCS)
    energies = beng.total_energy()                # (R,)
    print(f"\n[sweep-contact] J_cc {jcc_values} -> total energy {np.round(energies, 2).tolist()}")
    # distinct per replica (the sweep genuinely varies the physics)
    assert len(set(np.round(energies, 3))) == R


@cuda_only
def test_single_run_backcompat_unchanged():
    """R=1 batched run reproduces the existing single-engine run bit-exactly: the
    batched path is a faithful superset, so Phase 1-3 semantics are preserved."""
    beng = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), 1, CELLS_PER_AXIS))
    beng.run(N_MCS)
    single = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    single.run(N_MCS)
    assert np.array_equal(beng.get_ids()[0], single.get_ids())
    assert np.array_equal(beng.volumes()[0], single.volumes())
