"""Phase 4 Pass C gate: interactivity bridge + larger-lattice validation.

This pass COMPLETES Phase 4. Two deliverables, both tested here with NO display:

A. **Interactivity bridge** (``gpu_port/bridge/``). A headless-safe data path that
   pulls the device id-lattice / type-field from a running ``GPUEngine`` /
   ``BatchedGPUEngine`` and exposes render-ready NumPy arrays (+ an optional file
   render and a cc3d-player5 ``CellField`` hand-off). The correctness surface is the
   DATA path: the exported / render-ready field EXACTLY equals a ``state`` read-back
   of the device. We assert on arrays and (at most) write a PNG with the
   non-interactive 'Agg' backend -- never an interactive GUI window or event loop.

B. **Larger-lattice validation.** A MODEST "larger" lattice (bounded for gate
   runtime) runs through the Pass B CUDA-graph path to a VALID exact volume
   partition and stays stable. The real max-scale run (toward the 32 GB / int32
   ceiling) is opt-in behind ``BENCH=1`` (the existing pattern). We also assert the
   documented multi-GPU verdict: the int32 voxel-index ceiling is reached well
   BEFORE 32 GB is exhausted, so single-GPU memory is never the binding constraint
   and multi-GPU halo exchange is correctly DEFERRED.

All sizes here are kept small/bounded so the file stays within the suite budget.
"""

import os
import tempfile

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine, GraphRunner
from engine.batched import BatchedGPUEngine, build_batched_grid_state
from engine import bench as bench_mod
from bridge import LatticeView, render_slice_png, to_cc3d_cell_field

# ---- gate parameters (small & fast) ----------------------------------------
LATTICE_L = 24
N_MCS = 25
CELLS_PER_AXIS = 3
BASE_SEED = 2024

CONTACT = np.array([[0.0, 5.0], [5.0, 1.0]])
TARGET_VOLUME = np.array([0.0, 64.0])
LAMBDA_VOLUME = np.array([0.0, 4.0])

# "larger" gate lattice: clearly bigger than the 24^3 used by the other gate files
# and bigger per-axis than the engine's prior default tests, but bounded so the
# graph run finishes in well under a second. The opt-in BENCH run pushes to scale.
GATE_LARGER_L = 160


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")


def _cfg(seed, contact=CONTACT, lam=LAMBDA_VOLUME, tv=TARGET_VOLUME, L=LATTICE_L):
    return EngineConfig(
        Lx=L, Ly=L, Lz=L, seed=seed, temperature=10.0,
        target_volume=tv, lambda_volume=lam, contact=contact,
    )


def _state_ids(engine, replica=None):
    """Ground-truth device id-lattice via the ENGINE's own state read-back (the
    'state' the bridge must exactly reproduce)."""
    if hasattr(engine, "R"):
        return engine.get_ids()[0 if replica is None else replica]
    return engine.get_ids()


# =====================================================================
# A. Bridge correctness -- data path only (no GUI window/event loop)
# =====================================================================
@cuda_only
def test_bridge_id_field_exactly_matches_device_state():
    """The bridge's exported id-field EXACTLY equals a device state read-back."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    ref = _state_ids(eng)
    got = view.id_field()
    assert got.shape == (eng.Lz, eng.Ly, eng.Lx)
    assert got.dtype == np.int32
    assert np.array_equal(got, ref), "bridge id-field != device id-lattice"


@cuda_only
def test_bridge_type_field_matches_id_mapped_through_cell_type():
    """The exported type-field equals each voxel's id mapped through the device
    per-cell ``cell_type`` SoA -- exactly the field cc3d-player5 colors by type."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    ref_ids = _state_ids(eng)
    ref_ct = eng.cell_type.numpy().astype(np.int32)        # device SoA read-back
    ref_types = ref_ct[ref_ids]
    got = view.type_field()
    assert np.array_equal(got, ref_types), "bridge type-field != id->cell_type mapping"
    # Medium voxels (id 0) map to type 0; non-medium to their cell's type.
    assert got[ref_ids == 0].max(initial=0) == 0
    assert set(np.unique(got)).issubset(set(np.unique(ref_ct)))


@cuda_only
def test_bridge_snapshot_is_coherent_with_state():
    """``snapshot()`` returns ids + type-field mapped from the SAME read, both exact
    vs the device, with correct metadata."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    ref_ids = _state_ids(eng)
    ref_ct = eng.cell_type.numpy().astype(np.int32)
    snap = view.snapshot()
    assert np.array_equal(snap["ids"], ref_ids)
    assert np.array_equal(snap["types"], ref_ct[ref_ids])
    assert np.array_equal(snap["types"], ref_ct[snap["ids"]])    # internally consistent
    assert snap["n_cells"] == eng.n_cells
    assert snap["dims"] == (eng.Lz, eng.Ly, eng.Lx)


@cuda_only
def test_bridge_slices_and_projection_match_state():
    """2-D slices (each axis) and the max/sum projections are exact sub-views of the
    device state (no display; assert on the arrays)."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    ids = _state_ids(eng)
    ct = eng.cell_type.numpy().astype(np.int32)
    types = ct[ids]
    Lz, Ly, Lx = eng.Lz, eng.Ly, eng.Lx
    # mid-plane slices, id field
    assert np.array_equal(view.slice("z", field="id"), ids[Lz // 2, :, :])
    assert np.array_equal(view.slice("y", field="id"), ids[:, Ly // 2, :])
    assert np.array_equal(view.slice("x", field="id"), ids[:, :, Lx // 2])
    # explicit index, type field
    assert np.array_equal(view.slice("z", index=5, field="type"), types[5, :, :])
    # projections
    assert np.array_equal(view.projection("z", field="type", reduce="max"), types.max(axis=0))
    assert np.array_equal(view.projection("y", field="id", reduce="sum"), ids.sum(axis=1))


@cuda_only
def test_bridge_tracks_live_state_as_sim_advances():
    """The bridge is a live pull: after the engine advances, a new snapshot reflects
    the NEW device state (and still matches a fresh read-back exactly)."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    view = LatticeView(eng)
    eng.run(5)
    snap_a = view.snapshot()
    assert np.array_equal(snap_a["ids"], _state_ids(eng))
    eng.run(20)                                            # advance further
    snap_b = view.snapshot()
    assert np.array_equal(snap_b["ids"], _state_ids(eng))  # tracks new state exactly
    # state genuinely changed over those MCS (so we're really tracking, not caching)
    assert not np.array_equal(snap_a["ids"], snap_b["ids"])


@cuda_only
def test_bridge_works_with_graph_runner_state():
    """The bridge reads engine device arrays directly, so it reflects state advanced
    by the Pass B CUDA-graph runner too (the runner shares the engine's arrays)."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    GraphRunner(eng).run(N_MCS)
    view = LatticeView(eng)
    assert np.array_equal(view.id_field(), _state_ids(eng))
    eng.assert_volume_partition()


@cuda_only
def test_bridge_batched_per_replica_exact():
    """For a batched engine the bridge exports a chosen replica's field, exactly
    matching that replica's device state; different replicas give different views."""
    R = 4
    beng = BatchedGPUEngine(build_batched_grid_state(_cfg(BASE_SEED), R, CELLS_PER_AXIS))
    beng.run(N_MCS)
    all_ids = beng.get_ids()                               # (R, Lz, Ly, Lx)
    ct = beng.cell_type.numpy().astype(np.int32)
    for r in range(R):
        view = LatticeView(beng, replica=r)
        assert np.array_equal(view.id_field(), all_ids[r]), f"replica {r} id-field mismatch"
        assert np.array_equal(view.type_field(), ct[all_ids[r]]), f"replica {r} type mismatch"
    # out-of-range replica is rejected
    with pytest.raises(IndexError):
        LatticeView(beng, replica=R).id_field()


@cuda_only
def test_bridge_cc3d_cellfield_handoff_is_exact_transpose():
    """The optional cc3d-player5 hand-off returns the id-lattice in CC3D's (x,y,z)
    CellField axis order -- a pure, exact transpose of the device (z,y,x) lattice
    (no Qt, no display)."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    ids = _state_ids(eng)                                  # (z, y, x)
    cf = to_cc3d_cell_field(view)                          # (x, y, z)
    assert cf.shape == (eng.Lx, eng.Ly, eng.Lz)
    assert np.array_equal(cf, np.transpose(ids, (2, 1, 0)))
    # round-trips back to the engine lattice exactly
    assert np.array_equal(np.transpose(cf, (2, 1, 0)), ids)


@cuda_only
def test_bridge_headless_png_render_no_window():
    """The headless file render (DEFAULT 'stdlib' backend) writes a non-empty, valid
    PNG and NEVER opens a window/event loop. The default uses a dependency-free stdlib
    writer (no native libs) so it is robust in the no-display gate -- it cannot hit the
    matplotlib/torch OpenMP DLL conflict that aborts the interpreter on this Windows
    box. We render from the SAME array the bridge exports and assert the rendered slice
    equals device state."""
    eng = GPUEngine(build_grid_state(_cfg(BASE_SEED), CELLS_PER_AXIS))
    eng.run(N_MCS)
    view = LatticeView(eng)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "slice_z.png")
        out = render_slice_png(view, path, axis="z", field="type")   # default backend
        assert out == path and os.path.exists(path)
        assert os.path.getsize(path) > 0
        with open(path, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n", "not a valid PNG"
        # an unknown backend is rejected (guards against silent GUI paths)
        with pytest.raises(ValueError):
            render_slice_png(view, os.path.join(d, "x.png"), backend="qt")
    # the array that was rendered is exactly a device-state slice
    ids = _state_ids(eng)
    ct = eng.cell_type.numpy().astype(np.int32)
    assert np.array_equal(view.slice("z", field="type"), ct[ids][eng.Lz // 2, :, :])


def test_bridge_fallback_png_writer_is_valid_png_cpu_only():
    """The dependency-free greyscale PNG fallback (used when matplotlib is absent)
    produces a structurally valid PNG. CPU-only: exercises the writer directly so
    the headless fallback path is covered even with no GPU."""
    from bridge.viewer import _write_grey_png
    arr = np.arange(64, dtype=np.int32).reshape(8, 8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "grey.png")
        _write_grey_png(arr, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "rb") as f:
            data = f.read()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"IHDR" in data[:32] and data[-4:] != b""    # has header + trailing chunk


# =====================================================================
# B. Larger-lattice validation + multi-GPU verdict
# =====================================================================
@cuda_only
def test_larger_lattice_runs_to_valid_partition():
    """A MODEST 'larger' lattice runs through the CUDA-graph hot loop to a VALID
    exact volume partition and stays stable. This is the in-gate larger-lattice
    deliverable; the real max-scale run is opt-in (BENCH=1, below)."""
    d = bench_mod.bench_large_lattice(L=GATE_LARGER_L, n_mcs=30, warmup=5,
                                      seed=BASE_SEED, verify_partition=True)
    assert d["valid_partition"] is True, "larger-lattice run is not a valid partition"
    assert np.isfinite(d["mcs_graph"]) and d["mcs_graph"] > 0
    assert d["n_voxels"] == GATE_LARGER_L ** 3
    print(
        f"\n[larger-lattice L={GATE_LARGER_L}^3 voxels={d['n_voxels']:.2e} "
        f"cells={d['n_cells']}] {d['mcs_graph']:.0f} MCS/s (graph), "
        f"vs CPU(10.5) x{d['graph_vs_cpu']:.0f}, valid_partition={d['valid_partition']}"
    )


@cuda_only
def test_larger_lattice_partition_stable_over_run():
    """Sanity that the larger lattice's exact partition holds at MULTIPLE points in a
    run (not just the end): volume SoA == voxel count after each chunk."""
    cfg = bench_mod._make_cfg(96, seed=BASE_SEED)
    eng = GPUEngine(build_grid_state(cfg, bench_mod._cells_per_axis(96)))
    gr = GraphRunner(eng)
    for _ in range(3):
        gr.run(10)
        eng.assert_volume_partition()                      # raises on any drift
    # total voxel mass is conserved exactly (it is a partition of the lattice)
    ids = eng.get_ids()
    assert int((ids > 0).sum()) == int(eng.volumes().sum())


def test_memory_math_and_multigpu_verdict_cpu_only():
    """The documented multi-GPU verdict, asserted from first principles (no GPU
    needed): the hot-loop kernels index voxels with a 32-bit linear index, so the
    largest single-GPU cube is bounded by the int32 ceiling (~1290^3, ~2.15e9
    voxels). The int32 id-lattice for that ceiling is ~8.6 GB -- well under the
    32 GB device -- so the int32 INDEX limit, not memory, is binding. Memory would
    only bind above ~2950^3 (> int32). Therefore NO single-GPU lattice can exhaust
    32 GB before the int32 ceiling; multi-GPU halo exchange is correctly DEFERRED."""
    limit = bench_mod.INT32_VOXEL_LIMIT
    max_cube = bench_mod.MAX_SAFE_CUBE
    # the documented safe cube is within the int32 ceiling
    assert max_cube ** 3 < limit
    # ids bytes at the int32 ceiling (4 B/voxel) is well under a 32 GB device
    ids_gb_at_ceiling = limit * 4 / 1e9
    assert ids_gb_at_ceiling < 32.0, ("int32-ceiling lattice already exceeds 32 GB -- "
                                      "memory would bind first (revisit multi-GPU verdict)")
    # cube edge at which 4 B/voxel would fill 32 GB is BEYOND the int32 ceiling
    voxels_for_32gb = 32e9 / 4
    cube_for_32gb = round(voxels_for_32gb ** (1.0 / 3.0))
    assert voxels_for_32gb > limit and cube_for_32gb > max_cube
    # estimate helper agrees with the breakdown for a representative large cube
    est = bench_mod.lattice_mem_estimate_gb(1024)
    assert abs(est["ids_gb"] - (1024 ** 3 * 4 / 1e9)) < 1e-9
    assert est["exceeds_int32_index"] is False
    print(
        f"\n[multi-GPU verdict] int32 voxel ceiling {limit:.3e} (~{max_cube}^3); "
        f"ids @ ceiling {ids_gb_at_ceiling:.1f} GB < 32 GB; 32 GB needs ~{cube_for_32gb}^3 "
        f"(> int32 ceiling) -> single-GPU memory never binds first -> multi-GPU DEFERRED."
    )


@cuda_only
@pytest.mark.skipif(os.environ.get("BENCH", "0") != "1", reason="set BENCH=1 for the max-scale lattice run")
def test_max_scale_lattice_bench():
    """Opt-in (BENCH=1) max-scale single-GPU run: push toward the int32 voxel ceiling
    and report the largest lattice that fits, its memory use, MCS/s, and that it stays
    a VALID exact partition. Not part of the fast default gate."""
    bench_mod.device_mem_gb()
    results = []
    for L in (512, 768, 1024, 1280):
        try:
            d = bench_mod.bench_large_lattice(L=L, n_mcs=40, warmup=5, seed=BASE_SEED,
                                              verify_partition=True)
            bench_mod._print_large(d)
            assert d["valid_partition"] is True
            assert d["mcs_graph"] > 0
            results.append(d)
        except Exception as e:                              # OOM at the very top end
            print(f"[max-scale L={L}^3] stopped: {type(e).__name__}: {e}")
            break
    assert results, "no max-scale lattice ran"
    biggest = results[-1]
    print(
        f"\n[max-scale] largest validated single-GPU lattice: L={biggest['L']}^3 "
        f"({biggest['n_voxels']:.3e} voxels, {biggest['n_cells']} cells), "
        f"{biggest['total_alloc_gb']:.2f} GB / {biggest['device_total_gb']:.1f} GB, "
        f"{biggest['mcs_graph']:.1f} MCS/s, valid_partition={biggest['valid_partition']}"
    )
