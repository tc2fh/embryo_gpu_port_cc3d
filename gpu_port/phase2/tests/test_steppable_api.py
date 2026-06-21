"""GPU steppable API gate: the cell.dict-equivalent SoA registry and the worked
non-FPP example (FloorFreeAreaSteppable) run end-to-end on device.

This validates the API seam that Phase 3 (GPU FPP + cohesotaxis) will plug into:
per-cell SoA scratch arrays + a Warp kernel computing a steppable observable +
a GPU reduction, scheduled around the engine MCS loop -- no per-object host calls.
"""

import numpy as np
import pytest

from engine import EngineConfig, state_from_id_lattice, GPUEngine
from engine.steppables import CellDict, GPUSteppable, SteppableManager, FloorFreeAreaSteppable


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_cell_dict_roundtrip():
    """The cell.dict-equivalent SoA registry stores/retrieves per-cell fields."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16)
    eng = GPUEngine(__import__("engine").build_grid_state(cfg, 2))
    cd = CellDict(eng.n_cells, device=eng.device)
    cd.register("LinkTime", "int32", 0)
    cd.register("stiffness", "float32", 1.5)
    assert cd.get("LinkTime").shape[0] == eng.n_cells + 1
    assert np.all(cd.get("LinkTime") == 0)
    assert np.allclose(cd.get("stiffness"), 1.5)
    vals = np.arange(eng.n_cells + 1, dtype=np.int32)
    cd.set("LinkTime", vals)
    assert np.array_equal(cd.get("LinkTime"), vals)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_floor_free_area_steppable_end_to_end():
    """Worked non-FPP example: a substrate floor under an exposed region. The
    GPU steppable counts exposed substrate voxels (Medium directly above) each MCS.
    Compared against an explicit NumPy recompute on the same lattice."""
    L = 20
    ids = np.zeros((L, L, L), dtype=np.int32)
    # a flat substrate floor at z=0 (each voxel its own 1-voxel substrate cell,
    # like the Embryo hollow sphere), partially covered by one big cell above.
    next_id = 1
    cell_type = [0]
    for y in range(L):
        for x in range(L):
            ids[0, y, x] = next_id
            cell_type.append(4)  # SUBSTRATE
            next_id += 1
    # a covering cell over a 6x6 patch at z=1 (covers those substrate voxels)
    cover_id = next_id
    cell_type.append(1)  # LEADING
    ids[1, 5:11, 5:11] = cover_id
    cell_type = np.array(cell_type, dtype=np.int32)

    n_types = 5
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=n_types,
        target_volume=np.array([0.0, 125.0, 125.0, 125.0, 1.0]),
        lambda_volume=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
        contact=np.full((n_types, n_types), 10.0),
        frozen=np.array([4], dtype=np.int32),
    )
    state = state_from_id_lattice(cfg, ids, cell_type)
    eng = GPUEngine(state)

    mgr = SteppableManager(eng)
    ffa = mgr.register(FloorFreeAreaSteppable(eng, substrate_type=4, frequency=1))
    mgr.start()

    # at mcs 0 (before any flips): exposed substrate voxels = floor minus covered
    free0 = ffa.step(0)
    covered = 36  # 6x6 patch directly above floor
    assert free0 == L * L - covered == 400 - 36 == 364

    # explicit NumPy recompute on the current lattice agrees
    cur = eng.get_ids()
    exposed = 0
    for y in range(L):
        for x in range(L):
            cid = cur[0, y, x]
            if cid != 0 and cell_type[cid] == 4:
                above = cur[1, y, x] if 1 < L else 0
                if above == 0:
                    exposed += 1
    assert free0 == exposed

    # run a few MCS via the manager; history accumulates and stays in range
    mgr.run(3)
    assert len(ffa.history) >= 3
    for _, fa in ffa.history:
        assert 0 <= fa <= L * L


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_steppable_manager_runs_engine():
    """The manager advances the engine MCS and preserves the partition invariant."""
    cfg = EngineConfig(Lx=18, Ly=18, Lz=18, seed=4,
                       target_volume=np.array([0.0, 64.0]),
                       lambda_volume=np.array([0.0, 4.0]),
                       contact=np.array([[0.0, 5.0], [5.0, 1.0]]))
    eng = GPUEngine(__import__("engine").build_grid_state(cfg, 3))

    class _Recorder(GPUSteppable):
        def __init__(self, engine):
            super().__init__(engine)
            self.seen = []

        def step(self, mcs):
            self.seen.append(mcs)

    mgr = SteppableManager(eng)
    rec = mgr.register(_Recorder(eng))
    mgr.start()
    mgr.run(5)
    assert rec.seen == [0, 1, 2, 3, 4]
    assert eng.assert_volume_partition()
