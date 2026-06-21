"""Geometry gate: the Embryo non-FPP start() constructors reproduce the expected
initial condition (cell counts + shell shape), and match the ACTUAL reference
functions in EmbryoSteppables.py voxel-for-voxel.

These tests are CPU-only (host-side geometry init); they do not require CUDA.
"""

import os
import re
import math

import numpy as np

from engine import geometry as G
from engine import EngineConfig, state_from_id_lattice


REF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "Embryo_Model_dev", "Embryo", "Simulation", "EmbryoSteppables.py",
)


# --------------------------------------------------------- mock CC3D surface
class _Cell:
    def __init__(self, cid, ctype):
        self.id = cid
        self.type = ctype


class _Field:
    def __init__(self, owner):
        self.o = owner

    def __setitem__(self, key, cell):
        x, y, z = key
        n = self.o.n
        if isinstance(x, slice):
            self.o.ids[z, y, x] = cell.id
        elif 0 <= x < n and 0 <= y < n and 0 <= z < n:
            self.o.ids[z, y, x] = cell.id


class _MockSelf:
    LEADING, PASSIVE, RING, SUBSTRATE, MEDIUM = 1, 2, 3, 4, 0

    def __init__(self, n):
        self.n = n
        self.ids = np.zeros((n, n, n), dtype=np.int32)
        self.types = [0]
        self._next = 1

    def new_cell(self, ctype):
        cid = self._next
        self._next += 1
        self.types.append(int(ctype))
        return _Cell(cid, int(ctype))

    @property
    def cell_field(self):
        return _Field(self)


def _load_reference_geometry():
    """Extract + exec only the standalone geometry functions from the reference
    steppables file (it imports cc3d at module scope, so we can't import it)."""
    src = open(REF_PATH, "r").read()
    ns = {"math": math, "np": np}
    for name in ("create_hollow_sphere", "select_voxel", "mesendoderm_sphere_circumference"):
        m = re.search(rf"^def {name}\(.*?(?=^def |^class |\Z)", src, re.S | re.M)
        assert m, f"could not extract {name} from reference"
        exec(compile(m.group(0), REF_PATH, "exec"), ns)
    return ns


def _reference_start_lattice(n=100):
    ns = _load_reference_geometry()
    ref = _MockSelf(n)
    vs, rs = 1, 0
    seq = [
        (58, 50, G.LEADING),
        (52, 48, G.PASSIVE),
        (58, 45, G.PASSIVE), (52, 45, G.PASSIVE), (46, 45, G.PASSIVE),
        (57, 40, G.PASSIVE), (52, 40, G.PASSIVE), (46, 40, G.PASSIVE), (40, 40, G.PASSIVE),
        (56, 35, G.PASSIVE), (51, 35, G.PASSIVE), (46, 35, G.PASSIVE), (40, 35, G.PASSIVE),
    ]
    for radius, zlevel, ctype in seq:
        ns["mesendoderm_sphere_circumference"](n, radius, zlevel, vs, rs, ref, ctype)
    ns["create_hollow_sphere"](n, 50, 1, ref)
    return ref.ids, np.asarray(ref.types, dtype=np.int32)


def test_hollow_sphere_is_a_shell():
    """create_hollow_sphere produces a spherical shell of 1-voxel Substrate cells
    at the expected radius +/- tolerance, centered in the cube."""
    b = G._LatticeBuilder(40)
    G.create_hollow_sphere(b, 40, radius=15, tolerance=1)
    types = b.finalize_types()
    occ = np.argwhere(b.ids > 0)  # (z,y,x)
    assert occ.shape[0] > 0
    center = 40 // 2
    d = np.linalg.norm(occ[:, ::-1] - center, axis=1)  # (x,y,z) distance
    assert d.min() >= 15 - 1 - 1e-6 and d.max() <= 15 + 1 + 1e-6
    # every shell cell is a single voxel of Substrate
    assert np.all(types[b.ids[b.ids > 0]] == G.SUBSTRATE)
    vols = np.bincount(b.ids.reshape(-1), minlength=len(types))
    assert np.all(vols[1:] == 1), "hollow-sphere cells must be 1 voxel each"


def test_mesendoderm_ring_block_count_and_size():
    """mesendoderm_sphere_circumference makes num_blocks cubic cells of edge 5."""
    b = G._LatticeBuilder(100)
    nb = G.mesendoderm_sphere_circumference(b, 100, radius=58, z_level=50,
                                            voxel_spacing=1, region_spacing=0,
                                            celltype=G.LEADING)
    expected = int(2 * math.pi * 58 / (5 + 1))
    assert nb == expected == 60
    types = b.finalize_types()
    assert np.sum(types == G.LEADING) == nb
    # each block is up to 5^3 = 125 voxels (clipped at borders)
    vols = np.bincount(b.ids.reshape(-1), minlength=len(types))
    assert vols[1:].max() <= 125 and vols[1:].max() >= 100


def test_embryo_start_counts():
    """The full start() geometry has the expected per-type cell counts."""
    state, info = G.build_embryo_start(cube_size=100)
    tc = info["type_counts"]
    assert tc[G.LEADING] == 60          # leading ring at radius 58
    assert tc[G.PASSIVE] == 618         # 12 passive rings
    assert tc[G.SUBSTRATE] == 62333     # hollow-sphere shell voxels (1 cell each)
    assert tc[G.RING] == 0
    assert state.n_cells == 60 + 618 + 62333
    # Substrate is frozen, Leading/Passive are not
    fm = state.cfg.frozen_mask()
    assert fm[G.SUBSTRATE] == 1 and fm[G.LEADING] == 0 and fm[G.PASSIVE] == 0


def test_embryo_geometry_matches_reference_voxel_exact():
    """The ported start() id-lattice is identical (voxel-for-voxel, id-for-id) to
    running the ACTUAL EmbryoSteppables.py geometry functions."""
    ref_ids, ref_types = _reference_start_lattice(100)
    state, _ = G.build_embryo_start(cube_size=100)
    assert np.array_equal(state.ids, ref_ids), "id-lattice differs from reference"
    assert np.array_equal(state.cell_type, ref_types), "cell types differ from reference"


def test_embryo_state_volume_partition_exact():
    """The constructed state's per-cell volume == lattice voxel count exactly."""
    state, _ = G.build_embryo_start(cube_size=100)
    counts = np.bincount(state.ids.reshape(-1), minlength=state.n_cells + 1)
    assert np.array_equal(counts[1:], state.volume[1:].astype(np.int64))
