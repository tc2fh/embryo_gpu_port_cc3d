"""Embryo non-FPP geometry constructors, ported to build a GPU-engine id-lattice.

Faithful ports of the two ``EmbryoSteppables.py`` geometry helpers that define the
initial condition (read-only reference: ``Embryo_Model_dev/.../EmbryoSteppables.py``):

* ``create_hollow_sphere(cube_size, radius, tolerance, ...)`` -- a shell of
  Substrate voxels where ``(radius-tol) <= dist(center) <= (radius+tol)``. In CC3D
  each ``select_voxel`` call makes a NEW single-voxel Substrate cell, so the shell
  is a cloud of 1-voxel cells. We reproduce that exactly (one id per shell voxel).

* ``mesendoderm_sphere_circumference(cube_size, radius, z_level, voxel_spacing,
  region_spacing, celltype)`` -- ``num_blocks`` cubic cells of edge 5 placed on a
  circle of radius ``(radius-region_spacing)*0.8`` at height ``z_level``; later
  calls overwrite earlier voxels (``cell_field[...] = cell``), matching CC3D
  assignment order.

The full Embryo ``start()`` builds the mesendoderm rings first, then the ectoderm
sphere LAST (so the sphere overwrites overlapping mesendoderm voxels) -- we
reproduce that ordering in ``build_embryo_start()``.

These run on the host (CPU) producing the int32 id-lattice + per-cell type array,
then are handed to the GPU engine. Geometry init is a one-time, non-hot-path step;
keeping it host-side is the explicit host/device boundary from the plan.
"""

from __future__ import annotations

import math

import numpy as np

from .config import EngineConfig
from .state import EngineState, state_from_id_lattice


# Embryo cell-type ids (from Embryo.xml CellType plugin)
MEDIUM = 0
LEADING = 1
PASSIVE = 2
RING = 3
SUBSTRATE = 4


class _LatticeBuilder:
    """Mutable id-lattice + per-cell type list that mimics CC3D ``new_cell`` /
    ``cell_field[...] = cell`` assignment semantics (last write wins)."""

    def __init__(self, cube_size: int):
        self.n = cube_size
        # ids[z,y,x]; 0 = Medium
        self.ids = np.zeros((cube_size, cube_size, cube_size), dtype=np.int32)
        self.cell_type = [0]  # index 0 = Medium
        self._next_id = 1

    def new_cell(self, ctype: int) -> int:
        cid = self._next_id
        self._next_id += 1
        self.cell_type.append(int(ctype))
        return cid

    def set_voxel(self, x: int, y: int, z: int, cid: int):
        if 0 <= x < self.n and 0 <= y < self.n and 0 <= z < self.n:
            self.ids[z, y, x] = cid

    def set_block(self, x0, x1, y0, y1, z0, z1, cid: int):
        self.ids[z0:z1, y0:y1, x0:x1] = cid

    def finalize_types(self) -> np.ndarray:
        return np.asarray(self.cell_type, dtype=np.int32)


def create_hollow_sphere(builder: _LatticeBuilder, cube_size: int, radius: float,
                         tolerance: float, open_cap: int | None = None):
    """Port of EmbryoSteppables.create_hollow_sphere: one 1-voxel Substrate cell
    per shell voxel (matches per-voxel ``new_cell`` / ``select_voxel``)."""
    center = cube_size // 2
    lo = radius - tolerance
    hi = radius + tolerance
    # iterate in the SAME order as the reference (x,y,z) so ids are reproducible
    for x in range(cube_size):
        for y in range(cube_size):
            for z in range(cube_size):
                distance = math.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                if lo <= distance <= hi:
                    if open_cap is None or z < open_cap:
                        cid = builder.new_cell(SUBSTRATE)
                        builder.set_voxel(x, y, z, cid)


def mesendoderm_sphere_circumference(builder: _LatticeBuilder, cube_size: int,
                                     radius: float, z_level: int, voxel_spacing: int,
                                     region_spacing: int, celltype: int):
    """Port of EmbryoSteppables.mesendoderm_sphere_circumference: num_blocks cubic
    cells of edge 5 around a circle at z_level."""
    center = cube_size // 2
    block_size = 5
    num_blocks = int(2 * math.pi * (radius - region_spacing) / (block_size + voxel_spacing))
    for i in range(num_blocks):
        angle = 2 * math.pi * i / num_blocks
        x_center = int(center + (radius - region_spacing) * 0.8 * math.cos(angle))
        y_center = int(center + (radius - region_spacing) * 0.8 * math.sin(angle))
        x0 = max(x_center - block_size // 2, 0)
        y0 = max(y_center - block_size // 2, 0)
        z0 = max(z_level - block_size // 2, 0)
        x1 = min(x0 + block_size, cube_size)
        y1 = min(y0 + block_size, cube_size)
        z1 = min(z0 + block_size, cube_size)
        cid = builder.new_cell(celltype)
        builder.set_block(x0, x1, y0, y1, z0, z1, cid)
    return num_blocks


def build_embryo_start(cube_size: int = 100, temperature: float = 10.0) -> tuple[EngineState, dict]:
    """Reproduce EmbryoSteppable.start() geometry exactly (mesendoderm rings then
    ectoderm shell last). Returns (EngineState, info) where info has per-type cell
    counts and the ring block counts for validation.

    Volume/contact parameters match Embryo.xml (all contact J = 10; target volumes
    Leading/Passive/Ring = 125, Substrate = 1; lambda = 1). Substrate is frozen.
    """
    b = _LatticeBuilder(cube_size)

    radius = 50
    tolerance = 1
    leading_level = 50
    voxel_spacing = 1
    region_spacing = 0
    leading_fill_radius = 58

    ring_counts = {}
    # mesendoderm rings, in the exact order of EmbryoSteppable.start()
    ring_counts["leading"] = mesendoderm_sphere_circumference(
        b, cube_size, leading_fill_radius, leading_level, voxel_spacing, region_spacing, LEADING)
    passive_blocks = 0
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 52, 48, voxel_spacing, region_spacing, PASSIVE)
    # 1 layer below leading
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 58, 45, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 52, 45, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 46, 45, voxel_spacing, region_spacing, PASSIVE)
    # 2 layers below leading
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 57, 40, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 52, 40, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 46, 40, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 40, 40, voxel_spacing, region_spacing, PASSIVE)
    # 3 layers below leading
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 56, 35, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 51, 35, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 46, 35, voxel_spacing, region_spacing, PASSIVE)
    passive_blocks += mesendoderm_sphere_circumference(b, cube_size, 40, 35, voxel_spacing, region_spacing, PASSIVE)
    ring_counts["passive"] = passive_blocks

    # ectoderm shell LAST so it overwrites overlapping mesendoderm voxels
    n_before = b._next_id
    create_hollow_sphere(b, cube_size, radius, tolerance, open_cap=None)
    ring_counts["substrate_voxels"] = b._next_id - n_before

    cell_type = b.finalize_types()
    n_types = 5
    target_volume = np.array([0.0, 125.0, 125.0, 125.0, 1.0])
    lambda_volume = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    contact = np.full((n_types, n_types), 10.0)
    np.fill_diagonal(contact[1:, 1:], 10.0)
    contact[0, 0] = 10.0  # Medium-Medium 10 per XML (does not affect dynamics)

    cfg = EngineConfig(
        Lx=cube_size, Ly=cube_size, Lz=cube_size,
        temperature=temperature,
        n_types=n_types,
        target_volume=target_volume,
        lambda_volume=lambda_volume,
        contact=contact,
        frozen=np.array([0, SUBSTRATE], dtype=np.int32),  # Medium + Substrate frozen
        contact_neighbor_order=3,   # GPU sweep validated at order<=3 (8-color)
        flip_neighbor_order=3,
        tracker_neighbor_order=1,   # Embryo BoundaryPixelTracker/NeighborTracker order 1
    )
    state = state_from_id_lattice(cfg, b.ids, cell_type)

    # per-type cell counts
    type_counts = {t: int(np.sum(cell_type == t)) for t in range(n_types)}
    info = {
        "ring_counts": ring_counts,
        "type_counts": type_counts,
        "n_cells": state.n_cells,
    }
    return state, info
