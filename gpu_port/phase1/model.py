"""Shared synthetic CPM+FPP model definition for the Phase 1 spike.

This module defines a small, reproducible synthetic Cellular Potts model used by
BOTH the GPU (Warp) prototype and the CPU (NumPy) reference, so the two can be
compared on identical initial conditions, parameters and link topology.

Energy semantics mirror the vendored CC3D source (read-only reference):

* Volume      ``E = lambda_V * (V - V_target)^2``  (per non-medium cell)
* Contact     ``E = sum over neighbor pairs of J(type_a, type_b)`` (Medium = type 0)
              ``ContactPlugin::changeEnergy`` sums J over the NeighborOrder shell.
* FPP (link)  ``E_link = offset + lambda * (L - L_target)^2`` with
              ``L = || COM_a - COM_b ||_2`` (``FocalPointPlasticityPlugin::potentialFunction``
              and ``distInvariantCM`` -> plain Euclidean for non-periodic BC).
* Metropolis  Boltzmann: accept if dE <= 0 else with prob exp(-dE / T)
              (``DefaultAcceptanceFunction``; k = 1, offset = 0).

The lattice is the int32 cell-id field (0 = Medium) as the single source of truth.
Per-cell state is Structure-of-Arrays. COM is the single source of truth for FPP
link length (stored as integer accumulators xsum/ysum/zsum plus volume).

This is a THROWAWAY prototype favouring clarity over architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# 26-neighbor (Moore, NeighborOrder<=3) offsets, excluding the center.
# Contact energy sums over this shell (matches NeighborOrder<=3 -> 8-color need).
# ----------------------------------------------------------------------------
def _moore_offsets() -> np.ndarray:
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=np.int32)  # (26, 3)


MOORE_OFFSETS = _moore_offsets()

# 6-neighbor von-Neumann offsets used to pick the *target* neighbor for a flip
# (the source pixel copies its id into a randomly chosen adjacent pixel). Using
# the face-neighbor set for the copy attempt keeps the CPU and GPU identical and
# is the usual CPM flip-attempt connectivity.
VONNEUMANN_OFFSETS = np.asarray(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
    dtype=np.int32,
)


@dataclass
class ModelConfig:
    """Synthetic model parameters shared by CPU and GPU implementations."""

    L: int = 32                     # cubic lattice edge (L^3 lattice)
    cells_per_axis: int = 3         # cells arranged on a 3x3x3 grid -> 27 cells
    temperature: float = 10.0       # CPM temperature (matches Embryo model T=10)

    # Volume constraint (single cell type for all real cells).
    target_volume: float = 64.0     # ~ (32/3)^3 ~ 38; use a round target
    lambda_volume: float = 2.0

    # Contact energy matrix indexed by [type_a, type_b].
    # type 0 = Medium, type 1 = Cell. Adhesion: cell-cell cheaper than cell-medium
    # so cells stay aggregated (sorting/compaction signal).
    j_medium_cell: float = 16.0
    j_cell_cell: float = 4.0
    j_medium_medium: float = 0.0

    # FPP spring parameters (single type-pair here).
    fpp_lambda: float = 5.0
    fpp_target_length: float = 12.0   # rest length of springs (lattice units)
    fpp_offset: float = 0.0
    fpp_max_length: float = 30.0      # links longer than this break (informational)

    seed: int = 12345

    @property
    def n_cells(self) -> int:
        return self.cells_per_axis ** 3

    def contact_matrix(self) -> np.ndarray:
        """2x2 contact energy J[type_a, type_b]; symmetric."""
        j = np.zeros((2, 2), dtype=np.float64)
        j[0, 0] = self.j_medium_medium
        j[0, 1] = j[1, 0] = self.j_medium_cell
        j[1, 1] = self.j_cell_cell
        return j


@dataclass
class ModelState:
    """Initial condition: id-lattice + per-cell SoA + link topology.

    ids:   (L,L,L) int32, 0 = Medium, 1..n_cells = cell ids.
    types: (n_cells+1,) int32, types[0] = 0 (Medium), rest = 1 (Cell).
    Per-cell arrays are length n_cells+1, index 0 is the (unused) Medium slot.
    """

    cfg: ModelConfig
    ids: np.ndarray
    types: np.ndarray
    target_volume: np.ndarray
    lambda_volume: np.ndarray
    # initial COM accumulators (sum of coords) and volume per cell
    xsum: np.ndarray
    ysum: np.ndarray
    zsum: np.ndarray
    volume: np.ndarray
    # link topology: undirected pairs (a, b) with a < b
    link_pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))


def _seed_cells(cfg: ModelConfig) -> np.ndarray:
    """Place ``cells_per_axis^3`` cubic cells on a regular grid inside the lattice.

    Each cell is a contiguous block so that the initial state is well defined and
    identical for CPU and GPU. Medium fills the gaps.
    """
    L = cfg.L
    n = cfg.cells_per_axis
    ids = np.zeros((L, L, L), dtype=np.int32)

    # cube edge for each cell and spacing between cube centers
    block = int(round(cfg.target_volume ** (1.0 / 3.0)))  # e.g. 4 for target 64
    block = max(2, block)
    spacing = L // n
    margin = (spacing - block) // 2
    cid = 0
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                cid += 1
                x0 = ix * spacing + margin
                y0 = iy * spacing + margin
                z0 = iz * spacing + margin
                ids[
                    z0 : z0 + block,
                    y0 : y0 + block,
                    x0 : x0 + block,
                ] = cid
    return ids


def _com_accumulators(ids: np.ndarray, n_cells: int):
    """Compute xsum/ysum/zsum (sum of coords) and volume per cell from the lattice."""
    xsum = np.zeros(n_cells + 1, dtype=np.float64)
    ysum = np.zeros(n_cells + 1, dtype=np.float64)
    zsum = np.zeros(n_cells + 1, dtype=np.float64)
    volume = np.zeros(n_cells + 1, dtype=np.float64)
    zz, yy, xx = np.nonzero(ids)
    cids = ids[zz, yy, xx]
    np.add.at(volume, cids, 1.0)
    np.add.at(xsum, cids, xx.astype(np.float64))
    np.add.at(ysum, cids, yy.astype(np.float64))
    np.add.at(zsum, cids, zz.astype(np.float64))
    return xsum, ysum, zsum, volume


def _build_links(cfg: ModelConfig) -> np.ndarray:
    """Connect each cell to its axis-adjacent neighbors on the 3x3x3 cell grid.

    Produces an undirected spring network (a 3D grid graph of cells). Pairs are
    returned as (a, b) with a < b, sorted, deterministically.
    """
    n = cfg.cells_per_axis

    def cid(ix, iy, iz):
        return iz * n * n + iy * n + ix + 1  # 1-based cell id

    pairs = set()
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                a = cid(ix, iy, iz)
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    if jx < n and jy < n and jz < n:
                        b = cid(jx, jy, jz)
                        lo, hi = (a, b) if a < b else (b, a)
                        pairs.add((lo, hi))
    arr = np.asarray(sorted(pairs), dtype=np.int32)
    return arr


def build_state(cfg: ModelConfig | None = None) -> ModelState:
    """Construct the shared initial state for the synthetic model."""
    if cfg is None:
        cfg = ModelConfig()
    ids = _seed_cells(cfg)
    n_cells = cfg.n_cells
    types = np.zeros(n_cells + 1, dtype=np.int32)
    types[1:] = 1  # all real cells are type 1
    target_volume = np.zeros(n_cells + 1, dtype=np.float64)
    target_volume[1:] = cfg.target_volume
    lambda_volume = np.zeros(n_cells + 1, dtype=np.float64)
    lambda_volume[1:] = cfg.lambda_volume
    xsum, ysum, zsum, volume = _com_accumulators(ids, n_cells)
    links = _build_links(cfg)
    return ModelState(
        cfg=cfg,
        ids=ids,
        types=types,
        target_volume=target_volume,
        lambda_volume=lambda_volume,
        xsum=xsum,
        ysum=ysum,
        zsum=zsum,
        volume=volume,
        link_pairs=links,
    )


def link_lengths_from_state(
    xsum: np.ndarray,
    ysum: np.ndarray,
    zsum: np.ndarray,
    volume: np.ndarray,
    link_pairs: np.ndarray,
) -> np.ndarray:
    """Euclidean COM-to-COM length of each link (single source of truth = COM)."""
    vol = np.where(volume > 0, volume, 1.0)
    cx = xsum / vol
    cy = ysum / vol
    cz = zsum / vol
    a = link_pairs[:, 0]
    b = link_pairs[:, 1]
    dx = cx[a] - cx[b]
    dy = cy[a] - cy[b]
    dz = cz[a] - cz[b]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def cell_coms(xsum, ysum, zsum, volume) -> np.ndarray:
    """Return (n_cells+1, 3) COM array; medium row is 0."""
    vol = np.where(volume > 0, volume, 1.0)
    return np.stack([xsum / vol, ysum / vol, zsum / vol], axis=1)
