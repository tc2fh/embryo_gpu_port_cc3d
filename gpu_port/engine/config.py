"""Configuration + constant tables for the CC3D-GPU production engine.

This module is the single place that defines:

* ``EngineConfig`` -- the model parameters (lattice size, temperature, per-type
  Volume parameters, the type x type Contact matrix, neighbor orders, RNG seed).
* The **neighbor-offset shells**, built to match CC3D's ``BoundaryStrategy``
  distance-sorted ordering on the 3D square lattice:

      order 1 -> distance 1   -> 6   face   neighbors
      order 2 -> distance v2  -> +12 edge   neighbors  (cumulative 18)
      order 3 -> distance v3  -> +8  corner neighbors  (cumulative 26 == Moore)
      order 4 -> distance 2   -> +6  axial-2 neighbors (cumulative 32)

  (verified against ``CompuCell3D/.../Boundary/BoundaryStrategy.cpp``
  ``getOffsetsAndDistances``: offsets are accumulated then sorted by Euclidean
  distance, and ``neighborOrderIndex`` buckets them by distance shell.)

These offsets are consumed by the engine as **flat ``int32`` device arrays**
(``off[3*n + axis]``) because this Warp build has no fixed-size integer matrix
constant type (carry-forward constraint from Phase 1).

Color-scheme decision (documented, see PHASE2_FINDINGS.md):
  The 8-color (2x2x2) checkerboard makes all same-color voxels mutually outside
  each other's Moore (NeighborOrder<=3) neighborhood, so they flip in parallel
  with stable neighbor reads. This is correct for a flip-target connectivity of
  order <= 3. NeighborOrder=4 (the Embryo XML) reaches distance-2 axial voxels
  and would need 27 colors; per the plan we DEFAULT the GPU sweep to order<=3
  / 8 colors and validate equivalence vs an order<=3 CPU reference. The contact
  *energy* neighborhood order is configurable independently and may be <= the
  sweep coloring order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Neighbor-offset shells (square 3D lattice), distance-sorted like CC3D.
# ---------------------------------------------------------------------------


def _all_offsets_up_to(max_axis: int = 2):
    """All integer offsets in the cube [-max_axis, max_axis]^3 minus the center,
    sorted by Euclidean distance then lexicographically (stable, reproducible)."""
    offs = []
    for dz in range(-max_axis, max_axis + 1):
        for dy in range(-max_axis, max_axis + 1):
            for dx in range(-max_axis, max_axis + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d2 = dx * dx + dy * dy + dz * dz
                offs.append((d2, dx, dy, dz))
    offs.sort(key=lambda t: (t[0], t[3], t[2], t[1]))
    return offs


# Distance^2 thresholds that close each neighbor order on the square lattice.
# order 1: d2<=1 ; order 2: d2<=2 ; order 3: d2<=3 ; order 4: d2<=4 ; ...
_ORDER_MAX_D2 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def neighbor_offsets(order: int) -> np.ndarray:
    """Offsets (N,3) int32 forming the cumulative NeighborOrder shell `order`.

    Mirrors CC3D ``getNeighborDirect`` over ``0..maxNeighborIndex`` for the given
    order on the square lattice (distance-sorted, center excluded).
    """
    if order not in _ORDER_MAX_D2:
        raise ValueError(f"unsupported neighbor order {order}")
    max_d2 = _ORDER_MAX_D2[order]
    max_axis = int(np.floor(np.sqrt(max_d2)))
    offs = [(dx, dy, dz) for (d2, dx, dy, dz) in _all_offsets_up_to(max_axis) if d2 <= max_d2]
    return np.asarray(offs, dtype=np.int32)


# Convenience constants.
MOORE_OFFSETS = neighbor_offsets(3)          # 26 (NeighborOrder<=3)
FACE_OFFSETS = neighbor_offsets(1)           # 6  (NeighborOrder 1)
ORDER4_OFFSETS = neighbor_offsets(4)         # 32 (NeighborOrder<=4)


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Model configuration for the GPU CPM engine (Volume + Contact).

    Type 0 is always Medium. ``contact`` is a dense (n_types, n_types) symmetric
    matrix of adhesion energies J[t1, t2]. ``target_volume`` / ``lambda_volume``
    are per-type (length n_types); the Medium entry (index 0) is ignored.
    ``frozen`` marks types that never flip (Embryo Substrate, XML ``Freeze=""``).
    """

    Lx: int = 32
    Ly: int = 32
    Lz: int = 32
    temperature: float = 10.0

    n_types: int = 2
    # per-type Volume parameters (index by type id; index 0 = Medium, unused)
    target_volume: np.ndarray = field(default_factory=lambda: np.array([0.0, 64.0]))
    lambda_volume: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.0]))
    # type x type contact energy matrix (symmetric)
    contact: np.ndarray = field(
        default_factory=lambda: np.array([[0.0, 16.0], [16.0, 4.0]])
    )
    # cell types that are frozen (never selected as a flip source or target).
    # Medium (type 0) is implicitly NEVER frozen and always participates, matching
    # CC3D Potts3D (checkIfFrozen is gated on a non-null cell). Default: none frozen.
    frozen: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # neighbor order for the CONTACT energy sum (Embryo XML uses 4; the GPU sweep
    # validation uses 3 -- see color-scheme decision above).
    contact_neighbor_order: int = 3
    # neighbor order used to pick the flip-target voxel (Potts uses maxNeighborIndex).
    # MUST be <= 3 for the 8-color checkerboard to be correct.
    flip_neighbor_order: int = 3
    # neighbor order for boundary-pixel + neighbor-contact trackers (Embryo: 1).
    tracker_neighbor_order: int = 1

    # flip2DimRatio (CC3D Potts default 1.0 in 3D): attempts per MCS = ratio*Lx*Ly*Lz
    flip2_dim_ratio: float = 1.0

    seed: int = 12345

    def __post_init__(self):
        self.target_volume = np.asarray(self.target_volume, dtype=np.float64)
        self.lambda_volume = np.asarray(self.lambda_volume, dtype=np.float64)
        self.contact = np.asarray(self.contact, dtype=np.float64)
        self.frozen = np.asarray(self.frozen, dtype=np.int32)
        assert self.contact.shape == (self.n_types, self.n_types), "contact matrix shape"
        assert self.target_volume.shape[0] >= self.n_types
        assert self.lambda_volume.shape[0] >= self.n_types
        if self.flip_neighbor_order > 3:
            raise ValueError(
                "flip_neighbor_order > 3 requires a 27-color scheme (not implemented; "
                "see PHASE2_FINDINGS.md color-scheme decision). Use <= 3 for 8-color."
            )

    @property
    def n_voxels(self) -> int:
        return self.Lx * self.Ly * self.Lz

    def frozen_mask(self) -> np.ndarray:
        """(n_types,) int32 mask: 1 where the type is frozen. Index 0 (Medium) is
        forced to 0 -- Medium is never frozen regardless of the ``frozen`` list."""
        m = np.zeros(self.n_types, dtype=np.int32)
        for t in self.frozen:
            ti = int(t)
            if 0 < ti < self.n_types:   # never freeze Medium (type 0)
                m[ti] = 1
        return m
