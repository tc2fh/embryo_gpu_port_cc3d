"""CC3D-GPU production engine (Phase 2).

GPU-resident Cellular Potts engine: int32 id-lattice source of truth + per-cell
Structure-of-Arrays + contact-energy matrix; 8-color checkerboard Metropolis
(NeighborOrder<=3) with Philox RNG; energy = Volume + Contact; on-GPU volume/COM
trackers (exact int64 fixed-point COM); per-MCS boundary-pixel + neighbor-contact
CSR; a GPU steppable API skeleton; and the Embryo non-FPP geometry constructors.

FocalPointPlasticity + cohesotaxis are Phase 3 (not here).
"""

from .config import (
    EngineConfig,
    neighbor_offsets,
    MOORE_OFFSETS,
    FACE_OFFSETS,
    ORDER4_OFFSETS,
)
from .state import (
    EngineState,
    state_from_id_lattice,
    build_grid_state,
    com_accumulators,
)
from .engine import GPUEngine, run_gpu
from .batched import (
    BatchedGPUEngine,
    BatchedState,
    build_batched_grid_state,
    batched_state_from_lattices,
    run_batched,
)
from .cpu_reference import CPUReference
from .fpp import FPPLinks, grid_graph_links
from .graph import GraphRunner, BatchedGraphRunner
from . import geometry
from . import steppables
from . import fpp
from . import cohesotaxis
from . import graph
from . import bench

__all__ = [
    "EngineConfig",
    "neighbor_offsets",
    "MOORE_OFFSETS",
    "FACE_OFFSETS",
    "ORDER4_OFFSETS",
    "EngineState",
    "state_from_id_lattice",
    "build_grid_state",
    "com_accumulators",
    "GPUEngine",
    "run_gpu",
    "BatchedGPUEngine",
    "BatchedState",
    "build_batched_grid_state",
    "batched_state_from_lattices",
    "run_batched",
    "CPUReference",
    "FPPLinks",
    "grid_graph_links",
    "GraphRunner",
    "BatchedGraphRunner",
    "geometry",
    "steppables",
    "fpp",
    "cohesotaxis",
    "graph",
    "bench",
]
