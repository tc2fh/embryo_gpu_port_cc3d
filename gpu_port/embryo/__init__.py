"""Full GPU Embryo model (Phase 3, Pass C).

This package wires the reusable GPU engine (``gpu_port/engine/``) into a complete
GPU-resident CompuCell3D Embryo simulation driver, porting the remaining
``EmbryoSteppables.py`` steppables that Passes A/B did not cover.

PORT-GAP (vs ``Embryo_Model_dev/Embryo/Simulation/EmbryoSteppables.py``):

ALREADY PORTED in ``gpu_port/engine/`` (REUSED here, not duplicated):
  * geometry ``EmbryoSteppable.start()`` + ``create_hollow_sphere`` /
    ``mesendoderm_sphere_circumference`` -> ``engine.geometry.build_embryo_start``
    (voxel-exact at 63011 cells: 60 Leading / 618 Passive / 62333 Substrate).
  * ``LeadingEdgeSteppable`` lamellipodia link + cohesotaxis selection +
    Poisson turnover -> ``engine.steppables.LamellipodiaSteppable`` +
    ``engine.cohesotaxis`` (the ``create_lamellipodia_link`` pipeline).
  * ``SubstrateSteppable`` floor-free-area observable -> the windowed
    ``ClosureSteppable`` here (faithful CellArray window) + the engine
    ``FloorFreeAreaSteppable`` (whole-floor variant).
  * FocalPointPlasticity spring energy -> ``engine.kernels.fpp_delta_cell`` +
    ``engine.fpp.FPPLinks`` (quadratic ``lambda*(L-target)^2``; the XML's nested
    ``LinkConstituentLaw`` Lambda*Length is dead config -- see model.py NOTE).

PORTED NOW (this package):
  * ``LeadingEdgeSteppable`` / ``PassiveSteppable`` TISSUE links + INTERCALATION
    turnover + ``PassiveSteppable`` passive-cell-substrate links ->
    ``embryo.steppables.TissueLinkSteppable`` / ``PassiveSubstrateSteppable``.
  * The complete driver ``embryo.model.EmbryoModel`` (engine + every steppable),
    plus a reduced-scale wound-closure scene ``build_closure_scene`` used by the
    in-gate statistical validation.

SKIPPED (documented, no physics): ``EmbryoSteppable.step`` (TIFF I/O only),
``ActinRingSteppable`` (empty body), the ``ifDynamicStiffness`` / ``ifPythonCall``
/ ``ifDataSave`` / plotting branches (all OFF in the model defaults).
"""

from .params import EmbryoParams
from .steppables import TissueLinkSteppable, PassiveSubstrateSteppable, ClosureSteppable
from .model import EmbryoModel, build_closure_scene, build_scaled_embryo
from .batched_device import BatchedDeviceEmbryoModel

__all__ = [
    "EmbryoParams",
    "TissueLinkSteppable",
    "PassiveSubstrateSteppable",
    "ClosureSteppable",
    "EmbryoModel",
    "build_closure_scene",
    "build_scaled_embryo",
    "BatchedDeviceEmbryoModel",
]
