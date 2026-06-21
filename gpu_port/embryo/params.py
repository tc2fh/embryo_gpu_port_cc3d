"""Embryo model parameters, copied verbatim from the ``EmbryoSteppables.py``
module head (read-only reference). Centralized so the GPU driver and the CPU
reference use bit-identical constants.

All values match ``Embryo_Model_dev/Embryo/Simulation/EmbryoSteppables.py`` with
the model's runtime flags (``ifCohesotaxis``/``ifDynamicStiffness``/``ifPythonCall``
all 0; ``ifPassiveSubstrate=1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbryoParams:
    # 1 mcs = 5 s (EmbryoSteppables.timestep)
    timestep: int = 5

    # --- Tissue (neighbor) link params: new_fpp_link(a,b,TissueLambda,TTarget,Tmax)
    tissue_lambda: float = 600.0
    tissue_target: float = 5.0      # TTarget_distance
    tissue_max: float = 10.0        # Tmax_distance
    max_neighbor_num: int = 4       # MaxNeighborNum

    # TissueRate = 1/10000 (intercalation tissue-link turnover, per MCS prob = it)
    tissue_rate: float = 1.0 / 10000.0

    # --- Lamellipodia link params (LeadingEdge): new_fpp_link(...,800,1,15)
    lamellipodia_lambda: float = 800.0
    lamellipodia_distance: int = 2   # LamellipodiaDistance (Manhattan shell order)
    ll_target: float = 1.0           # LLTargetDist
    ll_max: float = 15.0             # LLMaxDist

    # --- Passive cell-substrate link params: new_fpp_link(...,SLinkLambda,...)
    if_passive_substrate: int = 1
    slink_lambda: float = 10.0       # SLinkLambda
    slink_target: float = 1.0        # SLink_TargetDist
    slink_max: float = 5.0           # SLinkMaxDist
    # SubLinkRate = timestep * (1/360)
    sl_cycling_rate: float = 1.0 / 360.0

    # --- Lamellipodia Poisson turnover: LamellaeRate = timestep * (1/180)
    l_cycling_rate: float = 1.0 / 180.0

    # --- Cohesotaxis selection slope
    sigma: float = 8.0

    @property
    def lamellae_rate(self) -> float:
        return self.timestep * self.l_cycling_rate

    @property
    def sub_link_rate(self) -> float:
        return self.timestep * self.sl_cycling_rate

    # Per-MCS Poisson deletion probabilities (1 - exp(-rate)).
    @property
    def lamellae_delete_prob(self) -> float:
        return 1.0 - math.exp(-self.lamellae_rate)

    @property
    def tissue_delete_prob(self) -> float:
        # EmbryoSteppables uses np.random.uniform() < (1-exp(-TissueRate))
        return 1.0 - math.exp(-self.tissue_rate)

    @property
    def sub_link_delete_prob(self) -> float:
        return 1.0 - math.exp(-self.sub_link_rate)


DEFAULT = EmbryoParams()
