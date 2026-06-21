"""Make the production engine importable as ``import engine`` under pytest.

The gate command is ``pixi run python -m pytest -q gpu_port``. As with the Phase
2/3/4/5 conftests, inserting the ``gpu_port`` directory on ``sys.path`` lets the
Phase 6 tests import the engine package with a simple absolute import (``from
engine import ...``) regardless of pytest's rootdir. This does not touch the other
phases.
"""

import os
import sys

_PHASE6 = os.path.dirname(os.path.abspath(__file__))
_GPU_PORT = os.path.dirname(_PHASE6)  # .../gpu_port  (contains the `engine` package)
if _GPU_PORT not in sys.path:
    sys.path.insert(0, _GPU_PORT)
