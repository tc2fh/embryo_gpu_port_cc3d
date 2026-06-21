"""Make the production engine importable as ``import engine`` / ``import embryo``
under pytest for the Phase 8 tests.

The gate command is ``pixi run python -m pytest -q gpu_port``. As with the Phase
2/3/4/5/6 conftests, inserting the ``gpu_port`` directory on ``sys.path`` lets the
Phase 8 tests import the engine + embryo packages with simple absolute imports
(``from engine import ...`` / ``from embryo import ...``) regardless of pytest's
rootdir. This does not touch the other phases.
"""

import os
import sys

_PHASE8 = os.path.dirname(os.path.abspath(__file__))
_GPU_PORT = os.path.dirname(_PHASE8)  # .../gpu_port  (contains the `engine` package)
if _GPU_PORT not in sys.path:
    sys.path.insert(0, _GPU_PORT)
