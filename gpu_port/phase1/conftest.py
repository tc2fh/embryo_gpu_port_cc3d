"""Make the phase1 directory importable as top-level modules under pytest.

The gate command is ``pixi run python -m pytest -q gpu_port`` and there is no
package ``__init__.py`` chain from the repo root down to here (and Phase 1 may
only create files under ``gpu_port/phase1/``). Inserting this directory on
``sys.path`` lets the tests and engine modules use simple absolute imports
(``import model``, ``from cpm_gpu import ...``) regardless of rootdir.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
