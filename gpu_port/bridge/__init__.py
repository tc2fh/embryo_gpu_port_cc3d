"""Interactivity bridge for the live GPU CPM simulation (Phase 4 Pass C).

A lightweight, self-contained, HEADLESS-SAFE viewer data path: pull the device
id-lattice / type-field from a running ``GPUEngine`` / ``BatchedGPUEngine`` to the
host and expose render-ready arrays (and optional file renders). No interactive
window or event loop is ever opened, so it is safe in a no-display test gate. The
default file render uses a dependency-free stdlib PNG writer (no native libs, so it
cannot trip the matplotlib/torch OpenMP DLL conflict seen on some Windows setups); a
labeled matplotlib ``Agg`` figure is available opt-in. An optional cc3d-player5
hand-off (``to_cc3d_cell_field``) returns the lattice in CC3D's (x,y,z) ``CellField``
axis order; the player integration, if used, consumes that array on its side.

Primary API: ``LatticeView(engine)`` -> ``.snapshot()`` / ``.id_field()`` /
``.type_field()`` / ``.slice(...)`` / ``.projection(...)``; ``render_slice_png(...)``
for a file render.
"""

from .viewer import (
    LatticeView,
    render_slice_png,
    to_cc3d_cell_field,
)

__all__ = [
    "LatticeView",
    "render_slice_png",
    "to_cc3d_cell_field",
]
