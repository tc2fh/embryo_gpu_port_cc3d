"""Live 3-D viewer for a running GPU CPM / Embryo simulation -- the VTK 3-D
front-end, the closest match to CC3D Player's 3-D cell view.

CC3D Player renders cells in 3-D with VTK; this reuses the same VTK 9.x that ships
with cc3d-player5. Each cell *type* is drawn as an isosurface (vtkDiscreteMarching-
Cubes over the type-field), colored with the same palette as the 2-D viewer; the
substrate shell is drawn translucent (toggleable) so the moving leaders/passives
stay visible inside it. A QTimer advances the GPU sim and rebuilds the surfaces
from fresh device state each frame.

Like ``live_viewer``, this is a manual desktop tool and is NOT imported by
``bridge/__init__.py`` (which stays headless-safe). VTK is the only extra import,
and it is intentionally kept out of the test gate. Run it from ``gpu_port``::

    pixi run python -m bridge.live_viewer3d                       # closure scene
    pixi run python -m bridge.live_viewer3d --scene embryo --size 100
    pixi run python -m bridge.live_viewer3d --no-substrate        # hide the shell

Rendering split
---------------
``CellSurfaces`` (the field -> isosurfaces -> actors pipeline) is deliberately
separate from the Qt window so it can be rendered to an OFFSCREEN VTK window in a
test (no display, no Qt), which is how the 3-D path is verified. The Qt embedding
(``QVTKRenderWindowInteractor``) is the standard interactive wrapper around it.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support

from engine.geometry import LEADING, PASSIVE, RING, SUBSTRATE
from .viewer import LatticeView
from .live_viewer import _type_lut, build_scene, _build_app

# Cell types drawn as opaque colored surfaces (substrate is handled separately).
_DYNAMIC_TYPES = [LEADING, PASSIVE, RING]


class CellSurfaces:
    """VTK isosurface pipeline over a CPM type-field.

    ``update(type_field)`` refreshes the geometry from current device state;
    ``actors()`` returns the renderables to add to any ``vtkRenderer`` (an
    interactive QVTK one, or an offscreen one in tests).
    """

    def __init__(self, dims_zyx, show_substrate: bool = True):
        Lz, Ly, Lx = dims_zyx
        self.image = vtk.vtkImageData()
        self.image.SetDimensions(Lx, Ly, Lz)        # VTK indexes (x, y, z)
        lut = _type_lut()
        n = len(lut)

        # --- dynamic cells: one contour over all dynamic labels, colored by type
        self.dyn_contour = vtk.vtkDiscreteMarchingCubes()
        self.dyn_contour.SetInputData(self.image)
        for i, t in enumerate(_DYNAMIC_TYPES):
            self.dyn_contour.SetValue(i, t)
        vlut = vtk.vtkLookupTable()
        vlut.SetNumberOfTableValues(n)
        vlut.SetTableRange(0, n - 1)
        for t in range(n):
            r, g, b = (lut[t] / 255.0)
            vlut.SetTableValue(t, float(r), float(g), float(b), 1.0)
        vlut.Build()
        dyn_mapper = vtk.vtkPolyDataMapper()
        dyn_mapper.SetInputConnection(self.dyn_contour.GetOutputPort())
        dyn_mapper.SetLookupTable(vlut)
        dyn_mapper.SetScalarRange(0, n - 1)
        dyn_mapper.SetScalarModeToUseCellData()
        dyn_mapper.ScalarVisibilityOn()
        self.dyn_actor = vtk.vtkActor()
        self.dyn_actor.SetMapper(dyn_mapper)

        # --- substrate shell: translucent context (toggleable)
        self.sub_contour = vtk.vtkDiscreteMarchingCubes()
        self.sub_contour.SetInputData(self.image)
        self.sub_contour.SetValue(0, SUBSTRATE)
        sub_mapper = vtk.vtkPolyDataMapper()
        sub_mapper.SetInputConnection(self.sub_contour.GetOutputPort())
        sub_mapper.ScalarVisibilityOff()
        self.sub_actor = vtk.vtkActor()
        self.sub_actor.SetMapper(sub_mapper)
        sr, sg, sb = (lut[SUBSTRATE] / 255.0)
        prop = self.sub_actor.GetProperty()
        prop.SetColor(float(sr), float(sg), float(sb))
        prop.SetOpacity(0.12)
        self.sub_actor.SetVisibility(1 if show_substrate else 0)

    def update(self, type_field_zyx: np.ndarray):
        """Bind the current (Lz,Ly,Lx) type-field as the image scalars. C-order
        ravel of (z,y,x) iterates x fastest -> exactly VTK's (x,y,z) layout."""
        flat = np.ascontiguousarray(type_field_zyx).astype(np.int32).ravel()
        arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_INT)
        self.image.GetPointData().SetScalars(arr)
        self.image.Modified()

    def actors(self):
        return [self.dyn_actor, self.sub_actor]


class VTKLiveViewer:
    """Interactive 3-D window: embeds ``CellSurfaces`` in a QVTK widget and drives
    the GPU sim with a QTimer (Play/Pause/Step, substrate toggle)."""

    def __init__(self, model, mcs_per_frame: int = 5, interval_ms: int = 30,
                 max_mcs: int | None = None, show_substrate: bool = True,
                 title: str = "CC3D-GPU live 3D"):
        from pyqtgraph.Qt import QtWidgets, QtCore
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        self.model = model
        self.engine = getattr(model, "engine", model)
        self.view = LatticeView(self.engine)
        self.mcs_per_frame = int(mcs_per_frame)
        self.interval_ms = int(interval_ms)
        self.max_mcs = max_mcs
        self.mcs = 0
        self.playing = False

        self.surf = CellSurfaces((self.view.Lz, self.view.Ly, self.view.Lx), show_substrate)
        self.surf.update(self.view.type_field())

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(title)
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.vtk_widget = QVTKRenderWindowInteractor(central)
        root.addWidget(self.vtk_widget, stretch=1)
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.07, 0.07, 0.10)
        for a in self.surf.actors():
            self.ren.AddActor(a)
        self.ren.ResetCamera()
        rw = self.vtk_widget.GetRenderWindow()
        rw.AddRenderer(self.ren)
        self.iren = rw.GetInteractor()

        ctl = QtWidgets.QHBoxLayout()
        root.addLayout(ctl)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_step = QtWidgets.QPushButton("Step")
        ctl.addWidget(self.btn_play)
        ctl.addWidget(self.btn_step)
        self.cb_sub = QtWidgets.QCheckBox("substrate shell")
        self.cb_sub.setChecked(show_substrate)
        ctl.addWidget(self.cb_sub)
        ctl.addStretch(1)
        self.status = QtWidgets.QLabel("-")
        root.addWidget(self.status)

        self.btn_play.clicked.connect(self.toggle)
        self.btn_step.clicked.connect(self.step_once)
        self.cb_sub.stateChanged.connect(self._toggle_sub)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self._update_status()

    def _advance(self):
        self.model.run(self.mcs_per_frame, mcs_offset=self.mcs)
        self.mcs += self.mcs_per_frame

    def _redraw(self):
        self.surf.update(self.view.type_field())
        self.vtk_widget.GetRenderWindow().Render()
        self._update_status()

    def _toggle_sub(self, _state):
        self.surf.sub_actor.SetVisibility(1 if self.cb_sub.isChecked() else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def toggle(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(self.interval_ms)
        else:
            self.timer.stop()

    def step_once(self):
        self._advance()
        self._redraw()

    def _tick(self):
        if self.max_mcs is not None and self.mcs >= self.max_mcs:
            self.toggle()
            return
        self._advance()
        self._redraw()

    def _update_status(self):
        info = f"MCS {self.mcs}    cells {int(self.engine.n_cells)}"
        if hasattr(self.model, "num_active_links"):
            try:
                info += f"    active links {self.model.num_active_links()}"
            except Exception:
                pass
        self.status.setText(info)

    def show(self):
        self.win.resize(900, 840)
        self.win.show()
        self.iren.Initialize()


def run_live_3d(model, **kwargs):
    """Open the 3-D viewer and run the Qt event loop until the window closes."""
    app = _build_app()
    viewer = VTKLiveViewer(model, **kwargs)
    viewer.show()
    app.exec_()
    return viewer


def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Live 3-D GPU Embryo viewer (VTK). Watch the CPM sim in 3-D like "
                    "CC3D Player, fed straight from the GPU.")
    p.add_argument("--scene", choices=["closure", "embryo"], default="closure")
    p.add_argument("--size", type=int, default=None,
                   help="closure: lattice L (default 32); embryo: cube size (default 100).")
    p.add_argument("--mcs-per-frame", type=int, default=50)
    p.add_argument("--interval-ms", type=int, default=30)
    p.add_argument("--max-mcs", type=int, default=None)
    p.add_argument("--no-substrate", action="store_true",
                   help="hide the translucent substrate shell.")
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    model, info = build_scene(a.scene, a.size, a.temperature, a.seed)
    print(f"scene={a.scene} size={a.size} cells={info.get('n_cells')} "
          f"types={info.get('type_counts')}")
    print("opening 3-D window -- drag to rotate, scroll to zoom, Play to run.")
    run_live_3d(model, mcs_per_frame=a.mcs_per_frame, interval_ms=a.interval_ms,
                max_mcs=a.max_mcs, show_substrate=not a.no_substrate,
                title=f"CC3D-GPU live 3D -- {a.scene}")


if __name__ == "__main__":
    _main()
