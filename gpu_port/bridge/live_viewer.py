"""Live interactive viewer for a running GPU CPM / Embryo simulation -- the
"watch it evolve, like CC3D Player" front-end for the headless bridge.

Where `bridge.viewer` is the *headless* data path (pull-only, render-ready NumPy
arrays, no window), this module opens a real Qt window (pyqtgraph -- the same Qt
stack cc3d-player5 uses) that advances the GPU simulation in the background and
repaints a 2-D cell-type slice live, with play/pause/step and a slice slider.

Why pyqtgraph and NOT matplotlib
--------------------------------
matplotlib loads a second OpenMP runtime that conflicts with torch's on this box
(`libomp.dll` vs `libiomp5md.dll`) and *aborts the process* (OMP Error #15) unless
the unsafe, unsupported `KMP_DUPLICATE_LIB_OK=TRUE` is set -- which the OpenMP
runtime itself warns "may cause crashes or silently produce incorrect results",
unacceptable for a scientific sim. pyqtgraph renders through Qt and pulls in no
second OpenMP runtime, so there is no conflict and no workaround is needed.

This module is intentionally NOT imported by ``bridge/__init__.py`` (which stays
headless-safe for the test gate): it is a manual, interactive desktop tool. Run it
from the ``gpu_port`` directory, e.g.::

    cd gpu_port
    pixi run python -m bridge.live_viewer                      # reduced closure scene
    pixi run python -m bridge.live_viewer --scene embryo --size 100   # full Embryo
    pixi run python -m bridge.live_viewer --axis z --mcs-per-frame 10

Controls: **Play/Pause** runs the sim; **Step** advances one frame; the **slider**
scrubs the slice plane; the status line shows MCS / cell count / active links.
"""
from __future__ import annotations

import numpy as np

from engine.geometry import MEDIUM, LEADING, PASSIVE, RING, SUBSTRATE
from .viewer import LatticeView


# ---------------------------------------------------------------------------
# CC3D-Player-like cell-type colors, indexed by the engine's type constants.
# ---------------------------------------------------------------------------
def _type_lut() -> np.ndarray:
    """(n_types, 3) uint8 RGB lookup table indexed by cell *type* value."""
    n = max(MEDIUM, LEADING, PASSIVE, RING, SUBSTRATE) + 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    lut[MEDIUM] = (15, 15, 25)        # near-black background (medium)
    lut[LEADING] = (225, 55, 45)      # red    -- leading edge
    lut[PASSIVE] = (50, 180, 90)      # green  -- passive mesendoderm
    lut[RING] = (235, 205, 45)        # yellow -- actin ring (if present)
    lut[SUBSTRATE] = (70, 95, 165)    # blue   -- substrate (ectoderm shell)
    return lut


_TYPE_NAMES = {MEDIUM: "Medium", LEADING: "Leading", PASSIVE: "Passive",
               RING: "Ring", SUBSTRATE: "Substrate"}


def _build_app():
    """Get-or-create the single QApplication (safe to call repeatedly)."""
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class LiveViewer:
    """A live Qt window over a running simulation ``model``.

    ``model`` must expose ``.engine`` (a GPUEngine/BatchedGPUEngine) and
    ``.run(n_mcs, mcs_offset=...)`` -- the ``EmbryoModel`` interface. Frames are
    produced by advancing ``mcs_per_frame`` MCS then pulling a 2-D slice via the
    headless ``LatticeView`` (so what you see is exactly device state).
    """

    def __init__(self, model, axis: str = "z", index: int | None = None,
                 field: str = "type", mcs_per_frame: int = 5, interval_ms: int = 30,
                 max_mcs: int | None = None, title: str = "CC3D-GPU live"):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets, QtCore

        pg.setConfigOptions(imageAxisOrder="row-major")  # array[y, x] -> natural view

        self.model = model
        self.engine = getattr(model, "engine", model)
        self.view = LatticeView(self.engine)
        self.axis = axis.lower()
        self.field = field
        self.mcs_per_frame = int(mcs_per_frame)
        self.interval_ms = int(interval_ms)
        self.max_mcs = max_mcs
        self.mcs = 0
        self.playing = False
        self.lut = _type_lut()

        axis_len = {"z": self.view.Lz, "y": self.view.Ly, "x": self.view.Lx}[self.axis]
        self.index = self._densest_index(self.axis) if index is None else int(index)

        # --- window + image ---------------------------------------------------
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(title)
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        root.addWidget(self.glw, stretch=1)

        # --- controls ---------------------------------------------------------
        ctl = QtWidgets.QHBoxLayout()
        root.addLayout(ctl)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_step = QtWidgets.QPushButton("Step")
        ctl.addWidget(self.btn_play)
        ctl.addWidget(self.btn_step)
        ctl.addWidget(QtWidgets.QLabel(f"  {self.axis}-slice"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(axis_len - 1)
        self.slider.setValue(self.index)
        ctl.addWidget(self.slider, stretch=1)

        self.status = QtWidgets.QLabel("-")
        root.addWidget(self.status)
        legend = "   ".join(f"■ {_TYPE_NAMES[t]}" for t in
                            (LEADING, PASSIVE, RING, SUBSTRATE) if t in _TYPE_NAMES)
        root.addWidget(QtWidgets.QLabel(legend))

        self.btn_play.clicked.connect(self.toggle)
        self.btn_step.clicked.connect(self.step_once)
        self.slider.valueChanged.connect(self._on_slider)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)

        self._refresh()  # paint the initial state (MCS 0)

    # -- frame production ------------------------------------------------------
    def _advance(self):
        """Advance the simulation by one frame's worth of MCS (continuing the
        global MCS clock so RNG keys/turnover stay consistent)."""
        self.model.run(self.mcs_per_frame, mcs_offset=self.mcs)
        self.mcs += self.mcs_per_frame

    def _densest_index(self, axis: str) -> int:
        """Default slice = the plane (along ``axis``) holding the most *dynamic*
        (non-Medium, non-Substrate) cells, so the moving leaders/passives are in
        view immediately -- the closure ring sits at z=1-2, just above the frozen
        substrate floor; the embryo mesendoderm straddles the sphere middle. Falls
        back to densest non-Medium, then the mid-plane, for degenerate scenes."""
        tf = self.view.type_field()                       # (Lz, Ly, Lx)
        ax = {"z": 0, "y": 1, "x": 2}[axis]
        other = tuple(i for i in (0, 1, 2) if i != ax)
        counts = ((tf != MEDIUM) & (tf != SUBSTRATE)).sum(axis=other)
        if not counts.any():
            counts = (tf != MEDIUM).sum(axis=other)
        return int(np.argmax(counts)) if counts.any() else tf.shape[ax] // 2

    def _frame_image(self) -> np.ndarray:
        sl = self.view.slice(axis=self.axis, index=self.index, field=self.field)
        if self.field == "type":
            return self.lut[np.clip(sl, 0, len(self.lut) - 1)]   # (H, W, 3) uint8
        return sl                                                # raw id field

    def _refresh(self):
        img = self._frame_image()
        if img.ndim == 3:
            self.img.setImage(img, autoLevels=False)
        else:
            self.img.setImage(img, autoLevels=True)
        info = (f"MCS {self.mcs}    {self.axis}={self.index}    "
                f"cells {int(self.engine.n_cells)}")
        if hasattr(self.model, "num_active_links"):
            try:
                info += f"    active links {self.model.num_active_links()}"
            except Exception:
                pass
        if self.max_mcs is not None:
            info += f"    (stops at {self.max_mcs})"
        self.status.setText(info)

    # -- controls --------------------------------------------------------------
    def _on_slider(self, value: int):
        self.index = int(value)
        self._refresh()  # re-slice current state; does not advance the sim

    def toggle(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(self.interval_ms)
        else:
            self.timer.stop()

    def step_once(self):
        self._advance()
        self._refresh()

    def _tick(self):
        if self.max_mcs is not None and self.mcs >= self.max_mcs:
            self.toggle()   # auto-pause at the cap
            return
        self._advance()
        self._refresh()

    def show(self):
        self.win.resize(760, 820)
        self.win.show()


def run_live(model, **kwargs):
    """Open the live viewer over ``model`` and run the Qt event loop until the
    window is closed (blocking -- this is the interactive entry point)."""
    app = _build_app()
    viewer = LiveViewer(model, **kwargs)
    viewer.show()
    app.exec_()
    return viewer


def build_scene(scene: str = "closure", size: int | None = None,
                temperature: float = 10.0, seed: int = 1):
    """Build + ``start()`` an ``EmbryoModel`` for a named scene (shared by the 2-D
    and 3-D viewers). Returns ``(model, info)``.

      * ``closure`` -- reduced wound-closure scene (fast; default L=32).
      * ``embryo``  -- full scaled Embryo model (default cube size 100).
    """
    import embryo.model as em
    if scene == "closure":
        L = size or 32
        state, info = em.build_closure_scene(L=L, temperature=temperature, seed=seed)
        model = em.EmbryoModel(state, closure_window=info.get("window")).start()
    elif scene == "embryo":
        sz = size or 100
        state, info = em.build_scaled_embryo(cube_size=sz, temperature=temperature, seed=seed)
        model = em.EmbryoModel(state).start()
    else:
        raise ValueError(f"unknown scene {scene!r} (use 'closure' or 'embryo')")
    return model, info


# ---------------------------------------------------------------------------
# CLI: build an Embryo scene on the GPU and watch it live.
# ---------------------------------------------------------------------------
def _main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Live GPU Embryo viewer (pyqtgraph). Watch the CPM sim evolve "
                    "like CC3D Player, fed straight from the GPU.")
    p.add_argument("--scene", choices=["closure", "embryo"], default="closure",
                   help="'closure' = reduced wound-closure scene (fast, default); "
                        "'embryo' = the full scaled Embryo model.")
    p.add_argument("--size", type=int, default=None,
                   help="closure: lattice L (default 32); embryo: cube size (default 100).")
    p.add_argument("--axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--field", choices=["type", "id"], default="type")
    p.add_argument("--mcs-per-frame", type=int, default=5)
    p.add_argument("--interval-ms", type=int, default=30)
    p.add_argument("--max-mcs", type=int, default=None)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    model, info = build_scene(a.scene, a.size, a.temperature, a.seed)
    print(f"scene={a.scene} size={a.size} cells={info.get('n_cells')} "
          f"types={info.get('type_counts')}")
    print("opening live window -- Play to run, Step to advance, slider to scrub the slice.")
    run_live(model, axis=a.axis, field=a.field, mcs_per_frame=a.mcs_per_frame,
             interval_ms=a.interval_ms, max_mcs=a.max_mcs,
             title=f"CC3D-GPU live -- {a.scene}")


if __name__ == "__main__":
    _main()
