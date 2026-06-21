"""Headless-safe visualization bridge for the live GPU CPM simulation -- Phase 4 Pass C.

This is the interactivity bridge: a lightweight, self-contained way to *see* what a
running ``GPUEngine`` / ``BatchedGPUEngine`` is doing, without a Qt/cc3d-player5
display and without ever blocking on a GUI event loop. It is the data path that any
viewer (a notebook, a file watcher, or -- optionally -- cc3d-player5) reads from.

Design (why this shape)
-----------------------
* **Pull, don't push.** The engine owns the device arrays; the bridge *pulls* the
  id-lattice / type-field to the host on demand (``snapshot``) exactly as the
  engine's own ``get_ids()`` does. It never mutates engine state and never advances
  the simulation -- you interleave ``engine.run(k)`` and ``view.snapshot()`` yourself.
* **Render-ready arrays are the contract.** The primary API returns NumPy arrays
  (id-lattice, type-field, a 2-D slice, or a max/sum projection). Whether you then
  draw them with matplotlib, write a raw ``.npy``, or feed them to cc3d-player5's
  ``CellField`` is the caller's choice; the *correctness* surface is "the exported
  array equals the device state", which is testable with no display.
* **Headless-safe rendering, robust by default.** ``render_slice_png`` defaults to a
  dependency-free greyscale PNG writer (stdlib ``zlib`` only) -- NO native libraries,
  so it cannot block on a GUI and cannot hit the matplotlib/torch OpenMP-runtime DLL
  conflict that aborts the interpreter on some Windows setups. A labeled matplotlib
  figure (forced onto the non-interactive ``Agg`` backend) is available opt-in via
  ``backend="matplotlib"``. There is no ``plt.show()`` / event loop anywhere in this
  module, so importing or using it in a no-display test gate can never block.

The type-field
--------------
The id-lattice stores cell *ids* (0 = Medium, 1..n_cells). A CPM viewer normally
colors by cell *type* (Medium / Leading / Passive / ... ). ``type_field()`` maps
each voxel's id through the engine's per-cell ``cell_type`` SoA, reproducing the
field cc3d-player5 paints. Both the raw id-lattice and the type-field are exported
so a caller can pick id-resolution (per-cell) or type-resolution (per-type) views.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Engine adapters: normalize single vs batched engines to a common read surface.
# ---------------------------------------------------------------------------
def _is_batched(engine) -> bool:
    """A BatchedGPUEngine carries a replica axis (``R`` and ``n1``)."""
    return hasattr(engine, "R") and hasattr(engine, "n1")


def _engine_ids(engine, replica: int | None):
    """Host id-lattice (Lz,Ly,Lx) int32 for ``engine``.

    For a batched engine, ``replica`` selects which replica (default 0). This reads
    the device id-lattice exactly the way the engine's own ``get_ids()`` does, so
    the bridge can never diverge from engine state.
    """
    if _is_batched(engine):
        all_ids = engine.get_ids()                 # (R, Lz, Ly, Lx)
        r = 0 if replica is None else int(replica)
        if not (0 <= r < engine.R):
            raise IndexError(f"replica {r} out of range [0,{engine.R})")
        return np.ascontiguousarray(all_ids[r])
    return engine.get_ids()                          # (Lz, Ly, Lx)


def _engine_cell_type(engine) -> np.ndarray:
    """Per-cell type vector (n_cells+1,) int32, host copy (index 0 = Medium)."""
    return engine.cell_type.numpy().astype(np.int32)


def _engine_dims(engine):
    return int(engine.Lz), int(engine.Ly), int(engine.Lx)


# ---------------------------------------------------------------------------
# LatticeView -- the render-ready field provider (the bridge API).
# ---------------------------------------------------------------------------
class LatticeView:
    """Render-ready field provider for a running GPU CPM engine.

    Construct it once around a ``GPUEngine`` or ``BatchedGPUEngine``; call
    ``snapshot()`` (or the field accessors) whenever you want the current state.
    All accessors return fresh host NumPy arrays that EXACTLY mirror the device
    id-lattice / type SoA at call time -- this is the headless data path.

    Parameters
    ----------
    engine : GPUEngine | BatchedGPUEngine
        The live engine. Its device arrays are read (never written).
    replica : int, optional
        For a batched engine, which replica to view (default 0). Ignored for a
        single engine.
    """

    def __init__(self, engine, replica: int | None = None):
        self.engine = engine
        self.batched = _is_batched(engine)
        self.replica = 0 if (self.batched and replica is None) else replica
        self.Lz, self.Ly, self.Lx = _engine_dims(engine)

    # ----------------------------------------------------------- raw fields
    def id_field(self) -> np.ndarray:
        """Current id-lattice (Lz,Ly,Lx) int32 -- exactly the device ``ids``."""
        return _engine_ids(self.engine, self.replica)

    def type_field(self, ids: np.ndarray | None = None) -> np.ndarray:
        """Current type-field (Lz,Ly,Lx) int32: each voxel's cell id mapped through
        the per-cell ``cell_type`` SoA (0 = Medium). This is the field cc3d-player5
        colors by type. Pass a precomputed ``ids`` to avoid a second device read."""
        if ids is None:
            ids = self.id_field()
        ct = _engine_cell_type(self.engine)
        # ids are guaranteed in [0, n_cells]; gather types (vectorized, exact).
        return ct[ids]

    # ----------------------------------------------------------- 2-D views
    def slice(self, axis: str = "z", index: int | None = None, field: str = "id"):
        """A 2-D slice of the chosen ``field`` ('id' or 'type') along ``axis``
        ('x'|'y'|'z'). ``index`` defaults to the mid-plane. Returns a 2-D int32
        array (the natural in-plane orientation, suitable for ``imshow``)."""
        vol = self.id_field() if field == "id" else self.type_field()
        axis = axis.lower()
        if axis == "z":
            idx = self.Lz // 2 if index is None else int(index)
            return np.ascontiguousarray(vol[idx, :, :])          # (Ly, Lx)
        if axis == "y":
            idx = self.Ly // 2 if index is None else int(index)
            return np.ascontiguousarray(vol[:, idx, :])          # (Lz, Lx)
        if axis == "x":
            idx = self.Lx // 2 if index is None else int(index)
            return np.ascontiguousarray(vol[:, :, idx])          # (Lz, Ly)
        raise ValueError(f"axis must be x|y|z, got {axis!r}")

    def projection(self, axis: str = "z", field: str = "type", reduce: str = "max"):
        """A 2-D projection of ``field`` along ``axis`` ('max' or 'sum' reduction) --
        a cheap whole-lattice 'overview' render that needs no slice index. 'max'
        gives an occupancy/type silhouette; 'sum' gives column density."""
        vol = self.id_field() if field == "id" else self.type_field()
        ax = {"z": 0, "y": 1, "x": 2}[axis.lower()]
        if reduce == "max":
            return np.ascontiguousarray(vol.max(axis=ax))
        if reduce == "sum":
            return np.ascontiguousarray(vol.sum(axis=ax))
        raise ValueError(f"reduce must be max|sum, got {reduce!r}")

    # ----------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """A single coherent host snapshot of the current state: both the id-lattice
        and the type-field (mapped from the SAME id read), plus light metadata. The
        canonical 'give me a render-ready view of the live sim' call.

        Returns a dict with ``ids`` (Lz,Ly,Lx int32), ``types`` (Lz,Ly,Lx int32),
        ``n_cells``, ``dims`` (Lz,Ly,Lx), ``replica`` (None for a single engine),
        and ``cell_type`` (the per-cell type SoA used for the mapping)."""
        ids = self.id_field()
        ct = _engine_cell_type(self.engine)
        return {
            "ids": ids,
            "types": ct[ids],
            "cell_type": ct,
            "n_cells": int(self.engine.n_cells),
            "dims": (self.Lz, self.Ly, self.Lx),
            "replica": self.replica,
        }


# ---------------------------------------------------------------------------
# Headless rendering helpers (file/array only -- never a window/event loop).
# ---------------------------------------------------------------------------
def _matplotlib_agg():
    """Import matplotlib forced onto the non-interactive 'Agg' (file-only) backend.
    Returns the ``pyplot`` module, or ``None`` if matplotlib is unavailable. Using
    Agg guarantees no window is ever created -- safe in a no-display gate."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)            # file-only, no GUI, no event loop
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def render_slice_png(view: LatticeView, path: str, axis: str = "z",
                     index: int | None = None, field: str = "type",
                     backend: str = "stdlib") -> str:
    """Render a 2-D slice of the live state to a PNG file (headless).

    NEVER opens a window or runs an event loop, so it is safe under a no-display
    test gate. Reads the SAME render-ready array ``LatticeView`` exports, so the file
    content corresponds exactly to device state at call time. Returns the path.

    ``backend``:
      * ``"stdlib"`` (default) -- a dependency-free greyscale PNG written with the
        stdlib (``zlib`` only). This path pulls in NO native libraries, so it cannot
        hit the matplotlib/torch OpenMP-runtime conflict that aborts the interpreter
        on some Windows setups (observed here: ``libiomp5md.dll`` vs ``libomp.dll``).
        It is the robust default for the live-sim bridge.
      * ``"matplotlib"`` -- a labeled, colormapped figure via matplotlib's
        non-interactive ``Agg`` backend (still no window). Opt-in only, for
        environments where matplotlib is safe to load alongside torch. Falls back to
        the stdlib writer if matplotlib is unavailable.
    """
    img = view.slice(axis=axis, index=index, field=field)
    if backend == "matplotlib":
        plt = _matplotlib_agg()
        if plt is not None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
            ax.imshow(img, origin="lower", interpolation="nearest",
                      cmap="tab20" if field == "type" else "nipy_spectral")
            ax.set_title(f"{field} slice {axis}={'mid' if index is None else index}")
            ax.set_xlabel("x" if axis in ("z", "y") else "y")
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            return path
        # matplotlib requested but unavailable -> fall through to the stdlib writer
    elif backend != "stdlib":
        raise ValueError(f"backend must be 'stdlib' or 'matplotlib', got {backend!r}")
    _write_grey_png(img, path)                               # robust, no native deps
    return path


def _write_grey_png(arr2d: np.ndarray, path: str):
    """Write a 2-D array as an 8-bit greyscale PNG with no third-party deps.
    Normalizes ``arr2d`` to 0..255. Uses ``zlib`` + the PNG chunk format (stdlib
    only) so the headless fallback never needs matplotlib or PIL."""
    import struct
    import zlib

    a = np.asarray(arr2d)
    a = a.astype(np.float64)
    span = a.max() - a.min()
    if span > 0:
        a = (a - a.min()) / span
    g = (a * 255.0).round().astype(np.uint8)
    h, w = g.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)     # 8-bit greyscale
    raw = bytearray()
    for row in g:
        raw.append(0)                                        # filter type 0
        raw.extend(row.tobytes())
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig)
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", idat))
        f.write(_chunk(b"IEND", b""))


def to_cc3d_cell_field(view: LatticeView) -> np.ndarray:
    """Return the id-lattice in the (x,y,z) axis order cc3d-player5 expects for its
    ``CellField`` (CC3D indexes ``field[x,y,z]``; the engine stores (z,y,x)).

    This is the OPTIONAL cc3d-player5 hand-off: a CC3D ``CellField``/``CellG`` view
    can be populated from this array. It is a pure axis transpose of the exported
    id-lattice (no display, no Qt) -- the player integration, if used, consumes this
    array on its side. Returned array is contiguous (x,y,z) int32.
    """
    ids = view.id_field()                                    # (z, y, x)
    return np.ascontiguousarray(np.transpose(ids, (2, 1, 0)))  # (x, y, z)
