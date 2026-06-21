"""CUDA-Graph capture of the per-MCS hot loop -- Phase 4 Pass B.

Why
---
Each eager MCS issues 8 kernel launches (the 8-color checkerboard sweep) from
Python. At small/medium lattice sizes the per-launch host overhead (Python ->
Warp -> CUDA driver, x8 per MCS) is a meaningful fraction of the step time. A
CUDA graph records that device work ONCE and replays it as a single driver
submission, removing the per-launch Python/driver overhead on every subsequent
MCS.

What is INSIDE vs OUTSIDE the captured graph (the capture boundary)
-------------------------------------------------------------------
A captured CUDA graph must be PURE DEVICE WORK: no host synchronisation, no host
allocation, and no host-side compaction may happen mid-capture. The per-MCS hot
loop -- the 8 color Metropolis launches plus the on-device step-counter increment
-- is exactly that, so it is what we capture:

    INSIDE  (captured, replayed):  8x metropolis_color_dev_mcs_kernel
                                   + 1x incr_mcs_kernel   (mcs_dev[0] += 1)

    OUTSIDE (host loop, per call): recompute_trackers() / FPP rebuild() /
                                   neighbor-CSR + link-CSR host compaction,
                                   observable read-back, steppable link
                                   create/delete (topology changes).

The host-side rebuilds do host compaction (`numpy()` copies, prefix sums) so they
CANNOT live in the captured region -- they stay in the Python layer around the
graph replay, identical to the eager engine's per-MCS boundary.

The device-resident ``mcs`` trick
----------------------------------
A captured graph bakes in every *scalar* kernel argument, so a graph recorded
with a literal ``mcs`` would replay the SAME Philox key every MCS. We therefore
read ``mcs`` from a 1-element device array (``mcs_dev``) and append an on-device
``mcs_dev[0] += 1`` to the captured graph. Replaying the single graph N times then
walks mcs = m0, m0+1, ..., m0+N-1 -- the exact sequence the eager host loop uses --
so a graph run is BIT-EXACT to an eager run with the same seed (same kernels, same
RNG keys, just replayed). ``color`` already varies WITHIN one captured MCS (8
distinct launches with literal colors), so it needs no device indirection.

`wp.capture_if` (device-side conditional graph nodes)
-----------------------------------------------------
This Warp 1.14.0 build DOES expose ``wp.capture_if`` and it works on this device
(CUDA 12.4+ conditional nodes). It is the right tool for DEVICE-side branches whose
bodies are pure device launches. It is NOT applicable to FPP link create/delete:
those do HOST compaction (rebuild the link CSR from a Python inventory), which is a
capture-boundary violation regardless of conditional nodes. FPP links are static
within a sweep (engine docstring), so the captured sweep simply binds the current
link CSR; if a steppable changes link topology, the graph is re-captured (cheap,
amortised over the many MCS between topology changes). ``capture_if`` remains
available for future device-only conditional work (e.g. an on-device early-out).
"""

from __future__ import annotations

import warp as wp

from . import kernels as K
from .engine import GPUEngine
from .batched import BatchedGPUEngine

wp.init()


# ---------------------------------------------------------------------------
# Single-engine graph runner
# ---------------------------------------------------------------------------
class GraphRunner:
    """Capture + replay the per-MCS device hot loop of a single ``GPUEngine``.

    The runner SHARES the engine's device arrays (it does not copy state), so
    ``engine.get_ids()`` / ``volumes()`` / ``coms()`` reflect the graph-advanced
    state directly. The eager engine is left fully usable.
    """

    def __init__(self, engine: GPUEngine):
        self.engine = engine
        self.device = engine.device
        # device-resident mcs counter (the captured graph reads + increments it)
        self.mcs_dev = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.graph = None
        self._captured_fpp_enabled = None

    # ---- FPP binding (links static within a sweep; captured graph binds current)
    def _fpp_args(self):
        e = self.engine
        if e.fpp is not None and e.fpp.has_links():
            e.fpp.rebuild()   # host compaction -- done BEFORE capture, outside graph
            return (1, e.fpp.link_ptr, e.fpp.link_other, e.fpp.link_lambda, e.fpp.link_target)
        return (0, e._fpp_dummy_i32, e._fpp_dummy_i32, e._fpp_dummy_f32, e._fpp_dummy_f32)

    def _launch_color(self, color: int, fpp, snap):
        e = self.engine
        fpp_enabled, link_ptr, link_other, link_lambda, link_target = fpp
        snap_x, snap_y, snap_z, snap_v = snap
        wp.launch(
            K.metropolis_color_dev_mcs_kernel,
            dim=e._color_threads,
            inputs=[
                e.ids, e.cell_type, e.volume,
                e.xsum, e.ysum, e.zsum,
                e.target_volume, e.lambda_volume, e.contact,
                e.type_frozen,
                e.contact_off, e.n_contact,
                e.flip_off, e.n_flip,
                e.Lx, e.Ly, e.Lz,
                color, self.mcs_dev, e.base_seed, e.T,
                fpp_enabled, link_ptr, link_other, link_lambda, link_target,
                snap_x, snap_y, snap_z, snap_v,
            ],
            device=self.device,
        )

    def _snap_args(self, fpp_enabled):
        """Per-sweep COM/volume snapshot buffers (Phase 8). When FPP + the snapshot
        are on, allocate persistent buffers the captured graph copies into at the start
        of each MCS (a device->device ``wp.copy`` is captureable); otherwise alias the
        live arrays (byte-identical to the prior path)."""
        e = self.engine
        if fpp_enabled == 0 or not getattr(e, "fpp_com_snapshot", True):
            return (e.xsum, e.ysum, e.zsum, e.volume), False
        n1 = e.n_cells + 1
        if e._fpp_snap_xsum is None:
            e._fpp_snap_xsum = wp.zeros(n1, dtype=wp.int64, device=self.device)
            e._fpp_snap_ysum = wp.zeros(n1, dtype=wp.int64, device=self.device)
            e._fpp_snap_zsum = wp.zeros(n1, dtype=wp.int64, device=self.device)
            e._fpp_snap_volume = wp.zeros(n1, dtype=wp.float32, device=self.device)
        return (e._fpp_snap_xsum, e._fpp_snap_ysum,
                e._fpp_snap_zsum, e._fpp_snap_volume), True

    def _record(self, fpp, snap, snap_active):
        e = self.engine
        if snap_active:  # re-snapshot the live COM at the start of each replayed MCS
            wp.copy(snap[0], e.xsum); wp.copy(snap[1], e.ysum)
            wp.copy(snap[2], e.zsum); wp.copy(snap[3], e.volume)
        for color in self.engine.colors:
            self._launch_color(color, fpp, snap)
        wp.launch(K.incr_mcs_kernel, dim=1, inputs=[self.mcs_dev], device=self.device)

    def capture(self, mcs_offset: int = 0):
        """Record the per-MCS hot loop into a replayable CUDA graph.

        Graph capture RECORDS the launches without executing them, so it does NOT
        mutate the engine's lattice/SoA -- the recorded ops run only on
        ``capture_launch``. We must NOT do a state-mutating eager warm-up here (that
        would advance the real simulation by one MCS before replay). Kernel module
        compilation -- which genuinely cannot happen mid-capture -- is forced up
        front via ``wp.force_load`` plus ``force_module_load=True`` on the capture.
        ``mcs_dev`` is set so the first replay runs ``mcs_offset``.
        """
        fpp = self._fpp_args()        # FPP rebuild = host compaction, BEFORE capture
        snap, snap_active = self._snap_args(fpp[0])
        wp.force_load(self.device)    # compile kernels now (not mid-capture)
        self.mcs_dev.fill_(wp.int32(mcs_offset))
        with wp.ScopedCapture(device=self.device, force_module_load=True) as cap:
            self._record(fpp, snap, snap_active)
        self.graph = cap.graph
        self._captured_fpp_enabled = fpp[0]
        return self.graph

    def run(self, n_mcs: int, mcs_offset: int = 0):
        """Replay the captured per-MCS graph ``n_mcs`` times (capturing first if
        needed). Pure device replay -- the only host op is the final synchronise."""
        if self.graph is None:
            self.capture(mcs_offset=mcs_offset)
        else:
            self.mcs_dev.fill_(wp.int32(mcs_offset))
        for _ in range(n_mcs):
            wp.capture_launch(self.graph)
        wp.synchronize()


# ---------------------------------------------------------------------------
# Batched graph runner (parameter-sweep throughput)
# ---------------------------------------------------------------------------
class BatchedGraphRunner:
    """Capture + replay the per-MCS device hot loop of a ``BatchedGPUEngine`` (all
    R replicas advance per graph replay). Shares the engine's device arrays."""

    def __init__(self, engine: BatchedGPUEngine):
        self.engine = engine
        self.device = engine.device
        self.mcs_dev = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.graph = None

    def _record(self):
        e = self.engine
        dim = e.R * e._color_threads
        for color in e.colors:
            wp.launch(
                K.metropolis_color_batched_dev_mcs_kernel,
                dim=dim,
                inputs=[
                    e.ids, e.cell_type, e.cell_type_r,
                    e.volume, e.xsum, e.ysum, e.zsum,
                    e.target_volume, e.lambda_volume,
                    e.contact, e.n_types,
                    e.type_frozen,
                    e.contact_off, e.n_contact,
                    e.flip_off, e.n_flip,
                    e.Lx, e.Ly, e.Lz,
                    e.nvox, e.n1, e.R,
                    e._color_threads,
                    color, self.mcs_dev, e.base_seed_r, e.temperature,
                ],
                device=self.device,
            )
        wp.launch(K.incr_mcs_kernel, dim=1, inputs=[self.mcs_dev], device=self.device)

    def capture(self, mcs_offset: int = 0):
        # capture records (does not execute) the launches -> no state mutation;
        # compile kernels up front so module load never happens mid-capture.
        wp.force_load(self.device)
        self.mcs_dev.fill_(wp.int32(mcs_offset))
        with wp.ScopedCapture(device=self.device, force_module_load=True) as cap:
            self._record()
        self.graph = cap.graph
        return self.graph

    def run(self, n_mcs: int, mcs_offset: int = 0):
        if self.graph is None:
            self.capture(mcs_offset=mcs_offset)
        else:
            self.mcs_dev.fill_(wp.int32(mcs_offset))
        for _ in range(n_mcs):
            wp.capture_launch(self.graph)
        wp.synchronize()
