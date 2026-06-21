"""Phase 8, Objective 2 (d) -- batched throughput: device-collapsed vs O(R)-host.

The headline sweep-throughput win: ``BatchedDeviceEmbryoModel`` (the Phase-8 collapse --
one keyed-global CSR + one batched fused cohesotaxis + per-replica device link kernels +
a device combine, NO per-MCS host data movement) vs ``BatchedEmbryoModel`` (the
pre-phase O(R)-host path -- per-replica Python loops with a contact-graph copyback +
host adjacency dicts each replica each MCS).

In-gate: a small R is timed and the device path is asserted FASTER than the host path
(the throughput improvement gate (d)), with the replica-MCS/s + sims/hour printed. The
full R=8/32/64 benchmark runs under ``BENCH=1`` (heavy; outside the ~2-min gate).
"""

import os
import time

import numpy as np
import pytest

import warp as wp

from engine import BatchedGPUEngine, BatchedState
from embryo import build_closure_scene, BatchedDeviceEmbryoModel
from embryo.batched_steppables import BatchedEmbryoModel
from embryo.params import DEFAULT as P


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
BENCH = os.environ.get("BENCH") == "1"
ENABLE = ("tissue", "lamellipodia", "passive_substrate")


def _bench(model_cls, cfg, ids, cell_type, R, n_mcs, warmup=2):
    bstate = BatchedState(cfg, np.repeat(ids[None], R, axis=0), cell_type)
    beng = BatchedGPUEngine(bstate)
    model = model_cls(beng, params=P, enable=ENABLE)
    model.start()
    model.run(warmup)               # warm up (kernel compile + first-touch allocs)
    wp.synchronize()
    t0 = time.perf_counter()
    model.run(n_mcs, mcs_offset=warmup)
    wp.synchronize()
    dt = time.perf_counter() - t0
    return R * n_mcs / dt            # replica-MCS/s


@cuda
def test_device_collapse_faster_than_host_loop():
    """In-gate throughput sanity: at a modest R the device-collapsed batched model
    (single batched launches: keyed-global CSR + batched tissue/substrate + batched
    fused cohesotaxis + device combine) is FASTER than the pre-phase O(R)-host model
    (the gate-(d) improvement), and both advance the same number of replica-MCS. Prints
    replica-MCS/s + sims/hour."""
    state, info = build_closure_scene(L=20, seed=5, n_leaders=6)
    cfg = state.cfg
    R, n_mcs = 16, 15
    host = _bench(BatchedEmbryoModel, cfg, state.ids, state.cell_type, R, n_mcs)
    dev = _bench(BatchedDeviceEmbryoModel, cfg, state.ids, state.cell_type, R, n_mcs)
    sims_hr = dev / 10000.0 * 3600.0
    print(f"\n[phase8 throughput] R={R}: host-loop {host:.0f} rep-MCS/s | "
          f"device {dev:.0f} rep-MCS/s ({dev/host:.2f}x) | {sims_hr:.0f} sims/hr @10k MCS")
    assert dev > host, (
        f"device-collapsed path ({dev:.0f}) not faster than O(R)-host ({host:.0f})")


@pytest.mark.skipif(not (_cuda_available() and BENCH), reason="set BENCH=1 for the heavy R=8/32/64 sweep")
def test_throughput_sweep_R_8_32_64():
    """Heavy benchmark (BENCH=1): replica-MCS/s + sims/hour at R=8/32/64, device vs
    O(R)-host. Reports the headline sweep-throughput win."""
    state, info = build_closure_scene(L=24, seed=5, n_leaders=6)
    cfg = state.cfg
    n_mcs = 20
    print("\n[phase8 BENCH] device-collapsed vs O(R)-host batched Embryo (L=24):")
    for R in (8, 32, 64):
        host = _bench(BatchedEmbryoModel, cfg, state.ids, state.cell_type, R, n_mcs)
        dev = _bench(BatchedDeviceEmbryoModel, cfg, state.ids, state.cell_type, R, n_mcs)
        print(f"  R={R:3d}: host {host:8.0f} | device {dev:8.0f} rep-MCS/s "
              f"({dev/host:.2f}x) | device {dev/10000*3600:7.0f} sims/hr @10k MCS")
        assert dev > host
