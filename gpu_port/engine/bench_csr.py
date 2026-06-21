"""Neighbor-contact CSR build benchmark -- host compaction vs on-device compaction.

Two measurements (run as ``pixi run python -m engine.bench_csr``):

1. **Isolated CSR build** (``bench_csr_isolated``): on the real scaled-Embryo geometry,
   time ``neighbor_contact_csr(method="host")`` vs ``method="device")`` (order-1, the
   production setting). Both re-run the same device hash kernel; they differ only in how
   the hash table is compacted into the CSR -- so the ratio is the win on the per-MCS CSR
   build. Reports ms/call, speedup, #contacts, hash ``cap``, and the host<-device copy
   volume each path moves across PCIe.

2. **End-to-end Embryo** (``bench_embryo_endtoend``): full ``EmbryoModel`` (cube_size=100),
   MCS/s with ``csr_method`` = host vs device, plus an instrumented per-MCS breakdown
   (device sweep / CSR build / steppable host loops) so the real wall-clock impact and the
   next bottleneck are both explicit. (The breakdown pass adds per-segment syncs, so its
   absolute MCS/s is lower than the un-instrumented headline -- use the headline for the
   speedup, the breakdown for the shares.)
"""

from __future__ import annotations

import time

import numpy as np

import warp as wp

wp.init()


def _time_calls(fn, reps: int) -> float:
    """Mean ms per call of ``fn`` (which returns host arrays, so it fully completes
    each call). Device is assumed warm."""
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    wp.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def bench_csr_isolated(cube_size: int = 100, n_mcs: int = 5, reps: int = 20,
                       order: int = 1, seed: int = 12345) -> dict:
    from embryo.model import build_scaled_embryo
    from engine import GPUEngine

    state, info = build_scaled_embryo(cube_size=cube_size, seed=seed)
    eng = GPUEngine(state)
    eng.run(n_mcs)  # settle to a representative contact topology

    # warm up both compaction paths (first call compiles kernels / allocates scratch)
    eng.neighbor_contact_csr(order=order, method="host")
    eng.neighbor_contact_csr(order=order, method="device")

    indptr, indices, data = eng.neighbor_contact_csr(order=order, method="device")
    n1 = eng.n_cells + 1
    n_contacts = int(indptr[-1])
    cap = int(eng._ht_cap)

    host_ms = _time_calls(lambda: eng.neighbor_contact_csr(order=order, method="host"), reps)
    dev_ms = _time_calls(lambda: eng.neighbor_contact_csr(order=order, method="device"), reps)

    # PCIe traffic each path moves to produce the CSR on the host
    host_copy_mb = cap * (8 + 4) / 1e6                      # ht_key int64 + ht_count int32
    dev_copy_mb = (n1 + (n1 + 1) + 2 * n_contacts) * 4 / 1e6  # counts + indptr up + indices+data
    return {
        "cube_size": cube_size, "n_cells": eng.n_cells, "n_voxels": eng.cfg.n_voxels,
        "order": order, "n_contacts": n_contacts, "cap": cap,
        "host_ms": host_ms, "device_ms": dev_ms,
        "speedup": host_ms / dev_ms if dev_ms > 0 else float("nan"),
        "host_copy_mb": host_copy_mb, "device_copy_mb": dev_copy_mb,
        "copy_reduction": host_copy_mb / dev_copy_mb if dev_copy_mb > 0 else float("nan"),
    }


def _run_plain(model, n_mcs: int, offset: int) -> float:
    """Un-instrumented MCS/s for model.run-equivalent loop (one sync at the end)."""
    eng = model.engine
    wp.synchronize()
    t0 = time.perf_counter()
    for m in range(n_mcs):
        mcs = offset + m
        eng.step_mcs(mcs)
        model._inject_shared_csr()
        for s in model.manager.steppables:
            if mcs % s.frequency == 0:
                s.step(mcs)
    wp.synchronize()
    return n_mcs / (time.perf_counter() - t0)


def _run_instrumented(model, n_mcs: int, offset: int) -> dict:
    """Per-segment per-MCS timing (sync after each segment to attribute device time)."""
    eng = model.engine
    t_sweep = t_csr = t_stepp = 0.0
    wp.synchronize()
    for m in range(n_mcs):
        mcs = offset + m
        t0 = time.perf_counter(); eng.step_mcs(mcs); wp.synchronize(); t_sweep += time.perf_counter() - t0
        t0 = time.perf_counter(); model._inject_shared_csr(); wp.synchronize(); t_csr += time.perf_counter() - t0
        t0 = time.perf_counter()
        for s in model.manager.steppables:
            if mcs % s.frequency == 0:
                s.step(mcs)
        wp.synchronize(); t_stepp += time.perf_counter() - t0
    n = float(n_mcs)
    return {"sweep_ms": t_sweep / n * 1000, "csr_ms": t_csr / n * 1000,
            "steppable_ms": t_stepp / n * 1000}


def bench_embryo_endtoend(cube_size: int = 100, n_mcs: int = 20, warmup: int = 3,
                          seed: int = 12345) -> dict:
    from embryo.model import build_scaled_embryo, EmbryoModel

    out = {"cube_size": cube_size, "n_mcs": n_mcs}
    for method in ("host", "device"):
        state, _ = build_scaled_embryo(cube_size=cube_size, seed=seed)
        model = EmbryoModel(state, csr_method=method)
        model.start()
        _run_plain(model, warmup, offset=0)                       # warm up
        mcs_s = _run_plain(model, n_mcs, offset=warmup)           # headline MCS/s
        brk = _run_instrumented(model, n_mcs, offset=warmup + n_mcs)  # breakdown
        out[method] = {"mcs_per_s": mcs_s, **brk}
    h, d = out["host"], out["device"]
    out["end_to_end_speedup"] = d["mcs_per_s"] / h["mcs_per_s"] if h["mcs_per_s"] > 0 else float("nan")
    out["csr_build_speedup"] = h["csr_ms"] / d["csr_ms"] if d["csr_ms"] > 0 else float("nan")
    return out


def main():
    print("=== Neighbor-contact CSR: host vs on-device compaction (RTX 5090) ===")
    for L in (50, 100):
        d = bench_csr_isolated(cube_size=L)
        print(
            f"[isolated cube={d['cube_size']}^3 cells={d['n_cells']} "
            f"contacts={d['n_contacts']} cap={d['cap']:,}] "
            f"host {d['host_ms']:7.2f} ms | device {d['device_ms']:7.2f} ms | "
            f"speedup x{d['speedup']:.1f} | "
            f"PCIe host {d['host_copy_mb']:.1f} MB -> device {d['device_copy_mb']:.2f} MB "
            f"(x{d['copy_reduction']:.0f} less)"
        )
    print("--- end-to-end full Embryo (100^3) ---")
    e = bench_embryo_endtoend()
    h, d = e["host"], e["device"]
    print(
        f"[host  ] {h['mcs_per_s']:6.2f} MCS/s | per-MCS: sweep {h['sweep_ms']:.1f} ms, "
        f"CSR {h['csr_ms']:.1f} ms, steppables {h['steppable_ms']:.1f} ms"
    )
    print(
        f"[device] {d['mcs_per_s']:6.2f} MCS/s | per-MCS: sweep {d['sweep_ms']:.1f} ms, "
        f"CSR {d['csr_ms']:.1f} ms, steppables {d['steppable_ms']:.1f} ms"
    )
    print(
        f"=> CSR build speedup x{e['csr_build_speedup']:.1f} | "
        f"end-to-end speedup x{e['end_to_end_speedup']:.2f}"
    )


if __name__ == "__main__":
    main()
