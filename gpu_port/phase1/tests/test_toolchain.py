"""Toolchain + interop gate for the Phase 1 GPU-FPP spike.

Asserts the three things the spike must prove about the GPU framework in this
env (win-64 / Python 3.12 / CUDA-13 / torch 2.12.1+cu130):

1. NVIDIA Warp imports.
2. A trivial ``@wp.kernel`` actually runs on the GPU and produces correct output.
3. ``wp.from_torch`` / ``wp.to_torch`` zero-copy interop round-trips a CUDA tensor
   (same device pointer; a Warp kernel mutating the buffer is visible in torch).

GPU tests skip cleanly (not fail) if no CUDA device is present, so the gate is
meaningful only where a GPU exists. On this machine CUDA IS present, so they run.
"""

import numpy as np
import pytest


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def test_warp_imports():
    import warp as wp  # noqa: F401

    assert wp.config.version is not None


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_warp_kernel_runs_on_gpu():
    import warp as wp

    wp.init()
    assert wp.get_cuda_device_count() >= 1

    @wp.kernel
    def _scale(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = a[i] * 2.0 + 1.0

    n = 4096
    a = wp.full(n, 3.0, dtype=wp.float32, device="cuda:0")
    out = wp.zeros(n, dtype=wp.float32, device="cuda:0")
    wp.launch(_scale, dim=n, inputs=[a, out], device="cuda:0")
    wp.synchronize()
    res = out.numpy()
    assert np.allclose(res, 7.0), f"kernel result wrong: {res[:5]}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_torch_warp_zero_copy_roundtrip():
    import torch

    import warp as wp

    wp.init()
    n = 2048
    t = torch.arange(n, dtype=torch.float32, device="cuda")

    # from_torch must alias the SAME device memory (zero copy)
    wa = wp.from_torch(t)
    assert wa.device.is_cuda
    assert wa.ptr == t.data_ptr(), "wp.from_torch did not alias torch memory"

    # mutate via a Warp kernel; change must be visible in the torch tensor
    @wp.kernel
    def _inc(x: wp.array(dtype=wp.float32)):
        i = wp.tid()
        x[i] = x[i] + 10.0

    wp.launch(_inc, dim=n, inputs=[wa], device="cuda:0")
    wp.synchronize()
    expected = torch.arange(n, dtype=torch.float32, device="cuda") + 10.0
    assert torch.allclose(t, expected), "zero-copy mutation not visible in torch"

    # to_torch round-trips back to the same pointer
    wb = wp.zeros(n, dtype=wp.float32, device="cuda:0")
    tb = wp.to_torch(wb)
    assert tb.is_cuda
    assert tb.data_ptr() == wb.ptr, "wp.to_torch did not alias warp memory"
