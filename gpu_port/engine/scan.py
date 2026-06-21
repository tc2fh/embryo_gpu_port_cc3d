"""Device exclusive prefix-sum primitives for CSR row pointers (Phase 6).

Every per-MCS CSR build in the engine -- the FPP link inventory (``fpp.py``), the
neighbor-contact CSR (``engine.py``), and the batched per-replica link inventory
(``batched_fpp.py``) -- turned a per-row ``degree`` vector into a CSR row pointer
with the host idiom ``ptr[1:] = np.cumsum(degree[:n])``. That copied the degree
array off the GPU, ran NumPy, and reallocated a fresh ``wp.array`` every step.

This module replaces those three sites with on-device scans that are
**byte-identical** to the ``np.cumsum`` they replace (integer add is associative,
so a sequential accumulate equals NumPy's bit-for-bit, and an inclusive
``wp.utils.array_scan`` into the ``ptr[1:]`` slice equals the shifted exclusive
prefix). No host roundtrip, no per-step realloc.

The CSR-pointer contract (shared by all helpers): given ``degree`` (length >= n+1,
only ``[0:n]`` read) write ``out_ptr`` (length >= n+1) with ``out_ptr[0] = 0`` and
``out_ptr[k] = sum(degree[0:k])`` for k in 1..n.
"""

from __future__ import annotations

import warp as wp

from . import kernels as K

wp.init()


# Cached int32 scratch for the int64 indptr path (array_scan has no int64, so we
# scan in int32 then widen). Keyed by (device, capacity); grown on demand. The CSR
# n axis is fixed per engine, so this allocates once and is reused every MCS.
_I64_SCRATCH: dict = {}


def _i32_scratch(device: str, n_plus_1: int) -> wp.array:
    key = device
    buf = _I64_SCRATCH.get(key)
    if buf is None or buf.shape[0] < n_plus_1:
        buf = wp.zeros(max(n_plus_1, 1024), dtype=wp.int32, device=device)
        _I64_SCRATCH[key] = buf
    return buf


def exclusive_scan_to_ptr_i32(degree: wp.array, n: int, out_ptr: wp.array,
                              device: str) -> None:
    """int32 single-segment scan -> ``out_ptr`` (the FPP ``link_ptr`` site).

    Uses ``wp.utils.array_scan`` (a real parallel scan): an INCLUSIVE scan of
    ``degree[0:n]`` written into the ``out_ptr[1:n+1]`` slice is exactly the shifted
    exclusive prefix, and ``out_ptr[0]`` is left at its (pre-zeroed) 0. The caller
    must pass an ``out_ptr`` of length >= n+1 whose slot 0 is already 0 (a fresh
    ``wp.zeros`` or an explicit ``out_ptr[0:1].zero_()``)."""
    if n <= 0:
        out_ptr.zero_()
        return
    wp.utils.array_scan(degree[0:n], out_ptr[1:n + 1], True)


def exclusive_scan_to_ptr_i64(degree: wp.array, n: int, out_ptr: wp.array,
                              device: str) -> None:
    """int64 single-segment scan -> ``out_ptr`` (the neighbor-CSR ``indptr`` site).

    int64 to match the ``neighbor_contact_csr`` return dtype, but built with the
    PARALLEL int32 ``array_scan`` (no int64 variant exists) into an int32 scratch,
    then widened to int64 -- both parallel. Byte-identical in value to
    ``out[1:]=np.cumsum(degree[:n])`` (the int32 prefix never overflows for any scene
    the rest of the CSR build handles, which already sizes int32-indexed buffers)."""
    if n <= 0:
        out_ptr.zero_()
        return
    tmp = _i32_scratch(device, n + 1)
    tmp[0:1].zero_()                             # only slot 0 (array_scan fills [1:n+1])
    wp.utils.array_scan(degree[0:n], tmp[1:n + 1], True)
    wp.launch(K.cast_i32_to_i64_kernel, dim=n + 1,
              inputs=[tmp, int(n + 1), out_ptr], device=device)


def segmented_exclusive_scan_to_ptr_i32(degree: wp.array, R: int, n1: int,
                                        out_ptr: wp.array, device: str) -> None:
    """int32 per-replica (segmented) scan -> per-replica LOCAL ``link_ptr`` (the
    batched FPP site). ``degree`` / ``out_ptr`` are flat ``(R*(n1+1),)`` replica-major;
    each replica's prefix resets at its segment boundary. One thread per replica."""
    if R <= 0 or n1 <= 0:
        out_ptr.zero_()
        return
    wp.launch(K.segmented_exclusive_scan_to_ptr_i32_kernel, dim=int(R),
              inputs=[degree, int(R), int(n1), out_ptr], device=device)
