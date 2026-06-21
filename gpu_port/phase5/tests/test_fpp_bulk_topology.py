"""Batched FPP topology edits gate (post-Phase-4 perf work).

Phase 3's ``FPPLinks.create_link`` / ``delete_link`` each ``np.append`` (create) or
mask+recompact (delete) the WHOLE link inventory per call. A steppable's per-MCS
relink/turnover loop issues thousands of those -> O(M*k) host time, which (with the
neighbor-CSR now on device) became the dominant per-MCS cost at full Embryo scale.

``create_links_bulk`` / ``delete_links_bulk`` apply a whole batch in one allocation /
one vectorized compaction. The contract verified here:

  1. a batch is BYTE-IDENTICAL to the equivalent sequence of single calls (so the
     device CSR -- and thus every downstream observable -- is unchanged), and
  2. the single-call wrappers still delegate to the bulk path (no behavior drift),
  3. batching 50k links is effectively instant (impossible under O(M^2) np.append).
"""

import time

import numpy as np
import pytest

from engine import EngineConfig, build_grid_state, GPUEngine
from engine.fpp import FPPLinks


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _csr_to_adjsets(ptr, other, n):
    adj = {c: set() for c in range(1, n + 1)}
    for c in range(1, n + 1):
        lo, hi = int(ptr[c]), int(ptr[c + 1])
        adj[c] = set(int(x) for x in other[lo:hi])
    return adj


def _inventory(links):
    return (links._a.copy(), links._b.copy(),
            links._lam.copy(), links._tgt.copy(), links._max.copy())


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_create_bulk_matches_single_calls():
    """k separate ``create_link`` calls and one ``create_links_bulk`` of the same k
    links yield byte-identical _a/_b/_lam/_tgt/_max (order + dtype + values)."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=2))  # 8 cells

    pairs = [(1, 2), (2, 3), (1, 8), (4, 5), (3, 7)]
    lams = [600.0, 10.0, 600.0, 5.0, 600.0]
    tgts = [5.0, 1.0, 5.0, 2.0, 5.0]
    mxs = [10.0, 5.0, 10.0, 100.0, 10.0]

    single = FPPLinks(eng)
    for (a, b), lm, tg, mx in zip(pairs, lams, tgts, mxs):
        single.create_link(a, b, lam=lm, target=tg, maxlen=mx)

    bulk = FPPLinks(eng)
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    bulk.create_links_bulk(a, b, lam=lams, target=tgts, maxlen=mxs)

    for s, k in zip(_inventory(single), _inventory(bulk)):
        assert s.dtype == k.dtype
        assert np.array_equal(s, k)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_create_bulk_scalar_broadcast_matches_defaults():
    """A scalar per-link param broadcasts to the whole batch, identical to passing
    that same scalar to each single ``create_link`` call (the common steppable
    case: one lambda/target/max for the entire relink batch)."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=2)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=2))

    pairs = [(1, 2), (3, 4), (5, 6)]
    single = FPPLinks(eng)
    for a, b in pairs:
        single.create_link(a, b, lam=600.0, target=5.0, maxlen=10.0)
    bulk = FPPLinks(eng)
    bulk.create_links_bulk([p[0] for p in pairs], [p[1] for p in pairs],
                           lam=600.0, target=5.0, maxlen=10.0)
    for s, k in zip(_inventory(single), _inventory(bulk)):
        assert np.array_equal(s, k)
    # None -> class defaults, same as single create_link(None)
    sd = FPPLinks(eng); sd.create_link(1, 2)
    kd = FPPLinks(eng); kd.create_links_bulk([1], [2])
    for s, k in zip(_inventory(sd), _inventory(kd)):
        assert np.array_equal(s, k)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_delete_bulk_matches_single_calls():
    """Deleting k links one-by-one vs ``delete_links_bulk`` of the same k yields an
    identical remaining inventory, including the 'remove ALL matching {a,b}'
    (duplicate) and unordered-endpoint semantics of the original."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=3)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=2))

    # include a duplicate link (2,3)/(3,2) and an order-flipped delete target
    base = [(1, 2), (2, 3), (1, 8), (4, 5), (3, 2), (6, 7)]

    single = FPPLinks(eng)
    bulk = FPPLinks(eng)
    for a, b in base:
        single.create_link(a, b)
        bulk.create_link(a, b)

    dels = [(2, 1), (3, 2), (7, 6)]   # flipped endpoints; (3,2) hits both dups
    for a, b in dels:
        single.delete_link(a, b)
    bulk.delete_links_bulk(dels)

    for s, k in zip(_inventory(single), _inventory(bulk)):
        assert np.array_equal(s, k)
    # only (1,8) and (4,5) survive
    assert single._a.shape[0] == 2


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_bulk_csr_byte_identical_after_rebuild():
    """The device CSR built from a bulk-edited inventory is byte-identical to the
    one built from the equivalent single-call inventory."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=4)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=2))
    n = eng.n_cells

    pairs = [(1, 2), (2, 3), (1, 8), (4, 5), (3, 7), (6, 7)]
    single = FPPLinks(eng, max_length_default=100.0)
    bulk = FPPLinks(eng, max_length_default=100.0)
    for a, b in pairs:
        single.create_link(a, b)
    bulk.create_links_bulk([p[0] for p in pairs], [p[1] for p in pairs])

    single.delete_link(1, 2)
    bulk.delete_links_bulk([(1, 2)])
    single.rebuild()
    bulk.rebuild()

    assert np.array_equal(single.link_ptr.numpy(), bulk.link_ptr.numpy())
    a1 = _csr_to_adjsets(single.link_ptr.numpy(), single.link_other.numpy(), n)
    a2 = _csr_to_adjsets(bulk.link_ptr.numpy(), bulk.link_other.numpy(), n)
    assert a1 == a2
    assert a1[1] == {8}  # (1,2) gone, (1,8) remains


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_bulk_create_is_not_quadratic():
    """50k links in one ``create_links_bulk`` is effectively instant; the same via
    50k single ``create_link`` (np.append) calls would be O(M^2). Guards the perf
    regression that motivated the batch API."""
    cfg = EngineConfig(Lx=32, Ly=32, Lz=32, seed=5)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=4))  # 64 cells
    links = FPPLinks(eng)

    k = 50_000
    rng = np.random.default_rng(0)
    a = rng.integers(1, eng.n_cells + 1, size=k).astype(np.int32)
    b = rng.integers(1, eng.n_cells + 1, size=k).astype(np.int32)

    t0 = time.perf_counter()
    links.create_links_bulk(a, b, lam=600.0, target=5.0, maxlen=10.0)
    dt = time.perf_counter() - t0

    assert links._a.shape[0] == k
    assert dt < 0.5, f"bulk create of {k} links took {dt:.3f}s (expected << 0.5s)"
