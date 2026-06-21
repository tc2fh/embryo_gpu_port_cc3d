"""FPP device link-inventory gate (Pass A, deliverable 1).

The FocalPointPlasticity link inventory is a **device-authoritative per-cell link
CSR** built with the Phase 1 pattern: atomic-append create + flag/compaction
delete. Links carry per-link parameters (lambda, target length, max length),
mirroring CC3D ``new_fpp_link(cell_a, cell_b, lambda, targetDist, maxDist)``.

A link is *active* iff its current COM-to-COM length <= its ``max_length``
(CC3D drops a link once the cell COMs separate past maxDistance); creation /
deletion happen at the per-MCS steppable boundary (the ``recompute_trackers``
seam), never inside the Metropolis inner loop. Within a color sweep the CSR is
static.

The CSR is a deterministic function of (topology, COMs, max_lengths), so we
validate it EXACTLY against an independent NumPy reference -- not statistically.
"""

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


def _ref_active_and_csr(pairs, lengths, max_lengths, n_cells):
    """Reference: which links are active, and the per-cell undirected CSR
    (each kept undirected link contributes an entry to BOTH endpoints)."""
    keep = lengths <= max_lengths
    n1 = n_cells + 1
    deg = np.zeros(n1, dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        if keep[i]:
            deg[a] += 1
            deg[b] += 1
    # adjacency sets per cell (order-independent comparison)
    adj = [set() for _ in range(n1)]
    for i, (a, b) in enumerate(pairs):
        if keep[i]:
            adj[a].add(int(b))
            adj[b].add(int(a))
    return keep, deg, adj


def _csr_to_adjsets(link_ptr, link_other, n_cells):
    n1 = n_cells + 1
    adj = [set() for _ in range(n1)]
    for cid in range(n1):
        for k in range(int(link_ptr[cid]), int(link_ptr[cid + 1])):
            adj[cid].add(int(link_other[k]))
    return adj


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_fpp_link_csr_build_matches_reference():
    """Build a CSR from an explicit topology + COMs; it must match the NumPy
    reference exactly (degrees + adjacency), including per-link max-length cuts."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=5)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    n = eng.n_cells

    # a deterministic link topology over the 27 cells (a chain + a few extras)
    pairs = []
    for c in range(1, n):
        pairs.append((c, c + 1))
    pairs += [(1, 5), (2, 10), (3, 27), (7, 20)]
    pairs = np.array(sorted({tuple(sorted(p)) for p in pairs}), dtype=np.int32)
    npairs = pairs.shape[0]

    lam = np.full(npairs, 7.0, dtype=np.float32)
    target = np.full(npairs, 3.0, dtype=np.float32)
    # mix of max lengths so some links are cut
    maxlen = np.full(npairs, 6.0, dtype=np.float32)
    maxlen[::3] = 1.5  # tight cut on every third link

    links = FPPLinks(eng, target_length_default=3.0, lambda_default=7.0, max_length_default=6.0)
    links.set_topology(pairs, lam, target, maxlen)
    links.rebuild()

    # reference lengths from the engine's own COMs (single source of truth)
    coms = np.zeros((n + 1, 3))
    coms[1:] = eng.coms()
    a = pairs[:, 0]; b = pairs[:, 1]
    d = coms[a] - coms[b]
    lengths = np.sqrt((d * d).sum(axis=1))

    ref_keep, ref_deg, ref_adj = _ref_active_and_csr(pairs, lengths, maxlen, n)

    link_ptr = links.link_ptr.numpy()
    link_other = links.link_other.numpy()

    # degrees match
    deg = np.diff(link_ptr)
    assert np.array_equal(deg.astype(np.int64), ref_deg), (
        f"per-cell link degree mismatch\n{deg.astype(np.int64)}\n{ref_deg}"
    )
    # adjacency sets match exactly (CSR order is unspecified -> compare as sets)
    gpu_adj = _csr_to_adjsets(link_ptr, link_other, n)
    for cid in range(n + 1):
        assert gpu_adj[cid] == ref_adj[cid], f"cell {cid} adjacency mismatch"

    # active-link count matches the kept count
    assert links.num_active() == int(ref_keep.sum())


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_fpp_link_create_delete_lifecycle():
    """create_link / delete_link at the steppable boundary update the inventory;
    after rebuild the CSR reflects the new topology exactly."""
    cfg = EngineConfig(Lx=16, Ly=16, Lz=16, seed=2)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=2))
    n = eng.n_cells  # 8 cells

    links = FPPLinks(eng, target_length_default=2.0, lambda_default=5.0, max_length_default=100.0)
    links.rebuild()
    assert links.num_active() == 0

    # create a few links (large max_length so none are cut)
    links.create_link(1, 2)
    links.create_link(2, 3)
    links.create_link(1, 8)
    links.rebuild()
    adj = _csr_to_adjsets(links.link_ptr.numpy(), links.link_other.numpy(), n)
    assert adj[1] == {2, 8}
    assert adj[2] == {1, 3}
    assert adj[3] == {2}
    assert links.num_active() == 3

    # delete one undirected link -> disappears from both endpoints
    links.delete_link(1, 2)
    links.rebuild()
    adj = _csr_to_adjsets(links.link_ptr.numpy(), links.link_other.numpy(), n)
    assert adj[1] == {8}
    assert adj[2] == {3}
    assert links.num_active() == 2


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_fpp_max_length_cut_is_dynamic():
    """A link longer than its max_length is dropped (delete); when the COMs come
    back within range on a later rebuild it is restored (create) -- the dynamic
    flag+compaction lifecycle."""
    cfg = EngineConfig(Lx=20, Ly=20, Lz=20, seed=1)
    eng = GPUEngine(build_grid_state(cfg, cells_per_axis=3))
    n = eng.n_cells

    coms = np.zeros((n + 1, 3)); coms[1:] = eng.coms()
    # pick two far-apart cells and one close pair
    far = (1, n)              # opposite corners of the 3x3x3 grid
    near = (1, 2)
    dfar = np.linalg.norm(coms[far[0]] - coms[far[1]])
    dnear = np.linalg.norm(coms[near[0]] - coms[near[1]])
    assert dfar > dnear

    pairs = np.array([far, near], dtype=np.int32)
    # max_length between the two distances: far link cut, near link kept
    cut = (dfar + dnear) / 2.0
    maxlen = np.array([cut, cut], dtype=np.float32)
    links = FPPLinks(eng)
    links.set_topology(pairs, np.full(2, 5.0, np.float32), np.full(2, 2.0, np.float32), maxlen)
    links.rebuild()
    assert links.num_active() == 1
    adj = _csr_to_adjsets(links.link_ptr.numpy(), links.link_other.numpy(), n)
    # cell 1 is shared between far=(1,n) and near=(1,2): after the far cut it keeps
    # only the near link; cell n (far's other endpoint) is fully disconnected.
    assert adj[near[0]] == {near[1]}      # cell 1 -> {2}
    assert adj[far[1]] == set()           # cell n -> {}

    # now relax the cut so BOTH are within range -> far link restored
    links.set_max_lengths(np.array([dfar + 1.0, dfar + 1.0], dtype=np.float32))
    links.rebuild()
    assert links.num_active() == 2
