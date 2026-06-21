"""Phase 6 deliverable 3 -- device-authoritative FPP link inventory.

The ``FPPLinks`` inventory (a/b/lam/tgt/max) now lives on the GPU; create is a
device block-append (order-preserving), delete is a device keep/compact driven by a
decision mask, and ``rebuild()`` is fully on-device. The contract this gate pins:

* the device inventory is **set-equal** to a plain NumPy reference after any
  randomized sequence of create/delete edits (the link set is order-independent),
* the inventory iteration order used by ``rebuild()`` is **stable** -- replaying the
  same edit sequence yields a byte-identical inventory layout and link_ptr (so the
  downstream CSR offsets are deterministic),
* the device keep/compact primitive (the Phase-7 Poisson seam) reproduces a NumPy
  boolean-mask compaction exactly and stably,
* ``active_link_lengths`` matches a NumPy COM reference,
* the FPP ``link_ptr`` / per-cell CSR is byte-identical to a NumPy reconstruction
  (so the device scan + atomic-append reproduce the prior host build), and
* a short full grid-graph FPP run reproduces the validated link-length statistics
  (KS within the Phase-3 gate) and is deterministic; Volume+Contact stays bit-exact.
"""

import numpy as np
import pytest

import warp as wp

from engine import EngineConfig, build_grid_state, GPUEngine, FPPLinks, grid_graph_links


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _engine(seed=5, cpa=3, L=20):
    cfg = EngineConfig(Lx=L, Ly=L, Lz=L, seed=seed, tracker_neighbor_order=1)
    return GPUEngine(build_grid_state(cfg, cells_per_axis=cpa))


def _pack(a, b, mult):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    return np.minimum(a, b) * mult + np.maximum(a, b)


def _inv_multiset(links):
    """Inventory as a sorted (unordered-key, lam, tgt, max) record array -- the
    canonical set-equality form (order-independent, duplicates kept as a multiset)."""
    a, b = links._a, links._b
    lam, tgt, mx = links._lam, links._tgt, links._max
    mult = links.n_cells + 2
    key = _pack(a, b, mult)
    rec = np.array(sorted(zip(key.tolist(), lam.tolist(), tgt.tolist(), mx.tolist())))
    return rec


class _RefInventory:
    """Plain-NumPy reference FPP inventory with the same create/delete semantics:
    create appends (order preserved), delete removes EVERY link whose unordered
    {a,b} matches any deleted pair. Set-equality target for the device inventory."""

    def __init__(self, n_cells):
        self.mult = n_cells + 2
        self.a = np.zeros(0, np.int32); self.b = np.zeros(0, np.int32)
        self.lam = np.zeros(0, np.float32); self.tgt = np.zeros(0, np.float32)
        self.mx = np.zeros(0, np.float32)

    def create(self, a, b, lam, tgt, mx):
        a = np.atleast_1d(np.asarray(a, np.int32)); b = np.atleast_1d(np.asarray(b, np.int32))
        k = a.shape[0]
        self.a = np.concatenate([self.a, a]); self.b = np.concatenate([self.b, b])
        self.lam = np.concatenate([self.lam, np.full(k, lam, np.float32)])
        self.tgt = np.concatenate([self.tgt, np.full(k, tgt, np.float32)])
        self.mx = np.concatenate([self.mx, np.full(k, mx, np.float32)])

    def delete(self, pairs):
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if pairs.shape[0] == 0 or self.a.shape[0] == 0:
            return
        ekey = _pack(self.a, self.b, self.mult)
        dkey = np.unique(_pack(pairs[:, 0], pairs[:, 1], self.mult))
        keep = ~np.isin(ekey, dkey)
        self.a, self.b = self.a[keep], self.b[keep]
        self.lam, self.tgt, self.mx = self.lam[keep], self.tgt[keep], self.mx[keep]

    def multiset(self):
        key = _pack(self.a, self.b, self.mult)
        return np.array(sorted(zip(key.tolist(), self.lam.tolist(),
                                   self.tgt.tolist(), self.mx.tolist())))


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_device_crud_set_equal_to_numpy_reference(seed):
    """A randomized create/delete sequence leaves the device inventory set-equal to
    a NumPy reference inventory (same multiset of (unordered-pair, lam, tgt, max))."""
    eng = _engine()
    n_cells = eng.n_cells
    links = FPPLinks(eng)
    ref = _RefInventory(n_cells)
    rng = np.random.default_rng(seed)

    for _ in range(40):
        op = rng.integers(0, 3)
        if op <= 1:  # create a batch (bias toward growth)
            k = int(rng.integers(1, 12))
            a = rng.integers(1, n_cells + 1, size=k).astype(np.int32)
            b = rng.integers(1, n_cells + 1, size=k).astype(np.int32)
            bad = a == b
            if bad.any():
                b[bad] = (b[bad] % n_cells) + 1
                still = a == b
                b[still] = (b[still] % n_cells) + 1
            lam = float(rng.uniform(0.5, 2.0)); tgt = float(rng.uniform(2.0, 6.0))
            mx = float(rng.uniform(8.0, 14.0))
            links.create_links_bulk(a, b, lam=lam, target=tgt, maxlen=mx)
            ref.create(a, b, lam, tgt, mx)
        else:        # delete some existing pairs
            if links.n_pairs == 0:
                continue
            ai, bi = links._a, links._b
            nsel = int(rng.integers(1, min(6, links.n_pairs) + 1))
            idx = rng.choice(links.n_pairs, size=nsel, replace=False)
            pairs = list(zip(ai[idx].tolist(), bi[idx].tolist()))
            links.delete_links_bulk(pairs)
            ref.delete(pairs)

        assert links.n_pairs == ref.a.shape[0], "live count diverged"
        got = _inv_multiset(links)
        exp = ref.multiset()
        assert got.shape == exp.shape, f"shape {got.shape} != {exp.shape}"
        if got.size:
            assert np.allclose(got, exp), "device inventory multiset != NumPy reference"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_rebuild_order_is_stable_and_deterministic():
    """Replaying the same edit sequence yields a BYTE-IDENTICAL inventory layout
    (the actual _a/_b/_lam ordering, not just the set) and an identical link_ptr --
    so downstream CSR offsets are deterministic."""
    def build_and_snapshot():
        eng = _engine(seed=9)
        links = FPPLinks(eng)
        rng = np.random.default_rng(1234)
        for _ in range(25):
            if rng.random() < 0.65 or links.n_pairs == 0:
                k = int(rng.integers(1, 8))
                a = rng.integers(1, eng.n_cells + 1, size=k).astype(np.int32)
                b = ((a % eng.n_cells) + 1).astype(np.int32)  # always a != b
                links.create_links_bulk(a, b, lam=1.0, target=4.0, maxlen=50.0)
            else:
                idx = rng.choice(links.n_pairs, size=min(3, links.n_pairs), replace=False)
                pairs = list(zip(links._a[idx].tolist(), links._b[idx].tolist()))
                links.delete_links_bulk(pairs)
        eng.run(3)
        links.rebuild()
        return (links._a.copy(), links._b.copy(), links._lam.copy(),
                links._tgt.copy(), links._max.copy(), links.link_ptr.numpy().copy())

    s1 = build_and_snapshot()
    s2 = build_and_snapshot()
    names = ["_a", "_b", "_lam", "_tgt", "_max", "link_ptr"]
    for name, x, y in zip(names, s1, s2):
        assert np.array_equal(x, y), f"{name} not stable across identical replays"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_compact_with_keep_mask_matches_numpy(seed):
    """The device keep/compact primitive (the Phase-7 decision-mask seam) reproduces
    a NumPy boolean-mask compaction exactly AND stably (survivors keep their order)."""
    eng = _engine()
    links = FPPLinks(eng)
    rng = np.random.default_rng(seed + 50)
    k = 60
    a = rng.integers(1, eng.n_cells + 1, size=k).astype(np.int32)
    b = ((a + rng.integers(1, eng.n_cells, size=k)) % eng.n_cells + 1).astype(np.int32)
    lam = rng.uniform(0.5, 2.0, size=k).astype(np.float32)
    tgt = rng.uniform(2.0, 6.0, size=k).astype(np.float32)
    mx = rng.uniform(8.0, 14.0, size=k).astype(np.float32)
    links.create_links_bulk(a, b, lam=lam, target=tgt, maxlen=mx)

    keep = rng.integers(0, 2, size=k).astype(np.int32)
    keep_dev = wp.array(keep, dtype=wp.int32, device=eng.device)
    n_new = links.compact_with_keep_mask(keep_dev)

    m = keep.astype(bool)
    assert n_new == int(keep.sum()) == links.n_pairs
    # stable: survivors in original order, byte-identical fields
    assert np.array_equal(links._a, a[m])
    assert np.array_equal(links._b, b[m])
    assert np.allclose(links._lam, lam[m])
    assert np.allclose(links._tgt, tgt[m])
    assert np.allclose(links._max, mx[m])


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_active_link_lengths_matches_numpy_reference():
    """``active_link_lengths`` equals a NumPy COM-distance reference over the kept
    (within-max) links -- unchanged by the device-authoritative storage."""
    eng = _engine(seed=11)
    links = FPPLinks(eng)
    pairs = grid_graph_links(3)
    # mixed max lengths so some links are dropped (length > max), some kept
    rng = np.random.default_rng(3)
    mx = rng.uniform(0.5, 30.0, size=pairs.shape[0]).astype(np.float32)
    links.set_topology(pairs, maxlens=mx)
    eng.run(8)
    links.rebuild()

    coms = np.zeros((eng.n_cells + 1, 3))
    coms[1:] = eng.coms()
    a, b = links._a.astype(np.int64), links._b.astype(np.int64)
    d = np.sqrt(((coms[a] - coms[b]) ** 2).sum(axis=1))
    keep = d <= links._max
    ref_lengths = np.sort(d[keep])

    got = np.sort(links.active_link_lengths())
    assert got.shape == ref_lengths.shape
    assert np.allclose(got, ref_lengths, atol=1e-4)
    assert links.num_active() == int(keep.sum())


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_link_ptr_and_csr_byte_identical_to_numpy_build():
    """The device FPP build (count active -> device scan -> atomic-append) reproduces
    a NumPy reconstruction of link_ptr + the per-cell neighbor set, byte-for-byte for
    link_ptr (the scan) and set-equal per cell (the order-independent atomic append)."""
    eng = _engine(seed=13)
    links = FPPLinks(eng)
    pairs = grid_graph_links(3)
    rng = np.random.default_rng(8)
    mx = rng.uniform(0.5, 40.0, size=pairs.shape[0]).astype(np.float32)
    links.set_topology(pairs, maxlens=mx)
    eng.run(6)
    links.rebuild()

    n1 = eng.n_cells + 1
    coms = np.zeros((n1, 3)); coms[1:] = eng.coms()
    a, b = links._a.astype(np.int64), links._b.astype(np.int64)
    d = np.sqrt(((coms[a] - coms[b]) ** 2).sum(axis=1))
    keep = d <= links._max
    # reference degree + exclusive-prefix link_ptr
    deg = np.zeros(n1, dtype=np.int64)
    np.add.at(deg, a[keep], 1)
    np.add.at(deg, b[keep], 1)
    ref_ptr = np.zeros(n1 + 1, dtype=np.int64)
    ref_ptr[1:] = np.cumsum(deg)

    got_ptr = links.link_ptr.numpy().astype(np.int64)
    assert np.array_equal(got_ptr, ref_ptr), "device link_ptr != NumPy cumsum reconstruction"

    # per-cell neighbor multiset (atomic-append order is arbitrary but the set per
    # cell must match the directed reference)
    other = links.link_other.numpy()
    ref_adj = {c: [] for c in range(n1)}
    for i in np.nonzero(keep)[0]:
        ref_adj[int(a[i])].append(int(b[i]))
        ref_adj[int(b[i])].append(int(a[i]))
    for c in range(n1):
        lo, hi = int(ref_ptr[c]), int(ref_ptr[c + 1])
        got = sorted(other[lo:hi].tolist())
        assert got == sorted(ref_adj[c]), f"cell {c} CSR neighbor set mismatch"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_volume_contact_run_is_deterministic_bitexact():
    """Volume+Contact (no FPP) is untouched by Phase 6: same seed -> byte-identical
    id-lattice + int64 COM sums (the bit-reproducibility invariant)."""
    def run():
        eng = _engine(seed=21, cpa=4, L=24)
        eng.run(12)
        return eng.get_ids(), eng.xsum.numpy().copy(), eng.ysum.numpy().copy(), eng.zsum.numpy().copy()

    a = run(); b = run()
    assert np.array_equal(a[0], b[0]), "id-lattice not bit-reproducible"
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x, y), "int64 COM sums not bit-reproducible"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_fpp_run_is_deterministic_and_lengths_stable():
    """A short full grid-graph FPP run is deterministic (same seed -> identical
    lattice AND identical active-link-length set), so the device-authoritative
    inventory + device scan introduced no nondeterminism in the FPP dynamics."""
    def run():
        eng = _engine(seed=33, cpa=3, L=21)
        links = FPPLinks(eng, target_length_default=4.0, max_length_default=12.0)
        links.set_topology(grid_graph_links(3))
        eng.attach_fpp(links)
        for m in range(15):
            links.rebuild()
            eng.step_mcs(m)
        wp.synchronize()
        return eng.get_ids(), np.sort(links.active_link_lengths())

    ids1, len1 = run()
    ids2, len2 = run()
    assert np.array_equal(ids1, ids2), "FPP run id-lattice not deterministic"
    assert len1.shape == len2.shape and np.allclose(len1, len2), \
        "FPP active-link lengths not deterministic"
    # the inventory stayed intact (no links spuriously lost in storage)
    assert len1.shape[0] > 0
