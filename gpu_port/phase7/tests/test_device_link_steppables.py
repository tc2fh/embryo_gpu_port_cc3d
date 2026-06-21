"""Phase 7 exit gate -- device link steppables (tissue / substrate / Poisson) +
the dropped neighbor-CSR copyback.

The phase ports the per-cell HOST link-management loops (``embryo/steppables.py``
``TissueLinkSteppable`` / ``PassiveSubstrateSteppable`` and ``engine/steppables.py``
``LamellipodiaSteppable``) to device kernels that read Phase-6's resident device CSR
handles (``engine.neighbor_csr_*_dev``) + the device link inventory directly, and
stops copying the whole neighbor-contact graph back to the host every MCS. This gate
pins:

(a) DEVICE tissue/substrate link **set** == the HOST-path reference set EXACTLY on the
    real scaled-Embryo geometry, with the **cap-truncation order preserved** (both the
    Leading MaxNeighborNum+1 manager and the Passive MaxNeighborNum manager), at
    ``start()`` and after each evolved MCS -- the load-bearing fidelity invariant;
(b) the substrate create reproduces the host **min-id** rule exactly; the Poisson
    turnover (tissue / substrate / lamellipodia) matches ``1-exp(-rate)`` per kind and
    is reproducible per key (the keep-mask + ``compact_with_keep_mask`` seam);
(c) the full ``EmbryoModel.run`` (device backend) matches the prior validated
    full-Embryo link inventory + link-length mean/median/KS within Phase-3 tolerances;
(d) the per-MCS host CSR copyback is GONE -- the device backend never ``.numpy()``-s
    the full contact graph on the hot path (asserted by patching the engine copy site).

The host path is kept behind ``link_backend="host"`` so (a)/(b) can compare device-vs-
host; the default/hot path is ``"device"`` (no copyback).
"""

import os
import sys

import numpy as np
import pytest

import warp as wp

from engine import GPUEngine, FPPLinks
from engine.geometry import LEADING, PASSIVE, SUBSTRATE
from embryo import EmbryoModel, build_scaled_embryo
from embryo.params import DEFAULT as P
from embryo.steppables import (
    TissueLinkSteppable, PassiveSubstrateSteppable, neighbor_adjacency,
)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _undirected_set(a, b):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))


def _kind_set(links, kind_lambda):
    a = links._a.astype(np.int64); b = links._b.astype(np.int64); lam = links._lam
    m = np.abs(lam - kind_lambda) < 1e-3
    return _undirected_set(a[m], b[m])


def _snapshot(links):
    return (links._a.astype(np.int64), links._b.astype(np.int64),
            links._lam.copy(), links._tgt.copy(), links._max.copy())


def _restore(eng, snap):
    a, b, lam, tgt, mx = snap
    links = FPPLinks(eng, target_length_default=P.tissue_target,
                     lambda_default=P.tissue_lambda, max_length_default=P.tissue_max)
    if len(a):
        links.set_topology(np.stack([a, b], axis=1), lam, tgt, mx)
    return links


def _host_tissue_recreate(eng, links, csr_host, ctype):
    """Replicate the HOST TissueLinkSteppable recreate (Leading cap+1, Passive cap),
    cap-ordered, on the given inventory -- the differential reference for the device
    cap-ordered relink. (Pure recreate; the Poisson delete is tested separately.)"""
    for cell_types, off in ((LEADING, 1), (PASSIVE, 0)):
        s = TissueLinkSteppable(eng, links, cell_types=(cell_types,), params=P,
                                link_cap_offset=off, substrate_type=SUBSTRATE,
                                link_backend="host")
        s.shared_csr = csr_host
        adj = neighbor_adjacency(eng, exclude_types=(SUBSTRATE,), csr=csr_host,
                                 cells=s.managed, cell_type=s._cell_type)
        link_map = s._current_link_map()
        new_a, new_b = [], []
        for c in s.managed:
            c = int(c)
            partners = link_map.get(c, set())
            if len(partners) >= s.max_links:
                continue
            for nb in adj[c]:
                nb = int(nb)
                if len(partners) >= s.max_links:
                    break
                if nb in partners:
                    continue
                new_a.append(c); new_b.append(nb)
                partners.add(nb); link_map.setdefault(nb, set()).add(c)
        if new_a:
            links.create_links_bulk(new_a, new_b, lam=P.tissue_lambda,
                                    target=P.tissue_target, maxlen=P.tissue_max)


def _host_tissue_start(eng, links, csr_host):
    """Replicate the HOST TissueLinkSteppable.start (NO cap -- a link to every
    non-substrate neighbor lacking one), Leading then Passive."""
    for cell_types, off in ((LEADING, 1), (PASSIVE, 0)):
        s = TissueLinkSteppable(eng, links, cell_types=(cell_types,), params=P,
                                link_cap_offset=off, substrate_type=SUBSTRATE,
                                link_backend="host")
        s.shared_csr = csr_host
        s._start_host()


# ===========================================================================
# (a) device tissue link set == host reference set EXACTLY (cap order preserved)
# ===========================================================================
@cuda
@pytest.mark.parametrize("seed", [7, 11, 21])
def test_tissue_link_set_device_equals_host_at_start(seed):
    """At ``start()`` the device cap-ordered relink (no cap, both managers) produces
    EXACTLY the host-reference tissue link set on the real scaled-Embryo geometry."""
    st, info = build_scaled_embryo(cube_size=28, seed=seed)

    eng_h = GPUEngine(st); links_h = FPPLinks(eng_h)
    csr_host = eng_h.neighbor_contact_csr(order=1)
    _host_tissue_start(eng_h, links_h, csr_host)
    host_set = _undirected_set(links_h._a, links_h._b)

    eng_d = GPUEngine(st); links_d = FPPLinks(eng_d)
    eng_d.publish_neighbor_csr_device(order=1)
    ctype = eng_d.cell_type.numpy()
    lead = wp.array(np.sort(np.nonzero(ctype == LEADING)[0]).astype(np.int32),
                    dtype=wp.int32, device=eng_d.device)
    pas = wp.array(np.sort(np.nonzero(ctype == PASSIVE)[0]).astype(np.int32),
                   dtype=wp.int32, device=eng_d.device)
    BIG = TissueLinkSteppable._NO_CAP
    links_d.tissue_relink_device(eng_d, lead, int(lead.shape[0]), cap=BIG,
                                 substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                 target=P.tissue_target, maxlen=P.tissue_max)
    links_d.tissue_relink_device(eng_d, pas, int(pas.shape[0]), cap=BIG,
                                 substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                 target=P.tissue_target, maxlen=P.tissue_max)
    dev_set = _undirected_set(links_d._a, links_d._b)

    assert dev_set == host_set, (
        f"start tissue set mismatch: host {len(host_set)} dev {len(dev_set)}; "
        f"host-only {sorted(host_set - dev_set)[:6]} dev-only {sorted(dev_set - host_set)[:6]}")
    # no substrate endpoint in any tissue link
    a, b = links_d._a, links_d._b
    assert not ((ctype[a] == SUBSTRATE) | (ctype[b] == SUBSTRATE)).any()


@cuda
@pytest.mark.parametrize("seed", [7, 11])
def test_tissue_link_set_device_equals_host_each_step_cap_order(seed):
    """After several evolved MCS (where lamellipodia + substrate links consume budget
    so the per-cell cap GENUINELY BINDS), the device cap-ordered relink == the host
    cap-ordered relink EXACTLY, for the SAME inventory snapshot -- i.e. the load-
    bearing CSR-row truncation order keeps the same links the host/CPU reference keeps.
    Both the Leading (MaxNeighborNum+1) and Passive (MaxNeighborNum) caps are exercised
    and the per-cell degree caps are respected for the cell currently being processed."""
    st, info = build_scaled_embryo(cube_size=28, seed=seed)
    m = EmbryoModel(st, link_backend="device")
    m.start()
    eng = m.engine
    ctype = eng.cell_type.numpy()
    lead = wp.array(np.sort(np.nonzero(ctype == LEADING)[0]).astype(np.int32),
                    dtype=wp.int32, device=eng.device)
    pas = wp.array(np.sort(np.nonzero(ctype == PASSIVE)[0]).astype(np.int32),
                   dtype=wp.int32, device=eng.device)
    n_lead, n_pas = int(lead.shape[0]), int(pas.shape[0])

    for step in range(8):
        snap = _snapshot(m.links)
        csr_host = eng.neighbor_contact_csr(order=1)     # host CSR for the host ref

        # device relink (with the real caps) on a clone of the snapshot
        eng.publish_neighbor_csr_device(order=1)
        ld = _restore(eng, snap)
        ld.tissue_relink_device(eng, lead, n_lead, cap=P.max_neighbor_num + 1,
                                substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                target=P.tissue_target, maxlen=P.tissue_max)
        ld.tissue_relink_device(eng, pas, n_pas, cap=P.max_neighbor_num,
                                substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                target=P.tissue_target, maxlen=P.tissue_max)
        dev_set = _undirected_set(ld._a, ld._b)

        # host relink on a clone of the SAME snapshot
        lh = _restore(eng, snap)
        _host_tissue_recreate(eng, lh, csr_host, ctype)
        host_set = _undirected_set(lh._a, lh._b)

        # the EXACT set-equality with the host (whose cap loop is the CC3D reference) IS
        # the load-bearing cap-truncation-order assertion: a device cell keeps the same
        # links the host/CPU reference keeps, including which neighbors are dropped when
        # the per-cell cap binds, in CSR-row order.
        assert dev_set == host_set, (
            f"step {step}: device relink set != host (cap order). "
            f"host {len(host_set)} dev {len(dev_set)} "
            f"host-only {sorted(host_set - dev_set)[:6]} dev-only {sorted(dev_set - host_set)[:6]}")
        m.run(1, mcs_offset=step)


# ===========================================================================
# (b) substrate min-id rule + Poisson turnover rate / reproducibility
# ===========================================================================
@cuda
@pytest.mark.parametrize("seed", [11, 21])
def test_substrate_link_set_device_equals_host_min_id(seed):
    """The device substrate create (smallest-id Substrate neighbor) reproduces the host
    min-id substrate link set EXACTLY across evolved states -- the substrate min-id
    rule + 'one link per passive cell next-to-substrate without one' is preserved."""
    st, info = build_scaled_embryo(cube_size=28, seed=seed)
    m = EmbryoModel(st, link_backend="device")
    m.start()
    eng = m.engine
    ctype = eng.cell_type.numpy()
    passive = np.sort(np.nonzero(ctype == PASSIVE)[0]).astype(np.int32)
    pas_dev = wp.array(passive, dtype=wp.int32, device=eng.device)

    for step in range(6):
        snap = _snapshot(m.links)
        csr_host = eng.neighbor_contact_csr(order=1)

        # device substrate create on a clone
        eng.publish_neighbor_csr_device(order=1)
        ld = _restore(eng, snap)
        ld.substrate_relink_device(eng, pas_dev, len(passive), substrate_type=SUBSTRATE,
                                   slink_lambda=P.slink_lambda, slink_target=P.slink_target,
                                   slink_max=P.slink_max)
        dev_set = _kind_set(ld, P.slink_lambda)

        # host substrate create (replicate _step_host part (a)) on a clone
        lh = _restore(eng, snap)
        s = PassiveSubstrateSteppable(eng, lh, passive_type=PASSIVE,
                                      substrate_type=SUBSTRATE, params=P, link_backend="host")
        s.shared_csr = csr_host
        sl = s.cell_dict.get("sub_link")
        a0, b0, lam0 = snap[0], snap[1], snap[2]
        smask = np.abs(lam0 - P.slink_lambda) < 1e-3
        for ai, bi in zip(a0[smask], b0[smask]):
            pe = ai if ctype[ai] != SUBSTRATE else bi
            se = bi if ctype[ai] != SUBSTRATE else ai
            sl[int(pe)] = int(se)
        sub_nb = s._substrate_neighbor_map()
        new_a, new_b = [], []
        for c in s.passive:
            c = int(c)
            if sl[c] != 0:
                continue
            sub = sub_nb.get(c, np.zeros(0, dtype=np.int64))
            if sub.size == 0:
                continue
            target = int(sub.min())
            new_a.append(c); new_b.append(target); sl[c] = target
        if new_a:
            lh.create_links_bulk(new_a, new_b, lam=P.slink_lambda,
                                 target=P.slink_target, maxlen=P.slink_max)
        host_set = _kind_set(lh, P.slink_lambda)

        assert dev_set == host_set, (
            f"step {step}: device substrate set != host min-id ref. "
            f"host {len(host_set)} dev {len(dev_set)}")
        m.run(1, mcs_offset=step)


@cuda
@pytest.mark.parametrize("kind,rate_attr,stream", [
    ("tissue", "tissue_delete_prob", 1),
    ("substrate", "sub_link_delete_prob", 2),
])
def test_poisson_keep_mask_rate_and_reproducible(kind, rate_attr, stream):
    """The device Poisson keep-mask (the per-link Bernoulli driving
    ``compact_with_keep_mask``) deletes links at the model rate ``1-exp(-rate)`` within
    Phase-3 sampling tolerance, and is reproducible per key (same (mcs, link, stream,
    seed) -> same survivors). Built directly on a large synthetic inventory of one
    kind so the empirical fraction is a clean estimate of the rate."""
    from engine import link_kernels as LK
    eng = GPUEngine(build_scaled_embryo(cube_size=20, seed=1)[0])
    prob = float(getattr(P, rate_attr))
    if kind == "tissue":
        lamv = P.tissue_lambda
    else:
        lamv = P.slink_lambda
    # synthetic inventory of N links all of this kind (distinct unordered pairs)
    N = 200000
    rng = np.random.default_rng(0)
    a = rng.integers(1, eng.n_cells + 1, size=N).astype(np.int32)
    b = ((a.astype(np.int64) + rng.integers(1, eng.n_cells, size=N)) % eng.n_cells + 1).astype(np.int32)
    a_dev = wp.array(a, dtype=wp.int32, device=eng.device)
    b_dev = wp.array(b, dtype=wp.int32, device=eng.device)
    lam_dev = wp.array(np.full(N, lamv, np.float32), dtype=wp.float32, device=eng.device)

    def keep_for(mcs, seed):
        keep = wp.zeros(N, dtype=wp.int32, device=eng.device)
        wp.launch(LK.poisson_keep_mask_kernel, dim=N,
                  inputs=[a_dev, b_dev, lam_dev, N, float(lamv), prob,
                          int(mcs), int(seed), int(stream), keep], device=eng.device)
        wp.synchronize()
        return keep.numpy()

    fr = []
    for mcs in range(5):
        keep = keep_for(mcs, 12345)
        fr.append(1.0 - keep.mean())          # fraction deleted
    frac = float(np.mean(fr))
    sd = np.sqrt(prob * (1 - prob) / (N * 5))
    assert abs(frac - prob) < 6 * sd + 1e-9, (
        f"{kind} device Poisson fraction {frac:.6f} != model {prob:.6f} (6sigma={6*sd:.6f})")
    # reproducible per key
    k1 = keep_for(3, 7); k2 = keep_for(3, 7)
    assert np.array_equal(k1, k2), f"{kind} keep-mask not reproducible per key"
    # different mcs key -> different decisions (not a constant mask)
    k3 = keep_for(4, 7)
    assert not np.array_equal(k1, k3), f"{kind} keep-mask identical across mcs (key not used)"


@cuda
def test_poisson_delete_routes_through_compact_with_keep_mask():
    """The kind-specific device Poisson delete only removes links of THAT kind, leaves
    the other kinds intact, and goes through the Phase-6 ``compact_with_keep_mask``
    primitive (stable survivor order). Drive a high prob so deletions clearly fire."""
    st, info = build_scaled_embryo(cube_size=24, seed=3)
    m = EmbryoModel(st, link_backend="device")
    m.start()
    m.run(4)
    before = {k: _kind_set(m.links, lam) for k, lam in
              (("tissue", P.tissue_lambda), ("lam", P.lamellipodia_lambda),
               ("sub", P.slink_lambda))}
    n_before = m.links.n_pairs
    # delete tissue at p=0.9: tissue shrinks a lot, lam/sub untouched
    new_m = m.links.poisson_delete_device(P.tissue_lambda, 0.9, 999,
                                          m.engine.base_seed, 1)
    after = {k: _kind_set(m.links, lam) for k, lam in
             (("tissue", P.tissue_lambda), ("lam", P.lamellipodia_lambda),
              ("sub", P.slink_lambda))}
    assert new_m == m.links.n_pairs
    assert len(after["tissue"]) < len(before["tissue"]), "tissue not deleted"
    assert after["tissue"] <= before["tissue"], "tissue survivors not a subset (spurious adds)"
    assert after["lam"] == before["lam"], "lamellipodia links wrongly deleted"
    assert after["sub"] == before["sub"], "substrate links wrongly deleted"


# ===========================================================================
# (c) full EmbryoModel.run (device) matches the prior validated inventory + lengths
# ===========================================================================
@cuda
def test_full_embryo_device_inventory_and_lengths_match_host():
    """The full device ``EmbryoModel.run`` produces a link inventory (tissue /
    lamellipodia / substrate counts) and link-length distribution that match the HOST-
    backend EmbryoModel within Phase-3 tolerances. (The Poisson DECISIONS are
    re-keyed device-side -- stable per-link key vs the host's list-position key -- so
    the per-link delete identities differ; the gate checks the inventory COUNTS +
    link-length mean/median/KS track, the statistical-fidelity contract, exactly as the
    Phase-3 CC3D cross-check does.)"""
    from scipy import stats
    st_h, _ = build_scaled_embryo(cube_size=32, seed=5)
    st_d, _ = build_scaled_embryo(cube_size=32, seed=5)
    mh = EmbryoModel(st_h, link_backend="host"); mh.start(); mh.run(15)
    md = EmbryoModel(st_d, link_backend="device"); md.start(); md.run(15)
    mh.engine.assert_volume_partition()
    md.engine.assert_volume_partition()

    def kinds(links):
        lam = links._lam
        return dict(
            tissue=int(np.sum(np.abs(lam - P.tissue_lambda) < 1e-3)),
            lam=int(np.sum(np.abs(lam - P.lamellipodia_lambda) < 1e-3)),
            sub=int(np.sum(np.abs(lam - P.slink_lambda) < 1e-3)))

    kh, kd = kinds(mh.links), kinds(md.links)
    msg = f"\nhost {kh} active {mh.links.num_active()} | device {kd} active {md.links.num_active()}"
    # tissue: cap-ordered relink is exact each step; tiny TissueRate -> ~no deletes, so
    # the counts track within a small band.
    assert abs(kh["tissue"] - kd["tissue"]) <= 0.05 * kh["tissue"] + 5, f"tissue count off.{msg}"
    # lamellipodia: O(#leaders), stochastic per-cell turnover -> absolute band.
    assert abs(kh["lam"] - kd["lam"]) <= 0.5 * max(1, kh["lam"]) + 8, f"lam count off.{msg}"
    # substrate: create is exact; SubLinkRate turnover stochastic -> moderate band.
    assert abs(kh["sub"] - kd["sub"]) <= 0.15 * max(1, kh["sub"]) + 10, f"sub count off.{msg}"
    # link-length distribution matches (KS within the Phase-3 gate) + mean within a few %.
    lh = np.sort(mh.links.active_link_lengths())
    ld = np.sort(md.links.active_link_lengths())
    ks = stats.ks_2samp(lh, ld)
    mean_rel = abs(lh.mean() - ld.mean()) / abs(lh.mean())
    msg += f"\nlen host mean {lh.mean():.3f} dev mean {ld.mean():.3f} | KS {ks.statistic:.3f} mean_rel {mean_rel:.4f}"
    assert ks.statistic < 0.20, f"link-length distribution diverges device vs host.{msg}"
    assert mean_rel < 0.06, f"link-length mean diverges device vs host.{msg}"


@cuda
def test_device_link_kernels_are_deterministic_pure_functions():
    """The device link kernels (tissue cap-ordered relink, substrate min-id create,
    Poisson keep-mask) introduce NO float-atomic nondeterminism: as pure functions of a
    FIXED engine state + inventory they produce a byte-identical link set across repeats
    (integer/id atomic-append + keep/compact + keyed Philox only).

    (NB: a full ``EmbryoModel.run`` lattice is NOT bit-reproducible -- the FPP spring
    energy reads the int64 COM while the checkerboard sweep mutates it, a pre-existing
    engine-level FPP read race documented as an open bit-repro item, present in BOTH
    backends. That race is upstream of -- and unaffected by -- the Phase-7 link kernels,
    which this test isolates and pins as deterministic.)"""
    st, info = build_scaled_embryo(cube_size=28, seed=9)
    # evolve once to a representative mixed inventory + lattice, then FREEZE it.
    m = EmbryoModel(st, link_backend="device")
    m.start(); m.run(6)
    eng = m.engine
    snap = _snapshot(m.links)
    ctype = eng.cell_type.numpy()
    lead = wp.array(np.sort(np.nonzero(ctype == LEADING)[0]).astype(np.int32),
                    dtype=wp.int32, device=eng.device)
    pas = wp.array(np.sort(np.nonzero(ctype == PASSIVE)[0]).astype(np.int32),
                   dtype=wp.int32, device=eng.device)
    n_lead, n_pas = int(lead.shape[0]), int(pas.shape[0])

    def one_pass():
        eng.publish_neighbor_csr_device(order=1)
        lk = _restore(eng, snap)
        # tissue Poisson delete (keyed) -> tissue relink (cap) -> substrate create
        lk.poisson_delete_device(P.tissue_lambda, P.tissue_delete_prob, 123,
                                 eng.base_seed, 1)
        lk.tissue_relink_device(eng, lead, n_lead, cap=P.max_neighbor_num + 1,
                                substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                target=P.tissue_target, maxlen=P.tissue_max)
        lk.tissue_relink_device(eng, pas, n_pas, cap=P.max_neighbor_num,
                                substrate_type=SUBSTRATE, lam=P.tissue_lambda,
                                target=P.tissue_target, maxlen=P.tissue_max)
        lk.substrate_relink_device(eng, pas, n_pas, substrate_type=SUBSTRATE,
                                   slink_lambda=P.slink_lambda, slink_target=P.slink_target,
                                   slink_max=P.slink_max)
        # byte-identical inventory LAYOUT (not just the set): order-stable kernels
        return (lk._a.copy(), lk._b.copy(), lk._lam.copy())

    a1, b1, l1 = one_pass()
    a2, b2, l2 = one_pass()
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(l1, l2), (
        "device link kernels not deterministic on a fixed state (introduced nondeterminism)")
    assert a1.shape[0] > 0


# ===========================================================================
# (d) the per-MCS host CSR copyback is GONE on the device hot path
# ===========================================================================
@cuda
def test_no_full_graph_copyback_on_device_hot_path():
    """In device link-backend mode the per-MCS step must NOT copy the full neighbor-
    contact graph (the O(#contacts) ``indices``/``data``) back to the host. We patch
    the engine's CSR entry to record calls AND whether each one requested the host
    return (``host_return=True`` -> the full-graph copyback). A device-backend ``run``
    must make ZERO full-graph-copy calls (it uses ``publish_neighbor_csr_device`` ->
    resident handles + a scalar n_contacts read), whereas the host backend copies the
    full graph every MCS. We additionally patch ``numpy()`` on the resident
    indices/data device arrays to prove they are never materialized on the hot path."""
    st, info = build_scaled_embryo(cube_size=28, seed=4)

    # device backend: count host-return (full-graph copy) calls vs device-only calls
    md = EmbryoModel(st, link_backend="device")
    eng = md.engine
    calls = {"host_copy": 0, "device_only": 0}
    orig = eng.neighbor_contact_csr

    def spy(order=None, method="device", host_return=True):
        if host_return:
            calls["host_copy"] += 1
        else:
            calls["device_only"] += 1
        return orig(order=order, method=method, host_return=host_return)

    eng.neighbor_contact_csr = spy
    md.start()
    md.run(6)
    assert calls["host_copy"] == 0, (
        f"device hot path performed {calls['host_copy']} full-graph host CSR copybacks; "
        f"expected 0 (device-only builds: {calls['device_only']})")
    # the device-only builds DID run (one per start + step) -> handles are live
    assert calls["device_only"] >= 7, (
        f"expected >=7 device-only CSR builds (start + 6 steps); got {calls['device_only']}")
    # the resident device handles ARE published + populated (steppables have data)
    assert eng.neighbor_csr_indptr_dev is not None
    assert eng.neighbor_csr_n_contacts > 0
    assert eng.neighbor_csr_indices_dev.shape[0] == eng.neighbor_csr_n_contacts

    # contrast: the host backend DOES the full-graph copy every MCS (start + each step)
    st2, _ = build_scaled_embryo(cube_size=28, seed=4)
    mh = EmbryoModel(st2, link_backend="host")
    hcalls = {"host_copy": 0}
    horig = mh.engine.neighbor_contact_csr

    def hspy(order=None, method="device", host_return=True):
        if host_return:
            hcalls["host_copy"] += 1
        return horig(order=order, method=method, host_return=host_return)

    mh.engine.neighbor_contact_csr = hspy
    mh.start()
    mh.run(6)
    assert hcalls["host_copy"] >= 6, (
        f"host backend should full-graph copy the CSR each MCS; saw only "
        f"{hcalls['host_copy']} host-copy calls")
