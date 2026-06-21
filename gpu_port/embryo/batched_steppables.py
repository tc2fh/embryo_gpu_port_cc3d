"""Batched Embryo steppables (Phase 5, Tier 2c+).

Replica-axis ports of the Embryo link dynamics. ``BatchedTissueLinkSteppable`` runs
the TISSUE-link intercalation (Poisson delete + neighbor recreate) for R replicas
that share the cell-type assignment but evolve independent lattices / link sets, so a
sweep can vary the per-replica **intercalation rate** (TissueRate) and the tissue
spring params (lambda / target / max).

Design: the heavy work is batched -- the CPM sweep (``BatchedGPUEngine.step_mcs``),
the per-replica neighbor-contact CSR (Tier 2a), and the per-replica link-CSR build /
spring energy (Tier 1/2b). The create/delete DECISION logic is host-orchestrated per
replica (looping R), reusing the vectorized single-replica ``neighbor_adjacency`` and
the keyed-Philox ``_bernoulli`` with each replica's own base seed -- so the topology
dynamics are deterministic given a fixed lattice (the Tier 2c equivalence test pins
this against the single ``TissueLinkSteppable``).

Scope note (tier 2c): this models TISSUE links only -- the per-cell link cap counts
tissue partners only (substrate + lamellipodia links are Tier 2d). The batched FPP
inventory therefore == the union of the per-replica tissue link sets.
"""

from __future__ import annotations

import numpy as np

from engine.batched import BatchedGPUEngine
from engine.batched_fpp import BatchedFPPLinks

from .params import EmbryoParams, DEFAULT
from .steppables import (
    neighbor_adjacency, _bernoulli, _STREAM_TISSUE, _STREAM_SUBLINK,
    _grouped_csr_neighbors,
)


class BatchedTissueLinkSteppable:
    """Batched TISSUE-link + intercalation dynamics over R replicas.

    ``delete_prob`` is the per-replica Poisson tissue-link delete probability (the
    swept TissueRate, length R; defaults to ``params.tissue_delete_prob`` for all).
    The steppable owns the batched FPP topology: each MCS it recomputes every
    replica's tissue link set and pushes it to ``links`` via ``set_per_replica_pairs``.
    """

    def __init__(self, engine: BatchedGPUEngine, links: BatchedFPPLinks,
                 cell_types, params: EmbryoParams = DEFAULT, link_cap_offset: int = 1,
                 substrate_type: int = 4, delete_prob=None, owns_topology: bool = True):
        self.engine = engine
        self.links = links
        self.p = params
        self.R = engine.R
        self.substrate_type = int(substrate_type)
        self.cell_types = set(int(t) for t in cell_types)
        self.max_links = params.max_neighbor_num + int(link_cap_offset)
        # owns_topology: push tissue links straight to the FPP CSR (standalone use).
        # A combined driver sets this False and assembles tissue+substrate itself.
        self.owns_topology = bool(owns_topology)
        ctype = engine.cell_type.numpy()
        self._cell_type = ctype                       # shared across replicas (static)
        self.managed = np.nonzero(np.isin(ctype, list(self.cell_types)))[0].astype(np.int64)
        # per-replica owned undirected tissue links (frozenset-style (a,b), a<b)
        self._tissue = [set() for _ in range(self.R)]
        # per-replica base seeds (same streams as R independent single engines)
        self._seeds = np.asarray(engine._seeds_np, dtype=np.int64)
        if delete_prob is None:
            delete_prob = np.full(self.R, params.tissue_delete_prob, dtype=np.float64)
        self.delete_prob = np.broadcast_to(np.asarray(delete_prob, dtype=np.float64),
                                           (self.R,)).copy()

    # ----------------------------------------------------------------- helpers
    def _key(self, a, b):
        return (a, b) if a < b else (b, a)

    def _partners(self, r):
        """managed cell -> set of its tissue partners in replica r (for cap + dedup)."""
        m = {int(c): set() for c in self.managed}
        for a, b in self._tissue[r]:
            if a in m:
                m[a].add(b)
            if b in m:
                m[b].add(a)
        return m

    def _push_topology(self):
        """Upload every replica's current tissue link set to the batched FPP CSR
        (standalone, tissue-only use)."""
        pairs, lam, tgt, mx = [], [], [], []
        for r in range(self.R):
            p, l, t, m = self.emit(r)
            pairs.append(p)
            lam.append(l)
            tgt.append(t)
            mx.append(m)
        self.links.set_per_replica_pairs(pairs, lambdas=lam, targets=tgt, maxlens=mx)

    def emit(self, r):
        """This steppable's contribution to replica r's FPP inventory:
        (pairs (n,2), lambda (n,), target (n,), max (n,)) -- the tissue links with
        the tissue spring params. Used by a combined driver to merge link kinds."""
        pairs = np.array(sorted(self._tissue[r]), dtype=np.int32).reshape(-1, 2)
        n = pairs.shape[0]
        lam = np.full(n, self.p.tissue_lambda, dtype=np.float32)
        tgt = np.full(n, self.p.tissue_target, dtype=np.float32)
        mx = np.full(n, self.p.tissue_max, dtype=np.float32)
        return pairs, lam, tgt, mx

    def _adj(self, indptr, indices):
        return neighbor_adjacency(
            self.engine, exclude_types=(self.substrate_type,),
            csr=(indptr, indices, None), cells=self.managed, cell_type=self._cell_type)

    def _recreate(self, r, adj):
        """Create tissue links from each managed cell to its non-substrate neighbors
        while under the per-cell cap (mirrors TissueLinkSteppable's create loop)."""
        partners = self._partners(r)
        for c in self.managed:
            c = int(c)
            pc = partners[c]
            if len(pc) >= self.max_links:
                continue
            for nb in adj[c]:
                nb = int(nb)
                if len(pc) >= self.max_links:
                    break
                if nb in pc:
                    continue
                self._tissue[r].add(self._key(c, nb))
                pc.add(nb)
                partners.setdefault(nb, set()).add(c)

    # -------------------------------------------------------------------- API
    def start(self):
        csr = self.engine.neighbor_contact_csr(order=1)
        for r in range(self.R):
            indptr, indices, _ = csr[r]
            self._recreate(r, self._adj(indptr, indices))
        if self.owns_topology:
            self._push_topology()
        return [len(t) for t in self._tissue]

    def step(self, mcs: int):
        csr = self.engine.neighbor_contact_csr(order=1)
        dev = self.engine.device
        for r in range(self.R):
            # (1) Poisson-delete each existing tissue link (intercalation)
            tissue_list = list(self._tissue[r])
            if tissue_list:
                dec = _bernoulli(len(tissue_list), float(self.delete_prob[r]), mcs,
                                 int(self._seeds[r]), _STREAM_TISSUE, dev)
                for i, ab in enumerate(tissue_list):
                    if dec[i] == 1:
                        self._tissue[r].discard(ab)
            # (2) recreate links to neighbors under the per-cell cap
            indptr, indices, _ = csr[r]
            self._recreate(r, self._adj(indptr, indices))
        if self.owns_topology:
            self._push_topology()
        return [len(t) for t in self._tissue]


class BatchedPassiveSubstrateSteppable:
    """Batched port of ``PassiveSubstrateSteppable``: per passive cell next to the
    Substrate with no current substrate link, create one to the (deterministically
    chosen smallest-id) Substrate order-1 neighbor; each MCS delete it with the
    per-replica Poisson SubLinkRate. Per-replica ``sub_link`` mirror = the substrate
    id each passive cell is linked to (0 = none)."""

    def __init__(self, engine: BatchedGPUEngine, links: BatchedFPPLinks,
                 passive_type: int = 2, substrate_type: int = 4,
                 params: EmbryoParams = DEFAULT, delete_prob=None,
                 owns_topology: bool = True):
        self.engine = engine
        self.links = links
        self.p = params
        self.R = engine.R
        self.passive_type = int(passive_type)
        self.substrate_type = int(substrate_type)
        self.owns_topology = bool(owns_topology)
        ctype = engine.cell_type.numpy()
        self._cell_type = ctype
        self.n1 = engine.n1
        self.passive = np.nonzero(ctype == self.passive_type)[0].astype(np.int64)
        # per-replica substrate-link target per cell (0 = none), like cell.dict['link']
        self._sl = np.zeros((self.R, self.n1), dtype=np.int32)
        self._seeds = np.asarray(engine._seeds_np, dtype=np.int64)
        if delete_prob is None:
            delete_prob = np.full(self.R, params.sub_link_delete_prob, dtype=np.float64)
        self.delete_prob = np.broadcast_to(np.asarray(delete_prob, dtype=np.float64),
                                           (self.R,)).copy()

    def _sub_nb(self, indptr, indices):
        ctype = self._cell_type
        st = self.substrate_type

        def keep_fn(nb):
            return (nb != 0) & (ctype[nb] == st)
        return _grouped_csr_neighbors(indptr, indices, self.passive, keep_fn)

    def emit(self, r):
        sl = self._sl[r]
        cs = self.passive[sl[self.passive] != 0]
        pairs = np.array([(int(c), int(sl[int(c)])) for c in cs],
                         dtype=np.int32).reshape(-1, 2)
        n = pairs.shape[0]
        lam = np.full(n, self.p.slink_lambda, dtype=np.float32)
        tgt = np.full(n, self.p.slink_target, dtype=np.float32)
        mx = np.full(n, self.p.slink_max, dtype=np.float32)
        return pairs, lam, tgt, mx

    def _push_topology(self):
        pairs, lam, tgt, mx = [], [], [], []
        for r in range(self.R):
            p, l, t, m = self.emit(r)
            pairs.append(p); lam.append(l); tgt.append(t); mx.append(m)
        self.links.set_per_replica_pairs(pairs, lambdas=lam, targets=tgt, maxlens=mx)

    def start(self):
        # CC3D PassiveSteppable.start makes no substrate links (created in step()).
        if self.owns_topology and self.p.if_passive_substrate:
            self._push_topology()
        return [0] * self.R

    def step(self, mcs: int):
        if not self.p.if_passive_substrate:
            return [0] * self.R
        csr = self.engine.neighbor_contact_csr(order=1)
        dev = self.engine.device
        empty = np.zeros(0, dtype=np.int64)
        for r in range(self.R):
            indptr, indices, _ = csr[r]
            sub_nb = self._sub_nb(indptr, indices)
            sl = self._sl[r]
            # (a) create a substrate link for next-to-substrate passive cells w/o one
            for c in self.passive:
                c = int(c)
                if sl[c] != 0:
                    continue
                sub = sub_nb.get(c, empty)
                if sub.size == 0:
                    continue
                sl[c] = int(sub.min())          # deterministic pick (smallest id)
            # (b) Poisson-delete existing substrate links (applied AFTER (a))
            have = np.nonzero(sl[self.passive] != 0)[0]
            if have.size:
                cells = self.passive[have]
                dec = _bernoulli(cells.size, float(self.delete_prob[r]), mcs,
                                 int(self._seeds[r]), _STREAM_SUBLINK, dev)
                for i, c in enumerate(cells):
                    if dec[i] == 1:
                        sl[int(c)] = 0
        if self.owns_topology:
            self._push_topology()
        return [int(np.count_nonzero(self._sl[r][self.passive])) for r in range(self.R)]


class BatchedEmbryoModel:
    """End-to-end batched Embryo driver (Tier 2): R replicas advanced together with
    batched CPM + FPP energy, while the host-orchestrated link dynamics (tissue
    intercalation + passive-substrate adhesion) run per replica and jointly own ONE
    batched FPP inventory.

    ``enable`` selects link dynamics. ``tissue_delete_prob`` / ``sub_delete_prob`` are
    optional length-R sweep axes (TissueRate / SubLinkRate). NOTE: cohesotaxis /
    lamellipodia (the ifCohesotaxis variant) is NOT yet batched -- see Tier 2e.
    """

    def __init__(self, engine: BatchedGPUEngine, params: EmbryoParams = DEFAULT,
                 enable=("tissue", "passive_substrate"),
                 leading_type: int = 1, passive_type: int = 2, substrate_type: int = 4,
                 tissue_delete_prob=None, sub_delete_prob=None):
        self.engine = engine
        self.p = params
        self.enable = set(enable)
        # one shared per-replica FPP inventory; per-link params come from each emit()
        self.links = BatchedFPPLinks(
            engine, target_length_default=params.tissue_target,
            lambda_default=params.tissue_lambda, max_length_default=params.tissue_max)
        self.steppables = []
        if "tissue" in self.enable:
            self.steppables.append(BatchedTissueLinkSteppable(
                engine, self.links, cell_types=(leading_type,), params=params,
                link_cap_offset=1, substrate_type=substrate_type,
                delete_prob=tissue_delete_prob, owns_topology=False))
            self.steppables.append(BatchedTissueLinkSteppable(
                engine, self.links, cell_types=(passive_type,), params=params,
                link_cap_offset=0, substrate_type=substrate_type,
                delete_prob=tissue_delete_prob, owns_topology=False))
        if "passive_substrate" in self.enable:
            self.steppables.append(BatchedPassiveSubstrateSteppable(
                engine, self.links, passive_type=passive_type, substrate_type=substrate_type,
                params=params, delete_prob=sub_delete_prob, owns_topology=False))

    def _push_combined(self):
        """Assemble every steppable's per-replica contribution into one FPP CSR."""
        R = self.engine.R
        pairs, lam, tgt, mx = [], [], [], []
        for r in range(R):
            ps, ls, ts, ms = [], [], [], []
            for s in self.steppables:
                p, l, t, m = s.emit(r)
                if p.shape[0]:
                    ps.append(p); ls.append(l); ts.append(t); ms.append(m)
            if ps:
                pairs.append(np.concatenate(ps, axis=0))
                lam.append(np.concatenate(ls)); tgt.append(np.concatenate(ts))
                mx.append(np.concatenate(ms))
            else:
                pairs.append(np.zeros((0, 2), np.int32))
                lam.append(np.zeros(0, np.float32)); tgt.append(np.zeros(0, np.float32))
                mx.append(np.zeros(0, np.float32))
        self.links.set_per_replica_pairs(pairs, lambdas=lam, targets=tgt, maxlens=mx)

    def start(self):
        for s in self.steppables:
            s.start()
        self._push_combined()
        self.engine.attach_fpp(self.links)
        return self

    def run(self, n_mcs: int, mcs_offset: int = 0):
        for m in range(n_mcs):
            mcs = mcs_offset + m
            self.engine.step_mcs(mcs)       # batched CPM + FPP energy (per-replica CSR)
            for s in self.steppables:
                s.step(mcs)                 # update each kind's per-replica link set
            self._push_combined()           # assemble tissue+substrate into the FPP CSR
        return self
