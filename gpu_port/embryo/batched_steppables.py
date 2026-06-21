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
from .steppables import neighbor_adjacency, _bernoulli, _STREAM_TISSUE


class BatchedTissueLinkSteppable:
    """Batched TISSUE-link + intercalation dynamics over R replicas.

    ``delete_prob`` is the per-replica Poisson tissue-link delete probability (the
    swept TissueRate, length R; defaults to ``params.tissue_delete_prob`` for all).
    The steppable owns the batched FPP topology: each MCS it recomputes every
    replica's tissue link set and pushes it to ``links`` via ``set_per_replica_pairs``.
    """

    def __init__(self, engine: BatchedGPUEngine, links: BatchedFPPLinks,
                 cell_types, params: EmbryoParams = DEFAULT, link_cap_offset: int = 1,
                 substrate_type: int = 4, delete_prob=None):
        self.engine = engine
        self.links = links
        self.p = params
        self.R = engine.R
        self.substrate_type = int(substrate_type)
        self.cell_types = set(int(t) for t in cell_types)
        self.max_links = params.max_neighbor_num + int(link_cap_offset)
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
        """Upload every replica's current tissue link set to the batched FPP CSR."""
        pairs = [np.array(sorted(self._tissue[r]), dtype=np.int32).reshape(-1, 2)
                 for r in range(self.R)]
        self.links.set_per_replica_pairs(pairs)

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
        self._push_topology()
        return [len(t) for t in self._tissue]
