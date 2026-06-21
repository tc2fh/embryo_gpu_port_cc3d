"""Batched (replica/sweep) GPU CPM engine -- Phase 4 Pass A.

``BatchedGPUEngine`` advances **R independent CPM replicas concurrently** in one
set of kernel launches. It is the parameter-sweep workhorse: a sweep is R replicas
that share an initial condition but may carry **different swept parameters** (per
replica: the contact matrix, the per-type volume lambda/target, temperature). Each
replica is its own simulation -- replicas never interact -- and its own
reproducible Philox stream (the replica index is folded into the RNG key via a
per-replica base seed).

Why a separate class (not a flag on ``GPUEngine``)
--------------------------------------------------
The single-replica ``GPUEngine`` (Phases 1-3, 49 tests) is left byte-for-byte
unchanged. This class reuses the same ``EngineConfig`` / ``EngineState`` and the
same energy semantics, but lays state out with a leading replica axis and launches
the batched kernels. ``R=1`` reproduces a single ``GPUEngine`` run **bit-exactly**
(same key, same per-voxel ``rand_init`` index), so it is a faithful superset.

Memory layout (coalescing-preserving)
--------------------------------------
The replica axis is the SLOWEST (leading) dimension:

* id-lattice : flat ``ids[r*nvox + lin_idx(x,y,z)]``      (int32, replica-major)
* per-cell   : flat ``arr[r*n1 + cid]``  for volume(f32), xsum/ysum/zsum(int64),
               target_volume(f32), lambda_volume(f32)
* contact    : flat ``contact[r*nt*nt + t1*nt + t2]``     (per-replica matrix)
* temperature: ``temperature[r]`` ; base seed: ``base_seed_r[r]``

Within a fixed replica the per-color voxel stride is exactly the single-engine
stride, so coalesced voxel reads/writes are preserved; replicas touch disjoint
address ranges, so the int64-COM / float-volume atomics never collide across the
batch axis (no float atomics *across* replicas -- per the Phase 3 carry-forward).

Reproducibility
---------------
``seed = base_seed_r[r] + mcs*131072 + color*16384`` and the second ``rand_init``
arg is the LOCAL voxel index (0..nvox-1). With ``base_seed_r[r] = base_seed +
r*REPLICA_SEED_STRIDE``, batched replica ``r`` is the SAME stream as a single
``GPUEngine`` seeded ``base_seed + r*REPLICA_SEED_STRIDE``. The int64 fixed-point
COM makes the trajectory bit-exact and reproducible.

Scope note: FocalPointPlasticity is single-replica only here; batched FPP (a
per-replica link CSR) is DEFERRED (it is not among the swept parameters this pass
targets, and the FPP inventory lives behind its own per-cell CSR builder).
"""

from __future__ import annotations

import numpy as np

import warp as wp
import warp.utils  # radix_sort_pairs (per-replica neighbor-CSR compaction)

from .config import EngineConfig, neighbor_offsets
from .state import EngineState, build_grid_state, state_from_id_lattice
from . import kernels as K
from . import scan as S

wp.init()


# ---------------------------------------------------------------------------
# Batched initial-condition container
# ---------------------------------------------------------------------------
class BatchedState:
    """R stacked initial conditions sharing one ``EngineConfig`` topology.

    ``ids`` is (R, Lz, Ly, Lx) int32; ``cell_type`` is the SHARED (n_cells+1,)
    per-cell type vector (cell identity/topology is part of the IC, not a swept
    parameter). Per-replica xsum/ysum/zsum/volume are derived from each replica's
    lattice. ``n_cells`` is shared across replicas.
    """

    def __init__(self, cfg: EngineConfig, ids: np.ndarray, cell_type: np.ndarray):
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        assert ids.ndim == 4, "batched ids must be (R, Lz, Ly, Lx)"
        self.cfg = cfg
        self.ids = ids
        self.R = int(ids.shape[0])
        self.cell_type = np.asarray(cell_type, dtype=np.int32)
        self.n_cells = int(len(self.cell_type) - 1)
        assert ids.shape[1:] == (cfg.Lz, cfg.Ly, cfg.Lx), "replica lattice shape mismatch"


def build_batched_grid_state(cfg: EngineConfig, R: int, cells_per_axis: int = 3) -> BatchedState:
    """Stack R copies of the standard cubic-cell grid IC along a leading replica
    axis (the clean shared IC used by the statistical gate). All replicas start
    identical; per-replica *parameters* (not geometry) are what a sweep varies."""
    single = build_grid_state(cfg, cells_per_axis)
    ids = np.repeat(single.ids[None, ...], R, axis=0)
    return BatchedState(cfg, ids, single.cell_type)


def batched_state_from_lattices(cfg: EngineConfig, ids_list, cell_type: np.ndarray) -> BatchedState:
    """Build a BatchedState from an explicit list/array of R id-lattices (each
    (Lz,Ly,Lx)) sharing one ``cell_type`` vector. Replicas may start from different
    geometries if desired."""
    ids = np.stack([np.ascontiguousarray(a, dtype=np.int32) for a in ids_list], axis=0)
    return BatchedState(cfg, ids, cell_type)


# ---------------------------------------------------------------------------
# Batched engine
# ---------------------------------------------------------------------------
class BatchedGPUEngine:
    # spacing between per-replica base seeds; large + coprime-ish with the
    # per-MCS (131072) and per-color (16384) strides so replica streams don't
    # alias the mcs/color sub-keys. Single-engine equivalence uses this same
    # stride (seed_r = base_seed + r*REPLICA_SEED_STRIDE).
    REPLICA_SEED_STRIDE = 2_000_003

    def __init__(
        self,
        state: BatchedState,
        per_replica_config=None,
        device: str = "cuda:0",
        fpp_com_snapshot: bool = True,
    ):
        """``state`` carries R replica lattices + shared topology. ``per_replica_config``
        (optional) is a length-R list of ``EngineConfig`` whose SWEPT fields
        (contact, lambda_volume, target_volume, temperature, seed) override the
        base ``state.cfg`` per replica. Non-swept structural fields (lattice size,
        neighbor orders, n_types) must match the base config across replicas.
        """
        self.cfg: EngineConfig = state.cfg
        self.device = device
        self.R = state.R
        self.Lx, self.Ly, self.Lz = self.cfg.Lx, self.cfg.Ly, self.cfg.Lz
        self.nvox = self.cfg.n_voxels
        self.n_cells = state.n_cells
        self.n_types = self.cfg.n_types
        n1 = self.n_cells + 1
        self.n1 = n1
        R = self.R

        cfgs = self._resolve_configs(per_replica_config)
        self._validate_structural(cfgs)
        self.cfgs = cfgs

        # --- flat replica-major lattice on device ---
        ids_flat = np.ascontiguousarray(state.ids.reshape(-1), dtype=np.int32)
        self.ids = wp.array(ids_flat, dtype=wp.int32, device=device)

        # --- shared per-cell type vector (cell identity is part of the IC) ---
        self.cell_type = wp.array(state.cell_type.astype(np.int32), dtype=wp.int32, device=device)
        self.cell_type_r = 0          # cell_type is shared across replicas

        # --- per-replica-per-cell SoA (volume / COM sums / target / lambda) ---
        # derive each replica's volume + int64 COM sums from its own lattice
        vol = np.zeros(R * n1, dtype=np.float32)
        xsum = np.zeros(R * n1, dtype=np.int64)
        ysum = np.zeros(R * n1, dtype=np.int64)
        zsum = np.zeros(R * n1, dtype=np.int64)
        tv = np.zeros(R * n1, dtype=np.float32)
        lv = np.zeros(R * n1, dtype=np.float32)
        ct = state.cell_type
        for r in range(R):
            srep = state_from_id_lattice(cfgs[r], state.ids[r], state.cell_type)
            sl = slice(r * n1, (r + 1) * n1)
            vol[sl] = srep.volume.astype(np.float32)
            xsum[sl] = srep.xsum.astype(np.int64)
            ysum[sl] = srep.ysum.astype(np.int64)
            zsum[sl] = srep.zsum.astype(np.int64)
            tv[sl] = cfgs[r].target_volume[ct].astype(np.float32)
            lv[sl] = cfgs[r].lambda_volume[ct].astype(np.float32)
        self.volume = wp.array(vol, dtype=wp.float32, device=device)
        self.xsum = wp.array(xsum, dtype=wp.int64, device=device)
        self.ysum = wp.array(ysum, dtype=wp.int64, device=device)
        self.zsum = wp.array(zsum, dtype=wp.int64, device=device)
        self.target_volume = wp.array(tv, dtype=wp.float32, device=device)
        self.lambda_volume = wp.array(lv, dtype=wp.float32, device=device)

        # --- per-replica contact matrices (flat) + per-replica temperature/seed ---
        nt = self.n_types
        contact = np.zeros(R * nt * nt, dtype=np.float32)
        temps = np.zeros(R, dtype=np.float32)
        seeds = np.zeros(R, dtype=np.int32)
        for r in range(R):
            contact[r * nt * nt:(r + 1) * nt * nt] = cfgs[r].contact.astype(np.float32).reshape(-1)
            temps[r] = float(cfgs[r].temperature)
            seeds[r] = np.int32(cfgs[r].seed)
        self.contact = wp.array(contact, dtype=wp.float32, device=device)
        self.temperature = wp.array(temps, dtype=wp.float32, device=device)
        self.base_seed_r = wp.array(seeds, dtype=wp.int32, device=device)
        self._seeds_np = seeds

        # frozen mask is type-keyed and shared (frozen is structural, not swept)
        self.type_frozen = wp.array(self.cfg.frozen_mask(), dtype=wp.int32, device=device)

        # --- constant neighbor-offset tables (flat int32), shared across replicas ---
        co = neighbor_offsets(self.cfg.contact_neighbor_order)
        fo = neighbor_offsets(self.cfg.flip_neighbor_order)
        to = neighbor_offsets(self.cfg.tracker_neighbor_order)
        self.contact_off = wp.array(co.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.flip_off = wp.array(fo.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.tracker_off = wp.array(to.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.n_contact = int(co.shape[0])
        self.n_flip = int(fo.shape[0])
        self.n_tracker = int(to.shape[0])

        self.colors = list(range(8))
        # threads per color per replica (same upper bound as the single engine)
        self._color_threads = ((self.Lx + 1) // 2) * ((self.Ly + 1) // 2) * ((self.Lz + 1) // 2)

        # FocalPointPlasticity (Tier 1). Attached via attach_fpp(); until then the
        # batched Metropolis kernel gets length-1 dummies and fpp_enabled=0.
        self.fpp = None
        self._fpp_dummy_i32 = wp.zeros(1, dtype=wp.int32, device=device)
        self._fpp_dummy_f32 = wp.zeros(1, dtype=wp.float32, device=device)
        # Phase 8: per-sweep COM/volume snapshot the batched FPP spring reads (frozen
        # pre-sweep copy -> bit-reproducible-per-replica FPP, no read of a mid-updated
        # COM accumulator). Disjoint per-replica slices (R*n1), so no cross-replica
        # interaction. fpp_com_snapshot=False aliases the live arrays (prior path).
        self.fpp_com_snapshot = bool(fpp_com_snapshot)
        self._fpp_snap_xsum = None
        self._fpp_snap_ysum = None
        self._fpp_snap_zsum = None
        self._fpp_snap_volume = None

        # batched neighbor-contact CSR scratch (Tier 2a). Per-replica hash region +
        # the single-replica compaction scratch reused across the per-replica loop.
        self._bht_key = None
        self._bht_count = None
        self._bht_cap = 0
        self._csr_row_counts = None
        self._csr_cursor = None
        self._csr_keys = None
        self._csr_data = None
        self._csr_indices = None
        self._csr_indptr_dev = None   # device indptr (n1+1 int64), Phase-6 device scan

        # batched keyed-global-sort CSR scratch + device handles (Phase 8, Objective 2).
        # The batched CSR is one big CSR over R*n1 global rows (row r*n1+cid).
        self._gcsr_row_counts = None
        self._gcsr_cursor = None
        self._gcsr_indptr = None
        self._gcsr_keys = None
        self._gcsr_data = None
        self._gcsr_indices = None
        self.csr_indptr_dev = None    # (R*n1+1) int64 GLOBAL CSR row pointer
        self.csr_indices_dev = None   # (n_contacts) int32 dst ids
        self.csr_n_contacts = 0
        self.csr_nrows = 0

    def attach_fpp(self, fpp):
        """Attach a ``BatchedFPPLinks``: its per-replica link CSR is rebuilt once per
        MCS (the steppable boundary) and read race-free by all 8 color kernels. With
        R=1 and matching params this reproduces a single ``GPUEngine`` + ``FPPLinks``
        run bit-exactly."""
        self.fpp = fpp
        fpp.rebuild()
        return fpp

    # ------------------------------------------------- batched neighbor-contact CSR
    def neighbor_contact_csr(self, order: int | None = None):
        """Per-replica common-surface-area CSR (Tier 2a). Returns a length-R list of
        ``(indptr, indices, data)`` -- each byte-identical to the single
        ``GPUEngine.neighbor_contact_csr`` on that replica's lattice (the building
        block the batched Embryo steppables consume per replica).

        One batched hash launch fills R independent per-replica hash regions; each
        region is then compacted with the SAME on-device pipeline as the single
        engine (count -> host cumsum -> compact -> global int64 radix sort ->
        device dst-extract), looped over replicas (cheap at sweep lattice sizes)."""
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        off_w = wp.array(off.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        n_off = int(off.shape[0])
        n1 = self.n1
        nvox = self.nvox
        R = self.R

        upper = min(int(nvox) * n_off, int(n1) * int(n1))
        cap = 1
        target = max(1024, upper * 2)
        while cap < target:
            cap <<= 1
        if self._bht_key is None or self._bht_cap != cap:
            self._bht_key = wp.zeros(R * cap, dtype=wp.int64, device=self.device)
            self._bht_count = wp.zeros(R * cap, dtype=wp.int32, device=self.device)
            self._bht_cap = cap
        self._bht_key.fill_(wp.int64(-1))
        self._bht_count.zero_()

        wp.launch(
            K.neighbor_contact_hash_batched_kernel,
            dim=R * nvox,
            inputs=[
                self.ids, self.Lx, self.Ly, self.Lz, nvox, R,
                off_w, n_off, wp.int64(n1), cap,
                self._bht_key, self._bht_count,
            ],
            device=self.device,
        )
        wp.synchronize()

        out = []
        for r in range(R):
            ks = self._bht_key[r * cap:(r + 1) * cap]
            cs = self._bht_count[r * cap:(r + 1) * cap]
            out.append(self._compact_csr_slice(ks, cs, cap, n1))
        return out

    def publish_neighbor_csr_device(self, order: int | None = None):
        """Batched neighbor-CSR via ONE keyed global radix sort (Phase 8, Objective 2):
        build all R replicas' contact CSR with a SINGLE sort (replica id packed in the
        key's high bits -> disjoint per-replica ranges) + ONE global exclusive scan,
        and publish DEVICE handles (no per-replica host loop, no host copyback):

          * ``csr_indptr_dev``  : (R*n1+1) int64 GLOBAL CSR row pointer. Row ``r*n1+c``
            holds cell ``c``'s out-neighbors in replica ``r``. Cumulative across all
            replicas (replica r's block is contiguous, because the sort grouped by r).
          * ``csr_indices_dev`` : (n_contacts) int32 dst ids (the global indices array).
          * ``csr_n_contacts``  : total #directed contacts across all replicas.

        These are exactly the seam the batched device link/cohesotaxis kernels read per
        (replica, cell). Returns ``n_contacts``."""
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        off_w = wp.array(off.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        n_off = int(off.shape[0])
        n1 = self.n1
        nvox = self.nvox
        R = self.R

        upper = min(int(nvox) * n_off, int(n1) * int(n1))
        cap = 1
        target = max(1024, upper * 2)
        while cap < target:
            cap <<= 1
        if self._bht_key is None or self._bht_cap != cap:
            self._bht_key = wp.zeros(R * cap, dtype=wp.int64, device=self.device)
            self._bht_count = wp.zeros(R * cap, dtype=wp.int32, device=self.device)
            self._bht_cap = cap
        self._bht_key.fill_(wp.int64(-1))
        self._bht_count.zero_()
        wp.launch(
            K.neighbor_contact_hash_batched_kernel,
            dim=R * nvox,
            inputs=[self.ids, self.Lx, self.Ly, self.Lz, nvox, R,
                    off_w, n_off, wp.int64(n1), cap, self._bht_key, self._bht_count],
            device=self.device,
        )

        # global-row degree (R*n1 rows) + global append cursor
        nrows = R * n1
        if (self._gcsr_row_counts is None or
                self._gcsr_row_counts.shape[0] < nrows + 1):
            self._gcsr_row_counts = wp.zeros(nrows + 1, dtype=wp.int32, device=self.device)
            self._gcsr_cursor = wp.zeros(1, dtype=wp.int32, device=self.device)
            self._gcsr_indptr = wp.zeros(nrows + 1, dtype=wp.int64, device=self.device)
        self._gcsr_row_counts.zero_()
        self._gcsr_cursor.zero_()

        # compact all R hash regions into one dense array with replica-keyed global keys
        upper_contacts = R * cap
        if self._gcsr_keys is None or self._gcsr_keys.shape[0] < 2 * upper_contacts:
            # provisional sizing; resized precisely once n_contacts is known
            self._gcsr_keys = wp.zeros(max(8, 2 * upper_contacts), dtype=wp.int64, device=self.device)
            self._gcsr_data = wp.zeros(max(8, 2 * upper_contacts), dtype=wp.int32, device=self.device)
        wp.launch(
            K.neighbor_csr_compact_batched_global_kernel,
            dim=R * cap,
            inputs=[self._bht_key, self._bht_count, R, cap, wp.int64(n1),
                    self._gcsr_cursor, self._gcsr_keys, self._gcsr_data],
            device=self.device,
        )
        wp.synchronize()
        n_contacts = int(self._gcsr_cursor.numpy()[0])
        if n_contacts == 0:
            self.csr_indptr_dev = self._gcsr_indptr
            self.csr_indptr_dev.zero_()
            self.csr_indices_dev = wp.zeros(0, dtype=wp.int32, device=self.device)
            self.csr_n_contacts = 0
            self.csr_nrows = nrows
            return 0

        # count per global-row degree from the (unsorted) compacted keys
        wp.launch(
            K.neighbor_csr_count_global_kernel,
            dim=n_contacts,
            inputs=[self._gcsr_keys, n_contacts, wp.int64(n1), self._gcsr_row_counts],
            device=self.device,
        )
        # ONE global exclusive scan over all R*n1 rows -> global indptr (int64)
        S.exclusive_scan_to_ptr_i64(self._gcsr_row_counts, nrows, self._gcsr_indptr,
                                    self.device)
        # ONE global radix sort: ascending global key == (replica, src) major, dst asc
        wp.utils.radix_sort_pairs(self._gcsr_keys, self._gcsr_data, n_contacts)
        if self._gcsr_indices is None or self._gcsr_indices.shape[0] < n_contacts:
            self._gcsr_indices = wp.zeros(n_contacts, dtype=wp.int32, device=self.device)
        wp.launch(
            K.neighbor_csr_extract_dst_global_kernel,
            dim=n_contacts,
            inputs=[self._gcsr_keys, n_contacts, wp.int64(n1), self._gcsr_indices],
            device=self.device,
        )
        self.csr_indptr_dev = self._gcsr_indptr
        self.csr_indices_dev = self._gcsr_indices[:n_contacts]
        self.csr_n_contacts = n_contacts
        self.csr_nrows = nrows
        return n_contacts

    def _compact_csr_slice(self, ht_key, ht_count, cap: int, n1: int):
        """Compact one replica's hash slice into (indptr, indices, data) -- the exact
        single-engine on-device pipeline (``GPUEngine._compact_csr_device``), pointed
        at a per-replica view. Scratch is cached + grown across the replica loop."""
        if self._csr_row_counts is None or self._csr_row_counts.shape[0] < n1 + 1:
            self._csr_row_counts = wp.zeros(n1 + 1, dtype=wp.int32, device=self.device)
            self._csr_cursor = wp.zeros(n1 + 1, dtype=wp.int32, device=self.device)
            self._csr_indptr_dev = wp.zeros(n1 + 1, dtype=wp.int64, device=self.device)
        self._csr_row_counts.zero_()
        wp.launch(
            K.neighbor_csr_count_kernel,
            dim=cap,
            inputs=[ht_key, cap, wp.int64(n1), self._csr_row_counts],
            device=self.device,
        )
        # exclusive prefix sum of the per-source degree -> indptr, ON DEVICE (Phase 6:
        # replaces the host np.cumsum; byte-identical, same as the single-engine path).
        # (The scan fully writes indptr, so no pre-zero needed.)
        S.exclusive_scan_to_ptr_i64(self._csr_row_counts, n1, self._csr_indptr_dev,
                                    self.device)
        wp.synchronize()
        indptr = self._csr_indptr_dev.numpy()
        n_contacts = int(indptr[-1])
        if n_contacts == 0:
            return indptr, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        need = 2 * n_contacts
        if self._csr_keys is None or self._csr_keys.shape[0] < need:
            self._csr_keys = wp.zeros(need, dtype=wp.int64, device=self.device)
            self._csr_data = wp.zeros(need, dtype=wp.int32, device=self.device)
        self._csr_cursor.zero_()
        wp.launch(
            K.neighbor_csr_compact_kernel,
            dim=cap,
            inputs=[ht_key, ht_count, cap, self._csr_cursor, self._csr_keys, self._csr_data],
            device=self.device,
        )
        wp.utils.radix_sort_pairs(self._csr_keys, self._csr_data, n_contacts)
        if self._csr_indices is None or self._csr_indices.shape[0] < n_contacts:
            self._csr_indices = wp.zeros(n_contacts, dtype=wp.int32, device=self.device)
        wp.launch(
            K.neighbor_csr_extract_dst_kernel,
            dim=n_contacts,
            inputs=[self._csr_keys, n_contacts, wp.int64(n1), self._csr_indices],
            device=self.device,
        )
        wp.synchronize()
        indices = self._csr_indices.numpy()[:n_contacts].astype(np.int64)
        data = self._csr_data[:n_contacts].numpy().astype(np.int64)
        return indptr, indices, data

    # ------------------------------------------------------------ config plumbing
    def _resolve_configs(self, per_replica_config):
        if per_replica_config is None:
            # all replicas share the base config but get a per-replica base seed
            # (so identical-param replicas are still INDEPENDENT streams, matching
            # R independent single runs seeded base_seed + r*stride).
            base = self.cfg
            cfgs = []
            for r in range(self.R):
                cfgs.append(self._with_seed(base, int(base.seed) + r * self.REPLICA_SEED_STRIDE))
            return cfgs
        assert len(per_replica_config) == self.R, (
            f"per_replica_config length {len(per_replica_config)} != R {self.R}"
        )
        # honor each provided config's swept fields; if a caller leaves all seeds
        # equal we still de-correlate replicas via the per-replica stride so the
        # sweep replicas are independent draws (documented behavior).
        cfgs = []
        seen_seeds = set()
        for r, c in enumerate(per_replica_config):
            s = int(c.seed)
            if s in seen_seeds:
                s = int(self.cfg.seed) + r * self.REPLICA_SEED_STRIDE
            seen_seeds.add(int(c.seed))
            cfgs.append(self._with_seed(c, s))
        return cfgs

    @staticmethod
    def _with_seed(cfg: EngineConfig, seed: int) -> EngineConfig:
        return EngineConfig(
            Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz, temperature=cfg.temperature,
            n_types=cfg.n_types, target_volume=cfg.target_volume,
            lambda_volume=cfg.lambda_volume, contact=cfg.contact, frozen=cfg.frozen,
            contact_neighbor_order=cfg.contact_neighbor_order,
            flip_neighbor_order=cfg.flip_neighbor_order,
            tracker_neighbor_order=cfg.tracker_neighbor_order,
            flip2_dim_ratio=cfg.flip2_dim_ratio, seed=int(seed),
        )

    def _validate_structural(self, cfgs):
        base = self.cfg
        for r, c in enumerate(cfgs):
            assert (c.Lx, c.Ly, c.Lz) == (base.Lx, base.Ly, base.Lz), \
                f"replica {r} lattice size differs (structural, not swept)"
            assert c.n_types == base.n_types, f"replica {r} n_types differs"
            assert c.contact_neighbor_order == base.contact_neighbor_order, \
                f"replica {r} contact_neighbor_order differs"
            assert c.flip_neighbor_order == base.flip_neighbor_order, \
                f"replica {r} flip_neighbor_order differs"

    # ----------------------------------------------------------- FPP COM snapshot
    def _refresh_fpp_snapshot(self):
        """Copy the live per-replica COM/volume into the per-sweep FPP snapshot (Phase
        8). Disjoint per-replica slices, so each replica's spring reads only its own
        frozen COM -> bit-reproducible per replica. Returns the live arrays (alias) if
        the snapshot is off (prior path)."""
        if not self.fpp_com_snapshot:
            return self.xsum, self.ysum, self.zsum, self.volume
        n = self.R * self.n1
        if self._fpp_snap_xsum is None:
            self._fpp_snap_xsum = wp.zeros(n, dtype=wp.int64, device=self.device)
            self._fpp_snap_ysum = wp.zeros(n, dtype=wp.int64, device=self.device)
            self._fpp_snap_zsum = wp.zeros(n, dtype=wp.int64, device=self.device)
            self._fpp_snap_volume = wp.zeros(n, dtype=wp.float32, device=self.device)
        wp.copy(self._fpp_snap_xsum, self.xsum)
        wp.copy(self._fpp_snap_ysum, self.ysum)
        wp.copy(self._fpp_snap_zsum, self.zsum)
        wp.copy(self._fpp_snap_volume, self.volume)
        return (self._fpp_snap_xsum, self._fpp_snap_ysum,
                self._fpp_snap_zsum, self._fpp_snap_volume)

    # ------------------------------------------------------------------ sweep
    def step_mcs(self, mcs: int):
        """One Monte Carlo step for ALL replicas: 8 color launches over R*voxels.

        FPP links are STATIC within a sweep: if a ``BatchedFPPLinks`` is attached, its
        per-replica CSR is rebuilt ONCE here (the per-MCS boundary), then read
        race-free by all 8 color kernels (same contract as the single engine)."""
        if self.fpp is not None and self.fpp.has_links():
            self.fpp.rebuild()
            fpp_enabled = 1
            link_ptr = self.fpp.link_ptr
            link_other = self.fpp.link_other
            link_lambda = self.fpp.link_lambda
            link_target = self.fpp.link_target
            pay_stride = self.fpp.link_pay_stride
            snap_x, snap_y, snap_z, snap_v = self._refresh_fpp_snapshot()
        else:
            fpp_enabled = 0
            link_ptr = self._fpp_dummy_i32
            link_other = self._fpp_dummy_i32
            link_lambda = self._fpp_dummy_f32
            link_target = self._fpp_dummy_f32
            pay_stride = 1
            snap_x, snap_y, snap_z, snap_v = (
                self.xsum, self.ysum, self.zsum, self.volume)

        dim = self.R * self._color_threads
        for color in self.colors:
            wp.launch(
                K.metropolis_color_batched_kernel,
                dim=dim,
                inputs=[
                    self.ids, self.cell_type, self.cell_type_r,
                    self.volume, self.xsum, self.ysum, self.zsum,
                    self.target_volume, self.lambda_volume,
                    self.contact, self.n_types,
                    self.type_frozen,
                    self.contact_off, self.n_contact,
                    self.flip_off, self.n_flip,
                    self.Lx, self.Ly, self.Lz,
                    self.nvox, self.n1, self.R,
                    self._color_threads,
                    color, mcs, self.base_seed_r, self.temperature,
                    fpp_enabled, link_ptr, link_other, link_lambda, link_target,
                    pay_stride,
                    snap_x, snap_y, snap_z, snap_v,
                ],
                device=self.device,
            )

    def run(self, n_mcs: int, mcs_offset: int = 0):
        for m in range(n_mcs):
            self.step_mcs(mcs_offset + m)
        wp.synchronize()

    # ------------------------------------------------------------ observables
    def get_ids(self) -> np.ndarray:
        """(R, Lz, Ly, Lx) int32 id-lattices (host copy)."""
        return self.ids.numpy().reshape(self.R, self.Lz, self.Ly, self.Lx).copy()

    def volumes(self) -> np.ndarray:
        """(R, n_cells) per-cell volume excluding Medium."""
        v = self.volume.numpy().reshape(self.R, self.n1)
        return v[:, 1:].copy()

    def com_sums(self):
        """Raw int64 COM accumulators (xsum, ysum, zsum), each (R, n_cells+1).
        Bit-exact reproducibility is asserted on these (no float division)."""
        xs = self.xsum.numpy().reshape(self.R, self.n1).copy()
        ys = self.ysum.numpy().reshape(self.R, self.n1).copy()
        zs = self.zsum.numpy().reshape(self.R, self.n1).copy()
        return xs, ys, zs

    def coms(self) -> np.ndarray:
        """(R, n_cells, 3) per-cell COM (from int64 sums / volume), excl. Medium."""
        vol = self.volume.numpy().reshape(self.R, self.n1).astype(np.float64)
        v = np.where(vol > 0, vol, 1.0)
        xs, ys, zs = self.com_sums()
        cx = xs.astype(np.float64) / v
        cy = ys.astype(np.float64) / v
        cz = zs.astype(np.float64) / v
        return np.stack([cx, cy, cz], axis=2)[:, 1:, :]

    def surface_areas(self, order: int | None = None) -> np.ndarray:
        """(R, n_cells) per-cell surface area (differing-cell shell-neighbor pairs)."""
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        off_w = wp.array(off.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        surf = wp.zeros(self.R * self.n1, dtype=wp.int32, device=self.device)
        wp.launch(
            K.surface_batched_kernel,
            dim=self.R * self.nvox,
            inputs=[
                self.ids, self.Lx, self.Ly, self.Lz, self.nvox, self.n1,
                off_w, int(off.shape[0]), surf,
            ],
            device=self.device,
        )
        wp.synchronize()
        s = surf.numpy().reshape(self.R, self.n1)
        return s[:, 1:].copy()

    def total_energy(self) -> np.ndarray:
        """(R,) total Hamiltonian per replica = Volume energy + Contact energy."""
        vol = self.volume.numpy().reshape(self.R, self.n1).astype(np.float64)
        tv = self.target_volume.numpy().reshape(self.R, self.n1).astype(np.float64)
        lv = self.lambda_volume.numpy().reshape(self.R, self.n1).astype(np.float64)
        e_vol = np.sum(lv[:, 1:] * (vol[:, 1:] - tv[:, 1:]) ** 2, axis=1)
        accum = wp.zeros(self.R, dtype=wp.float32, device=self.device)
        wp.launch(
            K.total_contact_energy_batched_kernel,
            dim=self.R * self.nvox,
            inputs=[
                self.ids, self.cell_type, self.cell_type_r,
                self.contact, self.n_types,
                self.contact_off, self.n_contact,
                self.Lx, self.Ly, self.Lz, self.nvox, self.n1, accum,
            ],
            device=self.device,
        )
        wp.synchronize()
        e_contact = accum.numpy().astype(np.float64)
        return (e_vol + e_contact).astype(np.float64)

    # --------------------------------------------------------- consistency API
    def assert_volume_partition(self):
        """For EVERY replica: per-cell volume SoA == that replica's lattice voxel
        count exactly, and int64 COM sums == a fresh recompute (atomics never drift
        across the batch axis)."""
        ids = self.ids.numpy().reshape(self.R, self.nvox)
        soa_vol = self.volume.numpy().reshape(self.R, self.n1).astype(np.float64)
        for r in range(self.R):
            counts = np.bincount(ids[r], minlength=self.n1).astype(np.float64)
            if not np.array_equal(counts[1:], soa_vol[r, 1:]):
                bad = np.nonzero(counts[1:] != soa_vol[r, 1:])[0][:8] + 1
                raise AssertionError(
                    f"replica {r}: volume SoA != lattice voxel count for cells "
                    f"{bad.tolist()} (soa={soa_vol[r, bad].tolist()} counts={counts[bad].tolist()})"
                )
        # COM sums: fresh recompute and exact int64 compare
        v2 = wp.zeros(self.R * self.n1, dtype=wp.float32, device=self.device)
        xs = wp.zeros(self.R * self.n1, dtype=wp.int64, device=self.device)
        ys = wp.zeros(self.R * self.n1, dtype=wp.int64, device=self.device)
        zs = wp.zeros(self.R * self.n1, dtype=wp.int64, device=self.device)
        wp.launch(
            K.recompute_volume_com_batched_kernel,
            dim=self.R * self.nvox,
            inputs=[self.ids, self.Lx, self.Ly, self.Lz, self.nvox, self.n1, v2, xs, ys, zs],
            device=self.device,
        )
        wp.synchronize()
        live = (self.xsum.numpy(), self.ysum.numpy(), self.zsum.numpy())
        fresh = (xs.numpy(), ys.numpy(), zs.numpy())
        for name, a, b in zip(("xsum", "ysum", "zsum"), live, fresh):
            a = a.reshape(self.R, self.n1)
            b = b.reshape(self.R, self.n1)
            if not np.array_equal(a[:, 1:], b[:, 1:]):
                rr = np.nonzero(np.any(a[:, 1:] != b[:, 1:], axis=1))[0][:4]
                raise AssertionError(f"COM {name} drifted from recompute for replicas {rr.tolist()}")
        return True


def run_batched(
    cfg: EngineConfig,
    n_mcs: int,
    state: BatchedState,
    per_replica_config=None,
    device: str = "cuda:0",
) -> BatchedGPUEngine:
    eng = BatchedGPUEngine(state, per_replica_config=per_replica_config, device=device)
    eng.run(n_mcs)
    return eng
