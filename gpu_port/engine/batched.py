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

from .config import EngineConfig, neighbor_offsets
from .state import EngineState, build_grid_state, state_from_id_lattice
from . import kernels as K

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

    def attach_fpp(self, fpp):
        """Attach a ``BatchedFPPLinks``: its per-replica link CSR is rebuilt once per
        MCS (the steppable boundary) and read race-free by all 8 color kernels. With
        R=1 and matching params this reproduces a single ``GPUEngine`` + ``FPPLinks``
        run bit-exactly."""
        self.fpp = fpp
        fpp.rebuild()
        return fpp

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
        else:
            fpp_enabled = 0
            link_ptr = self._fpp_dummy_i32
            link_other = self._fpp_dummy_i32
            link_lambda = self._fpp_dummy_f32
            link_target = self._fpp_dummy_f32
            pay_stride = 1

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
