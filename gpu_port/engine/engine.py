"""GPU-resident CPM engine: 8-color checkerboard Metropolis (Volume + Contact)
with on-GPU volume/COM trackers and per-MCS boundary-pixel + neighbor-contact CSR.

This is the production engine (Phase 2 core). State lives on the GPU:

* ``ids``        flat int32 id-lattice  (source of truth)
* per-cell SoA   ``cell_type``, ``volume`` (float32), ``xsum/ysum/zsum`` (int64)
* ``contact``    (n_types, n_types) float32 matrix
* constant neighbor-offset tables as flat int32 device arrays.

One MCS = run the 8 checkerboard colors once (each a kernel launch). Colors are
the 2x2x2 parity classes; same-color voxels are mutually outside each other's
Moore (order<=3) neighborhood, so they flip in parallel with stable neighbor
reads and never write the same voxel -- the GPU analogue of CC3D's OpenMP subgrid
checkerboard (Potts3D.cpp). Validity requires the flip-target neighbor order to
be <= 3 (enforced by EngineConfig).

COM uses int64 fixed-point accumulation of integer pixel coordinates -> exact and
reproducible. ``assert_volume_partition()`` checks the per-cell volume SoA equals
the lattice voxel count exactly.
"""

from __future__ import annotations

import numpy as np

import warp as wp
import warp.utils  # radix_sort_pairs (global int64 sort for the on-device CSR build)

from .config import EngineConfig, neighbor_offsets
from .state import EngineState
from . import kernels as K

wp.init()


class GPUEngine:
    def __init__(self, state: EngineState, device: str = "cuda:0"):
        self.cfg: EngineConfig = state.cfg
        self.device = device
        self.Lx, self.Ly, self.Lz = self.cfg.Lx, self.cfg.Ly, self.cfg.Lz
        self.n_cells = state.n_cells
        n1 = self.n_cells + 1

        # --- lattice + SoA on device ---
        ids_flat = np.ascontiguousarray(state.ids.reshape(-1), dtype=np.int32)
        self.ids = wp.array(ids_flat, dtype=wp.int32, device=device)
        self.cell_type = wp.array(state.cell_type.astype(np.int32), dtype=wp.int32, device=device)
        self.volume = wp.array(state.volume.astype(np.float32), dtype=wp.float32, device=device)
        self.xsum = wp.array(state.xsum.astype(np.int64), dtype=wp.int64, device=device)
        self.ysum = wp.array(state.ysum.astype(np.int64), dtype=wp.int64, device=device)
        self.zsum = wp.array(state.zsum.astype(np.int64), dtype=wp.int64, device=device)

        # per-cell target/lambda volume from per-type config
        tv = np.zeros(n1, dtype=np.float32)
        lv = np.zeros(n1, dtype=np.float32)
        ct = state.cell_type
        tv[:] = self.cfg.target_volume[ct]
        lv[:] = self.cfg.lambda_volume[ct]
        self.target_volume = wp.array(tv, dtype=wp.float32, device=device)
        self.lambda_volume = wp.array(lv, dtype=wp.float32, device=device)

        # contact matrix + frozen mask
        self.contact = wp.array(self.cfg.contact.astype(np.float32), dtype=wp.float32, device=device)
        self.type_frozen = wp.array(self.cfg.frozen_mask(), dtype=wp.int32, device=device)

        # --- constant neighbor-offset tables (flat int32) ---
        co = neighbor_offsets(self.cfg.contact_neighbor_order)
        fo = neighbor_offsets(self.cfg.flip_neighbor_order)
        to = neighbor_offsets(self.cfg.tracker_neighbor_order)
        self.contact_off = wp.array(co.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.flip_off = wp.array(fo.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.tracker_off = wp.array(to.flatten().astype(np.int32), dtype=wp.int32, device=device)
        self.n_contact = int(co.shape[0])
        self.n_flip = int(fo.shape[0])
        self.n_tracker = int(to.shape[0])

        self.base_seed = int(self.cfg.seed)
        self.T = float(self.cfg.temperature)
        self.colors = list(range(8))
        # threads per color launch: safe upper bound (kernel early-returns extras)
        self._color_threads = ((self.Lx + 1) // 2) * ((self.Ly + 1) // 2) * ((self.Lz + 1) // 2)

        # tracker scratch (allocated lazily)
        self._is_boundary = None
        self._boundary_count = None
        self._pair_counts = None  # legacy dense path (kept for reference/tests)

        # FocalPointPlasticity (Phase 3). Attached via attach_fpp(); until then the
        # FPP block in the kernel is disabled and fed length-1 dummy arrays.
        self.fpp = None
        self._fpp_dummy_i32 = wp.zeros(1, dtype=wp.int32, device=device)
        self._fpp_dummy_f32 = wp.zeros(1, dtype=wp.float32, device=device)

        # hashed neighbor-CSR scratch (allocated lazily; replaces the dense matrix)
        self._ht_key = None
        self._ht_count = None
        self._ht_cap = 0

        # on-device CSR compaction scratch (allocated lazily; grown on demand)
        self._csr_row_counts = None   # per-source degree (n1+1 int32)
        self._csr_cursor = None       # per-row append cursor (n1+1 int32)
        self._csr_keys = None         # dense packed keys src*n1+dst (>= 2*n_contacts int64)
        self._csr_data = None         # dense CSR counts  (>= 2*n_contacts int32)

    # ---------------------------------------------------------------- FPP attach
    def attach_fpp(self, fpp):
        """Attach an FPPLinks inventory; its per-cell CSR is folded into the
        Metropolis energy and rebuilt once per MCS (links static within a sweep)."""
        self.fpp = fpp
        fpp.rebuild()
        return fpp

    # ------------------------------------------------------------------ sweep
    def step_mcs(self, mcs: int):
        """One Monte Carlo step: 8 color launches (one full lattice sweep).

        FPP links are STATIC within a sweep: if FPP is attached, its per-cell CSR
        is rebuilt ONCE here (the per-MCS boundary), then read race-free by all 8
        color kernels. Create/delete of links happens in steppables (which run
        around the MCS), not inside this inner loop.
        """
        if self.fpp is not None and self.fpp.has_links():
            self.fpp.rebuild()
            fpp_enabled = 1
            link_ptr = self.fpp.link_ptr
            link_other = self.fpp.link_other
            link_lambda = self.fpp.link_lambda
            link_target = self.fpp.link_target
        else:
            fpp_enabled = 0
            link_ptr = self._fpp_dummy_i32
            link_other = self._fpp_dummy_i32
            link_lambda = self._fpp_dummy_f32
            link_target = self._fpp_dummy_f32

        for color in self.colors:
            wp.launch(
                K.metropolis_color_kernel,
                dim=self._color_threads,
                inputs=[
                    self.ids, self.cell_type, self.volume,
                    self.xsum, self.ysum, self.zsum,
                    self.target_volume, self.lambda_volume, self.contact,
                    self.type_frozen,
                    self.contact_off, self.n_contact,
                    self.flip_off, self.n_flip,
                    self.Lx, self.Ly, self.Lz,
                    color, mcs, self.base_seed, self.T,
                    fpp_enabled, link_ptr, link_other, link_lambda, link_target,
                ],
                device=self.device,
            )

    def run(self, n_mcs: int, mcs_offset: int = 0):
        for m in range(n_mcs):
            self.step_mcs(mcs_offset + m)
        wp.synchronize()

    # ------------------------------------------------------------ observables
    def get_ids(self) -> np.ndarray:
        """Return the id-lattice as (Lz,Ly,Lx) int32 (host copy)."""
        return self.ids.numpy().reshape(self.Lz, self.Ly, self.Lx).copy()

    def volumes(self) -> np.ndarray:
        """Per-cell volume (excludes Medium), host copy."""
        return self.volume.numpy()[1:].copy()

    def coms(self) -> np.ndarray:
        """Per-cell COM (n_cells,3) excluding Medium, from int64 sums / volume."""
        vol = self.volume.numpy().astype(np.float64)
        v = np.where(vol > 0, vol, 1.0)
        cx = self.xsum.numpy().astype(np.float64) / v
        cy = self.ysum.numpy().astype(np.float64) / v
        cz = self.zsum.numpy().astype(np.float64) / v
        return np.stack([cx, cy, cz], axis=1)[1:]

    def surface_areas(self, order: int | None = None) -> np.ndarray:
        """Per-cell surface area (count of differing-cell shell-neighbor pairs)."""
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        off_w = wp.array(off.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        surf = wp.zeros(self.n_cells + 1, dtype=wp.int32, device=self.device)
        wp.launch(
            K.surface_kernel,
            dim=self.cfg.n_voxels,
            inputs=[self.ids, self.Lx, self.Ly, self.Lz, off_w, int(off.shape[0]), surf],
            device=self.device,
        )
        wp.synchronize()
        return surf.numpy()[1:].copy()

    def total_energy(self) -> float:
        """Total Hamiltonian = Volume energy + Contact energy (host reduction of
        per-cell volume + a GPU contact reduction)."""
        # Volume energy (exact, cheap on host)
        vol = self.volume.numpy().astype(np.float64)
        tv = self.target_volume.numpy().astype(np.float64)
        lv = self.lambda_volume.numpy().astype(np.float64)
        e_vol = float(np.sum(lv[1:] * (vol[1:] - tv[1:]) ** 2))
        # Contact energy (GPU)
        accum = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(
            K.total_energy_kernel,
            dim=self.cfg.n_voxels,
            inputs=[
                self.ids, self.cell_type, self.volume,
                self.target_volume, self.lambda_volume, self.contact,
                self.contact_off, self.n_contact,
                self.Lx, self.Ly, self.Lz, accum,
            ],
            device=self.device,
        )
        wp.synchronize()
        e_contact = float(accum.numpy()[0])
        return e_vol + e_contact

    # ------------------------------------------------ neighbor-contact CSR (hash)
    def neighbor_contact_csr(self, order: int | None = None, method: str = "device"):
        """Common-surface-area CSR between cells, built with a device hash over
        directed (self,neighbor) pairs -> O(#contacts) memory (NOT the dense
        (n_cells+1)^2 matrix, which is ~16 GB at 63k cells). Exact: it reproduces
        the dense/CPU directed contact matrix.

        Returns ``(indptr, indices, data)`` with indices sorted ascending within
        each source row, indices including Medium (id 0). dtype int64 (all three).

        ``method`` selects how the device hash table is compacted into the CSR:

        * ``"device"`` (default): compact + sort ON the GPU and transfer only the
          O(#contacts) result -- a count kernel (per-source degree) -> host cumsum
          of the tiny per-row degree -> atomic-cursor scatter -> per-row ascending
          sort. Mirrors the FPP link-CSR build. Avoids copying the cap-sized hash
          table (~200 MB/MCS at full Embryo scale) and the host ``np.lexsort``.
        * ``"host"``: the original Phase-3 path -- copy the whole cap-sized table to
          the host and compact there (boolean mask -> ``np.lexsort`` -> ``np.bincount``).
          Kept as the reference the device path is asserted byte-equal to, and as a
          fallback. Produces identical output to ``"device"``.
        """
        if order is None:
            order = self.cfg.tracker_neighbor_order
        off = neighbor_offsets(order)
        off_w = wp.array(off.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        n_off = int(off.shape[0])
        n1 = self.n_cells + 1

        # capacity: a power of two comfortably above the max possible distinct
        # directed pairs. An upper bound is (#voxels * shell size); we also never
        # need more than n1*n1. Use the smaller, rounded up to a power of two,
        # with a 2x load-factor headroom (open addressing needs slack).
        nvox = self.cfg.n_voxels
        upper = min(int(nvox) * n_off, int(n1) * int(n1))
        cap = 1
        target = max(1024, upper * 2)
        while cap < target:
            cap <<= 1

        if self._ht_key is None or self._ht_cap != cap:
            self._ht_key = wp.zeros(cap, dtype=wp.int64, device=self.device)
            self._ht_count = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._ht_cap = cap
        self._ht_key.fill_(wp.int64(-1))
        self._ht_count.zero_()

        wp.launch(
            K.neighbor_contact_hash_kernel,
            dim=nvox,
            inputs=[
                self.ids, self.Lx, self.Ly, self.Lz,
                off_w, n_off, wp.int64(n1), cap,
                self._ht_key, self._ht_count,
            ],
            device=self.device,
        )

        if method == "host":
            return self._compact_csr_host(n1)
        if method == "device":
            return self._compact_csr_device(cap, n1)
        raise ValueError(f"unknown method {method!r}; use 'device' or 'host'")

    def _compact_csr_host(self, n1: int):
        """Host compaction of the hash table (original Phase-3 path): copy the whole
        cap-sized table to host and sort/bincount there. The device path's exactness
        reference."""
        wp.synchronize()
        key = self._ht_key.numpy()
        cnt = self._ht_count.numpy()
        occ = key >= 0
        keys = key[occ].astype(np.int64)
        data = cnt[occ].astype(np.int64)
        src = (keys // n1).astype(np.int64)
        dst = (keys % n1).astype(np.int64)
        # sort by (src, dst) so each CSR row's indices are ascending
        order_idx = np.lexsort((dst, src))
        src = src[order_idx]; dst = dst[order_idx]; data = data[order_idx]
        indptr = np.zeros(n1 + 1, dtype=np.int64)
        # counts per source row -> prefix sum
        row_counts = np.bincount(src, minlength=n1)[:n1]
        indptr[1:] = np.cumsum(row_counts)
        return indptr, dst.astype(np.int64), data.astype(np.int64)

    def _compact_csr_device(self, cap: int, n1: int):
        """On-device compaction of the hash table: count per-source degree -> host
        cumsum -> stream occupied (packed-key, count) into dense arrays -> ONE global
        radix sort on the int64 packed key src*n1+dst. Ascending key == ascending
        (src, dst) (dst < n1), so the single sort yields the CSR layout directly --
        no segmented sort over n1 tiny skewed rows. Transfers only the per-row counts
        (n1 ints) + the compact O(#contacts) result. Byte-identical to
        ``_compact_csr_host`` (same ascending-within-row order)."""
        # scratch (lazy; row_counts/cursor sized n1+1 like the FPP build)
        if self._csr_row_counts is None or self._csr_row_counts.shape[0] < n1 + 1:
            self._csr_row_counts = wp.zeros(n1 + 1, dtype=wp.int32, device=self.device)
            self._csr_cursor = wp.zeros(n1 + 1, dtype=wp.int32, device=self.device)
        self._csr_row_counts.zero_()

        wp.launch(
            K.neighbor_csr_count_kernel,
            dim=cap,
            inputs=[self._ht_key, cap, wp.int64(n1), self._csr_row_counts],
            device=self.device,
        )
        wp.synchronize()

        # exclusive prefix sum of the per-source degree -> indptr (host; n1 is tiny
        # and off the hot per-flip path -- the FPP-build approach).
        row_counts = self._csr_row_counts.numpy()[:n1]
        indptr = np.zeros(n1 + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(row_counts)
        n_contacts = int(indptr[-1])
        if n_contacts == 0:
            return indptr, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

        # dense sort buffers (cached, grown on demand). radix_sort_pairs double-
        # buffers, so key/value storage must hold 2*count elements.
        need = 2 * n_contacts
        if self._csr_keys is None or self._csr_keys.shape[0] < need:
            self._csr_keys = wp.zeros(need, dtype=wp.int64, device=self.device)
            self._csr_data = wp.zeros(need, dtype=wp.int32, device=self.device)
        self._csr_cursor.zero_()  # slot 0 is the single global append counter

        wp.launch(
            K.neighbor_csr_compact_kernel,
            dim=cap,
            inputs=[self._ht_key, self._ht_count, cap,
                    self._csr_cursor, self._csr_keys, self._csr_data],
            device=self.device,
        )
        # one global radix sort of the int64 packed key (co-moving the counts): groups
        # by src and orders dst ascending within each row in a single O(#contacts) pass.
        wp.utils.radix_sort_pairs(self._csr_keys, self._csr_data, n_contacts)
        wp.synchronize()

        keys = self._csr_keys.numpy()[:n_contacts]
        indices = (keys % np.int64(n1)).astype(np.int64)   # dst id (Medium 0 included)
        data = self._csr_data.numpy()[:n_contacts].astype(np.int64)
        return indptr, indices, data

    # ------------------------------------------------------ tracker maintenance
    def recompute_trackers(self):
        """Recompute boundary-pixel flags + per-cell boundary count and the
        neighbor-contact (common-surface) matrix from the current lattice.

        Returns a dict with:
          ``boundary_count`` (n_cells+1,)  voxels on each cell's boundary
          ``neighbor_csr``   (indptr, indices, data) CSR of common-surface areas
                             between cells (indices include Medium id 0).
        Recomputed once per MCS for the Python/steppable layer.
        """
        nvox = self.cfg.n_voxels
        n1 = self.n_cells + 1
        if self._is_boundary is None:
            self._is_boundary = wp.zeros(nvox, dtype=wp.int32, device=self.device)
            self._boundary_count = wp.zeros(n1, dtype=wp.int32, device=self.device)
        self._boundary_count.zero_()

        wp.launch(
            K.boundary_pixel_flag_kernel,
            dim=nvox,
            inputs=[
                self.ids, self.Lx, self.Ly, self.Lz,
                self.tracker_off, self.n_tracker,
                self._is_boundary, self._boundary_count,
            ],
            device=self.device,
        )
        wp.synchronize()

        boundary_count = self._boundary_count.numpy().copy()
        # neighbor-contact CSR via the scalable hashed build (no dense n^2 matrix)
        indptr, indices, data = self.neighbor_contact_csr()
        return {
            "boundary_count": boundary_count,
            "is_boundary": self._is_boundary.numpy().reshape(self.Lz, self.Ly, self.Lx).copy(),
            "neighbor_csr": (indptr, indices.astype(np.int64), data.astype(np.int64)),
        }

    # --------------------------------------------------------- consistency API
    def assert_volume_partition(self):
        """Assert the per-cell volume SoA equals the lattice voxel count exactly,
        and the int64 COM sums equal a fresh recompute (atomics never drifted)."""
        ids = self.ids.numpy()
        counts = np.bincount(ids, minlength=self.n_cells + 1).astype(np.float64)
        soa_vol = self.volume.numpy().astype(np.float64)
        if not np.array_equal(counts[1:], soa_vol[1:]):
            bad = np.nonzero(counts[1:] != soa_vol[1:])[0][:8] + 1
            raise AssertionError(
                f"volume SoA != lattice voxel count for cells {bad.tolist()} "
                f"(soa={soa_vol[bad].tolist()} counts={counts[bad].tolist()})"
            )
        # COM sums: recompute fresh and compare exactly (int64)
        v2 = wp.zeros(self.n_cells + 1, dtype=wp.float32, device=self.device)
        xs = wp.zeros(self.n_cells + 1, dtype=wp.int64, device=self.device)
        ys = wp.zeros(self.n_cells + 1, dtype=wp.int64, device=self.device)
        zs = wp.zeros(self.n_cells + 1, dtype=wp.int64, device=self.device)
        wp.launch(
            K.recompute_volume_com_kernel,
            dim=self.cfg.n_voxels,
            inputs=[self.ids, self.Lx, self.Ly, self.Lz, v2, xs, ys, zs],
            device=self.device,
        )
        wp.synchronize()
        for name, live, fresh in (
            ("xsum", self.xsum, xs), ("ysum", self.ysum, ys), ("zsum", self.zsum, zs)
        ):
            a = live.numpy()
            b = fresh.numpy()
            if not np.array_equal(a[1:], b[1:]):
                bad = np.nonzero(a[1:] != b[1:])[0][:8] + 1
                raise AssertionError(
                    f"COM {name} drifted from lattice recompute for cells {bad.tolist()}"
                )
        return True


def run_gpu(cfg: EngineConfig, n_mcs: int, state: EngineState, device: str = "cuda:0") -> GPUEngine:
    eng = GPUEngine(state, device=device)
    eng.run(n_mcs)
    return eng
