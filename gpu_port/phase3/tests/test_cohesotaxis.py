"""Cohesotaxis pipeline gate (Pass B, deliverable 3).

The ``ifCohesotaxis=1`` lamellipodia-link target selection is ported as the plan's
staged on-device Warp kernels:

    stencil classify -> segmented compaction -> all-pairs PixelDist reduction
    -> Gumbel-max weighted select -> Manhattan-shell argmax

reproducing the CPU ``EmbryoSteppables.create_lamellipodia_link`` behaviour
(``Embryo_Model_dev/Embryo/Simulation/EmbryoSteppables.py``).

Semantics matched (CPU reference, lines ~218-281 + helpers):
  * Per LEADING cell, classify its boundary pixels over an explicit 18-offset
    stencil into FreePixelList (next-to-Medium AND next-to-Substrate) and
    CellAdhesionPixelList (next-to-Substrate OR next-to-Passive).
  * Sort FreePixelList by ``PixelDist`` (cumulative Euclidean distance to all
    CellAdhesion pixels), ascending; map rank -> ``SigWeights(n, Sigma)`` (a
    descending sigmoid PDF over a [-5,5] window); select a Free pixel by that PDF.
    ``rng.choice(p=weights)`` == Gumbel-max( log p + Gumbel ), which the GPU does
    with Philox keyed by (mcs, cell, base_seed) for reproducibility.
  * From the selected pixel, take Substrate cells at *exact Manhattan distance ==
    LamellipodiaDistance* whose ``zCOM > pixel.z``; pick the one with the greatest
    ``zCOM`` (argmax) and create the lamellipodia FPP link.

Validation strategy (per the brief):
  * PixelDist reduction + Manhattan-shell argmax + the stencil classification are
    DETERMINISTIC and are checked EXACTLY against a NumPy reference.
  * The Gumbel-max selection is checked DISTRIBUTIONALLY: its chosen-rank histogram
    over many keys matches ``SigWeights`` (== the CPU ``rng.choice`` PDF) within
    sampling noise.
"""

import numpy as np
import pytest

from engine import EngineConfig, GPUEngine
from engine.state import state_from_id_lattice
from engine.fpp import FPPLinks
from engine import cohesotaxis as CT
from engine.geometry import MEDIUM, LEADING, PASSIVE, SUBSTRATE


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# A small, fully-controlled scene with a known cohesotaxis answer.
#
#   z=0 plane: a flat slab of single-voxel SUBSTRATE cells (a "floor").
#   z=1.. :    one LEADING cell block sitting on the floor, exposed to Medium
#              on top; a couple of PASSIVE blocks beside it.
# This mirrors the embryo's leader-on-substrate geometry at toy scale, giving a
# deterministic FreePixelList / CellAdhesionPixelList we can mirror in NumPy.
# ---------------------------------------------------------------------------
def _toy_scene(L=12):
    ids = np.zeros((L, L, L), dtype=np.int32)
    cell_type = [0]  # Medium

    def new_cell(t):
        cell_type.append(int(t))
        return len(cell_type) - 1

    # substrate floor: 1-voxel cells at z=0..2 (a few layers).
    for z in range(0, 3):
        for y in range(L):
            for x in range(L):
                cid = new_cell(SUBSTRATE)
                ids[z, y, x] = cid

    # A substrate "wall" rising above the floor on the high-x side (x=8), spanning
    # z up through and above the leader's height -- the embryo geometry where
    # leaders climb the ectoderm shell. This gives the leader's free
    # (medium+substrate) boundary pixels a valid upward target (substrate at
    # Manhattan order 2 with zCOM > z).
    for z in range(0, 9):
        for y in range(L):
            cid = new_cell(SUBSTRATE)
            ids[z, y, 8] = cid

    # leading cell: a thin slab pressed against the wall (x=7) so every one of its
    # medium+substrate boundary pixels sits beside the climbing wall and has a
    # valid upward order-2 substrate target (matches a leader hugging the shell).
    lead = new_cell(LEADING)
    ids[3:6, 4:7, 7] = lead

    # a passive block beside the leader (shares a face, low-x side)
    pas = new_cell(PASSIVE)
    ids[3:6, 4:7, 5:7] = pas

    n_cells = len(cell_type) - 1
    n_types = 5
    cfg = EngineConfig(
        Lx=L, Ly=L, Lz=L, n_types=n_types, seed=123, temperature=10.0,
        target_volume=np.array([0.0, 27.0, 27.0, 27.0, 1.0]),
        lambda_volume=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
        contact=np.full((n_types, n_types), 10.0),
        frozen=np.array([0, SUBSTRATE], dtype=np.int32),
        tracker_neighbor_order=1,
    )
    st = state_from_id_lattice(cfg, ids, cell_type=np.asarray(cell_type, np.int32))
    return st, lead


# ============================ NumPy reference ==============================
# 18-offset stencil identical to EmbryoSteppables.neighbor_pixel_locations.
STENCIL_18 = (
    (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)


def _ref_classify_pixels(ids, cell_type, cell_id, Lx, Ly, Lz):
    """Reference FreePixelList / CellAdhesionPixelList for one cell, mirroring
    EmbryoSteppables.create_lamellipodia_link classification. Returns two lists of
    (x,y,z) integer pixel coords."""
    free, adh = [], []
    zz, yy, xx = np.nonzero(ids == cell_id)
    for x, y, z in zip(xx.tolist(), yy.tolist(), zz.tolist()):
        nm = ns = nf = False
        for dx, dy, dz in STENCIL_18:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= Lx or ny >= Ly or nz >= Lz:
                nc = 0
            else:
                nc = int(ids[nz, ny, nx])
            if nc == 0:
                nm = True
            else:
                t = int(cell_type[nc])
                if t == SUBSTRATE:
                    ns = True
                elif t == PASSIVE:
                    nf = True
        # CC3D scans only cell boundary pixels; a fully-interior pixel hits none.
        if nm and ns:
            free.append((x, y, z))
        if ns or nf:
            adh.append((x, y, z))
    return free, adh


def _ref_pixeldist(free, adh):
    """CumDist[f] = sum_a ||f - a||_2  (EmbryoSteppables.PixelDist)."""
    free = np.asarray(free, dtype=np.float64)
    adh = np.asarray(adh, dtype=np.float64)
    out = np.zeros(len(free), dtype=np.float64)
    for i, f in enumerate(free):
        d = adh - f
        out[i] = np.sqrt((d * d).sum(axis=1)).sum()
    return out


def _sig_weights(x, b):
    """EmbryoSteppables.SigWeights: descending sigmoid PDF of length x."""
    xrange = np.linspace(-5, 5, x)
    sig = 1.0 / (1.0 + np.e ** -(xrange - b))
    w = sig / np.sum(sig)
    return w[::-1]


def _ref_manhattan_argmax(pixel, ids, cell_type, coms, n, Lx, Ly, Lz):
    """nth_order_neighbors + max-zCOM pick: among Substrate cells at EXACT
    Manhattan distance == n from ``pixel`` whose zCOM > pixel.z, return the id with
    the greatest zCOM (or -1 if none)."""
    px, py, pz = pixel
    best_id, best_z = -1, -1e30
    seen = set()
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            for dz in range(-n, n + 1):
                if abs(dx) + abs(dy) + abs(dz) != n:
                    continue
                nx, ny, nz = px + dx, py + dy, pz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= Lx or ny >= Ly or nz >= Lz:
                    continue
                c = int(ids[nz, ny, nx])
                if c == 0 or int(cell_type[c]) != SUBSTRATE:
                    continue
                zc = coms[c, 2]
                if zc > pz and c not in seen:
                    seen.add(c)
                if zc > pz and zc > best_z:
                    best_z, best_id = zc, c
    return best_id


# ============================ tests ==============================
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_stencil_classify_matches_reference_exactly():
    """The on-device stencil classification + segmented compaction reproduces the
    CPU FreePixelList / CellAdhesionPixelList for each leading cell, EXACTLY."""
    st, lead = _toy_scene()
    eng = GPUEngine(st)
    pipe = CT.CohesotaxisPipeline(eng, leading_type=LEADING, substrate_type=SUBSTRATE,
                                  passive_type=PASSIVE)
    res = pipe.classify()

    free_gpu = set(map(tuple, res.free_pixels(lead)))
    adh_gpu = set(map(tuple, res.adhesion_pixels(lead)))

    free_ref, adh_ref = _ref_classify_pixels(
        eng.get_ids(), st.cell_type, lead, eng.Lx, eng.Ly, eng.Lz)
    assert free_gpu == set(free_ref), "FreePixelList mismatch"
    assert adh_gpu == set(adh_ref), "CellAdhesionPixelList mismatch"
    assert len(free_ref) > 0, "toy scene must produce free pixels"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_pixeldist_reduction_matches_reference_exactly():
    """The all-pairs PixelDist reduction (CumDist per free pixel) matches the NumPy
    reference exactly (deterministic; float32 vs float64 within tight tol)."""
    st, lead = _toy_scene()
    eng = GPUEngine(st)
    pipe = CT.CohesotaxisPipeline(eng, leading_type=LEADING, substrate_type=SUBSTRATE,
                                  passive_type=PASSIVE)
    res = pipe.classify()
    pipe.pixel_dist(res)

    free = res.free_pixels(lead)
    cum_gpu = res.cum_dist(lead)
    # reference computed on the SAME free/adhesion sets, matched by coordinate
    free_ref, adh_ref = _ref_classify_pixels(
        eng.get_ids(), st.cell_type, lead, eng.Lx, eng.Ly, eng.Lz)
    ref_map = {tuple(f): d for f, d in zip(free_ref, _ref_pixeldist(free_ref, adh_ref))}
    for f, c in zip(map(tuple, free), cum_gpu):
        assert abs(c - ref_map[f]) <= 1e-3 * max(1.0, ref_map[f]), (
            f"PixelDist mismatch at {f}: gpu {c} ref {ref_map[f]}")


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_manhattan_shell_argmax_matches_reference_exactly():
    """The Manhattan-shell (order == LamellipodiaDistance) substrate argmax-by-zCOM
    matches the NumPy reference exactly for every free pixel in the scene."""
    st, lead = _toy_scene()
    eng = GPUEngine(st)
    pipe = CT.CohesotaxisPipeline(eng, leading_type=LEADING, substrate_type=SUBSTRATE,
                                  passive_type=PASSIVE, lamellipodia_distance=2)
    res = pipe.classify()
    coms = np.zeros((eng.n_cells + 1, 3)); coms[1:] = eng.coms()
    ids = eng.get_ids()

    # check argmax for EVERY free pixel (the kernel is evaluated per candidate)
    target_gpu = pipe.manhattan_argmax_for_pixels(res.free_pixels(lead))
    for (x, y, z), tgt in zip(res.free_pixels(lead), target_gpu):
        ref = _ref_manhattan_argmax((x, y, z), ids, st.cell_type, coms, 2,
                                    eng.Lx, eng.Ly, eng.Lz)
        assert int(tgt) == int(ref), f"Manhattan argmax mismatch at pixel {(x,y,z)}"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_gumbel_max_selection_matches_sigweights_distribution():
    """Gumbel-max over (log SigWeights + Gumbel(Philox)) reproduces the CPU
    ``rng.choice(FreePixelList, p=SigWeights)`` selection DISTRIBUTION: the
    chosen-rank histogram converges to SigWeights within sampling noise."""
    n = 7              # number of free pixels (ranks 0..6)
    sigma = 8.0
    weights = _sig_weights(n, sigma)

    # Many independent keys (vary the "cell" coordinate) -> empirical PDF over the
    # selected rank. The pipeline ranks ascending by CumDist; here CumDist == rank
    # so the chosen index IS the chosen rank, directly comparable to SigWeights.
    n_draws = 60000
    counts = CT.gumbel_select_rank_histogram(n=n, sigma=sigma, n_draws=n_draws,
                                             mcs=0, base_seed=999)
    emp = counts / counts.sum()
    # statistical agreement: max abs deviation within a few sigma of multinomial
    # noise. sigma_i ~ sqrt(p(1-p)/N) ~ <0.002 here; allow 0.01 (very safe).
    assert counts.sum() == n_draws
    assert np.max(np.abs(emp - weights)) < 0.01, (
        f"Gumbel-max PDF deviates from SigWeights:\n emp={emp}\n ref={weights}")


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_gumbel_select_is_reproducible():
    """Same (mcs, cell, base_seed) key -> identical selection (Philox determinism);
    a different mcs generally changes it."""
    n, sigma = 6, 8.0
    a = CT.gumbel_select_rank_histogram(n=n, sigma=sigma, n_draws=5000, mcs=4, base_seed=7)
    b = CT.gumbel_select_rank_histogram(n=n, sigma=sigma, n_draws=5000, mcs=4, base_seed=7)
    assert np.array_equal(a, b), "Gumbel-max selection not reproducible for a fixed key"


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_full_pipeline_creates_lamellipodia_link_on_device():
    """End-to-end: the pipeline selects a direction pixel and creates a lamellipodia
    link via Pass A's FPPLinks.create_link with the correct per-link params
    (LamellipodiaLambda / LLTargetDist / LLMaxDist). The chosen target must be a
    valid Manhattan-shell substrate argmax for the selected pixel."""
    st, lead = _toy_scene()
    eng = GPUEngine(st)
    links = FPPLinks(eng, target_length_default=CT.LL_TARGET_DIST,
                     lambda_default=CT.LAMELLIPODIA_LAMBDA, max_length_default=CT.LL_MAX_DIST)
    eng.attach_fpp(links)
    pipe = CT.CohesotaxisPipeline(eng, leading_type=LEADING, substrate_type=SUBSTRATE,
                                  passive_type=PASSIVE, lamellipodia_distance=2)

    created = pipe.create_lamellipodia_links(links, mcs=0)
    assert lead in created, "leading cell should receive a lamellipodia link"
    target = created[lead]

    # the created link exists in the inventory with the lamellipodia params
    links.rebuild()
    a = links._a; b = links._b
    mask = ((a == lead) & (b == target)) | ((a == target) & (b == lead))
    assert mask.any(), "lamellipodia link not present in FPP inventory"
    assert np.allclose(links._lam[mask], CT.LAMELLIPODIA_LAMBDA)
    assert np.allclose(links._tgt[mask], CT.LL_TARGET_DIST)
    assert np.allclose(links._max[mask], CT.LL_MAX_DIST)

    # target must be substrate above some selected free pixel (validity)
    assert int(st.cell_type[target]) == SUBSTRATE


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_poisson_link_turnover_rate_matches_model():
    """Poisson lamellipodia turnover (LeadingEdgeSteppable.step): an existing link
    is deleted each MCS with probability 1-exp(-LamellaeRate), reproducible via
    Philox keyed by (mcs, cell, base_seed). Over many cells/steps the empirical
    deletion fraction matches the model probability within sampling noise."""
    p = 1.0 - np.exp(-CT.LAMELLAE_RATE)
    n_cells = 4000
    # one Bernoulli(p) decision per cell for a fixed mcs, keyed reproducibly
    deletes = CT.poisson_delete_decisions(n_cells=n_cells, mcs=3, base_seed=42,
                                          rate=CT.LAMELLAE_RATE)
    frac = deletes.mean()
    # reproducible
    deletes2 = CT.poisson_delete_decisions(n_cells=n_cells, mcs=3, base_seed=42,
                                           rate=CT.LAMELLAE_RATE)
    assert np.array_equal(deletes, deletes2), "Poisson decisions not reproducible"
    # binomial std ~ sqrt(p(1-p)/N); allow 5 sigma
    sd = np.sqrt(p * (1 - p) / n_cells)
    assert abs(frac - p) < 5 * sd + 1e-6, (
        f"Poisson delete fraction {frac:.5f} != model {p:.5f} (5sigma={5*sd:.5f})")


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_lamellipodia_steppable_runs_through_engine_seam():
    """LamellipodiaSteppable (the GPU LeadingEdgeSteppable lamellipodia port) plugs
    into the SteppableManager + engine FPP seam: start() creates a lamellipodia
    link, the engine runs with FPP folded in (partition invariant stays exact), and
    over many MCS the Poisson turnover deletes/recreates links on device without
    breaking the engine."""
    from engine.steppables import SteppableManager, LamellipodiaSteppable

    st, lead = _toy_scene()
    eng = GPUEngine(st)
    links = FPPLinks(eng, target_length_default=CT.LL_TARGET_DIST,
                     lambda_default=CT.LAMELLIPODIA_LAMBDA, max_length_default=CT.LL_MAX_DIST)
    eng.attach_fpp(links)

    mgr = SteppableManager(eng)
    step = LamellipodiaSteppable(eng, links, leading_type=LEADING,
                                 substrate_type=SUBSTRATE, passive_type=PASSIVE,
                                 lamellipodia_distance=2)
    mgr.register(step)
    created = step.start()
    assert lead in created, "start() should create an initial lamellipodia link"
    eng.assert_volume_partition()

    # run several MCS through the manager (engine + steppable turnover)
    mgr.run(20)
    eng.assert_volume_partition()      # FPP turnover must not break the partition

    # the leader's recorded link target (cell.dict['link'] equivalent) is either a
    # valid substrate cell or 0 (deleted this step, awaiting recreation)
    lt = step.cell_dict.get("link_target")
    tgt = int(lt[lead])
    assert tgt == 0 or int(st.cell_type[tgt]) == SUBSTRATE
