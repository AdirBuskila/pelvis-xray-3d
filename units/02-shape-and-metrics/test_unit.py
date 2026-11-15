"""Checks for unit 02 (shape and metrics).

Every check_* takes YOUR function and tests it on small cases whose answer was worked out by
hand (or follows from a property, like "a shape compared with itself is perfect"). There is no
reference solution in here: the expected values are numbers, not code.

  notebook:  import test_unit as T;  T.check_dice(dice)
  pytest:    pytest units/02-shape-and-metrics   (tests the copies you moved into src/xray3d;
                                                  functions not moved yet are skipped)
"""
from __future__ import annotations

import numpy as np

from xray3d import data, metrics, shape
from xray3d.checks import call, close, run, skip_if_missing


def _have_hero() -> bool:
    return data.ct_path(data.HERO).exists()


# ---------------------------------------------------------------- part A: the cube

def check_bbox_center_mm(fn):
    m = np.zeros((10, 10, 10), bool)
    m[2:5, 3:8, 0:1] = True          # indices 2..4, 3..7, 0..0  ->  index-space centre (3, 5, 0)

    def identity():
        close(call(fn, m, np.eye(4)), [3, 5, 0], "identity affine: the centre of indices 2..4, 3..7, 0..0")

    def scaled_and_shifted():
        a = np.diag([2.0, 2.0, 3.0, 1.0])
        a[:3, 3] = [10, -20, 5]
        close(call(fn, m, a), [16, -10, 5], "2 x 2 x 3 mm voxels, origin (10, -20, 5)")

    def flipped():
        a = np.diag([-1.5, -1.5, 1.5, 1.0])
        a[:3, 3] = [100, 50, 0]
        close(call(fn, m, a), [95.5, 42.5, 0], "an LPS-style affine (negative x and y steps)")

    def not_the_centroid():
        # a bounding box ignores how the voxels are spread inside it; a centroid doesn't
        m2 = np.zeros((10, 10, 10), bool)
        m2[0, 0, 0] = True
        m2[8, 0, 0] = True
        m2[7, 0, 0] = True
        close(call(fn, m2, np.eye(4)), [4, 0, 0], "voxels at x = 0, 7, 8: the box centre is 4 (the mean would be 5)")

    run("bbox_center_mm", identity, scaled_and_shifted, flipped, not_the_centroid)


def check_cube_grid(fn):
    def shape_and_corners():
        c = np.array([10.0, -20.0, 30.0])
        g = np.asarray(call(fn, c))
        assert g.shape == (64, 64, 64, 3), f"the default cube is 64^3 points of (x, y, z): expected (64, 64, 64, 3), got {g.shape}"
        close(g[0, 0, 0], c - 157.5, "first voxel centre = centre - 160 mm + half a 5 mm voxel")
        close(g[63, 63, 63], c + 157.5, "last voxel centre")
        close(g.reshape(-1, 3).mean(axis=0), c, "the mean of all voxel centres is the cube centre")

    def axes_in_order():
        g = np.asarray(call(fn, np.zeros(3)))
        close(g[1, 0, 0] - g[0, 0, 0], [5, 0, 0], "one step along array axis 0 moves 5 mm along x (R)")
        close(g[0, 1, 0] - g[0, 0, 0], [0, 5, 0], "one step along array axis 1 moves 5 mm along y (A)")
        close(g[0, 0, 1] - g[0, 0, 0], [0, 0, 5], "one step along array axis 2 moves 5 mm along z (S)")

    def tiny():
        g = np.asarray(call(fn, np.zeros(3), 10.0, 2))
        close(g[0, 0, 0], [-2.5, -2.5, -2.5], "a 10 mm cube of 2^3 voxels: centres at -2.5 and +2.5")
        close(g[1, 0, 1], [2.5, -2.5, 2.5], "voxel (1, 0, 1)")

    def box():
        # the same function makes the CT box for DiffDRR: 480 x 480 x 400 mm in 2 mm voxels
        g = np.asarray(call(fn, np.array([0.0, 0, 100]), np.array([480.0, 480, 400]), np.array([240, 240, 200])))
        assert g.shape == (240, 240, 200, 3), f"size and n can differ per axis: expected (240, 240, 200, 3), got {g.shape}"
        close(g[0, 0, 0], [-239, -239, -99], "first centre of the box")
        close(g[-1, -1, -1], [239, 239, 299], "last centre of the box")

    run("cube_grid", shape_and_corners, axes_in_order, tiny, box)


def _toy():
    # every voxel holds its own index as digits: vol[i, j, k] = 100 i + 10 j + k
    i, j, k = np.meshgrid(np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    return (100 * i + 10 * j + k).astype(float)


def check_resample(fn):
    vol = _toy()
    a = np.diag([2.0, 2.0, 2.0, 1.0])
    a[:3, 3] = [10, 20, 30]           # voxel (i, j, k) sits at (10 + 2i, 20 + 2j, 30 + 2k) mm

    def on_a_voxel():
        close(call(fn, vol, a, np.array([[12.0, 24, 36]]), 1), [123], "the world position of voxel (1, 2, 3)")

    def between_voxels():
        close(call(fn, vol, a, np.array([[13.0, 24, 36]]), 1), [173], "half-way between voxels (1,2,3) and (2,2,3), linear")
        close(call(fn, vol, a, np.array([[13.2, 24, 36]]), 0), [223], "order=0: the nearest voxel, (2, 2, 3)")

    def outside():
        close(call(fn, vol, a, np.array([[1000.0, 0, 0]]), 1, -1000.0), [-1000], "far outside the volume -> the fill value")

    def keeps_shape():
        pts = np.tile(np.array([12.0, 24, 36]), (2, 3, 1))       # (2, 3, 3): a 2 x 3 grid of points
        got = np.asarray(call(fn, vol, a, pts, 1))
        assert got.shape == (2, 3), f"points of shape (2, 3, 3) give values of shape (2, 3), got {got.shape}"

    def rotated_affine():
        # array axis 0 runs along world y, axis 1 along world -x: voxel (1, 2, 3) sits at (-2, 1, 3)
        r = np.array([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        close(call(fn, vol, r, np.array([[-2.0, 1, 3]]), 1), [123], "an affine that swaps and flips axes")

    run("resample", on_a_voxel, between_voxels, outside, keeps_shape, rotated_affine)


def check_occupancy(fn):
    m = np.zeros((40, 40, 40), bool)
    m[10:30, 10:30, 10:30] = True     # 1 mm voxels: the block's surface is half-way between voxel 9 and 10 (9.5 mm)

    def block():
        occ = np.asarray(call(fn, m, np.eye(4), np.array([19.5, 19.5, 19.5]), 40.0, 8))
        assert occ.shape == (8, 8, 8), f"n=8 gives an (8, 8, 8) cube, got {occ.shape}"
        assert occ.dtype == np.uint8, f"the dataset stores occupancy as uint8 (0/1), got {occ.dtype}"
        # cell centres at 2, 7, 12, 17, 22, 27, 32, 37 mm; inside [9.5, 29.5]: 12, 17, 22, 27 -> indices 2..5
        want = np.zeros((8, 8, 8), np.uint8)
        want[2:6, 2:6, 2:6] = 1
        assert np.array_equal(occ, want), f"expected a 4^3 block at indices 2..5, got {int(occ.sum())} voxels"

    def fractional():
        # the centre shifted to x = 22.2 mm: cell centres at x = 4.7, 9.7, ..., 34.7. At 9.7 the mask interpolates to
        # 0.7 (in), at 29.7 to 0.3 (out) -> x indices 1..4. A bool mask that isn't cast to float loses the 0.7.
        occ = np.asarray(call(fn, m, np.eye(4), np.array([22.2, 19.5, 19.5]), 40.0, 8))
        want = np.zeros((8, 8, 8), np.uint8)
        want[1:5, 2:6, 2:6] = 1
        assert np.array_equal(occ, want), (f"cube centre at x = 22.2: expected x indices 1..4, got {int(occ.sum())} voxels "
                                           "(values between 0 and 1 must survive interpolation: resample the mask as float)")

    def layout_does_not_matter():
        # the same patient stored as RAS and as LPS (x and y flipped in the array): the cube is in world mm, so equal
        ras = np.zeros((30, 30, 30), bool)
        ras[5:25, 8:15, 3:20] = True
        ras[20:25, 15:25, 3:8] = True       # an asymmetric bit, so a flip would show
        a_ras = np.eye(4)
        flip = np.diag([-1.0, -1, 1, 1])
        flip[:3, 3] = [29, 29, 0]           # index 29 - i in the LPS array is index i in the RAS one
        lps = ras[::-1, ::-1, :]
        a_lps = a_ras @ flip
        c = np.array([15.0, 15, 15])
        o1 = np.asarray(call(fn, ras, a_ras, c, 32.0, 16))
        o2 = np.asarray(call(fn, lps, a_lps, c, 32.0, 16))
        assert o1.sum() > 0, "the RAS version came out empty"
        assert np.array_equal(o1, o2), "RAS and LPS storage of the same mask gave different cubes: use the affine both ways"

    def real_pelvis():
        if not _have_hero():
            print("  (hero masks not on disk: real-pelvis check skipped)")
            return
        masks = data.load_mask(data.HERO, data.PELVIS)
        pelvis = masks["hip_left"] | masks["hip_right"] | masks["sacrum"]
        aff = data.load_nifti(data.mask_path(data.HERO, "sacrum")).affine
        centre = np.array([-20.832, 133.668, 136.55])      # s1390's pelvis bounding-box centre (mm), worked out once
        occ = np.asarray(call(fn, pelvis, aff, centre))
        want = pelvis.sum() * 1.5 ** 3 / 5.0 ** 3          # the same volume counted in 5 mm voxels
        assert abs(occ.sum() - want) < 0.05 * want, f"s1390: {int(occ.sum())} occupied voxels, its volume is worth {want:.0f}"
        o = np.argwhere(occ)
        mid = (o.min(0) + o.max(0)) / 2
        assert np.all(np.abs(mid - 31.5) <= 1.0), f"the pelvis should sit in the middle of the cube, its box centre is at index {mid}"

    run("occupancy", block, fractional, layout_does_not_matter, real_pelvis)


def check_mesh_from_occupancy(fn):
    def block():
        occ = np.zeros((4, 4, 4), np.uint8)
        occ[1:3, 1:3, 1:3] = 1           # centres at -2.5 and +2.5 mm; the surface is half-way to the empty ones
        v, f = (np.asarray(x) for x in call(fn, occ, 5.0, np.array([100.0, 0, 0])))
        assert v.ndim == 2 and v.shape[1] == 3, f"vertices should be (V, 3), got {v.shape}"
        assert f.ndim == 2 and f.shape[1] == 3, f"faces should be (F, 3) triangles, got {f.shape}"
        assert np.issubdtype(f.dtype, np.integer) and f.max() < len(v), "faces index into the vertex array"
        close(v.min(axis=0), [95, -5, -5], "the 2^3 block (10 mm wide) centred at (100, 0, 0): lowest vertex")
        close(v.max(axis=0), [105, 5, 5], "highest vertex")

    def corner_voxel_is_closed():
        # a voxel in the corner of the array: without care the surface is open where the array ends
        occ = np.zeros((4, 4, 4), np.uint8)
        occ[0, 0, 0] = 1
        v, f = (np.asarray(x) for x in call(fn, occ, 5.0, np.zeros(3)))
        close(v.mean(axis=0), [-7.5, -7.5, -7.5], "the mesh of voxel (0,0,0) is centred on that voxel's centre", atol=1e-3)
        close(np.abs(v - [-7.5, -7.5, -7.5]).max(), 2.5, "and reaches half a voxel (2.5 mm) from it", atol=1e-3)
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2), "the surface has holes (an edge used by only one triangle): pad the cube with empty voxels"

    def empty():
        v, f = (np.asarray(x) for x in call(fn, np.zeros((4, 4, 4), np.uint8), 5.0, np.zeros(3)))
        assert len(v) == 0 and len(f) == 0, "an empty cube has no surface: return empty arrays (don't crash)"

    run("mesh_from_occupancy", block, corner_voxel_is_closed, empty)


# ---------------------------------------------------------------- part B: metrics

def _cubes():
    a = np.zeros((10, 10, 10), np.uint8)
    a[2:6, 2:6, 2:6] = 1                # 64 voxels
    b = np.roll(a, 1, axis=0)           # the same cube one voxel along: overlap 3 x 4 x 4 = 48
    return a, b


def check_dice(fn):
    a, b = _cubes()

    def shifted():
        close(call(fn, a, b), 0.75, "two 4^3 cubes, one voxel apart: 2 * 48 / (64 + 64)")

    def same_and_disjoint():
        close(call(fn, a, a), 1.0, "a shape and itself")
        close(call(fn, a, np.roll(a, 5, axis=1)), 0.0, "no overlap")

    def both_empty():
        z = np.zeros_like(a)
        close(call(fn, z, z), 1.0, "nothing predicted where there's nothing: perfect (and no division by zero)")
        close(call(fn, a, z), 0.0, "empty prediction")

    run("dice", shifted, same_and_disjoint, both_empty)


def check_iou(fn):
    a, b = _cubes()

    def shifted():
        close(call(fn, a, b), 0.6, "48 / (64 + 64 - 48)")

    def edge_cases():
        z = np.zeros_like(a)
        close(call(fn, a, a), 1.0, "a shape and itself")
        close(call(fn, z, z), 1.0, "both empty")
        close(call(fn, a, z), 0.0, "empty prediction")

    run("iou", shifted, edge_cases)


def check_surface_voxels(fn):
    def cubes():
        for side, want in [(1, 1), (3, 26), (5, 98)]:
            m = np.zeros((9, 9, 9), bool)
            m[2:2 + side, 2:2 + side, 2:2 + side] = True
            s = np.asarray(call(fn, m))
            assert s.shape == m.shape, f"the surface is a mask of the same shape, got {s.shape}"
            assert int(s.sum()) == want, f"a {side}^3 cube has {want} surface voxels ({side}^3 minus the {max(side - 2, 0)}^3 inside), got {int(s.sum())}"
            assert not np.any(s & ~m), "surface voxels must be voxels of the mask"

    def array_border():
        s = np.asarray(call(fn, np.ones((4, 4, 4), np.uint8)))
        assert int(s.sum()) == 56, f"a full 4^3 array: outside the array counts as empty, so 64 - 8 = 56 are on the surface, got {int(s.sum())}"

    def six_neighbours():
        m = np.zeros((9, 9, 9), bool)
        m[2:5, 2:5, 2:5] = True
        m[2, 2, 2] = False                 # a corner gone: the centre still has all 6 face neighbours
        assert int(np.asarray(call(fn, m)).sum()) == 25, "6-neighbourhood: a 3^3 cube minus a corner has 25 surface voxels (the centre is inside)"

    def empty():
        assert int(np.asarray(call(fn, np.zeros((5, 5, 5), bool))).sum()) == 0, "an empty mask has no surface"

    run("surface_voxels", cubes, array_border, six_neighbours, empty)


def _dot(*ijk, shape=(8, 8, 8)):
    m = np.zeros(shape, bool)
    for p in ijk:
        m[p] = True
    return m


def check_surface_distances(fn):
    def one_voxel_each():
        a, b = _dot((1, 1, 1)), _dot((1, 4, 5))       # 0, 3, 4 voxels apart
        close(call(fn, a, b, 1.0), [5.0], "1 mm voxels: a 3-4-5 triangle")
        close(call(fn, a, b, 2.0), [10.0], "2 mm voxels")
        close(call(fn, a, b, (1.0, 2.0, 3.0)), [np.sqrt(6.0 ** 2 + 12.0 ** 2)], "voxels of 1 x 2 x 3 mm: 3 steps of 2 mm, 4 of 3 mm")

    def direction_matters():
        cube = np.zeros((8, 8, 8), bool)
        cube[0:3, 0:3, 0:3] = True                     # 26 surface voxels
        corner = _dot((0, 0, 0))
        d_ab = np.sort(np.asarray(call(fn, cube, corner, 1.0)))
        d_ba = np.asarray(call(fn, corner, cube, 1.0))
        assert d_ab.shape == (26,), f"one distance per surface voxel of the FIRST mask (26), got {d_ab.shape}"
        close(d_ab.max(), 2 * np.sqrt(3), "the far corner of the cube is 2*sqrt(3) from voxel (0,0,0)")
        close(d_ba, [0.0], "from the corner voxel to the cube's surface: it's on it")

    def itself():
        m = np.zeros((8, 8, 8), bool)
        m[2:5, 2:5, 2:5] = True
        close(call(fn, m, m, 5.0), np.zeros(26), "a surface to itself: 26 zeros")

    def to_nothing():
        d = np.asarray(call(fn, _dot((1, 1, 1)), np.zeros((8, 8, 8), bool), 5.0))
        assert d.shape == (1,) and np.isinf(d[0]), f"no surface to reach: the distance is infinite, got {d}"

    def to_the_surface_not_the_inside():
        b = np.zeros((8, 8, 8), bool)
        b[1:6, 1:6, 1:6] = True            # a 5^3 cube; voxel (3, 3, 3) is its centre, 2 voxels from its surface
        close(call(fn, _dot((3, 3, 3)), b, 1.0), [2.0], "a voxel deep inside b is 2 mm from b's SURFACE (not 0)")

    run("surface_distances", one_voxel_each, direction_matters, itself, to_the_surface_not_the_inside, to_nothing)


def _row(last):
    """19 isolated voxels along x, 3 apart (x = 0, 3, ..., 54), plus one at x = last (20 in all)."""
    m = np.zeros((64, 5, 5), bool)
    m[0:55:3, 2, 2] = True
    m[last, 2, 2] = True
    return m


# gt and pred differ in one voxel of 20: gt's is at x = 57, pred's at x = 59 (2 voxels = 10 mm apart).
# Distances, both directions: 38 zeros and two 10 mm.
ROW_GT, ROW_PRED = _row(57), _row(59)


def _both_empty_and_one_empty(fn, name, both, one, *extra):
    z = np.zeros((8, 8, 8), bool)
    close(call(fn, z, z, *extra), both, f"{name}: both empty")
    got = float(call(fn, _dot((1, 1, 1)), z, *extra))
    if np.isinf(one):
        assert np.isinf(got), f"{name}: an empty prediction should give inf (no surface to measure to), got {got}"
    else:
        close(got, one, f"{name}: empty prediction")


def check_hd95(fn):
    def identical():
        close(call(fn, ROW_GT, ROW_GT, 5.0), 0.0, "a shape and itself")

    def single_voxels():
        close(call(fn, _dot((1, 1, 1)), _dot((1, 1, 4)), 5.0), 15.0, "two single voxels 3 cells (15 mm) apart")

    def percentile():
        # np.percentile of [0]*19 + [10]: position 0.95 * 19 = 18.05, so 0 + 0.05 * 10 = 0.5
        close(call(fn, ROW_GT, ROW_PRED, 5.0), 0.5, "19 zeros and one 10 mm in each direction, np.percentile(d, 95)")

    def symmetric():
        a, b = _cubes()
        b2 = np.zeros_like(a)
        b2[2:8, 2:8, 2:8] = 1
        close(call(fn, a, b2, 5.0), call(fn, b2, a, 5.0), "hd95(gt, pred) == hd95(pred, gt): use both directions")

    def directed_not_pooled():
        # pred lacks gt's voxel at x = 57; it is 3 voxels (15 mm) from pred's x = 54. gt -> pred: 19 zeros and 15 mm,
        # np.percentile -> 0.05 * 15 = 0.75; pred -> gt: 19 zeros. Pooling the 39 would give 0.
        close(call(fn, ROW_GT, _row(54), 5.0), 0.75, "the larger of the two directed 95th percentiles")

    run("hd95", identical, single_voxels, percentile, symmetric, directed_not_pooled,
        lambda: _both_empty_and_one_empty(fn, "hd95", 0.0, np.inf, 5.0))


def check_assd(fn):
    def identical_and_single():
        close(call(fn, ROW_GT, ROW_GT, 5.0), 0.0, "a shape and itself")
        close(call(fn, _dot((1, 1, 1)), _dot((1, 1, 4)), 5.0), 15.0, "two single voxels 15 mm apart")

    def row():
        close(call(fn, ROW_GT, ROW_PRED, 5.0), 0.5, "(10 + 10) mm over 40 surface voxels")

    def pooled():
        # 26 distances one way (a 3^3 cube to its corner voxel), one zero the other way: the mean over all 27
        cube = np.zeros((8, 8, 8), bool)
        cube[0:3, 0:3, 0:3] = True
        ijk = np.argwhere(cube)
        ijk = ijk[~np.all(ijk == 1, axis=1)]          # the 26 surface voxels (not the centre one)
        want = np.linalg.norm(ijk, axis=1).sum() / 27 # their distances to (0,0,0), divided by 26 + 1
        close(call(fn, cube, _dot((0, 0, 0)), 1.0), want, "the average over ALL surface voxels of both shapes (27)")

    run("assd", identical_and_single, row, pooled, lambda: _both_empty_and_one_empty(fn, "assd", 0.0, np.inf, 5.0))


def check_nsd(fn):
    def row():
        close(call(fn, ROW_GT, ROW_PRED, 1.5, 5.0), 0.95, "38 of 40 surface voxels within 1.5 mm")
        close(call(fn, ROW_GT, ROW_PRED, 10.0, 5.0), 1.0, "all 40 within 10 mm (<= counts)")

    def one_voxel_off():
        a, b = _dot((1, 1, 1)), _dot((1, 1, 2))
        close(call(fn, a, b, 1.5, 5.0), 0.0, "one voxel (5 mm) apart, tolerance 1.5 mm: nothing counts")
        close(call(fn, a, b, 5.0, 5.0), 1.0, "the same, tolerance 5 mm: everything counts")

    def identical():
        close(call(fn, ROW_GT, ROW_GT, 1.5, 5.0), 1.0, "a shape and itself")

    def pooled():
        cube = np.zeros((8, 8, 8), bool)
        cube[0:3, 0:3, 0:3] = True
        # cube -> corner: 7 of 26 within 1.5 mm (itself, 3 at 1 mm, 3 at 1.41 mm); corner -> cube: 1 of 1 -> 8 / 27
        close(call(fn, cube, _dot((0, 0, 0)), 1.5, 1.0), 8 / 27, "pooled over all 27 surface voxels (not the mean of 7/26 and 1/1)")

    run("nsd", row, one_voxel_off, identical, pooled, lambda: _both_empty_and_one_empty(fn, "nsd", 1.0, 0.0, 1.5, 5.0))


def check_chamfer(fn):
    def hand_case():
        a = np.array([[0.0, 0, 0]])
        b = np.array([[3.0, 4, 0], [0, 0, 1]])
        # a -> b: 1;  b -> a: 5 and 1, mean 3;  chamfer = (1 + 3) / 2
        close(call(fn, a, b), 2.0, "a = {0}, b = {(3,4,0), (0,0,1)}")
        close(call(fn, b, a), 2.0, "symmetric")

    def shifted_cloud():
        rng = np.random.default_rng(0)
        a = rng.integers(0, 50, size=(200, 3)) * 10.0     # points on a 10 mm lattice
        a = np.unique(a, axis=0)
        close(call(fn, a, a), 0.0, "a cloud and itself")
        close(call(fn, a, a + [1.0, 0, 0]), 1.0, "the same cloud moved 1 mm: every nearest neighbour is 1 mm away")

    def empty():
        got = float(call(fn, np.zeros((0, 3)), np.array([[0.0, 0, 0]])))
        assert np.isinf(got), f"one cloud empty: inf, got {got}"
        close(call(fn, np.zeros((0, 3)), np.zeros((0, 3))), 0.0, "both clouds empty: 0.0")

    run("chamfer", hand_case, shifted_cloud, empty)


def _ce_case(side):
    """A 64^3 cube (5 mm voxels) with a few 'acetabulum' voxels around a femoral head of radius 25 mm.

    The head centre is a voxel centre: index (14, 32, 22) -> (-87.5, 2.5, -47.5) mm. For the right hip
    everything is mirrored in x. Lateral = away from the midline: -x on the left, +x on the right.
    """
    s = -1 if side == "left" else 1
    centre = np.array([87.5 * s, 2.5, -47.5])
    occ = np.zeros((64, 64, 64), np.uint8)

    def put(lateral_mm, up_mm, dy_mm=0.0):
        p = centre + [lateral_mm * s, dy_mm, up_mm]
        occ[tuple(np.round((p + 160) / 5 - 0.5).astype(int))] = 1

    put(0, 25)          # straight above the head: 0 deg
    put(15, 20)         # 25 mm away, 15 lateral and 20 up: atan(15/20) = 36.87 deg  <- the lateral edge
    put(-20, 15)        # medial: -53.13 deg (with lateral and medial swapped this would win)
    put(25, -10)        # lateral but BELOW the centre: not the roof
    put(35, 25)         # 43 mm away: outside the band around the head (54.5 deg, ignored)
    put(20, 15, 5.0)    # one coronal layer further back (53 deg, ignored)
    return occ, centre


def check_ce_angle(fn):
    def left():
        occ, c = _ce_case("left")
        close(call(fn, occ, c, 25.0, "left", 5.0), np.degrees(np.arctan2(15, 20)), "left hip: the most lateral roof voxel at 15 mm out, 20 mm up", atol=1e-4)

    def right():
        occ, c = _ce_case("right")
        close(call(fn, occ, c, 25.0, "right", 5.0), np.degrees(np.arctan2(15, 20)), "right hip (mirrored): same angle", atol=1e-4)

    def nothing_there():
        occ, c = _ce_case("left")
        got = float(call(fn, np.zeros_like(occ), c, 25.0, "left", 5.0))
        assert np.isnan(got), f"no roof voxels: the angle is undefined (nan), got {got}"

    run("ce_angle", left, right, nothing_there)


KEYS_ORDER = ["dice", "iou", "hd95", "assd", "nsd_1.5mm", "nsd_5mm", "chamfer"]
KEYS = set(KEYS_ORDER)


def check_evaluate(fn):
    gt = np.zeros((16, 16, 16), np.uint8)
    gt[4:10, 4:10, 4:10] = 1

    def perfect():
        m = call(fn, gt, gt, 5.0)
        assert set(m) == KEYS, f"keys should be {sorted(KEYS)}, got {sorted(m)}"
        for k, want in {"dice": 1, "iou": 1, "hd95": 0, "assd": 0, "nsd_1.5mm": 1, "nsd_5mm": 1, "chamfer": 0}.items():
            close(m[k], want, f"GT vs GT: {k} must be perfect")

    def worst():
        m = call(fn, gt, np.zeros_like(gt), 5.0)
        for k in ("dice", "iou", "nsd_1.5mm", "nsd_5mm"):
            close(m[k], 0.0, f"empty prediction: {k}")
        for k in ("hd95", "assd", "chamfer"):
            assert np.isinf(m[k]), f"empty prediction: {k} should be inf, got {m[k]}"

    def one_voxel_shift():
        m = call(fn, gt, np.roll(gt, 1, axis=0), 5.0)
        close(m["dice"], 5 / 6, "a 6^3 block moved one voxel: overlap 5/6")
        close(m["hd95"], 5.0, "moved one voxel = 5 mm")
        close(m["nsd_5mm"], 1.0, "everything is within 5 mm")
        assert m["nsd_1.5mm"] < 1.0, "at 1.5 mm the moved faces don't count"
        assert 0 < m["chamfer"] <= 5.0, f"the mesh moved 5 mm, so Chamfer is between 0 and 5, got {m['chamfer']}"

    run("evaluate", perfect, worst, one_voxel_shift)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_bbox_center_mm():
    check_bbox_center_mm(skip_if_missing(shape, "bbox_center_mm"))


def test_cube_grid():
    check_cube_grid(skip_if_missing(shape, "cube_grid"))


def test_resample():
    check_resample(skip_if_missing(shape, "resample"))


def test_occupancy():
    check_occupancy(skip_if_missing(shape, "occupancy"))


def test_mesh_from_occupancy():
    check_mesh_from_occupancy(skip_if_missing(shape, "mesh_from_occupancy"))


def test_dice():
    check_dice(skip_if_missing(metrics, "dice"))


def test_iou():
    check_iou(skip_if_missing(metrics, "iou"))


def test_surface_voxels():
    check_surface_voxels(skip_if_missing(metrics, "surface_voxels"))


def test_surface_distances():
    check_surface_distances(skip_if_missing(metrics, "surface_distances"))


def test_hd95():
    check_hd95(skip_if_missing(metrics, "hd95"))


def test_assd():
    check_assd(skip_if_missing(metrics, "assd"))


def test_nsd():
    check_nsd(skip_if_missing(metrics, "nsd"))


def test_chamfer():
    check_chamfer(skip_if_missing(metrics, "chamfer"))


def test_ce_angle():
    check_ce_angle(skip_if_missing(metrics, "ce_angle"))


def test_evaluate():
    check_evaluate(skip_if_missing(metrics, "evaluate"))


def test_dataset_geometry():
    """The processed dataset (if built here): YOUR project_pinhole + dataset.camera reproduce the pelvis silhouettes."""
    import pytest

    from xray3d import dataset, geometry, prep

    pp = skip_if_missing(geometry, "project_pinhole")
    if not dataset.exists():
        pytest.skip("processed dataset v1 not built here")
    for sid in dataset.subjects()[:3]:
        d = dataset.load_subject(sid)
        for v in dataset.VIEWS:
            iou_ = prep.silhouette_iou(d, v, pp)
            assert iou_ >= 0.9, f"{sid} {v}: silhouette IoU {iou_:.3f} < 0.9"
