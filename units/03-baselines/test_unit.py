"""Checks for unit 03 (baselines: the visual hull and nearest-neighbour retrieval).

Every check_* takes YOUR function and tests it on small cases whose answer was worked out by
hand (or follows from a property, like "a voxel can only survive if both views see it"). There is no
reference solution in here: the expected values are numbers, not code.

  notebook:  import test_unit as T;  T.check_visual_hull(visual_hull)
  pytest:    pytest units/03-baselines   (tests the copies you moved into src/xray3d;
                                          functions not moved yet are skipped)

The toy camera used below
-------------------------
A tiny 4 x 4 detector and a 4 x 4 x 4 grid of 10 mm voxels centred on the origin (centres at -15, -5, 5,
15 mm). The source is 1,000 km away, so the rays are parallel for all practical purposes, and the focal
length is chosen so that 10 mm at the origin = 1 pixel. The axes are the dataset's (dataset.camera):

  "ap"  : columns run patient's Right -> Left (-x), rows head -> feet (-z), beam along -y
          voxel (i, j, k)  ->  pixel (row 3 - k, column 3 - i), whatever j is
  "lat" : columns run Anterior -> Posterior (-y), rows head -> feet (-z), beam along +x
          voxel (i, j, k)  ->  pixel (row 3 - k, column 3 - j), whatever i is
"""
from __future__ import annotations

import numpy as np

from xray3d import baselines
from xray3d.checks import call, close, run, skip_if_missing

FAR = 1.0e6                  # mm from the source to the origin: parallel rays, in practice
CELL = 10.0                  # toy voxel size (mm), and the size of one pixel at the origin


def toy_camera(view: str):
    """K, R, t of the toy camera (see the module docstring). Same conventions as dataset.camera."""
    f = FAR / CELL
    K = np.array([[f, 0.0, 1.5], [0.0, f, 1.5], [0.0, 0.0, 1.0]])       # principal point: middle of 4 pixels
    if view == "ap":
        R = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
        source = np.array([0.0, FAR, 0.0])
    else:
        R = np.array([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]])
        source = np.array([-FAR, 0.0, 0.0])
    return K, R, -R @ source


def toy_grid():
    """(4, 4, 4, 3) voxel centres of the toy grid (mm)."""
    ax = (np.arange(4) - 1.5) * CELL
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)


TOY = {v: toy_camera(v) for v in ("ap", "lat")}


# ---------------------------------------------------------------- part A: the visual hull

def check_pixel_index(fn):
    def toy_voxels():
        row, col, inside = (np.asarray(x) for x in call(fn, toy_grid(), TOY["ap"], 4))
        assert row.shape == (4, 4, 4) and col.shape == (4, 4, 4) and inside.shape == (4, 4, 4), \
            f"one (row, col, inside) per point: (4, 4, 4) each for a (4, 4, 4, 3) grid, got {row.shape}, {col.shape}, {inside.shape}"
        assert np.issubdtype(row.dtype, np.integer) and np.issubdtype(col.dtype, np.integer), \
            "row and col are pixel INDICES: integers (round, don't truncate)"
        assert inside.all(), "every toy voxel lands on the 4 x 4 toy detector"
        i, j, k = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        assert np.array_equal(col, 3 - i), "AP: the column comes from x (Right -> Left), so voxel i lands in column 3 - i (round to the nearest pixel: int() turns 2.9999999 into 2)"
        assert np.array_equal(row, 3 - k), "AP: the row comes from z (head -> feet), so voxel k lands in row 3 - k"
        row, col, _ = (np.asarray(x) for x in call(fn, toy_grid(), TOY["lat"], 4))
        assert np.array_equal(col, 3 - j), "LAT: voxel j (Anterior) lands in column 3 - j"

    def rounding_and_the_edges():
        # x = 19 mm -> u = -0.4 (rounds to column 0: still on the detector); x = 21 -> u = -0.6 (column -1: off it)
        # x = -19 -> u = 3.4 (column 3); x = -21 -> u = 3.6 (column 4 of a 4-pixel detector: off it)
        pts = np.array([[19.0, 0, 5], [21.0, 0, 5], [-19.0, 0, 5], [-21.0, 0, 5], [0.0, 0, 1000.0]])
        row, col, inside = (np.asarray(x) for x in call(fn, pts, TOY["ap"], 4))
        assert inside.tolist() == [True, False, True, False, False], \
            f"pixel centres are at 0, 1, 2, 3, so a pixel covers centre +- 0.5; expected inside = [T, F, T, F, F], got {inside.tolist()}"
        assert col[0] == 0 and col[2] == 3, f"u = -0.4 rounds to column 0 and u = 3.4 to column 3, got {col[0]} and {col[2]}"
        assert row[0] == 1, f"z = 5 mm is pixel row 1 (rows run head -> feet), got {row[0]}"

    def real_camera():
        # the dataset's AP camera, worked out with the pinhole formula: u = c + f * x_cam / z_cam
        from xray3d import dataset

        g = dataset.geometry()
        f, c, sid = g["sdd_mm"] / g["pixel_mm"], (g["detector_px"] - 1) / 2, g["source_mm"]["ap"][1]
        pts = np.array([[101.0, 100, -48], [101.0, -100, -48]])          # the same x, z: near and far from the source
        want_col = np.round(c + f * (-pts[:, 0]) / (sid - pts[:, 1])).astype(int)
        want_row = np.round(c + f * (-pts[:, 2]) / (sid - pts[:, 1])).astype(int)
        row, col, inside = (np.asarray(x) for x in call(fn, pts, dataset.camera("ap"), g["detector_px"]))
        assert inside.all(), "both points are well inside the AP image"
        close(col, want_col, "AP columns of a point near the source and the same point far from it (magnification)")
        close(row, want_row, "AP rows")
        assert abs(col[0] - c) > abs(col[1] - c), "the point nearer the source must land further from the image centre"

    run("pixel_index", toy_voxels, rounding_and_the_edges, real_camera)


def check_backproject(fn):
    img = np.arange(16, dtype=float).reshape(4, 4) + 1.0        # img[r, c] = 4 r + c + 1: every pixel different

    def smear_along_the_ray():
        out = np.asarray(call(fn, img, TOY["ap"], toy_grid()))
        assert out.shape == (4, 4, 4), f"one value per point: (4, 4, 4), got {out.shape}"
        i, j, k = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        close(out, img[3 - k, 3 - i], "AP: voxel (i, j, k) gets pixel (3 - k, 3 - i), the same for every j (along the beam)")
        out = np.asarray(call(fn, img, TOY["lat"], toy_grid()))
        close(out, img[3 - k, 3 - j], "LAT: voxel (i, j, k) gets pixel (3 - k, 3 - j), the same for every i")

    def off_the_detector():
        pts = np.array([[21.0, 0, 5], [0.0, 0, 1000.0], [5.0, 0, 5], [-21.0, 0, 5]])
        close(call(fn, img, TOY["ap"], pts), [0.0, 0.0, img[1, 1], 0.0],
              "a point that lands off the detector gets 0 (no ray of this image passes through it); (5, 0, 5) gets pixel (1, 1); "
              "(-21, 0, 5) lands in column 4 of this 4-pixel image: off the detector (use the image's own size, not 128)")

    def default_points_are_the_cube():
        from xray3d import dataset

        g = dataset.geometry()
        n = g["detector_px"]
        out = np.asarray(call(fn, np.ones((n, n)), dataset.camera("ap")))
        assert out.shape == (g["cube_n"],) * 3, f"without points_mm: every voxel centre of the cube, {(g['cube_n'],) * 3}, got {out.shape}"
        assert out[32, :, 32].min() == 1.0, "the middle of the cube is seen by the AP detector"

    run("backproject", smear_along_the_ray, off_the_detector, default_points_are_the_cube)


def _sil(*pixels):
    s = np.zeros((4, 4), bool)
    for p in pixels:
        s[p] = True
    return s


def check_visual_hull(fn):
    full = np.ones((4, 4), bool)

    def a_slab():
        h = np.asarray(call(fn, {"ap": _sil(*[(r, 0) for r in range(4)]), "lat": full}, TOY, toy_grid()))
        assert h.shape == (4, 4, 4), f"one value per voxel: (4, 4, 4), got {h.shape}"
        want = np.zeros((4, 4, 4), bool)
        want[3] = True                       # AP column 0 = the most Right voxels, i = 3, at every j and k
        assert np.array_equal(h.astype(bool), want), \
            f"AP sees only column 0 (x = +15 mm), LAT sees everything: the hull is the slab i = 3 (16 voxels), got {int(h.astype(bool).sum())}"

    def one_voxel():
        h = np.asarray(call(fn, {"ap": _sil((0, 3)), "lat": _sil((0, 1))}, TOY, toy_grid())).astype(bool)
        assert int(h.sum()) == 1 and h[0, 2, 3], \
            "AP pixel (0, 3) is the ray i = 0, k = 3; LAT pixel (0, 1) is j = 2, k = 3: only voxel (0, 2, 3) is on both rays"

    def views_must_agree_on_height():
        h = np.asarray(call(fn, {"ap": _sil((0, 3)), "lat": _sil((1, 1))}, TOY, toy_grid())).astype(bool)
        assert not h.any(), "AP sees something in row 0 (top), LAT only in row 1: no voxel is on both rays, the hull is empty"

    def a_hidden_hole_is_filled():
        # a 4^3 block with a 2 x 2 tunnel along z through its middle: every ray AP or LAT still crosses bone
        shape = np.ones((4, 4, 4), bool)
        shape[1:3, 1:3, :] = False
        sil_ap, sil_lat = full, full         # its shadows: the tunnel is invisible from the front AND the side
        h = np.asarray(call(fn, {"ap": sil_ap, "lat": sil_lat}, TOY, toy_grid())).astype(bool)
        assert h.all() and int(shape.sum()) == 48, \
            "the tunnel (a concavity no AP or LAT ray can see into) stays filled: the hull is all 64 voxels for a 48-voxel shape"

    def only_the_views_given():
        h = np.asarray(call(fn, {"ap": _sil((0, 3))}, TOY, toy_grid())).astype(bool)
        assert int(h.sum()) == 4 and h[0, :, 3].all(), "one view: the hull is the whole ray (i = 0, k = 3, every j: 4 voxels)"

    def dtype():
        h = np.asarray(call(fn, {"ap": full, "lat": full}, TOY, toy_grid()))
        assert h.dtype == np.uint8, f"return uint8 0/1, like the dataset's occupancy, got {h.dtype}"

    run("visual_hull", a_slab, one_voxel, views_must_agree_on_height, a_hidden_hole_is_filled, only_the_views_given, dtype)


# ---------------------------------------------------------------- part B: nearest neighbour

def _halves():
    """An 8 x 8 'AP' with a dark left half (0) and a bright right half (2); the 'LAT' is its transpose."""
    ap = np.zeros((8, 8))
    ap[:, 4:] = 2.0
    return ap, ap.T.copy()


def check_nn_features(fn):
    def hand_case():
        ap, lat = _halves()
        # downsampled to 2 x 2: [[0, 2], [0, 2]] and [[0, 0], [2, 2]]; mean 1, std 1 -> -1 and +1
        close(call(fn, ap, lat, 2), [-1, 1, -1, 1, -1, -1, 1, 1],
              "8 x 8 -> 2 x 2 block means, standardised, flattened row by row, AP first then LAT")

    def each_image_standardised():
        rng = np.random.default_rng(0)
        ap, lat = rng.random((16, 16)), 50 + 10 * rng.random((16, 16))
        f = np.asarray(call(fn, ap, lat, 4), float)
        assert f.shape == (32,), f"two 4 x 4 images -> 32 numbers, got {f.shape}"
        for part, name in [(f[:16], "AP"), (f[16:], "LAT")]:
            close([part.mean(), part.std()], [0.0, 1.0], f"the {name} part has mean 0 and std 1 (standardise AFTER downsampling)", atol=1e-9)

    def brightness_does_not_matter():
        rng = np.random.default_rng(1)
        ap, lat = rng.random((16, 16)), rng.random((16, 16))
        close(call(fn, 3 * ap + 7, 0.5 * lat - 2, 4), call(fn, ap, lat, 4),
              "a brighter / higher-contrast copy of the same images gives the same features")

    def flat_image():
        f = np.asarray(call(fn, np.full((8, 8), 5.0), None, 2), float)
        assert f.shape == (4,) and np.all(np.isfinite(f)), f"a constant image: std 0, no NaN please (zeros), got {f}"
        close(f, np.zeros(4), "a constant image has no structure: all zeros")

    def batches_and_one_view():
        ap, lat = np.zeros((5, 8, 8)), np.zeros((5, 8, 8))
        ap[:, :, 4:] = 2.0
        f2 = np.asarray(call(fn, ap, lat, 2))
        f1 = np.asarray(call(fn, ap, None, 2))
        assert f2.shape == (5, 8), f"5 subjects, two 2 x 2 images each -> (5, 8), got {f2.shape}"
        assert f1.shape == (5, 4), f"lat=None: AP only -> (5, 4), got {f1.shape}"
        close(f1[3], [-1, 1, -1, 1], "every row of a batch is one subject's features")

    def the_dataset_size():
        f = np.asarray(call(fn, np.random.default_rng(2).random((128, 128)), np.ones((128, 128)), 32))
        assert f.shape == (2048,), f"two 128 x 128 images at size 32 -> 2 * 32 * 32 = 2048, got {f.shape}"

    run("nn_features", hand_case, each_image_standardised, brightness_does_not_matter, flat_image,
        batches_and_one_view, the_dataset_size)


def check_nearest_neighbour(fn):
    train = np.array([[0.0, 0], [3, 4], [10, 0]])

    def hand_case():
        q = np.array([[1.0, 1], [3, 3], [9, 1], [3, 4]])
        got = np.asarray(call(fn, train, q))
        assert got.shape == (4,), f"one index per query: (4,), got {got.shape}"
        assert np.issubdtype(got.dtype, np.integer), f"indices are integers, got {got.dtype}"
        assert got.tolist() == [0, 1, 2, 1], f"nearest training rows: (1,1) -> 0, (3,3) -> 1, (9,1) -> 2, (3,4) is row 1 itself; got {got.tolist()}"

    def euclidean_not_manhattan():
        # (5, 0): L2 distance to (0, 0) is 5, to (3, 4) is sqrt(4 + 16) = 4.47 -> row 1
        #         (L1 would say 5 vs 6 -> row 0)
        got = np.asarray(call(fn, train[:2], np.array([[5.0, 0]])))
        assert got.tolist() == [1], f"(5, 0) is closer to (3, 4) than to (0, 0) in straight-line (L2) distance, got {got.tolist()}"

    run("nearest_neighbour", hand_case, euclidean_not_manhattan)


def check_oracle_neighbour(fn):
    gt = np.zeros((4, 4, 4), np.uint8)
    gt[0:2, 0:2, 0:2] = 1                                     # 8 voxels
    t = np.zeros((4, 4, 4, 4), np.uint8)
    # t[0] empty:                              Dice 0
    t[1, 1:3, 0:2, 0:2] = 1                    # moved one voxel: overlap 4, Dice 8 / 16 = 0.5
    t[2] = 1                                   # everything (contains gt): Dice 16 / 72 = 0.22
    t[3, 0:2, 0:2, 0:1] = 1                    # half of gt: Dice 8 / 12 = 0.67  <- the best

    def hand_case():
        got = call(fn, t, gt)
        assert int(got) == 3, f"Dice with gt: 0, 0.5, 0.22, 0.67 -> index 3 (a shape that CONTAINS gt is not the best match), got {got}"

    def itself():
        t2 = t.copy()
        t2[0] = gt
        assert int(call(fn, t2, gt)) == 0, "if gt itself is in the training set, the oracle finds it"

    run("oracle_neighbour", hand_case, itself)


def check_mean_shape(fn):
    a = np.zeros((3, 2, 2, 2), np.uint8)
    a[0, 0] = 1                     # shape 0: the half i = 0
    a[1, :, 0] = 1                  # shape 1: the half j = 0
    a[2, 0, 0] = 1                  # shape 2: the quarter i = 0, j = 0

    def majority():
        m = np.asarray(call(fn, a))
        assert m.shape == (2, 2, 2), f"one cube out: (2, 2, 2), got {m.shape}"
        assert m.dtype == np.uint8, f"uint8 0/1 like an occupancy, got {m.dtype}"
        want = np.zeros((2, 2, 2), np.uint8)
        want[0, 0] = 1              # (0, 0, *): 3 votes of 3 -> in;  (0, 1, *) and (1, 0, *): 1 of 3 -> out
        assert np.array_equal(m, want), f"voxels inside at least half of the shapes: only i = 0, j = 0 (3 votes of 3), got {np.argwhere(m).tolist()}"

    def ties_count_in():
        m = np.asarray(call(fn, a[:2]))
        # 2 shapes: (0, 0, *) has 2 votes, (0, 1, *) and (1, 0, *) one each (= 0.5) -> in;  (1, 1, *) none
        want = np.ones((2, 2, 2), np.uint8)
        want[1, 1] = 0
        assert np.array_equal(m, want), "a mean of exactly 0.5 counts as inside (>= level, like occupancy in unit 02)"

    def level():
        m = np.asarray(call(fn, a, 0.3))
        assert int(m.sum()) == 6, f"level 0.3: every voxel with at least 1 vote of 3 (6 voxels), got {int(m.sum())}"

    run("mean_shape", majority, ties_count_in, level)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_pixel_index():
    check_pixel_index(skip_if_missing(baselines, "pixel_index"))


def test_backproject():
    check_backproject(skip_if_missing(baselines, "backproject"))


def test_visual_hull():
    check_visual_hull(skip_if_missing(baselines, "visual_hull"))


def test_nn_features():
    check_nn_features(skip_if_missing(baselines, "nn_features"))


def test_nearest_neighbour():
    check_nearest_neighbour(skip_if_missing(baselines, "nearest_neighbour"))


def test_oracle_neighbour():
    check_oracle_neighbour(skip_if_missing(baselines, "oracle_neighbour"))


def test_mean_shape():
    check_mean_shape(skip_if_missing(baselines, "mean_shape"))
