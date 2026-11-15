"""Checks for unit 07 (simulated vs real X-rays: the gap, re-projection, uncertainty, error analysis).

Every check_* takes YOUR function and tests it on small cases whose answer was worked out by
hand (or follows from a property, like "a detector's gain can't change the line integral"). There is
no reference solution in here: the expected values are numbers, not code. No check touches DeepFluoro.

  notebook:  import test_unit as T;  T.check_log_transform(log_transform)
  pytest:    pytest units/07-sim-vs-real   (tests the copies you moved into src/xray3d/realxray.py;
                                            functions not moved yet are skipped)
"""
from __future__ import annotations

import math

import numpy as np

from xray3d import realxray
from xray3d.checks import call, close, run, skip_if_missing

NAN = float("nan")


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- part A: the gap

def check_log_transform(fn):
    dark, flat = 100.0, 1100.0

    def hand_worked():
        # raw = dark + (flat - dark) * exp(-p)  <=>  p = -ln((raw - dark) / (flat - dark))
        raw = np.array([100 + 1000 * np.exp(-2.0), 1100.0, 100 + 1000 * np.exp(-0.5)])
        close(call(fn, raw, dark, flat), [2.0, 0.0, 0.5], "dark 100, flat 1100", atol=1e-6)

    def keeps_shape():
        raw = np.full((2, 3), 600.0)
        got = np.asarray(call(fn, raw, dark, flat))
        assert got.shape == (2, 3), f"a (2, 3) image in should give (2, 3) out, got {got.shape}"
        close(got, np.full((2, 3), np.log(2.0)), "raw 600 = half way between dark and flat -> p = ln 2", atol=1e-6)

    def gain_does_not_matter():
        # a detector with 3x the gain reads 3x everything, dark and flat included: same photons, same p
        raw = np.array([150.0, 400.0, 900.0])
        close(call(fn, 3 * raw, 3 * dark, 3 * flat), call(fn, raw, dark, flat), "gain x3 (raw, dark and flat)", atol=1e-9)

    def wrong_flat_is_a_constant():
        # an error in the flat level only adds the same number to every pixel (standardise removes it);
        # an error in the dark level does NOT: that's why the dark level matters and the flat level hardly does
        raw = np.array([150.0, 400.0, 900.0])
        diff = np.asarray(call(fn, raw, dark, 2100.0)) - np.asarray(call(fn, raw, dark, flat))
        close(diff, np.full(3, np.log(2.0)), "flat 2100 instead of 1100: p shifts by ln(2000/1000) everywhere", atol=1e-9)

    def outside_the_range():
        got = np.asarray(call(fn, np.array([50.0, 100.0, 1500.0]), dark, flat), float)
        assert np.isfinite(got).all(), f"pixels at or below the dark level (or above flat) must not give inf/nan: {got}"
        assert got[0] > 10 and got[1] > 10, f"a pixel that got no photons is 'infinitely' attenuated: p should be huge (got {got[:2]})"
        assert abs(got[2]) < 1e-12, f"a pixel brighter than the flat field would give p < 0: clip it to 0 (got {got[2]})"

    run("log_transform", hand_worked, keeps_shape, gain_does_not_matter, wrong_flat_is_a_constant, outside_the_range)


def check_gap(fn):
    def three_specimens():
        sim = {"a": 0.8, "b": 0.7, "c": 0.9}
        real = {"a": 0.6, "b": 0.7, "c": 0.5, "d": 0.4}        # d has no simulated twin: it can't be paired
        g = call(fn, sim, real)
        for k in ("per_specimen", "n", "mean", "sd", "worse"):
            assert k in g, f"the result should have the key {k!r} (got {sorted(g)})"
        assert sorted(g["per_specimen"]) == ["a", "b", "c"], f"only specimens in BOTH dicts count, got {sorted(g['per_specimen'])}"
        close([g["per_specimen"][s] for s in "abc"], [0.2, 0.0, 0.4], "drop per specimen (sim - real)", atol=1e-9)
        close([g["n"], g["mean"], g["sd"], g["worse"]], [3, 0.2, 0.2, 2], "n, mean, sd (n - 1), number worse on real", atol=1e-9)

    def lower_is_better():
        # HD95 in mm: 10 mm on the DRR, 25 mm on the real frame is a DROP of 15 mm (positive = worse on real)
        g = call(fn, {"a": 10.0}, {"a": 25.0}, higher_is_better=False)
        close([g["per_specimen"]["a"], g["mean"], g["worse"]], [15.0, 15.0, 1], "HD95 10 -> 25 mm", atol=1e-9)
        assert _isnan(g["sd"]), f"with one specimen there is no standard deviation: return nan (got {g['sd']})"

    def nan_is_skipped():
        g = call(fn, {"a": 0.8, "b": NAN}, {"a": 0.7, "b": 0.5})
        assert g["n"] == 1 and sorted(g["per_specimen"]) == ["a"], "a nan metric (e.g. an undefined angle) can't be paired"
        close(g["mean"], 0.1, "the one finite pair", atol=1e-9)

    run("gap", three_specimens, lower_is_better, nan_is_skipped)


# ---------------------------------------------------------------- part B: checking a prediction

FAR, CELL = 1.0e6, 10.0     # unit 03's toy camera: parallel rays, 10 mm at the origin = 1 pixel


def toy_camera(view: str = "ap"):
    """A 4 x 4 detector looking at a 4 x 4 x 4 grid of 10 mm voxels (cube 40 mm), same axes as dataset.camera.
    "ap": voxel (i, j, k) -> pixel (row 3 - k, column 3 - i), whatever j is."""
    f = FAR / CELL
    K = np.array([[f, 0.0, 1.5], [0.0, f, 1.5], [0.0, 0.0, 1.0]])
    if view == "ap":
        R, source = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]]), np.array([0.0, FAR, 0.0])
    else:
        R, source = np.array([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]]), np.array([-FAR, 0.0, 0.0])
    return K, R, -R @ source


def check_reprojection_iou(fn):
    cam = toy_camera("ap")

    def one_voxel():
        occ = np.zeros((4, 4, 4), np.uint8)
        occ[1, 2, 3] = 1                                       # -> pixel (row 0, column 2)
        sil = np.zeros((4, 4), bool)
        sil[0, 2] = True
        close(call(fn, occ, sil, cam, cube_mm=40.0), 1.0, "voxel (1, 2, 3) vs pixel (0, 2)")
        sil[3, 3] = True                                       # a silhouette pixel nothing projects to
        close(call(fn, occ, sil, cam, cube_mm=40.0), 0.5, "the same, plus one extra silhouette pixel: 1 / 2")
        valid = np.ones((4, 4), bool)
        valid[3, 3] = False                                    # ... outside the field of view: it doesn't count
        close(call(fn, occ, sil, cam, valid=valid, cube_mm=40.0), 1.0, "the extra pixel outside `valid`")

    def a_ray_is_one_pixel():
        occ = np.zeros((4, 4, 4), np.uint8)
        occ[2, :, 1] = 1                                       # 4 voxels along the AP beam -> one pixel (2, 1)
        sil = np.zeros((4, 4), bool)
        sil[2, 1] = True
        close(call(fn, occ, sil, cam, cube_mm=40.0), 1.0, "a column of voxels along the beam")

    def empty():
        occ = np.zeros((4, 4, 4), np.uint8)
        close(call(fn, occ, np.zeros((4, 4), bool), cam, cube_mm=40.0), 1.0, "nothing vs nothing (by convention)")
        sil = np.zeros((4, 4), bool)
        sil[1, 1] = True
        close(call(fn, occ, sil, cam, cube_mm=40.0), 0.0, "an empty prediction vs a silhouette")

    def no_holes_at_real_size():
        # our real camera and cube: a solid 100 x 70 x 120 mm box. Its exact silhouette is the convex hull of its 8
        # projected corners. 5 mm voxels land ~1.7 px apart on the detector: one point per voxel leaves holes.
        from matplotlib.path import Path
        from scipy.spatial import ConvexHull

        from xray3d import dataset

        K, R, t = dataset.camera("ap")
        occ = np.zeros((64, 64, 64), np.uint8)
        occ[22:42, 25:39, 20:44] = 1
        lo, hi = np.array([22, 25, 20]) * 5.0 - 160, np.array([42, 39, 44]) * 5.0 - 160
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uvw = (corners @ R.T + t) @ K.T
        uv = uvw[:, :2] / uvw[:, 2:]
        hull = Path(uv[ConvexHull(uv).vertices])
        cc, rr = np.meshgrid(np.arange(128), np.arange(128))
        sil = hull.contains_points(np.stack([cc.ravel(), rr.ravel()], 1)).reshape(128, 128)
        got = float(call(fn, occ, sil, (K, R, t)))
        assert got > 0.9, (f"a solid box vs its exact silhouette gave IoU {got:.3f} (expected > 0.9): did the projected "
                           "points leave holes? project several points per voxel, not just its centre")

    run("reprojection_iou", one_voxel, a_ray_is_one_pixel, empty, no_holes_at_real_size)


def _toy_models():
    import torch
    from torch import nn

    class DropToy(nn.Module):
        """logit +10 where dropout keeps the 1, -10 where it drops it (p = 0.5): sigmoid ~ a coin flip per voxel."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.5)

        def forward(self, x):
            return 20.0 * (0.5 * self.drop(torch.ones_like(x))) - 10.0

    class NormToy(nn.Module):
        """A BatchNorm with fixed running statistics: in eval mode logits = x - 5, in training mode they'd change."""
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm3d(1, affine=False)
            self.bn.running_mean.fill_(5.0)
            self.bn.running_var.fill_(1.0)
            self.drop = nn.Dropout3d(0.0)

        def forward(self, x):
            return self.drop(self.bn(x))

    return DropToy(), NormToy()


def check_mc_dropout_predict(fn):
    import torch

    def coin_flips():
        torch.manual_seed(0)
        drop, _ = _toy_models()
        x = torch.zeros(1, 1, 4, 4, 4)
        mean, var = call(fn, drop, x, 400)
        assert tuple(mean.shape) == (1, 1, 4, 4, 4) and tuple(var.shape) == (1, 1, 4, 4, 4), \
            f"mean and variance should have the model output's shape (1, 1, 4, 4, 4), got {tuple(mean.shape)}, {tuple(var.shape)}"
        m, v = float(mean.mean()), float(var.mean())
        assert 0.45 < m < 0.55, (f"each voxel is ~0 or ~1 with probability 1/2, so the mean probability should be ~0.5, "
                                 f"got {m:.3f}: is dropout really ON during the passes? (and are these probabilities, not logits?)")
        assert 0.22 < v < 0.27, f"a coin flip between 0 and 1 has variance 1/4, got {v:.3f}"
        assert not mean.requires_grad, "no gradients are needed for prediction: wrap the passes in torch.no_grad()"

    def only_dropout_is_switched_on():
        torch.manual_seed(0)
        _, norm = _toy_models()
        x = torch.randn(2, 1, 3, 3, 3) * 2 + 5
        mean, var = call(fn, norm, x, 5)
        close(mean.numpy(), torch.sigmoid(x - 5).numpy(), "a BatchNorm model: mean = sigmoid(x - 5) (its eval statistics)", atol=1e-5)
        close(var.numpy(), np.zeros(tuple(x.shape)), "no dropout -> no variance", atol=1e-10)
        assert not norm.bn.training, "only the dropout layers may be in training mode: model.train() switches BatchNorm too"

    def left_in_eval_mode():
        drop, _ = _toy_models()
        call(fn, drop, torch.zeros(1, 1, 2, 2, 2), 3)
        assert not any(m.training for m in drop.modules()), \
            "after the passes put the model back in eval mode: the next ordinary prediction must be deterministic"

    run("mc_dropout_predict", coin_flips, only_dropout_is_switched_on, left_in_eval_mode)


def check_uncertainty_vs_error(fn):
    var = np.array([0.1, 0.1, 0.0, 0.0])
    prob = np.array([0.9, 0.2, 0.8, 0.1])     # predicted occupancy (p > 0.5): 1, 0, 1, 0
    gt = np.array([0, 0, 1, 0])               # -> only voxel 0 is wrong

    def hand_worked():
        got = call(fn, var, prob, gt)
        for k in ("r", "var_wrong", "var_right", "n"):
            assert k in got, f"the result should have the key {k!r} (got {sorted(got)})"
        close([got["r"], got["var_wrong"], got["var_right"], got["n"]], [1 / np.sqrt(3), 0.1, 0.1 / 3, 4],
              "r = corr(var, error) = 1/sqrt(3); mean var on wrong / right voxels; voxels used", atol=1e-9)

    def region():
        got = call(fn, var, prob, gt, region=np.array([1, 1, 0, 1], bool))
        close([got["r"], got["n"]], [0.5, 3], "only voxels 0, 1, 3", atol=1e-9)

    def three_d_and_perfect():
        rng = np.random.default_rng(0)
        g = rng.random((5, 5, 5)) > 0.5
        p = np.where(rng.random((5, 5, 5)) > 0.8, 1.0 - g, g * 1.0) * 0.8 + 0.1     # wrong on ~20 % of voxels
        err = ((p > 0.5) != g).astype(float)
        close(call(fn, err * 0.3, p, g)["r"], 1.0, "variance exactly where the errors are -> r = 1", atol=1e-9)

    def no_errors():
        got = call(fn, var, np.array([0.1, 0.2, 0.8, 0.1]), gt)
        assert _isnan(got["r"]) and _isnan(got["var_wrong"]), \
            f"with no wrong voxel there's no correlation and no 'mean variance when wrong': nan, not an error (got {got})"

    run("uncertainty_vs_error", hand_worked, region, three_d_and_perfect, no_errors)


# ---------------------------------------------------------------- part C: error analysis

def check_ce_error(fn):
    def fake_ce(occ, head, radius, side):
        """A stand-in for ce_angle: the number of voxels (x2 on the right). It also checks the arguments."""
        assert np.shape(head) == (3,), f"ce_angle takes the head centre (3,), got shape {np.shape(head)}"
        assert np.ndim(radius) == 0, "ce_angle takes the radius as one number"
        return float(np.sum(occ)) * (1 if side == "left" else 2)

    gt, pred = np.zeros(10, np.uint8), np.zeros(10, np.uint8)
    gt[:3], pred[:5] = 1, 1
    heads = np.array([[10.0, 0, 5, 20], [-10.0, 0, 5, 20]])

    def both_sides():
        got = call(fn, gt, pred, heads, ce_fn=fake_ce)
        close([got["left"], got["right"], got["mean"]], [2.0, 4.0, 3.0], "|5 - 3| and |10 - 6| degrees")

    def missing_head():
        h = heads.copy()
        h[1] = np.nan                                         # the right femur wasn't in the scan
        got = call(fn, gt, pred, h, ce_fn=fake_ce)
        assert _isnan(got["right"]), f"no right femoral head -> the right error is nan (got {got['right']})"
        close(got["mean"], 2.0, "the mean over the sides that exist")

    def undefined_angle():
        def no_roof(occ, head, radius, side):
            return NAN if np.sum(occ) == 5 else 1.0            # the prediction has no acetabular roof
        got = call(fn, gt, pred, heads, ce_fn=no_roof)
        assert _isnan(got["left"]) and _isnan(got["right"]) and _isnan(got["mean"]), \
            f"if the angle can't be measured on the prediction the error is unknown (nan), not 0: got {got}"

    run("ce_error", both_sides, missing_head, undefined_angle)


def check_error_summary(fn):
    rows = [{"subject": "a", "dice": 0.9, "ce_err": 1.0}, {"subject": "b", "dice": 0.8, "ce_err": 2.0},
            {"subject": "c", "dice": 0.7, "ce_err": 3.0}, {"subject": "d", "dice": 0.6, "ce_err": NAN}]

    def hand_worked():
        got = call(fn, rows, k=2)
        for k in ("spearman", "worst_dice", "worst_ce", "both"):
            assert k in got, f"the result should have the key {k!r} (got {sorted(got)})"
        close(got["spearman"], -1.0, "dice falls exactly as the angle error rises (a, b, c; d has no angle)")
        assert list(got["worst_dice"]) == ["d", "c"], f"lowest Dice first: expected ['d', 'c'], got {got['worst_dice']}"
        assert list(got["worst_ce"]) == ["c", "b"], f"largest angle error first (nan can't be ranked): expected ['c', 'b'], got {got['worst_ce']}"
        assert list(got["both"]) == ["c"], f"in both lists: expected ['c'], got {got['both']}"

    def ties():
        r = [{"subject": s, "dice": d, "ce_err": e} for s, d, e in [("a", 0.9, 1.0), ("b", 0.9, 2.0), ("c", 0.5, 5.0)]]
        close(call(fn, r, k=1)["spearman"], -np.sqrt(3) / 2, "tied Dice share their rank (2.5, 2.5, 1)", atol=1e-9)

    def rank_not_value():
        # one huge angle error: Pearson on the values would be dominated by it, Spearman isn't
        r = [{"subject": s, "dice": d, "ce_err": e} for s, d, e in [("a", 0.9, 1.0), ("b", 0.8, 2.0), ("c", 0.7, 90.0)]]
        close(call(fn, r, k=1)["spearman"], -1.0, "ranks only: (3, 2, 1) vs (1, 2, 3)", atol=1e-9)

    run("error_summary", hand_worked, ties, rank_not_value)


# ---------------------------------------------------------------- pytest: the same checks on src/xray3d/realxray.py

def test_log_transform():
    check_log_transform(skip_if_missing(realxray, "log_transform"))


def test_gap():
    check_gap(skip_if_missing(realxray, "gap"))


def test_reprojection_iou():
    check_reprojection_iou(skip_if_missing(realxray, "reprojection_iou"))


def test_mc_dropout_predict():
    check_mc_dropout_predict(skip_if_missing(realxray, "mc_dropout_predict"))


def test_uncertainty_vs_error():
    check_uncertainty_vs_error(skip_if_missing(realxray, "uncertainty_vs_error"))


def test_ce_error():
    check_ce_error(skip_if_missing(realxray, "ce_error"))


def test_error_summary():
    check_error_summary(skip_if_missing(realxray, "error_summary"))
