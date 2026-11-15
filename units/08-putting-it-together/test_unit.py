"""Checks for unit 08 (putting it together: two X-rays -> a mesh, and the results story).

Every check_* takes YOUR function and tests it on cases whose answer was worked out by hand. The networks
used here are FAKE: tiny modules whose output is fixed in advance (logit 0 -> probability 0.5, ln 3 -> 0.75,
a box of +10 logits -> a box of bone), so the expected answer is known without any trained model. There is
no reference solution in here.

  notebook:  import test_unit as T;  T.check_prepare_inputs(prepare_inputs)
  pytest:    pytest units/08-putting-it-together   (tests the copies you moved into src/xray3d/pipeline.py;
                                                    anything not moved yet is skipped)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from xray3d import pipeline
from xray3d.checks import call, close, run, skip_if_missing

LN3 = math.log(3.0)
LN15 = math.log(1.5)          # sigmoid(ln 1.5) = 1.5 / 2.5 = 0.6


def _img(seed, scale=1.0, offset=0.0):
    return (offset + scale * np.random.default_rng(seed).normal(size=(128, 128))).astype(np.float32)


def _std(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean()) / a.std()


# ---------------------------------------------------------------- part A: two X-rays -> a mesh

def check_prepare_inputs(fn):
    # AP: a bright, high-contrast image (like a raw line integral); LAT: a dim one. Different on purpose.
    ap, lat = _img(0, scale=50.0, offset=1000.0), _img(1, scale=0.2, offset=3.0)

    def shape_and_type():
        x = call(fn, ap, lat)
        assert isinstance(x, torch.Tensor), f"return a torch tensor (the network takes tensors), got {type(x).__name__}"
        assert tuple(x.shape) == (1, 2, 128, 128), (f"shape {tuple(x.shape)}: the network takes (B, V, 128, 128). "
                                                    "One subject is a batch of 1 (B = 1), two views (V = 2)")
        assert x.dtype == torch.float32, f"dtype {x.dtype}: the network's weights are float32"

    def per_image_standardised():
        x = call(fn, ap, lat).numpy()
        close(x[0, 0], _std(ap), "channel 0 must be the AP image, standardised on its own (mean 0, std 1)", atol=1e-3)
        close(x[0, 1], _std(lat), "channel 1 must be the LAT image, standardised on its own. (Standardising both "
              "together would leave the bright AP positive and the dim LAT negative)", atol=1e-3)

    def exposure_does_not_matter():
        a = call(fn, ap, lat).numpy()
        b = call(fn, 7.0 * ap - 300.0, lat).numpy()
        close(b, a, "a brighter / more contrasted AP must give the same input (that's what standardising is for)", atol=1e-3)

    def one_view():
        x = call(fn, ap)
        assert tuple(x.shape) == (1, 1, 128, 128), f"no LAT given: a 1-view input (1, 1, 128, 128), got {tuple(x.shape)}"
        x = call(fn, ap, lat, n_views=1)
        assert tuple(x.shape) == (1, 1, 128, 128), "n_views=1 with both images (a 1-view model on a pair): use only the AP"
        close(x.numpy()[0, 0], _std(ap), "n_views=1: channel 0 is the AP", atol=1e-3)

    def tensors_too():
        x = call(fn, torch.from_numpy(ap), torch.from_numpy(lat))
        close(x.numpy()[0, 1], _std(lat), "torch tensors as input should work too", atol=1e-3)

    def leaves_inputs_alone():
        a0, l0 = ap.copy(), lat.copy()
        call(fn, ap, lat)
        assert np.array_equal(ap, a0) and np.array_equal(lat, l0), "don't change the caller's images in place"

    def refuses_wrong_inputs():
        for args, kw, why in [((ap,), {"n_views": 2}, "n_views=2 but no LAT image"),
                              ((ap[:64, :64], lat[:64, :64]), {}, "64 x 64 images (the network was trained on 128 x 128)"),
                              ((np.stack([ap, ap]), lat), {}, "a 3D array as the AP image")]:
            try:
                call(fn, *args, **kw)
            except ValueError:
                continue
            raise AssertionError(f"{why}: raise a ValueError that says what's wrong, instead of passing it on to "
                                 "the network (which would fail later with a confusing message, or not at all)")

    run("prepare_inputs", shape_and_type, per_image_standardised, exposure_does_not_matter, one_view, tensors_too,
        leaves_inputs_alone, refuses_wrong_inputs)


class _FixedLogits(nn.Module):
    """A fake network: whatever it is given, it returns the same logits (one copy per sample of the batch)."""

    def __init__(self, logits):
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))        # a parameter, so the model has a device and a training flag
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.calls = []

    def forward(self, x):
        self.calls.append({"training": self.training, "grad": torch.is_grad_enabled(), "x": x.detach().clone()})
        return self.w * self.logits.expand(x.shape[0], *self.logits.shape)


# 8 voxels, logits chosen so the probabilities are round numbers
_LOGITS = torch.tensor([0.0, LN3, -LN3, math.log(9.0), -30.0, 30.0, LN15, -LN15]).reshape(1, 2, 2, 2)
_PROBS = np.array([0.5, 0.75, 0.25, 0.9, 0.0, 1.0, 0.6, 0.4]).reshape(2, 2, 2)


def check_predict_occupancy(fn):
    x = torch.zeros(2, 1, 4, 4)            # a batch of 2 (the fake network ignores the images)

    def probabilities():
        prob, occ = call(fn, _FixedLogits(_LOGITS), x)
        assert isinstance(prob, np.ndarray) and isinstance(occ, np.ndarray), \
            "return NumPy arrays (prob, occ): the metrics and the mesh work on NumPy (.cpu().numpy())"
        assert prob.shape == (2, 2, 2, 2), (f"prob has shape {prob.shape}: (B, D, H, W) = (2, 2, 2, 2), i.e. drop the "
                                            "channel axis of the (B, 1, D, H, W) logits")
        close(prob[0], _PROBS, "probabilities = sigmoid(logits): 0 -> 0.5, ln 3 -> 0.75, -ln 3 -> 0.25, ln 9 -> 0.9", atol=1e-5)

    def threshold():
        _, occ = call(fn, _FixedLogits(_LOGITS), x)
        assert occ.dtype == np.uint8, f"occ dtype {occ.dtype}: uint8 0/1, like the dataset's occ"
        close(occ[0], (_PROBS > 0.5).astype(float), "occupancy = prob > 0.5 (strictly: p = 0.5 exactly is NOT bone)")
        _, occ = call(fn, _FixedLogits(_LOGITS), x, threshold=0.7)
        close(occ[1], (_PROBS > 0.7).astype(float), "threshold=0.7: only 0.75, 0.9 and 1.0 survive")

    def eval_and_no_grad():
        model = _FixedLogits(_LOGITS).train()
        prob, _ = call(fn, model, x)
        assert model.calls, "predict_occupancy never called the model"
        assert not model.calls[-1]["training"], "call model.eval() first (dropout must be off when predicting)"
        assert not model.calls[-1]["grad"], "wrap the forward pass in torch.no_grad(): no training here"

    run("predict_occupancy", probabilities, threshold, eval_and_no_grad)


def _box_logits(lo, hi, n=64):
    """(1, n, n, n) logits: +10 inside the box lo <= (i, j, k) < hi, -10 outside."""
    v = torch.full((1, n, n, n), -10.0)
    v[0, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 10.0
    return v


def check_predict_from_xrays(fn):
    ap, lat = _img(2, 30.0, 500.0), _img(3, 2.0, 10.0)
    # a 4 x 4 x 4 box of bone at i = 40..43, j = 20..23, k = 30..33. Voxel i's centre is (i + 0.5) * 5 - 160 mm,
    # so the box's voxel centres run from x = 42.5 to 57.5 mm; the surface (level 0.5) is half-way to the empty
    # neighbours: x in [40, 60], y in [-60, -40], z in [-10, 10] mm.
    lo, hi = (40, 20, 30), (44, 24, 34)
    box = ([40.0, -60.0, -10.0], [60.0, -40.0, 10.0])

    def the_dict():
        out = call(fn, _FixedLogits(_box_logits(lo, hi)), ap, lat)
        assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
        missing = {"occ", "prob", "verts", "faces"} - set(out)
        assert not missing, f"the dict is missing {sorted(missing)}"
        assert np.shape(out["occ"]) == (64, 64, 64), f"occ: one subject's (64, 64, 64) cube (no batch axis), got {np.shape(out['occ'])}"
        assert np.shape(out["prob"]) == (64, 64, 64), f"prob: (64, 64, 64), got {np.shape(out['prob'])}"
        assert int(np.sum(out["occ"])) == 64, f"the fake network predicts a 4 x 4 x 4 box (64 voxels); occ has {int(np.sum(out['occ']))}"

    def mesh_in_mm():
        out = call(fn, _FixedLogits(_box_logits(lo, hi)), ap, lat)
        v, f = np.asarray(out["verts"], float), np.asarray(out["faces"])
        assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 0, f"verts: (N, 3) points, got shape {v.shape}"
        assert f.ndim == 2 and f.shape[1] == 3 and len(f) > 0, f"faces: (M, 3) vertex indices, got shape {f.shape}"
        close(v.min(0), box[0], "the mesh's lowest corner, in mm in the cube frame (origin at the cube centre). "
              "Voxel indices instead of mm would give [40, 20, 30]; forgetting the -160 mm would give [200, 100, 150]", atol=0.6)
        close(v.max(0), box[1], "the mesh's highest corner (mm, cube frame)", atol=0.6)

    def cell_size():
        out = call(fn, _FixedLogits(_box_logits(lo, hi)), ap, lat, cell_mm=2.5)
        v = np.asarray(out["verts"], float)
        close(v.min(0), [20.0, -30.0, -5.0], "cell_mm=2.5 (a 160 mm cube): every coordinate halves", atol=0.4)

    def inputs_reach_the_network():
        model = _FixedLogits(_box_logits(lo, hi))
        call(fn, model, ap, lat)
        x = model.calls[-1]["x"].numpy()
        assert x.shape == (1, 2, 128, 128), f"the network got {x.shape}; use prepare_inputs"
        close(x[0, 1], _std(lat), "the network's channel 1 should be the standardised LAT", atol=1e-3)
        call(fn, model, ap)
        assert tuple(model.calls[-1]["x"].shape) == (1, 1, 128, 128), "no LAT given: the network gets 1 view"
        assert not model.calls[-1]["training"] and not model.calls[-1]["grad"], "eval mode, no gradients (use predict_occupancy)"

    def nothing_predicted():
        out = call(fn, _FixedLogits(torch.full((1, 64, 64, 64), -10.0)), ap, lat)
        assert int(np.sum(out["occ"])) == 0 and len(out["verts"]) == 0, \
            "a network that predicts nothing gives an empty cube and an empty mesh (and no crash)"

    run("predict_from_xrays", the_dict, mesh_in_mm, cell_size, inputs_reach_the_network, nothing_predicted)


# ---------------------------------------------------------------- part B: the results story

def _rows():
    # one method, 5 subjects; t3 has an empty prediction (HD95 = inf), t5 a NaN Dice (a failed evaluation)
    vals = {"t1": (0.80, 6.0), "t2": (0.55, 22.0), "t3": (0.00, math.inf), "t4": (0.71, 9.0), "t5": (math.nan, 8.0)}
    rows = [{"method": "net", "subject": s, "dice": d, "hd95": h} for s, (d, h) in vals.items()]
    rows += [{"method": "other", "subject": "t1", "dice": 0.99, "hd95": 1.0}]
    return rows


def check_best_and_worst(fn):
    rows = _rows()
    subj = lambda rs: [r["subject"] for r in rs]

    def dice():
        best, worst = call(fn, rows, 2, method="net")
        assert subj(best) == ["t1", "t4"], f"best 2 by Dice (highest first): t1 (0.80), t4 (0.71); got {subj(best)}"
        assert subj(worst) == ["t3", "t2"], f"worst 2 by Dice (worst first): t3 (0.00), t2 (0.55); got {subj(worst)}"

    def hd95_lower_is_better():
        best, worst = call(fn, rows, 2, metric="hd95", method="net")
        assert subj(best) == ["t1", "t5"], (f"HD95 is a distance: LOWER is better. Best 2: t1 (6 mm), t5 (8 mm); got "
                                            f"{subj(best)}")
        assert subj(worst) == ["t3", "t2"], f"worst 2 by HD95: t3 (inf, nothing predicted), t2 (22 mm); got {subj(worst)}"

    def other_metrics_right_direction():
        # two rows, a clearly better than b on every metric. IoU and NSD are overlaps (HIGHER is better, like Dice);
        # ASSD and Chamfer are distances (LOWER is better, like HD95). The docstring lists which is which.
        ab = [{"method": "net", "subject": "a", "iou": 0.9, "nsd_1.5mm": 0.9, "nsd_5mm": 0.9, "assd": 1.0, "chamfer": 1.0},
              {"method": "net", "subject": "b", "iou": 0.2, "nsd_1.5mm": 0.2, "nsd_5mm": 0.2, "assd": 7.0, "chamfer": 7.0}]
        for m, why in [("iou", "IoU is an overlap: HIGHER is better (a 0.9, b 0.2)"),
                       ("nsd_1.5mm", "NSD is a fraction of surface within tolerance: HIGHER is better (a 0.9, b 0.2)"),
                       ("nsd_5mm", "NSD is a fraction of surface within tolerance: HIGHER is better (a 0.9, b 0.2)"),
                       ("assd", "ASSD is a distance: LOWER is better (a 1 mm, b 7 mm)"),
                       ("chamfer", "Chamfer is a distance: LOWER is better (a 1 mm, b 7 mm)")]:
            best, worst = call(fn, ab, 1, metric=m)
            assert subj(best) == ["a"] and subj(worst) == ["b"], (f"metric={m!r}: {why}. Best should be a, worst b; got "
                                                                  f"best {subj(best)}, worst {subj(worst)}")

    def nan_is_not_ranked():
        best, worst = call(fn, rows, 10, method="net")
        assert "t5" not in subj(best) + subj(worst), ("t5's Dice is NaN (the evaluation failed): it can't be ranked, "
                                                      "leave it out (NaN compares False with everything, so sorting puts it anywhere)")
        assert len(best) == 4 and len(worst) == 4, "k=10 but only 4 rankable rows: return all 4"

    def method_filter():
        best, _ = call(fn, rows, 1)
        assert subj(best) == ["t1"] and best[0]["method"] == "other", "method=None: rank all rows (the 'other' row has Dice 0.99)"
        best, _ = call(fn, rows, 1, method="net")
        assert best[0]["method"] == "net", "method='net': only that method's rows"

    def rows_unchanged():
        r = _rows()
        before = [dict(x) for x in r]
        call(fn, r, 2, method="net")
        assert len(r) == len(before) and all(x["subject"] == y["subject"] for x, y in zip(r, before)), \
            "don't sort the caller's list in place (use sorted(), not list.sort())"
        best, _ = call(fn, r, 1, method="net")
        assert best[0] is r[0] or best[0] == r[0], "return the rows themselves (dicts), so the caller sees every column"

    run("best_and_worst", dice, hd95_lower_is_better, other_metrics_right_direction, nan_is_not_ranked, method_filter, rows_unchanged)


def check_story_table(fn):
    # "net": 3 subjects, 2 seeds each. Per subject (mean over seeds): s1 0.7, s2 0.5, s3 0.8.
    net = [{"subject": s, "seed": k, "dice": d, "hd95": h} for s, k, d, h in [
        ("s1", 0, 0.8, 5.0), ("s1", 1, 0.6, 7.0), ("s2", 0, 0.5, math.inf), ("s2", 1, 0.5, 30.0),
        ("s3", 0, 0.9, 4.0), ("s3", 1, 0.7, 6.0)]]
    # "unbalanced": s1 three times (0.9 each), s2 once (0.3): the subject mean is 0.6, the row mean would be 0.75
    unbal = [{"subject": "s1", "dice": 0.9, "hd95": 1.0}] * 3 + [{"subject": "s2", "dice": 0.3, "hd95": 2.0}]
    one = [{"subject": "s9", "dice": 0.4, "hd95": 3.0}]

    def get():
        t = call(fn, {"net": net, "unbalanced": unbal, "one subject": one, "nothing": []})
        assert isinstance(t, list) and len(t) == 4, "return a list with one dict per group, in the groups' order"
        assert [r["label"] for r in t] == ["net", "unbalanced", "one subject", "nothing"], "keep the groups' order and labels"
        return t

    def counts():
        t = get()
        assert t[0]["n_subjects"] == 3 and t[0]["n_rows"] == 6, (
            f"net: n_subjects {t[0]['n_subjects']}, n_rows {t[0]['n_rows']}; expected 3 subjects (2 seeds each = 6 rows). "
            "The n of the table is the number of SUBJECTS: 2 seeds of the same pelvis are not 2 independent cases")

    def subject_first():
        t = get()
        close(t[0]["dice_mean"], (0.7 + 0.5 + 0.8) / 3, "net: Dice mean over subjects (each subject = the mean over its seeds)")
        close(t[1]["dice_mean"], 0.6, "unbalanced: subject s1 has 3 rows, s2 one. Average per subject first (0.9, 0.3), "
              "then over subjects: 0.6. (The mean over rows, 0.75, counts s1 three times)", atol=1e-6)

    def sample_sd():
        t = get()
        close(t[0]["dice_sd"], float(np.std([0.7, 0.5, 0.8], ddof=1)), "net: the SAMPLE standard deviation over the 3 "
              "subject means (divide by n - 1 = 2): 0.1528. (Dividing by n gives 0.1247; the sd over the 6 rows 0.1366)", atol=1e-4)
        assert math.isnan(t[2]["dice_sd"]), "one subject: no spread can be estimated, so sd is NaN (not 0)"
        close(t[2]["dice_mean"], 0.4, "one subject: the mean is its value")

    def nonfinite():
        t = get()
        assert t[0]["hd95_n"] == 2, (f"net: HD95 of s2 is inf in seed 0 (nothing predicted), so s2's mean is inf; "
                                     f"leave it out of the mean but say so: hd95_n = 2 (got {t[0]['hd95_n']})")
        close(t[0]["hd95_mean"], (6.0 + 5.0) / 2, "net: HD95 mean over the 2 finite subjects (s1 6.0, s3 5.0)")
        assert t[0]["dice_n"] == 3, "net: all 3 Dice values are finite: dice_n = 3"

    def empty_group():
        t = get()
        assert t[3]["n_subjects"] == 0 and t[3]["dice_n"] == 0 and math.isnan(t[3]["dice_mean"]), \
            "an empty group: 0 subjects, mean NaN (don't crash: a results file may not exist yet)"

    def other_metrics():
        t = call(fn, {"net": net}, metrics=("dice",))
        assert "dice_mean" in t[0] and "hd95_mean" not in t[0], "metrics=('dice',): only those columns"

    run("story_table", counts, subject_first, sample_sd, nonfinite, empty_group, other_metrics)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_prepare_inputs():
    check_prepare_inputs(skip_if_missing(pipeline, "prepare_inputs"))


def test_predict_occupancy():
    check_predict_occupancy(skip_if_missing(pipeline, "predict_occupancy"))


def test_predict_from_xrays():
    check_predict_from_xrays(skip_if_missing(pipeline, "predict_from_xrays"))


def test_best_and_worst():
    check_best_and_worst(skip_if_missing(pipeline, "best_and_worst"))


def test_story_table():
    check_story_table(skip_if_missing(pipeline, "story_table"))
