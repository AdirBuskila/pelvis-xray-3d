"""Checks for unit 06 (the controlled experiment: summary statistics and the paired comparison).

Every check_* takes YOUR function and tests it on tiny tables whose answers were worked out by hand
(the sample std of 0.78, 0.80, 0.82 is 0.02; a bootstrap of [0, 1] can only give means 0, 0.5 and 1, ...).
There is no reference solution in here.

  notebook:  import test_unit as T;  T.check_mean_std(mean_std)
  pytest:    pytest units/06-experiment   (tests the copies you moved into src/xray3d/experiment.py;
                                           anything not moved yet is skipped)

The toy results table used below (rows as run_experiment.load_rows gives them)
--------------------------------------------------------------------------------
    arm  seed  subject  dice  hd95
    A     0     x       0.6   10
    A     0     y       0.8   20
    A     1     x       0.5   12
    A     1     y       0.7   22
    B     0     x       0.4   30
    B     0     y       0.4   40
    B     1     x       0.2   30
    B     1     y       0.4   50
Arm A: seed 0 scores (0.6 + 0.8) / 2 = 0.7, seed 1 scores 0.6. Subject x averages (0.6 + 0.5) / 2 = 0.55 over seeds.
"""
from __future__ import annotations

import math
import random

import numpy as np

from xray3d import experiment
from xray3d.checks import call, close, run, skip_if_missing

_TOY = [("A", 0, "x", 0.6, 10), ("A", 0, "y", 0.8, 20), ("A", 1, "x", 0.5, 12), ("A", 1, "y", 0.7, 22),
        ("B", 0, "x", 0.4, 30), ("B", 0, "y", 0.4, 40), ("B", 1, "x", 0.2, 30), ("B", 1, "y", 0.4, 50)]


def toy_rows(shuffle_seed=None):
    rows = [{"method": f"net_{a}", "arm": a, "seed": s, "subject": sub, "dice": d, "hd95": h, "retrieved": ""}
            for a, s, sub, d, h in _TOY]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)     # a CSV promises no order
    return rows


def _num(x, what):
    assert np.ndim(x) == 0, f"{what}: expected one number, got {x!r}"
    return float(x)


def _dict(d, what):
    assert isinstance(d, dict), f"{what}: return a dict, got {type(d).__name__}"
    return d


# ---------------------------------------------------------------- part B: seeds and subjects

def check_mean_std(fn):
    def three_seeds():
        m, s = call(fn, [0.78, 0.80, 0.82])
        close(_num(m, "mean"), 0.80, "mean of 0.78, 0.80, 0.82")
        close(_num(s, "std"), 0.02, "sample std of 0.78, 0.80, 0.82: sqrt((0.02^2 + 0 + 0.02^2) / (3 - 1)) = 0.02", atol=1e-9)

    def sample_not_population():
        m, s = call(fn, [1, 2, 3, 4])
        close(m, 2.5, "mean of 1, 2, 3, 4")
        if abs(s - math.sqrt(1.25)) < 1e-6:
            raise AssertionError("std of 1, 2, 3, 4 came out 1.118: that divides by n. The sample std divides by n - 1 "
                                 "(np.std(..., ddof=1)): sqrt(5 / 3) = 1.291")
        close(s, math.sqrt(5 / 3), "sample std of 1, 2, 3, 4 = sqrt(5 / 3)")

    def one_value():
        m, s = call(fn, [0.7])
        close(m, 0.7, "mean of a single run")
        assert math.isnan(s), f"one run has no spread to estimate: return nan for its std (got {s}); 0 would claim it's exact"

    def any_iterable():
        m, s = call(fn, (v for v in (2.0, 4.0)))
        close([m, s], [3.0, math.sqrt(2)], "a generator of 2 and 4 (e.g. dict.values())")

    run("mean_std", three_seeds, sample_not_population, one_value, any_iterable)


def check_seed_scores(fn):
    def toy():
        got = _dict(call(fn, toy_rows(0), "A"), "seed_scores")
        assert sorted(got) == [0, 1], f"arm A has seeds 0 and 1, got keys {sorted(got)}"
        close([got[0], got[1]], [0.7, 0.6], "arm A: seed 0 = mean(0.6, 0.8), seed 1 = mean(0.5, 0.7) (one number per RUN)")

    def other_arm_and_metric():
        got = call(fn, toy_rows(1), "B", "hd95")
        close([got[0], got[1]], [35.0, 40.0], "arm B, metric hd95: seed 0 = mean(30, 40), seed 1 = mean(30, 50)")

    def sorted_keys():
        assert list(call(fn, toy_rows(2), "A")) == [0, 1], "keys in seed order (0, 1, ...), whatever the row order"

    run("seed_scores", toy, other_arm_and_metric, sorted_keys)


def check_per_subject_mean(fn):
    def toy():
        got = _dict(call(fn, toy_rows(3), "A"), "per_subject_mean")
        assert sorted(got) == ["x", "y"], f"arm A has subjects x and y, got {sorted(got)}"
        close([got["x"], got["y"]], [0.55, 0.75], "arm A: x = mean over seeds of (0.6, 0.5), y = mean of (0.8, 0.7)")

    def other_arm_and_metric():
        got = call(fn, toy_rows(4), "B", "hd95")
        close([got["x"], got["y"]], [30.0, 45.0], "arm B, hd95: x = mean(30, 30), y = mean(40, 50)")

    def same_grand_mean():
        # averaging seeds per subject, or subjects per seed, must give the same overall mean (a balanced table)
        a = np.mean(list(call(fn, toy_rows(5), "A").values()))
        close(a, 0.65, "the mean of the per-subject means of arm A is the arm's overall mean, 0.65")

    def sorted_keys():
        assert list(call(fn, toy_rows(6), "B")) == ["x", "y"], "keys in sorted subject order"

    run("per_subject_mean", toy, other_arm_and_metric, same_grand_mean, sorted_keys)


def check_paired_differences(fn):
    def matched_by_subject():
        a = {"x": 0.9, "y": 0.7, "z": 0.8}
        b = {"z": 0.6, "x": 0.8, "y": 0.7}            # same subjects, another order
        got = _dict(call(fn, a, b), "paired_differences")
        assert list(got) == ["x", "y", "z"], f"keys: the subjects, sorted; got {list(got)}"
        close([got["x"], got["y"], got["z"]], [0.1, 0.0, 0.2], "a - b subject by subject (match the NAMES, not the order)", atol=1e-9)

    def mismatch_is_an_error():
        try:
            call(fn, {"x": 0.9, "y": 0.7}, {"x": 0.8, "w": 0.1})
        except ValueError:
            return
        raise AssertionError("a and b hold different subjects (y vs w): raise ValueError. Pairing only means something "
                             "when both arms were scored on exactly the same patients")

    run("paired_differences", matched_by_subject, mismatch_is_an_error)


# ---------------------------------------------------------------- part C: is it real?

def check_bootstrap_ci(fn):
    def two_values():
        # resampling 2 subjects with replacement: means 0 (p 1/4), 0.5 (p 1/2), 1 (p 1/4)
        lo, hi = call(fn, [0.0, 1.0], n_boot=10_000, level=0.95, seed=0)
        close([_num(lo, "lo"), _num(hi, "hi")], [0.0, 1.0],
              "diffs [0, 1]: the 2.5th and 97.5th percentiles of {0, 0.5, 1} with probabilities 1/4, 1/2, 1/4")
        lo, hi = call(fn, [0.0, 1.0], n_boot=10_000, level=0.4, seed=0)
        close([lo, hi], [0.5, 0.5], "the same with level 0.4: the 30th and 70th percentiles both fall on 0.5")

    def resample_size_n():
        # 4 subjects: the mean of a resample is (number of 1s drawn) / 4, binomial(4, 1/4). P(mean <= 0.5) = 0.949,
        # P(mean <= 0.75) = 0.996, so the 97.5th percentile is 0.75; P(mean = 0) = 0.32 puts the 2.5th at 0.
        lo, hi = call(fn, [0.0, 0.0, 0.0, 1.0], n_boot=10_000, level=0.95, seed=1)
        close([lo, hi], [0.0, 0.75], "diffs [0, 0, 0, 1]: draw as many subjects as there are (4), with replacement")

    def constant():
        lo, hi = call(fn, [0.05] * 9, n_boot=2000, seed=0)
        close([lo, hi], [0.05, 0.05], "every subject improved by exactly 0.05: no uncertainty left", atol=1e-9)

    def reproducible_and_brackets_mean():
        d = [0.03, -0.01, 0.05, 0.02, 0.00, 0.04, 0.01, -0.02, 0.06]
        a = call(fn, d, n_boot=3000, seed=7)
        b = call(fn, d, n_boot=3000, seed=7)
        close(a, b, "same seed -> the same interval (use np.random.default_rng(seed))")
        assert a[0] < np.mean(d) < a[1], f"the interval {a} should contain the mean difference {np.mean(d):.4f}"
        assert a[0] < a[1], "lo must be below hi"

    run("bootstrap_ci", two_values, resample_size_n, constant, reproducible_and_brackets_mean)


def check_verdict(fn):
    def clear_win():
        close(call(fn, [0.80, 0.82, 0.81], [0.70, 0.71, 0.69], (0.08, 0.14)), 1,
              "A beats B by 0.11, seed std 0.01, the interval excludes 0: +1")
        close(call(fn, [0.70, 0.71, 0.69], [0.80, 0.82, 0.81], (-0.14, -0.08)), -1, "the same with A and B swapped: -1")

    def interval_crosses_zero():
        close(call(fn, [0.90, 0.90, 0.91], [0.80, 0.80, 0.81], (-0.02, 0.20)), 0,
              "the seeds differ by 0.10, but over patients the interval includes 0: within noise (0)")

    def within_seed_spread():
        close(call(fn, [0.80, 0.85, 0.75], [0.78, 0.77, 0.79], (0.01, 0.03)), 0,
              "A - B = 0.02, but A's seeds alone spread by std 0.05: another seed could flip it. Within noise (0)")

    def lower_is_better():
        close(call(fn, [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], (-12.0, -8.0), higher_is_better=False), 1,
              "HD95 (mm, lower is better): A is 10 mm lower, std 1, interval excludes 0: A is better (+1)")

    run("verdict", clear_win, interval_crosses_zero, within_seed_spread, lower_is_better)


def check_results_table(fn):
    base = [{"method": "mean_shape", "subject": "x", "dice": 0.50}, {"method": "mean_shape", "subject": "y", "dice": 0.60},
            {"method": "nn", "subject": "y", "dice": 0.30}, {"method": "nn", "subject": "x", "dice": 0.40}]

    def arms_and_baselines():
        t = call(fn, toy_rows(), base, "dice")
        assert isinstance(t, list) and all(isinstance(r, dict) for r in t), "return a list of dicts, one per row of the table"
        assert [r["method"] for r in t] == ["A", "B", "mean_shape", "nn"], \
            f"rows: the arms in order of appearance, then the baselines; got {[r['method'] for r in t]}"
        a, b, ms, nn = t
        close([a["mean"], a["std"]], [0.65, math.sqrt(0.005)],
              "arm A: seed scores 0.7 and 0.6 -> mean 0.65, sample std 0.0707 (the spread over SEEDS, not subjects)")
        close([b["mean"], b["std"]], [0.35, math.sqrt(0.005)], "arm B: seed scores 0.4 and 0.3")
        assert a["runs"] == 2 and a["subjects"] == 2, f"arm A: 2 runs on 2 subjects, got runs={a['runs']} subjects={a['subjects']}"
        close(ms["mean"], 0.55, "mean_shape: the mean over its subjects")
        close(nn["mean"], 0.35, "nn: the mean over its subjects")
        assert math.isnan(ms["std"]) and ms["runs"] == 1, \
            "a baseline is one deterministic run: no seed spread, so std is nan (not its spread over subjects) and runs = 1"

    def other_metric():
        t = {r["method"]: r for r in call(fn, toy_rows(8), [], "hd95")}
        assert sorted(t) == ["A", "B"], f"no baselines given: only the arms A and B, got {sorted(t)}"
        close([t["A"]["mean"], t["B"]["mean"]], [16.0, 37.5], "hd95: arm A seeds 15 and 17 -> 16; arm B 35 and 40 -> 37.5")

    def same_patients_in_every_row():
        try:
            call(fn, toy_rows(), base + [{"method": "hull", "subject": "x", "dice": 0.3}, {"method": "hull", "subject": "z", "dice": 0.3}], "dice")
        except ValueError:
            return
        raise AssertionError("baseline 'hull' was scored on x and z, the experiment on x and y: raise ValueError "
                             "(every row of an honest table is measured on the same patients)")

    run("results_table", arms_and_baselines, other_metric, same_patients_in_every_row)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_mean_std():
    check_mean_std(skip_if_missing(experiment, "mean_std"))


def test_seed_scores():
    check_seed_scores(skip_if_missing(experiment, "seed_scores"))


def test_per_subject_mean():
    check_per_subject_mean(skip_if_missing(experiment, "per_subject_mean"))


def test_paired_differences():
    check_paired_differences(skip_if_missing(experiment, "paired_differences"))


def test_bootstrap_ci():
    check_bootstrap_ci(skip_if_missing(experiment, "bootstrap_ci"))


def test_verdict():
    check_verdict(skip_if_missing(experiment, "verdict"))


def test_results_table():
    check_results_table(skip_if_missing(experiment, "results_table"))
