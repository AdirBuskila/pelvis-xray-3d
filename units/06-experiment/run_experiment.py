"""Unit 06's controlled experiment: train every arm x seed, then score each run on the TEST split.

Written by Claude (infrastructure). The concept code it runs is the students':
    xray3d.nets.XrayTo3D, losses.bce_dice_loss, training.train_step / validation_dice   (units 04-05, via trainer)
    xray3d.metrics.evaluate                                                           (unit 02)
The analysis of the CSV it writes (mean +- std, paired differences, the verdict) is the students' too
(xray3d.experiment, unit 06's notebook).

The design
----------
Three arms, three seeds each (0, 1, 2), everything else the trainer's DEFAULTS (architecture, 4,000 steps,
batch 8, Adam 1e-3, augmentation, the split):

    arm          n_views  train_frac   used by
    ap              1        1.0       comparison 1 (1 view vs 2 views)
    ap_lat          2        1.0       comparison 1 AND comparison 2 (it IS the "100 %" arm)
    ap_lat_f25      2        0.25      comparison 2 (training-set size, 25 % vs 100 %)

Two comparisons x two arms x three seeds would be 12 runs, but "2 views, 100 % of train" is the same
configuration in both comparisons, so the same three runs serve both: 9 runs.

Usage (from the repo root, or anywhere):
    python units/06-experiment/run_experiment.py                         # the real thing (GPU): 9 runs x 4,000 steps
    python units/06-experiment/run_experiment.py --seeds 0               # only seed 0 of every arm (split across sessions)
    python units/06-experiment/run_experiment.py --smoke                 # tiny CPU version, numbers meaningless
    python units/06-experiment/run_experiment.py --eval-only             # re-score every finished run (ignores the
                                                                         # test_eval.json cache), rewrite the CSV
    python units/06-experiment/run_experiment.py --restore-from /kaggle/input/<previous-output>/runs

Everything is resumable: a finished run (summary.json says it reached --iters) is skipped, an unfinished one
continues from its last.pt (trainer.train's resume), and a run whose test scores are already cached in its
folder (test_eval.json, for the same best.pt and the same test subjects) is not re-scored. The cache doesn't know
whether metrics.evaluate changed since, so after fixing a metric run --eval-only: it re-scores every run.

Output: one row per (run, test subject), the long format of unit 03's results/baselines.csv
    method, subject, retrieved, dice, iou, hd95, assd, nsd_1.5mm, nsd_5mm, chamfer   + arm, seed, n_views, train_frac
written to env.results_root() / "experiment_v1.csv" (the real run) or, for --smoke, to
trainer.runs_root() / "experiment_smoke.csv" (outside git: smoke numbers must never land in results/).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

# import this checkout's xray3d even if the package isn't installed (same as the notebooks' setup cell)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from xray3d import env, torchdata, trainer  # noqa: E402

# ----------------------------------------------------------------------------- the design

# arm name -> the trainer overrides that define it. ONLY these keys differ between arms.
ARMS = {
    "ap":         dict(n_views=1, train_frac=1.0),
    "ap_lat":     dict(n_views=2, train_frac=1.0),
    "ap_lat_f25": dict(n_views=2, train_frac=0.25),
}
# comparison -> (arm A, arm B); the notebook reports A - B
COMPARISONS = {
    "views": ("ap_lat", "ap"),          # does the second view help?
    "size":  ("ap_lat", "ap_lat_f25"),  # does 4x the training data help?
}
SEEDS = (0, 1, 2)
ITERS = 4000

# the tiny CPU configuration the notebook runs locally: it exercises every line of the pipeline in a few
# minutes, but 40 steps teach the network almost nothing, so its numbers mean NOTHING.
SMOKE = dict(seeds=(0, 1), iters=40, batch_size=4, lr=3e-3, eval_every=20, ckpt_every=20)

METRICS = ("dice", "iou", "hd95", "assd", "nsd_1.5mm", "nsd_5mm", "chamfer")   # the keys of metrics.evaluate
FIELDS = ["method", "subject", "retrieved", *METRICS, "arm", "seed", "n_views", "train_frac"]


def run_name(arm: str, seed: int, prefix: str = "exp") -> str:
    """The run's folder name under trainer.runs_root(), e.g. exp-ap_lat-s1 (smoke runs: smoke-ap_lat-s1)."""
    return f"{prefix}-{arm}-s{seed}"


def default_out(smoke: bool = False) -> Path:
    return trainer.runs_root() / "experiment_smoke.csv" if smoke else env.results_root() / "experiment_v1.csv"


# ----------------------------------------------------------------------------- training

def _finished(name: str, iters: int) -> bool:
    """True if the run's summary.json says it already reached `iters` steps."""
    p = trainer.run_dir(name) / "summary.json"
    if not p.exists():
        return False
    return int(json.loads(p.read_text()).get("step", 0)) >= iters


def train_all(arms=None, seeds=SEEDS, iters=ITERS, prefix="exp", device=None, root=None, verbose=True, **overrides):
    """Train (or resume, or skip) every arm x seed. Returns {run name: summary dict or None if skipped}.

    overrides: extra trainer keys applied to EVERY run (e.g. batch_size for the smoke test). They can't
    touch an arm's own keys: that would silently break the "change one variable" design.
    """
    arms = list(arms or ARMS)
    clash = set(overrides) & {k for a in arms for k in ARMS[a]}
    if clash:
        raise ValueError(f"{sorted(clash)} define the arms; don't override them for all runs")
    out = {}
    for arm in arms:
        for seed in seeds:
            name = run_name(arm, seed, prefix)
            if _finished(name, iters):
                if verbose:
                    print(f"[{name}] already finished ({iters} steps): skipped")
                out[name] = None
                continue
            t0 = time.time()
            out[name] = trainer.train(name, device=device, root=root, verbose=verbose,
                                      seed=seed, iters=iters, **ARMS[arm], **overrides)
            if verbose:
                print(f"[{name}] best val Dice {out[name]['best_val_dice']:.3f}  ({time.time() - t0:.0f} s this session)")
    return out


# ----------------------------------------------------------------------------- scoring on the test split

def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


@torch.no_grad()
def evaluate_run(name: str, root=None, device=None, batch_size: int = 8, verbose: bool = True,
                 force: bool = False) -> list[dict]:
    """Score a run's best.pt on every TEST subject with the students' metrics.evaluate.

    The test split is touched here and nowhere else: best.pt was chosen on the validation Dice, so the test
    numbers are an honest estimate. Rows are cached in <run>/test_eval.json, keyed by best.pt's step and the
    list of test subjects, so re-running the script doesn't re-score unchanged runs.
    force=True ignores the cache and scores again (the key can't tell that metrics.evaluate was fixed since).
    """
    from xray3d import metrics   # the students' unit 02 code
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rdir = trainer.run_dir(name)
    ckpt = trainer.load_checkpoint(rdir / "best.pt", device)
    cfg = ckpt["config"]
    test = torchdata.split_subjects(root)["test"]
    cache = rdir / "test_eval.json"
    key = {"best_step": int(ckpt["step"]), "subjects": test}
    if cache.exists() and not force:               # force: re-score even if the cache key matches
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["rows"]

    model = trainer.build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                   # dropout off: one deterministic prediction per subject
    ds = torchdata.XrayDataset(test, n_views=cfg["n_views"], augment=False, root=root)
    arm = next((a for a, kv in ARMS.items() if all(cfg[k] == v for k, v in kv.items())), "?")
    rows = []
    for i0 in range(0, len(ds), batch_size):
        xs, ys = zip(*(ds[i] for i in range(i0, min(i0 + batch_size, len(ds)))))
        logits = model(torch.stack(xs).to(device)).cpu()
        for j, (lg, y) in enumerate(zip(logits, ys)):
            pred = (lg[0] > 0).numpy().astype(np.uint8)      # logit > 0  <=>  sigmoid > 0.5
            gt = (y[0] > 0.5).numpy().astype(np.uint8)
            m = {k: float(v) for k, v in metrics.evaluate(gt, pred).items()}
            rows.append({"method": f"net_{arm}", "subject": test[i0 + j], "retrieved": "", **m,
                         "arm": arm, "seed": int(cfg["seed"]), "n_views": int(cfg["n_views"]),
                         "train_frac": float(cfg["train_frac"])})
    cache.write_text(json.dumps({"key": key, "rows": rows}, indent=0))
    if verbose:
        print(f"[{name}] test Dice {np.mean([r['dice'] for r in rows]):.3f} (mean of {len(rows)} subjects, "
              f"best.pt from step {key['best_step']})")
    return rows


def evaluate_all(arms=None, seeds=SEEDS, prefix="exp", out=None, root=None, device=None, verbose=True,
                 force=False) -> Path:
    """Score every run that has a best.pt and write the long CSV. Missing runs are reported, not fatal.
    force=True re-scores every run instead of reusing its test_eval.json (what --eval-only does)."""
    out = Path(out) if out else default_out(prefix == "smoke")
    rows, missing = [], []
    for arm in arms or ARMS:
        for seed in seeds:
            name = run_name(arm, seed, prefix)
            if not (trainer.run_dir(name) / "best.pt").exists():
                missing.append(name)
                continue
            rows += evaluate_run(name, root=root, device=device, verbose=verbose, force=force)
    if missing:
        print("not trained yet (left out of the CSV):", ", ".join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: _fmt(r[k]) for k in FIELDS} for r in rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


# ----------------------------------------------------------------------------- reading results, Kaggle sessions

NUMERIC = set(METRICS) | {"seed", "n_views", "train_frac"}


def load_rows(path) -> list[dict]:
    """A results CSV (experiment_v1.csv or baselines.csv) as a list of dicts, numbers as floats (seed and n_views as int)."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in NUMERIC & set(r):
                r[k] = int(r[k]) if k in ("seed", "n_views") else float(r[k])
            rows.append(r)
    return rows


def restore_runs(src) -> list[str]:
    """Copy run folders from an earlier session (e.g. /kaggle/input/<previous version's output>/runs) into
    trainer.runs_root(), so the next session resumes them. A folder that already exists here is left alone."""
    src, dst = Path(src), trainer.runs_root()
    copied = []
    for d in sorted(p for p in src.iterdir() if p.is_dir() and (p / "last.pt").exists()):
        if not (dst / d.name).exists():
            shutil.copytree(d, dst / d.name)
            copied.append(d.name)
    print(f"restored {len(copied)} run(s) from {src}: {', '.join(copied) or '-'}")
    return copied


def run_and_evaluate(arms=None, seeds=SEEDS, iters=ITERS, smoke=False, out=None, device=None, root=None,
                     train=True, evaluate=True, verbose=True, force_eval=False, **overrides) -> Path | None:
    """The whole experiment: train_all, then evaluate_all. What the notebook and the CLI call.
    force_eval=True re-scores every run even if its cached test scores look current."""
    prefix = "smoke" if smoke else "exp"
    if train:
        train_all(arms, seeds, iters, prefix=prefix, device=device, root=root, verbose=verbose, **overrides)
    if evaluate:
        return evaluate_all(arms, seeds, prefix=prefix, out=out, root=root, device=device, verbose=verbose,
                            force=force_eval)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default 0 1 2 (smoke: 0 1)")
    ap.add_argument("--iters", type=int, default=None, help=f"default {ITERS} (smoke: {SMOKE['iters']})")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU configuration; numbers meaningless")
    ap.add_argument("--out", default=None, help="CSV path (default: results/experiment_v1.csv; smoke: runs/experiment_smoke.csv)")
    ap.add_argument("--eval-only", action="store_true", help="don't train; re-score every finished run (ignores the cached test_eval.json)")
    ap.add_argument("--train-only", action="store_true", help="don't score on the test split")
    ap.add_argument("--restore-from", default=None, help="copy earlier run folders from here first (Kaggle sessions)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=None, help="processed dataset folder (default: torchdata.dataset_root())")
    a = ap.parse_args(argv)
    if a.restore_from:
        restore_runs(a.restore_from)
    extra = {k: v for k, v in SMOKE.items() if k not in ("seeds", "iters")} if a.smoke else {}
    seeds = tuple(a.seeds) if a.seeds else (SMOKE["seeds"] if a.smoke else SEEDS)
    iters = a.iters or (SMOKE["iters"] if a.smoke else ITERS)
    return run_and_evaluate(a.arms, seeds, iters, smoke=a.smoke, out=a.out, device=a.device, root=a.root,
                            train=not a.eval_only, evaluate=not a.train_only, force_eval=a.eval_only, **extra)


if __name__ == "__main__":
    main()
