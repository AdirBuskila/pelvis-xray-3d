"""The training loop around YOUR train_step: seeds, checkpoints, the CSV log, resume, Drive.

Written by Claude (infrastructure). The concept code it calls is the students':
    xray3d.nets.XrayTo3D        the network            (unit 04)
    xray3d.losses.bce_dice_loss the loss               (unit 05)
    xray3d.training.train_step  one optimisation step  (unit 05)
    xray3d.training.validation_dice    the validation Dice    (unit 05)

A run has a name and a folder, runs_root() / <name>:
    last.pt       the latest checkpoint (resume starts from it)
    best.pt       the checkpoint with the highest validation Dice so far
    log.csv       one row per evaluation: step, mean training loss since the last row, val Dice, time
    summary.json  the config and the final numbers, written when the run ends
Checkpoint format (the contract, units 06-08 read it):
    {"model": state_dict, "config": {...}, "step": n, "seed": s}   + "optimizer" and "best_val_dice" for resuming
Always loaded with torch.load(..., weights_only=True): a checkpoint is data, it must not be able to run code.

Typical use (from a notebook or units/05-training/train.py):
    from xray3d import trainer
    trainer.train("first-2view", n_views=2, seed=0, iters=4000)
    model = trainer.load_model("first-2view")          # best.pt, in eval mode
    trainer.plot_curves(["first-2view"])
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from xray3d import env, torchdata

# Everything that defines a run. train() takes these as keyword arguments; all of them go into the
# checkpoint's "config", so a run can always be rebuilt and compared with another one.
DEFAULTS = dict(
    n_views=2,        # 1 = AP only, 2 = AP + LAT
    fuse="concat",    # how the lifted views are combined: "concat" or "sum"
    dropout=0.1,      # Dropout3d on the fused volume (unit 07 uses it for MC-dropout uncertainty)
    seed=0,           # weights' initialisation, data order, augmentation
    iters=4000,       # optimisation steps (the proposal: 3,000-5,000)
    batch_size=8,
    lr=1e-3,          # Adam, constant
    train_frac=1.0,   # fraction of the training subjects (unit 06's second arm), nested subsets
    augment=True,     # intensity augmentation of the training images
    eval_every=250,   # validation Dice every this many steps (one row of log.csv)
    ckpt_every=250,   # save last.pt (and copy it to Drive on Colab) every this many steps
)
MODEL_KEYS = ("n_views", "fuse", "dropout")   # the subset of the config that XrayTo3D(**...) takes


# ----------------------------------------------------------------------------- folders

def runs_root() -> Path:
    """Where run folders go.

    Override with XRAY3D_RUNS. Default: env.processed_root() / "runs" (the contract), except on Kaggle,
    where processed_root() is the attached dataset (read-only): there it's /kaggle/working/runs, which Kaggle
    keeps as the notebook's output.
    """
    if os.environ.get("XRAY3D_RUNS"):
        return Path(os.environ["XRAY3D_RUNS"])
    if env.where() == "kaggle":
        return Path("/kaggle/working/runs")
    return env.processed_root() / "runs"


def run_dir(name: str) -> Path:
    return runs_root() / name


def drive_runs_root() -> Path | None:
    """On Colab with Google Drive mounted: MyDrive/pelvis-xray3d/runs. Otherwise None.

    Mount it in the notebook first:  from google.colab import drive; drive.mount("/content/drive")
    A Colab runtime is wiped when it ends; Drive is not. Hence the copy after every checkpoint.
    """
    mydrive = Path("/content/drive/MyDrive")
    if env.where() == "colab" and mydrive.exists():
        return mydrive / "pelvis-xray3d" / "runs"
    return None


def _sync_to_drive(rdir: Path) -> None:
    droot = drive_runs_root()
    if droot is None:
        return
    dst = droot / rdir.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in rdir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)


def _restore_from_drive(rdir: Path) -> None:
    """A new Colab runtime: bring the run folder back from Drive so the run can resume."""
    droot = drive_runs_root()
    if droot is None or (rdir / "last.pt").exists() or not (droot / rdir.name / "last.pt").exists():
        return
    rdir.mkdir(parents=True, exist_ok=True)
    for f in (droot / rdir.name).iterdir():
        if f.is_file():
            shutil.copy2(f, rdir / f.name)
    print(f"restored {rdir.name} from Drive")


# ----------------------------------------------------------------------------- seeds, model, checkpoints

def seed_everything(seed: int) -> None:
    """Same seed -> same initial weights, same data order, same augmentations.

    On a GPU some convolution kernels are still not bit-for-bit repeatable, so two runs with the same seed
    end up close but not identical. That's why unit 06 compares several seeds instead of trusting one run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: dict):
    """XrayTo3D from a run config (only the model keys are used). Imports the students' nets.py."""
    from xray3d import nets
    return nets.XrayTo3D(**{k: config[k] for k in MODEL_KEYS if k in config})


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict, best_val_dice: float) -> None:
    ckpt = {
        "model": model.state_dict(),
        "config": dict(config),
        "step": int(step),
        "seed": int(config["seed"]),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "best_val_dice": float(best_val_dice),
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # write-then-rename: a crash mid-save never leaves a broken last.pt


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def _ckpt_path(name_or_path: str | Path, which: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".pt":
        return p
    rdir = p if p.is_dir() else run_dir(str(name_or_path))
    return rdir / f"{which}.pt"


def load_model(name_or_path: str | Path, which: str = "best", device: str = "cpu"):
    """The trained network of a run, in eval mode.

    name_or_path : a run name (looked up in runs_root()), a run folder, or a .pt file
    which        : "best" (highest val Dice) or "last"
    """
    ckpt = load_checkpoint(_ckpt_path(name_or_path, which), device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


# ----------------------------------------------------------------------------- the log

LOG_FIELDS = ["step", "train_loss", "val_dice", "seconds"]   # seconds: training time so far (summed over resumes)


def _truncate_log(path: Path, step: int) -> None:
    """On resume: drop rows written after the checkpoint we resume from (they will be redone)."""
    if not path.exists():
        return
    rows = [r for r in csv.DictReader(path.open()) if int(r["step"]) <= step]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _append_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_log(name_or_dir: str | Path) -> dict[str, np.ndarray]:
    """log.csv of a run as arrays: {"step", "train_loss", "val_dice", "seconds"}."""
    p = Path(name_or_dir)
    path = (p if p.is_dir() else run_dir(str(name_or_dir))) / "log.csv"
    rows = list(csv.DictReader(path.open()))
    return {k: np.array([float(r[k]) for r in rows]) for k in LOG_FIELDS}


def plot_curves(names, axes=None, title: str = ""):
    """Training loss and validation Dice against the step, one line per run."""
    import matplotlib.pyplot as plt
    if isinstance(names, (str, Path)):
        names = [names]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        log = read_log(n)
        label = Path(n).name
        axes[0].plot(log["step"], log["train_loss"], marker=".", label=label)
        axes[1].plot(log["step"], log["val_dice"], marker=".", label=label)
    axes[0].set(xlabel="step", ylabel="training loss (BCE + soft Dice)", title="training loss")
    axes[1].set(xlabel="step", ylabel="Dice", title="validation Dice (sigmoid > 0.5)", ylim=(0, 1))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend()
    if title:
        axes[0].figure.suptitle(title)
    axes[0].figure.tight_layout()
    return axes


# ----------------------------------------------------------------------------- the loop

def train(name: str, resume: bool = True, device: str | None = None, root: str | Path | None = None,
          num_workers: int | None = None, verbose: bool = True, **overrides) -> dict:
    """Train one run (or resume it) and return its summary.

    name       : the run's folder name under runs_root()
    resume     : if <run>/last.pt exists, continue from it (same config required) instead of starting over
    device     : "cuda" / "cpu"; default: the GPU if there is one
    root       : the processed dataset folder (default: torchdata.dataset_root())
    overrides  : any key of DEFAULTS, e.g. n_views=1, seed=2, iters=3000, train_frac=0.25
    """
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown config keys {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **overrides}
    cfg["name"] = name
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = 2 if env.where() != "local" else 0
    from xray3d import losses, training  # the students' code; imported here so the module loads without it

    rdir = run_dir(name)
    _restore_from_drive(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    last, best, log_path = rdir / "last.pt", rdir / "best.pt", rdir / "log.csv"

    seed_everything(cfg["seed"])
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    step, best_dice = 0, -1.0
    if resume and last.exists():
        ckpt = load_checkpoint(last, device)
        old = {k: v for k, v in ckpt["config"].items() if k not in ("iters", "eval_every", "ckpt_every")}
        new = {k: v for k, v in cfg.items() if k not in ("iters", "eval_every", "ckpt_every")}
        if old != new:
            raise ValueError(f"run '{name}' exists with a different config:\n  saved {old}\n  now   {new}\n"
                             "use another name, or resume=False to start it over")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_dice = int(ckpt["step"]), float(ckpt.get("best_val_dice", -1.0))
        _truncate_log(log_path, step)
        if verbose:
            print(f"resuming '{name}' at step {step} (best val Dice so far {best_dice:.3f})")
    elif log_path.exists():
        log_path.unlink()  # starting over: the old log belongs to the old run

    # the data order is seeded too; after a resume it continues with a different (but still seeded) order
    train_loader, val_loader = torchdata.make_loaders(
        n_views=cfg["n_views"], batch_size=cfg["batch_size"], train_frac=cfg["train_frac"],
        seed=cfg["seed"] * 100_003 + step, augment=cfg["augment"], root=root, num_workers=num_workers)
    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"run '{name}': {len(train_loader.dataset)} train / {len(val_loader.dataset)} val subjects, "
              f"{cfg['n_views']} view(s), {n_par:,} parameters, device {device}, steps {step} -> {cfg['iters']}")
        print(f"  folder: {rdir}")

    batches = torchdata.infinite(train_loader)
    prev = read_log(rdir)["seconds"] if log_path.exists() else []
    losses_since, t0 = [], time.time() - (float(prev[-1]) if len(prev) else 0.0)   # "seconds" counts across resumes
    while step < cfg["iters"]:
        loss = training.train_step(model, next(batches), optimizer, losses.bce_dice_loss, device)
        if not np.isfinite(loss):
            raise FloatingPointError(f"loss is {loss} at step {step + 1}: lower the learning rate or check the loss")
        losses_since.append(loss)
        step += 1
        if step % cfg["eval_every"] == 0 or step == cfg["iters"]:
            val_dice = training.validation_dice(model, val_loader, device)
            row = {"step": step, "train_loss": round(float(np.mean(losses_since)), 5),
                   "val_dice": round(val_dice, 5), "seconds": round(time.time() - t0, 1)}
            _append_log(log_path, row)
            losses_since = []
            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(best, model, optimizer, step, cfg, best_dice)
            if verbose:
                print(f"  step {step:>5}  loss {row['train_loss']:.4f}  val Dice {val_dice:.3f}  "
                      f"(best {best_dice:.3f})  {row['seconds']:.0f} s")
        if step % cfg["ckpt_every"] == 0 or step == cfg["iters"]:
            save_checkpoint(last, model, optimizer, step, cfg, best_dice)
            _sync_to_drive(rdir)

    log = read_log(rdir)
    summary = {"name": name, "config": cfg, "step": step, "best_val_dice": best_dice,
               "final_val_dice": float(log["val_dice"][-1]) if len(log["step"]) else None,
               "folder": str(rdir), "device": device}
    (rdir / "summary.json").write_text(json.dumps(summary, indent=1))
    _sync_to_drive(rdir)
    return summary
