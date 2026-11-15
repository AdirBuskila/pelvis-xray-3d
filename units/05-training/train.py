"""Train one run from the command line (written by Claude; unit 05 uses it for the first real run; unit 06 has its own run_experiment.py).

    python units/05-training/train.py --views 2 --seed 0 --iters 4000 --name v2-s0
    python units/05-training/train.py --views 2 --seed 1 --train-frac 0.25 --name v2-f25-s1

It calls xray3d.trainer.train, which uses YOUR nets.XrayTo3D, losses.bce_dice_loss, training.train_step and
training.validation_dice (moved into src/xray3d in units 04-05). Running it again with the same --name resumes from that
run's last.pt. The run folder is printed at the start (trainer.run_dir(name)); the summary goes to stdout as JSON.
"""
import argparse
import json
import sys
from pathlib import Path

# import this checkout's xray3d even if the package isn't installed (same as the notebooks' setup cell)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xray3d import trainer  # noqa: E402


def main(argv=None):
    d = trainer.DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name (folder under trainer.runs_root())")
    ap.add_argument("--views", type=int, choices=(1, 2), default=d["n_views"], help="1 = AP only, 2 = AP + LAT")
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--iters", type=int, default=d["iters"])
    ap.add_argument("--train-frac", type=float, default=d["train_frac"], help="fraction of the training subjects (nested subsets)")
    ap.add_argument("--batch", type=int, default=d["batch_size"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--fuse", choices=("concat", "sum"), default=d["fuse"])
    ap.add_argument("--eval-every", type=int, default=d["eval_every"])
    ap.add_argument("--ckpt-every", type=int, default=d["ckpt_every"])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: the GPU if there is one)")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: 2 in the cloud, 0 locally)")
    ap.add_argument("--root", default=None, help="processed dataset folder (default: torchdata.dataset_root())")
    ap.add_argument("--restart", action="store_true", help="start over even if the run exists (default: resume)")
    a = ap.parse_args(argv)
    summary = trainer.train(
        a.name, resume=not a.restart, device=a.device, root=a.root, num_workers=a.workers,
        n_views=a.views, seed=a.seed, iters=a.iters, train_frac=a.train_frac, batch_size=a.batch, lr=a.lr,
        fuse=a.fuse, eval_every=a.eval_every, ckpt_every=a.ckpt_every, augment=not a.no_augment)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
