"""The processed dataset (v1) as PyTorch sees it: a Dataset, input standardisation, augmentation, loaders.

Written by Claude (infrastructure). Units 05-08 use it; the concept code (network, loss, training step)
is the students'. What lives where:

  <processed root>/v1/index.json      the list of subjects + the camera geometry (built by unit 02)
  <processed root>/v1/<sid>.npz       one subject: occ (64,64,64) uint8, ap/lat (128,128) float32, ...
  <repo>/splits/split_v1.json         which subject is train / val / test (unit 01, never regenerated)

One sample = (x, y):
  x : (V, 128, 128) float32, V = 1 (AP) or 2 (AP, LAT), each image standardised on its own
  y : (1, 64, 64, 64) float32 in {0, 1}, the pelvis occupancy in the canonical cube (RAS axes)
A DataLoader stacks them into (B, V, 128, 128) and (B, 1, 64, 64, 64): the network contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as nd
from torch.utils.data import DataLoader, Dataset

from xray3d import env

VIEWS = ("ap", "lat")  # the order of the views in x: channel 0 is always AP, channel 1 LAT


# ----------------------------------------------------------------------------- where the data is

def dataset_root(root: str | Path | None = None) -> Path:
    """The folder that holds index.json and the <sid>.npz files.

    Default: env.processed_root() / "v1" (the contract). If index.json sits directly in
    env.processed_root() (e.g. a Kaggle dataset uploaded without the v1 folder), that works too.
    """
    if root is not None:
        return Path(root)
    base = env.processed_root()
    if (base / "v1" / "index.json").exists() or not (base / "index.json").exists():
        return base / "v1"
    return base


def load_index(root: str | Path | None = None) -> dict:
    """index.json: {"version", "subjects", "geometry", ...}."""
    path = dataset_root(root) / "index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the dataset in unit 02 (build_dataset.ipynb), or on Colab run "
            "xray3d.dataset.ensure_local() first; on Kaggle attach the dataset 'pelvis-xray3d-v1'.")
    return json.loads(path.read_text())


def load_split(path: str | Path | None = None) -> dict[str, list[str]]:
    """The fixed patient-level split from unit 01 ({"train": [...], "val": [...], "test": [...]})."""
    path = Path(path) if path else env.repo_root() / "splits" / "split_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: it is written once by your split_subjects in unit 01")
    sp = json.loads(path.read_text())
    return {k: list(sp[k]) for k in ("train", "val", "test")}


def split_subjects(root: str | Path | None = None, split_path: str | Path | None = None) -> dict[str, list[str]]:
    """The split, restricted to subjects that are actually in the processed dataset.

    The split was made over all 1,228 ids before screening (unit 01), so most ids in it have no npz file
    (cut pelvis, metal, or simply not in the small subset). Screening only removes subjects from a part;
    no subject ever moves between train, val and test.
    """
    have = set(load_index(root)["subjects"])
    sp = load_split(split_path)
    return {k: sorted(s for s in v if s in have) for k, v in sp.items()}


def subset(subjects: list[str], frac: float, seed: int = 0) -> list[str]:
    """A fixed fraction of a subject list (unit 06's training-set-size arm).

    The subsets are NESTED: the 25 % set is the first quarter of one seeded shuffle, so it is contained in
    the 50 % set, which is contained in the full set. The seed here picks *which* subjects; it is kept the
    same for every training seed, so "25 % vs 100 %" compares sizes, not lucky picks.
    """
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    order = sorted(subjects)
    rng = np.random.default_rng(seed)
    order = [order[i] for i in rng.permutation(len(order))]
    n = max(1, math.ceil(frac * len(order)))
    return sorted(order[:n])


# ----------------------------------------------------------------------------- the input images

def standardise(img, eps: float = 1e-6):
    """Zero mean, unit standard deviation, per image (over the last two axes).

    Works on a NumPy array or a torch tensor of shape (..., H, W). The SAME function is applied to real
    X-rays in unit 07, so a DRR and a real frame reach the network on the same scale whatever the detector
    gain or the exposure was: adding a constant or multiplying by one disappears here.
    """
    if isinstance(img, torch.Tensor):
        x = img.float()
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - m) / (s + eps)
    x = np.asarray(img, dtype=np.float32)
    m = x.mean(axis=(-2, -1), keepdims=True)
    s = x.std(axis=(-2, -1), keepdims=True)
    return ((x - m) / (s + eps)).astype(np.float32)


def augment_intensity(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random changes of the image's INTENSITIES only, never its geometry.

    Why only intensities: the target cube is tied to the camera. Shifting or rotating the image without
    moving the occupancy the same way would teach the network a wrong geometry.
    Why not brightness or contrast: standardise() removes them anyway (a + b*img standardises to the
    same thing). What survives standardisation, and also differs between a DRR and a real X-ray:
      * gamma: a non-linear tone curve (real detectors and beam hardening are not linear in p),
      * blur: real images are less sharp (focal spot, scatter),
      * noise: quantum noise.
    img: one (H, W) line-integral image (before standardisation).
    """
    x = img.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:  # gamma works on [0, 1]; map there and back
        u = (x - lo) / (hi - lo)
        x = lo + (hi - lo) * u ** float(rng.uniform(0.7, 1.4))
    if rng.random() < 0.5:
        x = nd.gaussian_filter(x, sigma=float(rng.uniform(0.3, 1.2)))
    x = x + rng.normal(0.0, float(rng.uniform(0.0, 0.03)) * (x.std() + 1e-6), x.shape).astype(np.float32)
    return x


# ----------------------------------------------------------------------------- the Dataset

class XrayDataset(Dataset):
    """Samples (x, y) for a list of subjects (see the module docstring for the shapes).

    subjects : subject ids, e.g. split_subjects()["train"]
    n_views  : 1 -> x has only the AP image; 2 -> AP and LAT
    augment  : random intensity augmentation (training only; never for val/test)
    root     : folder with index.json + npz files (default: dataset_root())
    All subjects are read into memory once (300 subjects are ~120 MB), so an epoch never touches the disk.
    """

    def __init__(self, subjects, n_views: int = 2, augment: bool = False, root: str | Path | None = None):
        if n_views not in (1, 2):
            raise ValueError(f"n_views must be 1 or 2, got {n_views}")
        self.subjects = list(subjects)
        if not self.subjects:
            raise ValueError("no subjects: is the processed dataset there, and does the split contain them?")
        self.n_views, self.augment = n_views, augment
        self.root = dataset_root(root)
        self.images, self.targets = [], []
        for sid in self.subjects:
            with np.load(self.root / f"{sid}.npz") as z:
                self.images.append(np.stack([z[v].astype(np.float32) for v in VIEWS[:n_views]]))
                self.targets.append(z["occ"].astype(bool))
        self._rng = None

    def __len__(self) -> int:
        return len(self.subjects)

    def _generator(self) -> np.random.Generator:
        # numpy's generator is seeded from torch's, so torch.manual_seed (and the DataLoader's per-worker seed)
        # makes the augmentation reproducible too
        if self._rng is None:
            self._rng = np.random.default_rng(int(torch.initial_seed()) % 2**32)
        return self._rng

    def __getitem__(self, i: int):
        imgs = self.images[i]
        if self.augment:
            rng = self._generator()
            imgs = np.stack([augment_intensity(im, rng) for im in imgs])
        x = torch.from_numpy(standardise(imgs))
        y = torch.from_numpy(self.targets[i][None].astype(np.float32))
        return x, y


def _worker_init(worker_id: int) -> None:
    # every worker process gets its own copy of the dataset; reset its numpy generator so it is re-seeded
    # from that worker's torch seed (otherwise all workers would produce the same augmentations)
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = None


def make_loaders(n_views: int = 2, batch_size: int = 8, train_frac: float = 1.0, seed: int = 0,
                 augment: bool = True, root: str | Path | None = None, num_workers: int = 0,
                 subset_seed: int = 0, split_path: str | Path | None = None):
    """(train_loader, val_loader) for one run.

    train: shuffled with a generator seeded by `seed`, incomplete last batch dropped (BatchNorm dislikes a
           batch of 1), augmented. train_frac < 1 keeps a nested subset (see subset()).
    val:   in a fixed order, no augmentation, every subject once.
    """
    sp = split_subjects(root, split_path)
    train_ids = subset(sp["train"], train_frac, subset_seed) if train_frac < 1 else sp["train"]
    train_ds = XrayDataset(train_ids, n_views, augment=augment, root=root)
    val_ds = XrayDataset(sp["val"], n_views, augment=False, root=root)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, drop_last=True,
                       generator=g, num_workers=num_workers, worker_init_fn=_worker_init if num_workers else None,
                       persistent_workers=num_workers > 0)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train, val


def infinite(loader):
    """Iterate over a DataLoader forever (training counts iterations, not epochs)."""
    while True:
        yield from loader
