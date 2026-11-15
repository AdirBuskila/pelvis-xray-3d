"""Reading the processed dataset v1: the 64^3 pelvis cubes and their simulated X-ray pairs.

Written by Claude (infrastructure). Built by unit 02 (`units/02-shape-and-metrics/build_dataset.ipynb`,
code in `xray3d/prep.py`); read by every later unit. The format is the contract of
`docs/superpowers/specs/2026-11-12-units-02-08-design.md`:

    <processed_root>/v1/
        index.json        {"version": 1, "subjects": [...], "geometry": {...}, "created": ..., "source": ...}
        s1390.npz         one file per subject:
            occ           (64, 64, 64) uint8   pelvis occupancy (hip_left | hip_right | sacrum)
            ap, lat       (128, 128) float32   line integral p of the whole CT (DiffDRR, perspective)
            ap_bone, lat_bone                  the same rendering of the pelvis voxels only
            center_mm     (3,)                 the cube centre in the CT's world coordinates (mm)
            femoral_heads (2, 4)               extra (unit 02): [x, y, z, radius] in mm, cube frame,
                                               rows = (left, right); NaN if a femur was missing

The cube frame: origin at the cube centre, axes R (patient's Right), A (Anterior), S (Superior), in mm.
Voxel (i, j, k) of `occ` has its centre at ((i, j, k) + 0.5) * 5 mm - 160 mm.

Images: rows run head -> feet; AP columns run patient's Right -> Left; LAT columns Anterior -> Posterior.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from . import env

VERSION = 1
KAGGLE_DATASET = "pelvis-xray3d-v1"          # the name to give the dataset when saving it on Kaggle
DRIVE_ZIP = "pelvis-xray3d/pelvis-xray3d-v1.zip"   # where the zip lives on Google Drive (inside MyDrive)

# The imaging geometry, fixed for the whole project (also written into index.json, which wins if it differs).
GEOMETRY = {
    "cube_mm": 320.0,             # side of the canonical cube
    "cube_n": 64,                 # voxels per side -> 5 mm voxels
    "axes": "RAS",                # array axis 0 -> Right, 1 -> Anterior, 2 -> Superior
    "sdd_mm": 1020.0,             # source -> detector
    "sid_mm": 750.0,              # source -> isocentre (= the cube centre); magnification 1020 / 750 = 1.36
    "detector_px": 128,           # detector is 128 x 128 pixels ...
    "pixel_mm": 4.0,              # ... of 4 mm: a 512 mm detector, 376 mm wide at the isocentre (2.9 mm there).
                                  # The design said 3.0 mm, but then the widest pelvises (s1390: 312 mm, magnified
                                  # 1.4-1.5x at its front) don't fit the AP image; 3.6 mm is the minimum for the
                                  # 27 local subjects, 4.0 leaves a margin for the full dataset.
    "box_mm": [480.0, 480.0, 400.0],   # the CT given to DiffDRR: this box around the cube centre, air outside the scan
    "box_voxel_mm": 2.0,          # the box is resampled to 2 mm voxels before rendering
    "mu_water": 0.02,             # 1/mm, the value used by unit 01's hu_to_mu
    # Where the X-ray source sits for each view, in the cube frame (found by rendering marker voxels, see prep.py):
    "source_mm": {"ap": [0.0, 750.0, 0.0],      # in front of the patient, beam runs Anterior -> Posterior
                  "lat": [-750.0, 0.0, 0.0]},   # on the patient's Left, beam runs Left -> Right
    # DiffDRR pose of each view: read(..., orientation="AP"), rotation (Z, X, Y euler, radians), translation (mm)
    "diffdrr": {"ap": {"rotation": [0.0, 0.0, 0.0], "translation": [0.0, 750.0, 0.0]},
                "lat": {"rotation": [1.5707963267948966, 0.0, 0.0], "translation": [0.0, 750.0, 0.0]}},
}

VIEWS = ("ap", "lat")


# ---------------------------------------------------------------- where the dataset is

def root() -> Path:
    """Folder holding index.json and the .npz files.

    Normally <processed_root>/v1. A Kaggle dataset made from the folder itself (no v1/ inside)
    is accepted too, as is any folder under /kaggle/input whose name starts with pelvis-xray3d.
    """
    base = env.processed_root()
    for cand in (base / f"v{VERSION}", base):
        if (cand / "index.json").exists():
            return cand
    if env.where() == "kaggle":
        # a dataset made from a notebook's output keeps the output's folders (e.g. processed/v1/), so search
        for idx in sorted(Path("/kaggle/input").glob("**/index.json")):
            if "pelvis-xray3d" in str(idx) and next(idx.parent.glob("s*.npz"), None) is not None:
                return idx.parent
    return base / f"v{VERSION}"


def exists() -> bool:
    return (root() / "index.json").exists()


def ensure_local(drive_zip: str = DRIVE_ZIP) -> Path:
    """Make sure the processed dataset is readable here; return its folder.

    local  : it's already on disk (built by build_dataset.ipynb); otherwise say how to get it.
    kaggle : attach the dataset `pelvis-xray3d-v1` (Add input -> Your datasets); root() finds it.
    colab  : mount Google Drive and unzip MyDrive/<drive_zip> into /content/data/processed.
    """
    if exists():
        return root()
    where = env.where()
    if where == "colab":
        from google.colab import drive  # only exists on Colab

        drive.mount("/content/drive")
        z = Path("/content/drive/MyDrive") / drive_zip
        if not z.exists():
            raise FileNotFoundError(f"{z} not found. Upload pelvis-xray3d-v1.zip there (see build_dataset.ipynb).")
        dest = env.processed_root()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"unzipping {z} -> {dest} ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)          # the zip holds v1/index.json, v1/sXXXX.npz
        if exists():
            return root()
        raise FileNotFoundError(f"unzipped, but no v1/index.json under {dest}: was the zip made from the v1 folder's parent?")
    if where == "kaggle":
        raise FileNotFoundError("Attach the Kaggle dataset 'pelvis-xray3d-v1' to this notebook (right panel -> Add input).")
    raise FileNotFoundError(f"No processed dataset at {root()}. Run units/02-shape-and-metrics/build_dataset.ipynb first.")


# ---------------------------------------------------------------- reading

def load_index() -> dict:
    """index.json as a dict (subjects, geometry, how and when it was built)."""
    return json.loads((ensure_local() / "index.json").read_text())


def geometry() -> dict:
    """The imaging geometry: the one stored in index.json if the dataset exists, else the defaults above."""
    if exists():
        return {**GEOMETRY, **load_index().get("geometry", {})}
    return dict(GEOMETRY)


def load_subject(sid: str) -> dict[str, np.ndarray]:
    """All arrays of one subject: occ, ap, lat, ap_bone, lat_bone, center_mm, femoral_heads."""
    with np.load(ensure_local() / f"{sid}.npz") as z:
        return {k: z[k] for k in z.files}


def load_split() -> dict:
    """splits/split_v1.json (written once by unit 01's split_subjects, never regenerated)."""
    path = env.repo_root() / "splits" / "split_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: it's made in unit 01, part E.")
    return json.loads(path.read_text())


def subjects(split: str | None = None) -> list[str]:
    """Subject ids in the dataset, optionally only those of one part of the split ("train", "val", "test").

    The split was made over all 1,228 ids, so with the 27 local subjects a part can be small.
    """
    ids = list(load_index()["subjects"])
    if split is None:
        return ids
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', not {split!r}")
    part = set(load_split()[split])
    return [s for s in ids if s in part]


def load_arrays(ids, keys=("occ", "ap", "lat")) -> dict[str, np.ndarray]:
    """Stack the same arrays of several subjects: {key: (N, ...)}. Handy for baselines and evaluation."""
    out = {k: [] for k in keys}
    for sid in ids:
        d = load_subject(sid)
        for k in keys:
            out[k].append(d[k])
    return {k: np.stack(v) for k, v in out.items()}


# ---------------------------------------------------------------- cameras

def voxel_centers_mm(n: int | None = None, cube_mm: float | None = None) -> np.ndarray:
    """(n, n, n, 3) centre of every cube voxel in the cube frame (mm). Plumbing for the camera checks."""
    g = geometry()
    n = n or g["cube_n"]
    cube_mm = cube_mm or g["cube_mm"]
    ax = (np.arange(n) + 0.5) * cube_mm / n - cube_mm / 2
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)


def camera(view: str):
    """K, R, t of one view, in the cube frame, such that

        uv = xray3d.geometry.project_pinhole(points_mm, K, R, t)

    gives (u, v) = (column, row) in that view's 128 x 128 image (pixel centres at integers,
    so the image centre is (63.5, 63.5)). Points are in mm, origin at the cube centre, RAS axes.

    How it was found: prep.py renders a volume holding single bright voxels at known positions with
    DiffDRR and compares where they land with this model (they agree to ~0.2 px).
    """
    g = geometry()
    view = view.lower()
    f = g["sdd_mm"] / g["pixel_mm"]                  # focal length in pixels
    c = (g["detector_px"] - 1) / 2.0                 # principal point: the middle of the detector
    K = np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])
    # rows of R = the camera's axes written in the cube frame:
    #   x_cam = direction of growing column, y_cam = direction of growing row (feet), z_cam = beam direction
    if view == "ap":
        R = np.array([[-1.0, 0, 0],      # columns run Right -> Left      (-x)
                      [0, 0, -1.0],      # rows run head -> feet          (-z)
                      [0, -1.0, 0]])     # beam runs Anterior -> Posterior (-y)
    elif view == "lat":
        R = np.array([[0, -1.0, 0],      # columns run Anterior -> Posterior (-y)
                      [0, 0, -1.0],      # rows run head -> feet             (-z)
                      [1.0, 0, 0]])      # beam runs Left -> Right           (+x)
    else:
        raise ValueError(f"view must be 'ap' or 'lat', not {view!r}")
    source = np.asarray(g["source_mm"][view], float)
    t = -R @ source                                  # the source sits at the camera's origin
    return K, R, t
