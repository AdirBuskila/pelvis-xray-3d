"""Finding subjects and loading their CT volumes and masks (TotalSegmentator v2.0.1).

Layout on disk (the small subset, unzipped by data/scripts or fetched by `fetch_subjects`):

    <data_root>/raw/totalsegmentator_small/extracted/
        s1390/
            ct.nii.gz                    the CT, in Hounsfield units (int16)
            segmentations/
                hip_left.nii.gz          one 0/1 mask per structure, same grid as ct.nii.gz
                hip_right.nii.gz
                sacrum.nii.gz
                femur_left.nii.gz  ...   (117 structures in total)

There is no single label map with label IDs: every structure is its own mask file.
License: CC BY 4.0, Wasserthal et al. 2023. Cite it wherever a figure shows this data.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from . import env
from .remote_zip import open_remote_zip

SMALL_SUBSET_URL = "https://zenodo.org/records/10047263/files/Totalsegmentator_dataset_small_v201.zip"
FULL_DATASET_URL = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"

# The bones that make up "the pelvis" in this project, and the neighbours we sometimes show.
PELVIS = ("hip_left", "hip_right", "sacrum")
FEMURS = ("femur_left", "femur_right")

# Small-subset subjects with a complete pelvis inside the scan and no metal near it, best
# femur coverage first. Produced by data/scripts/screen_local.py on 2026-11-11 (102 subjects ->
# 77 with a left hip, 75 with all three bones -> 47 complete -> these 27). s1390 is the "hero pelvis" of the videos.
GOOD_SUBJECTS = (
    "s1390", "s1361", "s1387", "s1383", "s1400", "s1379", "s1391", "s1348", "s1287",
    "s0250", "s1086", "s1267", "s1364", "s1336", "s1373", "s0629", "s1405", "s0915",
    "s0928", "s0461", "s0878", "s0344", "s0950", "s1358", "s1365", "s1366", "s0726",
)
HERO = "s1390"


def subjects_dir() -> Path:
    return env.data_root() / "raw" / "totalsegmentator_small" / "extracted"


def list_subjects() -> list[str]:
    """Subject ids present on disk (folders named sXXXX)."""
    d = subjects_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("s"))


def all_subject_ids() -> list[str]:
    """All 1228 subject ids of the full dataset (committed in splits/subjects_v201.txt)."""
    path = env.repo_root() / "splits" / "subjects_v201.txt"
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def screening() -> dict | None:
    """Results of data/scripts/screen_local.py (complete pelvis? metal?), if it was run here."""
    path = env.data_root() / "raw" / "totalsegmentator_small" / "screen.json"
    return json.loads(path.read_text()) if path.exists() else None


def ct_path(sid: str) -> Path:
    return subjects_dir() / sid / "ct.nii.gz"


def mask_path(sid: str, structure: str) -> Path:
    return subjects_dir() / sid / "segmentations" / f"{structure}.nii.gz"


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Open a NIfTI file. Nothing is read from disk until you ask for the array."""
    return nib.load(str(path))


def load_ct(sid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (hu, affine): the CT as an int16 array in Hounsfield units, and its 4x4 affine.

    The affine maps a voxel index (i, j, k, 1) to a position in millimetres (x, y, z, 1).
    """
    img = load_nifti(ct_path(sid))
    return np.asanyarray(img.dataobj), img.affine


def load_mask(sid: str, structures: str | tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    """Return {structure: boolean mask} for one or more structures of a subject."""
    if isinstance(structures, str):
        structures = [structures]
    return {s: np.asanyarray(load_nifti(mask_path(sid, s)).dataobj) > 0 for s in structures}


def axcodes(affine: np.ndarray) -> tuple[str, str, str]:
    """Which anatomical direction each array axis points to, e.g. ('R', 'A', 'S').

    ('R', 'A', 'S') means: index i grows towards the patient's Right, j towards Anterior
    (the front), k towards Superior (the head). The opposite letters are L, P, I.
    """
    return tuple(nib.aff2axcodes(affine))


def fetch_subjects(ids, structures=PELVIS + FEMURS, url: str = SMALL_SUBSET_URL) -> None:
    """Download a few subjects (CT + the listed masks) straight out of the remote zip.

    For Colab/Kaggle, where the 3.2 GB zip isn't on disk. About 30-60 MB per subject.
    Files already present are skipped, so calling it twice is cheap.
    """
    wanted = set()
    for sid in ids:
        wanted.add(f"{sid}/ct.nii.gz")
        wanted.update(f"{sid}/segmentations/{s}.nii.gz" for s in structures)
    missing = {m for m in wanted if not (subjects_dir() / m).exists()}
    if not missing:
        print("all files already on disk")
        return
    z = open_remote_zip(url)
    for info in z.infolist():
        if info.filename in missing:
            dst = subjects_dir() / info.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as fo:
                fo.write(src.read())
            print(f"got {info.filename}  ({info.file_size / 1e6:.1f} MB)")
