"""Real fluoroscopy: the DeepFluoro cadaver dataset, brought into OUR network's input geometry (unit 07).

Written by Claude (infrastructure). The concept code it calls is the students':
    xray3d.physics.hu_to_mu                        (unit 01)  HU -> attenuation, for every DRR made here
    xray3d.shape.bbox_center_mm, occupancy         (unit 02)  where the cube goes, the 64^3 ground truth
    xray3d.shape.cube_grid, resample (via prep)    (unit 02)  the CT box for the simulated AP + LAT
    xray3d.realxray.log_transform                  (unit 07)  detector intensity -> line integral
LICENCE: DeepFluoro is CC BY-NC-ND 4.0 (Grupp et al. 2020). NOTHING derived from it may be committed: no
figures with outputs, no arrays. Everything this module writes goes to cache_dir() (under the data folder,
which git ignores). Only aggregate numbers go to results/. Its .pt files are pickles: always loaded with
torch.load(..., weights_only=True), so a file can't run code.

What is on disk (the `deepfluoro/` folder of the Hugging Face dataset eigenvivek/xvr-data):
    data/raw/deepfluoro/subject01 .. subject06/
        volume.nii.gz        the cadaver's CT (1 mm voxels, array axes L, P, S; nibabel gives RAS world mm)
        mask.nii.gz          3D labels: 1 L-hemipelvis, 2 R-hemipelvis, 3 vertebrae, 4 upper sacrum,
                             5 L-femur, 6 R-femur, 7 lower sacrum   (checked: 1 and 5 lie on the patient's left)
        xrays/NNN.dcm        one real fluoroscopy frame: 1536 x 1536 uint16, 0.194 mm pixels, sdd 1020 mm
        xrays/NNN.pt         {"pose": (1, 4, 4) DiffDRR pose, "intrinsics": {sdd, delx, dely, x0, y0,
                              height=width=1436}}: the frame's camera, found by the dataset's authors with an
                              offline 2D/3D registration. 1436 = the 1536 frame minus 50 px on every side
                              (the collimator's edge), the crop xvr uses ("--crop 100").
        segmentations_multichannel/NNN.pt   2D labels of that frame, bit i = label i + 1 (overlaps allowed)

How the camera is used (found by reading xvr's evaluation code, then checked here against the 2D labels):
    read(volume, orientation="AP") with the volume centred at the world origin (DiffDRR's default),
    DRR(..., reverse_x_axis=False), intrinsics from the .pt file. Every frame's source sits BEHIND the
    cadaver (posterior): the real views are PA, not AP, and the most oblique ones are ~40 deg off.

The geometry gap (what this module can and can't match), see `virtual_ap`:
    our network was trained on AP images: source 750 mm in FRONT of the pelvis centre, detector 1020 mm from
    the source, 128 px of 4 mm (376 mm at the centre). A real frame differs in four ways:
    1. direction: the C-arm isn't exactly along the CT's A-P axis.      -> fixed exactly: the cube's axes are
       turned so its "anterior" axis lies on the line source -> pelvis centre (the ground truth turns too).
    2. side: the source is posterior (PA).                              -> approximated: the image is mirrored
       through the coronal plane of the pelvis centre. Exact for a flat object; for depth it inverts which
       side is magnified more (back instead of front), a few % across the pelvis.
    3. distance: source -> pelvis centre is d (650-830 mm), not 750.    -> approximated: the image is scaled by
       d / 750, exact at the centre's depth, ~1 % off 100 mm in front of or behind it.
    4. field of view: the real detector is 278 mm wide (~205 mm at the pelvis): it covers about the central
       60-75 of our 128 px. Nothing can fix that: the outside is marked invalid and filled.
    Rotations about the source are exactly a homography, so 1 costs nothing; 2-4 are written down in
    `info` per specimen.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from . import dataset, env

LABELS = {1: "hip_left", 2: "hip_right", 3: "vertebrae", 4: "sacrum_upper", 5: "femur_left", 6: "femur_right",
          7: "sacrum_lower"}
PELVIS_LABELS = (1, 2, 4, 7)       # = TotalSegmentator's hip_left | hip_right | sacrum (our target)
FEMUR_LABELS = {"left": 5, "right": 6}
FRAME_PX = 1536                    # a DICOM frame is 1536 x 1536 ...
CROP = 50                          # ... minus 50 px on every side = the 1436 x 1436 detector the poses describe
BIN = 4                            # 1436 -> 359 px of 0.776 mm: plenty for our 4 mm pixels, 16x fewer rays
# frames whose ground-truth pose the dataset's authors flag as questionable (xvr's evaluate.py, issue #8)
EXCLUDE = {("subject01", "003"), ("subject01", "050"), ("subject04", "002"), ("subject04", "004")}
HF_BASE = "https://huggingface.co/datasets/eigenvivek/xvr-data/resolve/main/deepfluoro"
HF_TREE = "https://huggingface.co/api/datasets/eigenvivek/xvr-data/tree/main/deepfluoro"
VERSION = 1                        # bump to rebuild every cached specimen


# ---------------------------------------------------------------------------------------------- where things are

def root() -> Path:
    """data/raw/deepfluoro (or <XRAY3D_DATA>/raw/deepfluoro); on Kaggle /tmp/deepfluoro.

    Kaggle: env.data_root() is under /kaggle/working, and everything in /kaggle/working becomes the notebook's
    published Output. DeepFluoro is CC BY-NC-ND (no redistribution), so its raw files go to /tmp instead, which
    Kaggle doesn't keep (fetch() downloads them again in a new session, ~400 MB).
    """
    if env.where() == "kaggle":
        return Path("/tmp/deepfluoro")
    return env.data_root() / "raw" / "deepfluoro"


def cache_dir() -> Path:
    """Where the per-specimen arrays built here are kept. Never inside git.

    <processed root>/deepfluoro_v1, except on Kaggle, where the processed root is the read-only input
    dataset: there /tmp/deepfluoro_v1. Not /kaggle/working: whatever is there becomes the notebook's Output,
    which can be published, and the licence forbids sharing derived data. /tmp is not kept between sessions,
    so on Kaggle the specimens are rebuilt (~40 s each) in every new session.
    """
    if env.where() == "kaggle":
        return Path("/tmp") / f"deepfluoro_v{VERSION}"
    return env.processed_root() / f"deepfluoro_v{VERSION}"


def specimens() -> list[str]:
    """The specimens on disk: ['subject01', ...] (those with a CT)."""
    return sorted(p.name for p in root().glob("subject*") if (p / "volume.nii.gz").exists())


def frames(specimen: str) -> list[str]:
    """The frame ids of one specimen that have a pose: ['000', '001', ...]."""
    return sorted(p.stem for p in (root() / specimen / "xrays").glob("*.pt"))


def fetch(specimen_ids=None, verbose: bool = True) -> None:
    """Download what unit 07 needs from Hugging Face (Colab/Kaggle; locally it's already there).

    Per specimen: the CT and its mask (~30-70 MB), every frame's small pose file, and only then the ONE frame
    that unit 07 uses (choose_frame) with its 2D labels. Skips files that exist. The licence lets you download
    and use the data for non-commercial research; it does not let you publish anything derived from it.
    """
    import urllib.parse

    def get(rel: str) -> Path:
        dst = root() / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(f"{HF_BASE}/{urllib.parse.quote(rel)}", tmp)
            tmp.replace(dst)
        return dst

    ids = specimen_ids or [f"subject{i:02d}" for i in range(1, 7)]
    for sp in ids:
        t0 = time.time()
        get(f"{sp}/volume.nii.gz")
        get(f"{sp}/mask.nii.gz")
        with urllib.request.urlopen(f"{HF_TREE}/{sp}/xrays") as r:
            names = [Path(e["path"]).name for e in json.loads(r.read()) if e["path"].endswith(".pt")]
        for n in names:
            get(f"{sp}/xrays/{n}")
        fr, _ = choose_frame(sp)
        get(f"{sp}/xrays/{fr}.dcm")
        get(f"{sp}/segmentations_multichannel/{fr}.pt")
        if verbose:
            print(f"{sp}: CT, mask, {len(names)} poses, frame {fr}  ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------- loading

def load_ct(specimen: str) -> tuple[np.ndarray, np.ndarray]:
    """(hu, affine): the CT in HU (float32) and its voxel -> RAS world (mm) affine."""
    img = nib.load(root() / specimen / "volume.nii.gz")
    return np.asanyarray(img.dataobj).astype(np.float32), np.asarray(img.affine, float)


def load_mask(specimen: str) -> np.ndarray:
    """The 3D label volume (uint8, same grid as the CT; label numbers in LABELS)."""
    return np.asanyarray(nib.load(root() / specimen / "mask.nii.gz").dataobj).astype(np.uint8)


def load_pose(specimen: str, frame: str) -> tuple[np.ndarray, dict]:
    """(pose (4, 4), intrinsics dict) of one frame. weights_only=True: the file is data, not code."""
    import torch

    d = torch.load(root() / specimen / "xrays" / f"{frame}.pt", weights_only=True)
    intr = {k: (float(v) if not isinstance(v, int) else v) for k, v in d["intrinsics"].items()}
    return d["pose"][0].double().numpy(), intr


def load_raw(specimen: str, frame: str) -> np.ndarray:
    """The frame's detector values, (1536, 1536) float32: brighter = MORE photons arrived (bone is dark)."""
    import pydicom   # pip install pydicom (only unit 07 needs it)

    return pydicom.dcmread(root() / specimen / "xrays" / f"{frame}.dcm").pixel_array.astype(np.float32)


def load_labels2d(specimen: str, frame: str, labels=PELVIS_LABELS) -> np.ndarray:
    """(1536, 1536) bool: the union of the given 2D labels of a frame (from the multichannel file)."""
    import torch

    packed = torch.load(root() / specimen / "segmentations_multichannel" / f"{frame}.pt", weights_only=True)
    packed = packed.squeeze().numpy().astype(np.uint8)          # some files are (1, 1, H, W)
    out = np.zeros(packed.shape, bool)
    for lab in labels:
        out |= ((packed >> (lab - 1)) & 1).astype(bool)          # bit i is label i + 1
    return out


def crop(img: np.ndarray) -> np.ndarray:
    """1536 x 1536 frame -> the 1436 x 1436 detector the poses describe."""
    return img[CROP:FRAME_PX - CROP, CROP:FRAME_PX - CROP]


def bin_image(img: np.ndarray, k: int = BIN) -> np.ndarray:
    """Average k x k blocks (a detector with k-times bigger pixels)."""
    h, w = img.shape[0] // k, img.shape[1] // k
    return img[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))


def dark_flat(raw: np.ndarray) -> tuple[float, float]:
    """Estimates of the detector's dark level and flat (unattenuated) level for one frame.

    dark: what a pixel reads with NO photons. The collimator blocks the beam completely at the frame's outer
          edge, and those pixels all read the same small value: the frame's minimum.
    flat: what a pixel reads with nothing in the beam (I0). A pelvis frame has almost no open field, so the
          99.9th percentile of the detector area is the best we have. It is a guess, and that's fine: an error
          in flat only adds a constant to p = -log(...), and standardise() removes constants.
    """
    det = crop(raw)
    return float(raw.min()), float(np.percentile(det, 99.9))


# ---------------------------------------------------------------------------------------------- cameras

# DiffDRR's read(..., orientation="AP") reorientation (diffdrr/data.py): copied, and asserted equal when rendering
_AP_REORIENT = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def volume_center(affine: np.ndarray, shape) -> np.ndarray:
    """World position (mm) of the middle of the CT array: DiffDRR moves this point to its world origin."""
    return (np.asarray(affine, float) @ np.r_[(np.asarray(shape, float)[:3] - 1) / 2, 1.0])[:3]


def _detector(intr: dict, binning: int = BIN):
    """DiffDRR's Detector for a frame, with binned pixels (no CT needed: it only makes rays)."""
    import torch
    from diffdrr.detector import Detector

    h = int(intr["height"]) // binning
    return Detector(float(intr["sdd"]), h, h, float(intr["delx"]) * binning, float(intr["dely"]) * binning,
                    float(intr["x0"]), float(intr["y0"]), torch.tensor(_AP_REORIENT, dtype=torch.float32),
                    reverse_x_axis=False)


def frame_camera(pose: np.ndarray, intr: dict, iso_mm: np.ndarray, binning: int = BIN) -> dict:
    """The real camera of one frame, as plain linear algebra in the CT's RAS world (mm).

    Returns {"P": (3, 4) projection matrix  world -> (column, row) of the cropped, binned frame (pixel centres at
    integers), "source": (3,), "size": pixels per side}. Found from DiffDRR's own rays, so it agrees with its
    renderer by construction: every pixel (i, j) has a ray from the source S to a point T(i, j) on the detector,
    and T is affine in (i, j): T = T00 + j a + i b. A point X then lies on the ray of (j, i) when
    X - S = lam * [a b T00-S] (j, i, 1), so  (j, i, 1) ~ M^-1 (X - S)  with  M = [a, b, T00 - S].
    """
    import torch
    from diffdrr.pose import RigidTransform

    det = _detector(intr, binning)
    src, tgt = det(RigidTransform(torch.tensor(pose[None], dtype=torch.float32)), None)
    n = det.height
    S = src[0, 0].double().numpy() + iso_mm                            # DiffDRR's world = ours - iso
    T = tgt[0].double().numpy().reshape(n, n, 3) + iso_mm               # T[i, j]: row i, column j
    a, b = T[0, 1] - T[0, 0], T[1, 0] - T[0, 0]
    M = np.stack([a, b, T[0, 0] - S], axis=1)
    P = np.linalg.inv(M) @ np.hstack([np.eye(3), -S[:, None]])
    return {"P": P, "source": S, "size": n}


def project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N, 3) world points -> (N, 2) (column, row) with a 3 x 4 projection matrix."""
    x = np.asarray(pts, float) @ P[:, :3].T + P[:, 3]
    return x[:, :2] / x[:, 2:3]


def beam_angle_deg(source: np.ndarray, target: np.ndarray) -> float:
    """Angle (deg) between the line source -> target and the CT's anterior-posterior axis (either way)."""
    d = np.asarray(target, float) - np.asarray(source, float)
    return float(np.degrees(np.arccos(min(1.0, abs(d[1]) / np.linalg.norm(d)))))


def frame_table(specimen: str) -> list[dict]:
    """Every usable frame of a specimen: {"frame", "angle_deg", "coverage"} (skips EXCLUDE).

    angle_deg : between the line source -> pelvis centre and the CT's A-P axis
    coverage  : the fraction of the pelvis (every 50th labelled voxel) that projects onto the real detector.
                The C-arm was often aimed at one hip, so the straightest frame can miss half of the pelvis.
    """
    img = nib.load(root() / specimen / "volume.nii.gz")
    aff = np.asarray(img.affine, float)
    iso = volume_center(aff, img.shape)
    idx = np.argwhere(np.isin(load_mask(specimen), PELVIS_LABELS))[::50]
    pts = idx @ aff[:3, :3].T + aff[:3, 3]
    out = []
    for fr in frames(specimen):
        if (specimen, fr) in EXCLUDE:
            continue
        pose, intr = load_pose(specimen, fr)
        cam = frame_camera(pose, intr, iso)
        uv = project(cam["P"], pts)
        inside = (uv >= 0).all(axis=1) & (uv <= cam["size"] - 1).all(axis=1)
        out.append({"frame": fr, "angle_deg": beam_angle_deg(cam["source"], pts.mean(axis=0)),
                    "coverage": float(inside.mean())})
    return out


def choose_frame(specimen: str, max_angle_deg: float = 12.0) -> tuple[str, float]:
    """(frame id, angle in deg): the near-AP frame that sees the most pelvis.

    Among the frames within max_angle_deg of the A-P axis, the one with the highest coverage (frame_table);
    if there is none, the frame with the smallest angle. All DeepFluoro frames are PA (source behind the
    cadaver), so 'near AP' here means 'near a straight PA'.
    """
    rows = frame_table(specimen)
    near = [r for r in rows if r["angle_deg"] <= max_angle_deg]
    best = max(near, key=lambda r: r["coverage"]) if near else min(rows, key=lambda r: r["angle_deg"])
    return best["frame"], best["angle_deg"]


# ---------------------------------------------------------------------------------------------- the virtual AP camera

def cube_axes(source: np.ndarray, center: np.ndarray) -> np.ndarray:
    """(3, 3) Q whose columns are the cube's R, A, S axes in the world, for a real source behind the patient.

    A (column 1) points from the real source through the pelvis centre (out of the patient's front);
    R is the world's +x made perpendicular to it; S = R x A. With this cube, our AP camera looks along the
    real beam's line, from the front.
    """
    n = np.asarray(center, float) - np.asarray(source, float)
    n /= np.linalg.norm(n)
    if n[1] < 0:              # the source is in FRONT (an AP frame): A must still point to the front
        n = -n
    ex = np.array([1.0, 0, 0]) - n[0] * n
    ex /= np.linalg.norm(ex)
    return np.stack([ex, n, np.cross(ex, n)], axis=1)


def world_to_cube(Q: np.ndarray, center: np.ndarray) -> np.ndarray:
    """4 x 4 T with  x_cube = T @ x_world: origin at the centre, axes Q. `T @ ct_affine` = the CT in cube mm."""
    T = np.eye(4)
    T[:3, :3] = Q.T
    T[:3, 3] = -Q.T @ np.asarray(center, float)
    return T


def virtual_ap(P_real: np.ndarray, source: np.ndarray, center: np.ndarray, Q: np.ndarray) -> dict:
    """The homography from OUR AP image (128 px of 4 mm) to the real frame, and what it had to approximate.

    Our camera (dataset.camera("ap") in the cube frame Q, origin `center`) sits 750 mm in front of the centre on
    the line through the real source. The real source sits d mm BEHIND it on the same line. Steps:
      * mirror the world through the plane through the centre perpendicular to that line: the real source
        lands d mm in front (S'). The real camera seen through the mirror, P' = P_real o mirror, shows the
        mirrored patient exactly as the real frame shows the real one. Approximation 1 (PA -> AP): we pretend
        the mirrored patient is the patient (it is, except front and back swap places).
      * a camera at S' with our rotation differs from P' only by a rotation about the shared source: exactly a
        homography.
      * moving from S' (d mm) to our 750 mm: approximation 2, a zoom by d / 750 (exact at the centre's depth).
    Returns {"H": (3, 3) our (column, row, 1) -> real (column, row, 1), "d_mm", "zoom", "angle_deg"}.
    """
    g = dataset.geometry()
    center = np.asarray(center, float)
    n = Q[:, 1]                                                  # cube's anterior axis, in the world
    d = float(np.linalg.norm(center - source))
    F = np.eye(3) - 2 * np.outer(n, n)                           # the mirror (a reflection's linear part)
    K, R_ap, _ = dataset.camera("ap")
    R_w = R_ap @ Q.T                                             # world -> our camera's axes
    K0 = K.copy()
    K0[0, 0] = K0[1, 1] = K[0, 0] * d / g["sid_mm"]              # the zoom: a camera at d mm seeing like ours at 750
    H = P_real[:, :3] @ F @ R_w.T @ np.linalg.inv(K0)
    return {"H": H, "d_mm": d, "zoom": d / g["sid_mm"], "angle_deg": beam_angle_deg(source, center)}


def warp(img: np.ndarray, H: np.ndarray, n: int | None = None, sub: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Resample a real-detector image into our n x n image through H; returns (image, valid).

    Each of our pixels averages sub x sub samples (bilinear) spread over its area: our 4 mm pixels are ~5 real
    (binned) pixels wide, and point-sampling would alias. valid = every sample fell on the real detector.
    """
    from scipy import ndimage

    n = n or dataset.geometry()["detector_px"]
    off = (np.arange(sub) + 0.5) / sub - 0.5
    u = (np.arange(n)[:, None] + off[None, :]).ravel()                 # sub-columns
    uu, vv = np.meshgrid(u, u, indexing="xy")                          # uu: column, vv: row (n*sub, n*sub)
    xyz = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]).T @ H.T
    col, row = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
    m = img.shape[0]
    inside = (col >= 0) & (col <= m - 1) & (row >= 0) & (row <= m - 1)
    vals = ndimage.map_coordinates(np.asarray(img, np.float64), [row, col], order=1, mode="nearest")
    vals = np.where(inside, vals, np.nan).reshape(n, sub, n, sub)
    valid = np.isfinite(vals).all(axis=(1, 3))
    out = np.where(valid, np.nan_to_num(vals).mean(axis=(1, 3)), 0.0)   # valid pixels have all sub x sub samples
    return out.astype(np.float32), valid


def fill_outside(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Our images outside the real field of view: the MEAN of the valid pixels (0 after standardise()).

    A choice, not a fact: the network never saw a cut-off image. Any constant is as made-up as another; the
    mean at least doesn't add an edge of its own to the contrast. Applied the same way to every input that
    is compared with a real frame, so the comparison stays fair.
    """
    out = np.array(img, np.float32, copy=True)
    out[~valid] = float(out[valid].mean()) if valid.any() else 0.0
    return out


# ---------------------------------------------------------------------------------------------- rendering

def render_at_pose(hu: np.ndarray, specimen: str, pose: np.ndarray, intr: dict, binning: int = BIN) -> np.ndarray:
    """DRR of the WHOLE CT on the real frame's detector at its real pose: (1436/binning)^2 line integrals p.

    Same recipe as prep.render_views: YOUR hu_to_mu replaces DiffDRR's own density, so the output is the
    physical line integral p = sum(mu * length in mm). ~10-20 s on a laptop CPU.
    """
    import torch
    from diffdrr.data import read
    from diffdrr.drr import DRR
    from diffdrr.pose import RigidTransform
    from torchio import ScalarImage

    from .prep import _student

    hu_to_mu = _student("physics", "hu_to_mu")
    subject = read(str(root() / specimen / "volume.nii.gz"), None, None, "AP")   # centred at the origin
    assert np.allclose(subject.reorient.numpy(), _AP_REORIENT), "DiffDRR's AP reorientation changed"
    mu = torch.from_numpy(np.ascontiguousarray(hu_to_mu(hu, dataset.geometry()["mu_water"]), np.float32))
    subject.density = ScalarImage(tensor=mu[None], affine=subject.volume.affine)
    h = int(intr["height"]) // binning                       # the same detector as _detector(intr, binning)
    drr = DRR(subject, sdd=float(intr["sdd"]), height=h, delx=float(intr["delx"]) * binning,
              dely=float(intr["dely"]) * binning, x0=float(intr["x0"]), y0=float(intr["y0"]), reverse_x_axis=False)
    with torch.no_grad():
        img = drr(RigidTransform(torch.tensor(pose[None], dtype=torch.float32)))
    return img[0, 0].numpy().astype(np.float32)


# ---------------------------------------------------------------------------------------------- one specimen

def build_specimen(specimen: str, log_transform, frame: str | None = None, verbose: bool = True) -> dict:
    """Everything unit 07 needs about one specimen, in the network's geometry. ~30-60 s on a laptop CPU.

    log_transform : YOUR realxray.log_transform(raw, dark, flat) (passed in, so it works before it's moved)
    Returns a dict of arrays (cube frame of this specimen; all images 128 x 128 in our AP/LAT orientation):
        occ           (64, 64, 64) uint8  ground truth: DeepFluoro labels 1, 2, 4, 7 in the cube (YOUR occupancy)
        femoral_heads (2, 4)              [x, y, z, r] mm in the cube frame, rows (left, right) (for ce_angle)
        real_ap       the real frame (log-transformed with YOUR function), warped; 0 outside `valid`
        drr_ap        a DRR at the real pose on the real detector (whole CT), warped the same way
        sim_ap, sim_lat, sim_ap_bone, sim_lat_bone   prep.render_views in the cube: exact AP / LAT geometry
        valid         bool, where the real detector covers our AP image
        labels2d      the frame's own 2D pelvis labels (from the dataset), warped (> 0.5 of the pixel)
        raw_small     the cropped real frame binned to 359 px (display only), drr_small: the DRR there
    and "info" (json-able): frame, angle, distances, zoom, the fraction of our image the real frame covers.
    """
    from .prep import _student, build_box, femoral_head, render_views

    bbox_center_mm, occupancy = _student("shape", "bbox_center_mm"), _student("shape", "occupancy")
    t0 = time.time()
    g = dataset.geometry()
    fr, ang_frame = (frame, float("nan")) if frame else choose_frame(specimen)
    hu, aff = load_ct(specimen)
    lab = load_mask(specimen)
    pelvis = np.isin(lab, PELVIS_LABELS)
    pose, intr = load_pose(specimen, fr)
    iso = volume_center(aff, hu.shape)
    cam = frame_camera(pose, intr, iso)

    # the cube: centred on the pelvis's bounding-box centre measured IN the cube's own axes, which depend on the
    # centre (A points from the source through it). Three rounds of "axes from centre, centre from axes" converge.
    center = np.asarray(bbox_center_mm(pelvis, aff), float)
    for _ in range(3):
        Q = cube_axes(cam["source"], center)
        c_cube = np.asarray(bbox_center_mm(pelvis, world_to_cube(Q, center) @ aff), float)
        center = center + Q @ c_cube
    Q = cube_axes(cam["source"], center)
    T = world_to_cube(Q, center)
    A = T @ aff                                               # CT voxel -> cube-frame mm
    zero = np.zeros(3)
    occ = np.asarray(occupancy(pelvis, A, zero, g["cube_mm"], g["cube_n"])).astype(np.uint8)
    heads = np.full((2, 4), np.nan, np.float32)
    for r, side in enumerate(("left", "right")):
        c, rad = femoral_head(lab == FEMUR_LABELS[side], A)
        heads[r, :3], heads[r, 3] = c, rad

    # the simulated pair, exactly like the training data (prep.build_box + render_views, in the turned cube)
    mu_box, bone_box = build_box(hu, pelvis, A, zero)
    sim, sim_bone = render_views(mu_box), render_views(bone_box)
    del mu_box, bone_box

    # the real frame and the DRR at its pose, on the real detector, then into our AP image
    raw = load_raw(specimen, fr)
    dark, flat = dark_flat(raw)
    p_real = bin_image(np.asarray(log_transform(crop(raw), dark, flat), np.float64))
    p_drr = render_at_pose(hu, specimen, pose, intr)
    vap = virtual_ap(cam["P"], cam["source"], center, Q)
    real_ap, valid = warp(p_real, vap["H"])
    drr_ap, valid_d = warp(p_drr, vap["H"])
    lab2d, _ = warp(bin_image(crop(load_labels2d(specimen, fr)).astype(np.float32)), vap["H"])
    valid &= valid_d

    info = {"specimen": specimen, "frame": fr, "frame_angle_deg": round(ang_frame, 2),
            "cube_angle_deg": round(vap["angle_deg"], 2), "source_to_center_mm": round(vap["d_mm"], 1),
            "zoom": round(vap["zoom"], 4), "valid_fraction": round(float(valid.mean()), 4),
            "dark": dark, "flat": flat, "occ_voxels": int(occ.sum()),
            "occ_touches_cube": bool(occ[[0, -1]].any() or occ[:, [0, -1]].any() or occ[:, :, [0, -1]].any()),
            "seconds": round(time.time() - t0, 1), "version": VERSION}
    arrays = {"occ": occ, "femoral_heads": heads, "valid": valid,
              "real_ap": np.where(valid, real_ap, 0).astype(np.float32),
              "drr_ap": np.where(valid, drr_ap, 0).astype(np.float32),
              "labels2d": (lab2d > 0.5) & valid,
              "sim_ap": sim["ap"], "sim_lat": sim["lat"], "sim_ap_bone": sim_bone["ap"], "sim_lat_bone": sim_bone["lat"],
              "raw_small": bin_image(crop(raw)).astype(np.float32), "drr_small": p_drr,
              "cube_axes": Q, "center_mm": center}
    if verbose:
        print(f"{specimen}: frame {fr} ({info['cube_angle_deg']:.1f} deg from the CT's A-P axis), source "
              f"{info['source_to_center_mm']:.0f} mm behind the pelvis centre (zoom {info['zoom']:.3f}), real field "
              f"covers {100 * info['valid_fraction']:.0f} % of our AP image  ({info['seconds']:.0f} s)")
    return {**arrays, "info": info}


def _transform_name(fn) -> str:
    """The name the cache records for a log transform: 'log_transform' for yours, '<lambda>' for any lambda."""
    return getattr(fn, "__name__", type(fn).__name__)


def load_specimen(specimen: str, log_transform=None, rebuild: bool = False, verbose: bool = True) -> dict:
    """build_specimen, cached in cache_dir()/<specimen>.npz (+ .json). DeepFluoro-derived: never commit these.

    The cache stores the log_transform's OUTPUT, and its info records the function's NAME. A cache built by a
    differently named function (e.g. unit 07's `lambda raw, dark, flat: -raw` experiment, named '<lambda>') is
    treated as stale and rebuilt, so a normal load never silently returns the -raw images. The name can't tell
    two versions of the same function apart: after editing your log_transform (or swapping one lambda for
    another) pass rebuild=True.
    """
    d = cache_dir()
    npz, js = d / f"{specimen}.npz", d / f"{specimen}.json"
    # log_transform=None means "the students' realxray.log_transform": compare by name without importing it,
    # so a cached specimen still loads before the function has been moved into src/
    want = "log_transform" if log_transform is None else _transform_name(log_transform)
    if npz.exists() and js.exists() and not rebuild:
        info = json.loads(js.read_text())
        # a cache from before the name was recorded has no "log_transform" key: stale too (we can't tell)
        if info.get("version") == VERSION and info.get("log_transform") == want:
            with np.load(npz) as z:
                return {**{k: z[k] for k in z.files}, "info": info}
    if log_transform is None:
        from .prep import _student
        log_transform = _student("realxray", "log_transform")
    out = build_specimen(specimen, log_transform, verbose=verbose)
    out["info"]["log_transform"] = _transform_name(log_transform)    # what built these arrays (see above)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **{k: v for k, v in out.items() if k != "info"})
    js.write_text(json.dumps(out["info"], indent=1))
    return out


# ---------------------------------------------------------------------------------------------- network inputs

INPUTS = {   # name -> (AP image key, whether the real field-of-view cut is applied)
    "sim_full": ("sim_ap", False),      # what the network was trained on: exact AP geometry, full field
    "sim_cut":  ("sim_ap", True),       # the same image, cut to the real frame's field of view
    "drr_pose": ("drr_ap", True),       # a DRR at the REAL pose, warped like the real frame (geometry approximated)
    "real":     ("real_ap", True),      # the real frame, warped
}


def ap_image(spec: dict, name: str) -> np.ndarray:
    """One of the four AP inputs of INPUTS (128 x 128 line integrals, before standardisation)."""
    key, cut = INPUTS[name]
    img = spec[key]
    return fill_outside(img, spec["valid"]) if cut else np.asarray(img, np.float32)


def network_input(spec: dict, name: str, n_views: int):
    """(1, V, 128, 128) torch tensor for the network: the AP input `name` (+ the simulated LAT if V = 2),
    each standardised exactly as in training (torchdata.standardise)."""
    import torch

    from . import torchdata

    imgs = [ap_image(spec, name)] + ([spec["sim_lat"]] if n_views == 2 else [])
    return torch.from_numpy(torchdata.standardise(np.stack(imgs)))[None]
