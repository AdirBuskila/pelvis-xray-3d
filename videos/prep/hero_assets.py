"""Build the shared "hero pelvis" assets used across the series, from one TotalSegmentator subject.

Usage: python hero_assets.py <subject_dir> <out_dir>

Scene coordinates (three.js, y up): x = patient Right, y = Superior, z = Posterior.
A camera at -z looking +z therefore sees the patient from the front with the patient's right
on the viewer's left, the radiology AP convention. (R, S, P) is right-handed, so no winding flip.
1 mm = 0.01 scene units.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure, filters
from PIL import Image

subj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seg = subj / "segmentations"

img = nib.load(subj / "ct.nii.gz")
assert nib.aff2axcodes(img.affine) == ("R", "A", "S"), nib.aff2axcodes(img.affine)
sp = np.array(img.header.get_zooms()[:3], dtype=float)           # mm per voxel
ct = np.asanyarray(img.dataobj).astype(np.float32)

def load(name):
    return np.asanyarray(nib.load(seg / f"{name}.nii.gz").dataobj) > 0

parts = {
    "hip_left": load("hip_left"), "hip_right": load("hip_right"), "sacrum": load("sacrum"),
    "femur_left": load("femur_left"), "femur_right": load("femur_right"),
    "spine": load("vertebrae_L5") | load("vertebrae_L4") | load("vertebrae_L3"),
}
pelvis = parts["hip_left"] | parts["hip_right"] | parts["sacrum"]

# region of interest: everything we show, plus a margin
allm = np.zeros_like(pelvis)
for m in parts.values(): allm |= m
idx = np.argwhere(allm)
lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(ct.shape) - 1)
sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
ct, pelvis = ct[sl], pelvis[sl]
parts = {k: v[sl] for k, v in parts.items()}
center_mm = (np.array(ct.shape) * sp) / 2.0
pc = np.argwhere(pelvis).mean(0) * sp                              # pelvis centroid, mm
SCALE = 0.01

def to_scene(p_mm):
    """voxel-space mm (R, A, S axes) -> scene (x=R, y=S, z=P), centred on the pelvis"""
    q = p_mm - pc
    return np.stack([q[:, 0], q[:, 2], -q[:, 1]], axis=1) * SCALE

def write_obj(path, mask, sigma=1.1, step=1):
    if not mask.any():
        print("WARNING: empty mask, skipped", path.name); return 0
    m = filters.gaussian(mask.astype(np.float32), sigma=sigma)
    v, f, n, _ = measure.marching_cubes(m, level=0.5, spacing=tuple(sp), step_size=step)
    xyz = to_scene(v)
    nrm = np.stack([n[:, 0], n[:, 2], -n[:, 1]], axis=1)
    with open(path, "w") as fo:
        fo.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in xyz)
        fo.writelines(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in nrm)
        # marching_cubes winding faces outward in (i,j,k); the z flip mirrors, so reverse winding
        fo.writelines(f"f {a+1}//{a+1} {c+1}//{c+1} {b+1}//{b+1}\n" for a, b, c in f)
    return len(xyz)

counts = {}
for name, m in parts.items():
    counts[name] = write_obj(out / f"{name}.obj", m)
counts["pelvis"] = write_obj(out / "pelvis.obj", pelvis)
counts["femurs"] = write_obj(out / "femurs.obj", parts["femur_left"] | parts["femur_right"])
print("meshes", counts)

# ---------------- DRRs (parallel beam) ----------------
MU_W = 0.02  # 1/mm
mu = MU_W * (1 + np.clip(ct, -1000, None) / 1000.0)
def window(L, p=(2, 99.7), gamma=2.2):
    a, b = np.percentile(L, p); return np.clip((L - a) / (b - a), 0, 1) ** gamma
L_ap = mu.sum(axis=1) * sp[1]            # along A: image indexed (R, S)
L_lat = mu.sum(axis=0) * sp[0]           # along R: image indexed (A, S)
ap = window(L_ap).T[::-1, ::-1]          # rows: S top; cols: patient right on the viewer's left
lat = window(L_lat).T[::-1, ::-1]        # rows: S top; cols: anterior on the left
sat = (1 - np.exp(-L_ap)).T[::-1, ::-1]  # the saturated version (video 2 teaching point)
sat = (sat - sat.min()) / (sat.max() - sat.min())
for name, a in [("drr_ap", ap), ("drr_lat", lat), ("drr_ap_saturated", sat)]:
    Image.fromarray((a * 255).astype(np.uint8)).save(out / f"{name}.png")

# ---------------- CT volume for slice sweeps (windowed uint8, 2x downsampled) ----------------
ds = 2
ctd = ct[::ds, ::ds, ::ds]
W_LO, W_HI = -350, 1100                  # soft tissue + bone window
vol = (np.clip((ctd - W_LO) / (W_HI - W_LO), 0, 1) * 255).astype(np.uint8)
lab = np.zeros_like(vol)
for v_, name in enumerate(["hip_left", "hip_right", "sacrum", "femur_left", "femur_right", "spine"], start=1):
    lab[parts[name][::ds, ::ds, ::ds]] = v_
# Data3DTexture expects x fastest: store as (S, A, R) C-order so R varies fastest
np.ascontiguousarray(vol.transpose(2, 1, 0)).tofile(out / "ct_u8.raw")
np.ascontiguousarray(lab.transpose(2, 1, 0)).tofile(out / "labels_u8.raw")

# three orthogonal slices through the pelvis centroid (full res), PNG
ci = (pc / sp).astype(int)
def winpng(a, name):
    Image.fromarray((np.clip((a - W_LO) / (W_HI - W_LO), 0, 1) * 255).astype(np.uint8)).save(out / name)
winpng(ct[:, :, ci[2]].T[::-1, ::-1], "slice_axial.png")      # rows: A top? see below
winpng(ct[:, ci[1], :].T[::-1, ::-1], "slice_coronal.png")    # rows: S top, patient right on viewer's left
winpng(ct[ci[0], :, :].T[::-1, ::-1], "slice_sagittal.png")   # rows: S top, anterior on the left

# ---------------- occupancy grids of the pelvis (what the network predicts) ----------------
pidx = np.argwhere(pelvis)
plo, phi = pidx.min(0), pidx.max(0)
ext_mm = (phi - plo + 1) * sp
cube_mm = ext_mm.max() * 1.06                                # isotropic cube around the pelvis
c_mm = (plo + phi + 1) / 2 * sp
occ = {}
for n in (16, 32, 64):
    # sample the mask at the centre of each of n^3 cells (nearest after a light blur)
    g = (np.arange(n) + 0.5) / n * cube_mm - cube_mm / 2
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3) + c_mm
    vox = pts / sp
    blurred = filters.gaussian(pelvis.astype(np.float32), sigma=0.6 * (cube_mm / n) / sp[0])
    val = ndimage.map_coordinates(blurred, vox.T, order=1, mode="constant")
    on = np.argwhere(val.reshape(n, n, n) > 0.5)
    cell = cube_mm / n
    centers = to_scene(((on + 0.5) / n * cube_mm - cube_mm / 2 + c_mm))
    occ[n] = {"n": n, "cell": cell * SCALE, "count": int(len(on)),
              "centers": np.round(centers, 4).flatten().tolist()}
    json.dump(occ[n], open(out / f"occ{n}.json", "w"))
print("occupancy", {n: occ[n]["count"] for n in occ})

# ---------------- HU histogram (body only) ----------------
body = ct > -500
h, edges = np.histogram(np.clip(ct[body | (ct > -1100)], -1100, 1900), bins=150, range=(-1100, 1900))
json.dump({"counts": h.tolist(), "edges": edges.tolist()}, open(out / "hu_hist.json", "w"))

json.dump({
    "subject": subj.name, "spacing_mm": sp.tolist(), "crop_shape": list(map(int, ct.shape)),
    "vol_dims_xyz": [int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])],  # (R, A, S) after ds
    "vol_spacing_mm": (sp * ds).tolist(), "window_hu": [W_LO, W_HI],
    "scene_scale": SCALE,
    # scene-space box of the CT crop, for placing slice planes / volume
    "box_min": to_scene(np.array([[0, 0, 0]], float))[0].tolist(),
    "box_max": to_scene(np.array([np.array(ct.shape) * sp]))[0].tolist(),
    "ap_size": [ap.shape[1] * sp[0] * SCALE, ap.shape[0] * sp[2] * SCALE],
    "lat_size": [lat.shape[1] * sp[1] * SCALE, lat.shape[0] * sp[2] * SCALE],
    "occ_cube_mm": float(cube_mm),
    "labels": ["", "hip_left", "hip_right", "sacrum", "femur_left", "femur_right", "spine"],
}, open(out / "meta.json", "w"), indent=1)
print("done; crop", ct.shape, "ap", ap.shape, "lat", lat.shape)
