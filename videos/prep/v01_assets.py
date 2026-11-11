"""Video 1 assets ("what is an image"): real pixel/voxel numbers and a few small PNGs.

Usage: python v01_assets.py <subject_dir> <hero_dir> <out_dir>
  subject_dir: data/raw/totalsegmentator_small/extracted/s1390
  hero_dir:    public/data/hero (drr_ap.png is read from there)
Uses the same crop as hero_assets.py so numbers match the hero volume.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image

subj, hero, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
seg = subj / "segmentations"
img = nib.load(subj / "ct.nii.gz")
sp = np.array(img.header.get_zooms()[:3], float)
ct_full = np.asanyarray(img.dataobj).astype(np.int16)
load = lambda n: np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
names = ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]
masks = {n: load(n) for n in names}
spine = load("vertebrae_L5") | load("vertebrae_L4") | load("vertebrae_L3")
allm = spine.copy()
for m in masks.values(): allm |= m
idx = np.argwhere(allm)
lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(ct_full.shape) - 1)
sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
ct = ct_full[sl]
mk = {n: m[sl] for n, m in masks.items()}
pelvis = mk["hip_left"] | mk["hip_right"] | mk["sacrum"]
pc = np.argwhere(pelvis).mean(0)
ci = pc.astype(int)
info = {}

# ---- 1. a real pixel patch of the AP DRR (the zoom-in scene) ----
drr = np.array(Image.open(hero / "drr_ap.png"))           # (rows, cols) uint8
H, W = drr.shape
# patch on the rim of the right acetabulum / femoral head (viewer's left), where values change
N = 8
best = None
for r in range(int(H * 0.50), int(H * 0.75)):
    for c in range(int(W * 0.12), int(W * 0.40)):
        q = drr[r:r + N, c:c + N].astype(float)
        if q.max() > 245: continue                         # avoid display-clipped pixels
        sc = q.std()
        if best is None or sc > best[0]: best = (sc, r, c)
_, r0, c0 = best
patch = drr[r0:r0 + N, c0:c0 + N]
info["drr"] = {"shape": [H, W], "patch_rc": [r0, c0], "n": N, "values": patch.tolist(),
               "spacing_mm": float(sp[0]), "dtype": str(drr.dtype)}

# ---- 2. resolution: the same DRR with bigger pixels (block mean) ----
for f in (4, 8):
    h2, w2 = H // f, W // f
    d = drr[:h2 * f, :w2 * f].reshape(h2, f, w2, f).mean((1, 3)).astype(np.uint8)
    Image.fromarray(d).save(out / f"drr_ap_ds{f}.png")
    info[f"ds{f}"] = [h2, w2]
    if f == 8: info["ds8_values"] = d.tolist()           # the "table of numbers" in the intro

# ---- 3. RGB: axial slice with the pelvis labels coloured, and its three channels ----
W_LO, W_HI = -350, 1100
ax = ct[:, :, ci[2]]                                       # (R, A) at the centroid height
g = np.clip((ax - W_LO) / (W_HI - W_LO), 0, 1)
rgb = np.stack([g, g, g], -1)
cols = {"hip_left": (1.0, 0.55, 0.25), "hip_right": (0.3, 0.75, 1.0), "sacrum": (0.45, 1.0, 0.55)}
for n, c in cols.items():
    m = mk[n][:, :, ci[2]]
    rgb[m] = 0.35 * rgb[m] + 0.65 * np.array(c)
disp = lambda a: a.transpose(1, 0, *range(2, a.ndim))[::-1, ::-1]   # rows: anterior top; patient right on viewer's left
rgb8 = (disp(rgb) * 255).astype(np.uint8)
Image.fromarray(rgb8).save(out / "axial_rgb.png")
for k, ch in enumerate("rgb"):
    Image.fromarray(rgb8[:, :, k]).save(out / f"axial_{ch}.png")
# one real RGB triple inside hip_left on the displayed image
dm = disp(mk["hip_left"][:, :, ci[2]])
rr, cc = np.argwhere(dm)[len(np.argwhere(dm)) // 2]
info["rgb"] = {"shape": list(rgb8.shape), "probe_rc": [int(rr), int(cc)], "probe": rgb8[rr, cc].tolist(),
               "probe_gray": int(rgb8[rr, cc].mean())}
gray_probe = (g.T[::-1, ::-1] * 255).astype(np.uint8)[rr, cc]
info["rgb"]["gray_value_there"] = int(gray_probe)

# ---- 4. mask patch: real HU and 0/1 of hip_left on the axial slice, across the bone rim ----
axm = disp(mk["hip_left"][:, :, ci[2]]).astype(np.uint8)
axh = disp(ax)
# choose a 10x10 window centred on the mask boundary nearest the mask centroid row
ys, xs = np.nonzero(axm)
yc = int(np.median(ys)); row = axm[yc]
edge_c = int(np.nonzero(np.diff(row.astype(int)))[0][0])
M = 10
pr, pcl = yc - M // 2, edge_c - M // 2
info["mask"] = {"rc": [pr, pcl], "n": M, "hu": axh[pr:pr + M, pcl:pcl + M].astype(int).tolist(),
                "m": axm[pr:pr + M, pcl:pcl + M].tolist(), "slice_shape": list(axm.shape)}
Image.fromarray((axm * 255).astype(np.uint8)).save(out / "axial_mask_hip_left.png")
Image.fromarray((np.clip((axh - W_LO) / (W_HI - W_LO), 0, 1) * 255).astype(np.uint8)).save(out / "axial_gray.png")

# ---- 5. the real header numbers and label statistics ----
vox_ml = float(np.prod(sp)) / 1000.0
info["nifti"] = {
    "shape": list(map(int, img.shape)), "zooms": [float(z) for z in img.header.get_zooms()[:3]],
    "dtype": str(img.get_data_dtype()), "axcodes": list(nib.aff2axcodes(img.affine)),
    "affine": np.round(img.affine, 3).tolist(),
    "hu_min": int(ct_full.min()), "hu_max": int(ct_full.max()),
    "n_seg_files": len(list(seg.glob("*.nii.gz"))),
    "crop_lo": lo.tolist(), "crop_shape": list(ct.shape), "centroid_vox_crop": ci.tolist(),
    "centroid_vox_full": (ci + lo).tolist(),
}
A = img.affine
v = np.array([*(ci + lo), 1.0])
info["nifti"]["centroid_mm"] = np.round(A @ v, 1)[:3].tolist()
info["labels"] = {n: {"voxels": int(m.sum()), "ml": round(float(m.sum()) * vox_ml, 1)} for n, m in masks.items()}
info["voxel_ml"] = vox_ml
json.dump(info, open(out / "info.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k not in ("drr", "mask")}, indent=1))
print("patch", patch)
print("mask", info["mask"])
