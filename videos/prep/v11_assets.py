"""Video 11 assets: class imbalance numbers from the hero 64^3 occupancy grid, and augmented AP DRRs.

Usage (from videos/): .venv/Scripts/python.exe -I prep/v11_assets.py
Writes public/data/v11/: v11.json + aug_*.png
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent.parent
hero = root / "public/data/hero"
out = root / "public/data/v11"
out.mkdir(parents=True, exist_ok=True)

occ = json.load(open(hero / "occ64.json"))
n = occ["n"]
N = n ** 3
pos = occ["count"]
assert pos == len(occ["centers"]) // 3
f = pos / N

# predictor 1: "empty everywhere"  (p = 0 for every cell)
acc_empty = (N - pos) / N
dice_empty = 0.0                                # 2*0 / (0 + pos)
# predictor 2: constant probability p = f everywhere (the best constant under BCE)
bce_const = -(f * np.log(f) + (1 - f) * np.log(1 - f))
soft_dice_const = 2 * f * pos / (f * N + pos)   # = 2 f pos / (2 pos) = f
# BCE examples
bce = {p: float(-np.log(p)) for p in (0.9, 0.5, 0.1, 0.01)}

# ---------------- augmentation examples on the AP DRR ----------------
ap = np.array(Image.open(hero / "drr_ap.png")).astype(np.float32) / 255
rng = np.random.default_rng(0)
augs = {
    "aug_bright": np.clip(ap * 1.0 + 0.12, 0, 1),
    "aug_contrast": np.clip((ap - 0.5) * 1.45 + 0.5, 0, 1),
    "aug_noise": np.clip(ap + rng.normal(0, 0.06, ap.shape), 0, 1),
}
for k, a in augs.items():
    Image.fromarray((a * 255).astype(np.uint8)).save(out / f"{k}.png")

# ---------------- geometric augmentation: shift the AP image, and the target's AP silhouette ----------------
import nibabel as nib
from scipy import ndimage as nd
subj = root.parent / "data/raw/totalsegmentator_small/extracted/s1390"
seg = subj / "segmentations"
load = lambda nm: np.asanyarray(nib.load(seg / f"{nm}.nii.gz").dataobj) > 0
names = ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3"]
parts = {nm: load(nm) for nm in names}
allm = np.zeros_like(parts["sacrum"])
for m in parts.values(): allm |= m
ix = np.argwhere(allm)
lo = np.maximum(ix.min(0) - 12, 0); hi = np.minimum(ix.max(0) + 12, np.array(allm.shape) - 1)
sl = tuple(slice(a_, b_ + 1) for a_, b_ in zip(lo, hi))
pel = (parts["hip_left"] | parts["hip_right"] | parts["sacrum"])[sl]
sil = pel.any(axis=1).T[::-1, ::-1]                     # same orientation as drr_ap.png
assert sil.shape == ap.shape, (sil.shape, ap.shape)
edge = sil & ~nd.binary_erosion(sil, iterations=2)
def outline(mask, col):
    rgba = np.zeros(mask.shape + (4,), np.uint8); rgba[mask] = col; return rgba
SHIFT = 22                                              # pixels (= 33 mm) to the viewer's right
def shift(a, dx):
    o_ = np.zeros_like(a); o_[:, dx:] = a[:, :-dx]; return o_
Image.fromarray((shift(ap, SHIFT) * 255).astype(np.uint8)).save(out / "aug_shift.png")
Image.fromarray(outline(edge, (255, 200, 90, 255))).save(out / "target_outline.png")
Image.fromarray(outline(shift(edge, SHIFT), (255, 200, 90, 255))).save(out / "target_outline_shift.png")

json.dump({
    "shift_px": SHIFT, "n": n, "cells": N, "occupied": pos, "frac": f,
    "acc_empty": acc_empty, "dice_empty": dice_empty,
    "bce_const": float(bce_const), "soft_dice_const": float(soft_dice_const),
    "bce_examples": bce,
}, open(out / "v11.json", "w"), indent=1)
print(json.load(open(out / "v11.json")))
