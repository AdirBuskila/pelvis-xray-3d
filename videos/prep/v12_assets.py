"""Video 12 assets: the lateral DRR of the hero pelvis at 90, 95 and 100 degrees from the AP view
(the "second comparison" of the controlled experiment: tilt the lateral view).

Usage: python -I v12_assets.py <subject_dir> <out_dir>

Same recipe as hero_assets.py (parallel beam, mu from HU, percentile window, gamma 2.2), but the
volume is first rotated about the superior axis by (angle - 90) degrees. One fixed window (from the
90-degree image) is used for all three, so the only difference between them is the geometry.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from PIL import Image

subj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
img = nib.load(subj / "ct.nii.gz")
assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
sp = np.array(img.header.get_zooms()[:3], float)
ct = np.asanyarray(img.dataobj).astype(np.float32)
seg = subj / "segmentations"
m = None
for n in ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]:
    a = np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
    m = a if m is None else (m | a)
idx = np.argwhere(m)
lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(ct.shape) - 1)
ct = ct[:, max(lo[1] - 30, 0):hi[1] + 30, lo[2]:hi[2] + 1]   # whole body width in R; A and S cropped around the bones (drops the table)
mu = 0.02 * (1 + np.clip(ct, -1000, None) / 1000.0)
assert abs(sp[0] - sp[1]) < 1e-6, "rotation in the R-A plane assumes square voxels there"

res, win = {}, None
for ang in (90, 95, 100):
    v = ndimage.rotate(mu, ang - 90, axes=(0, 1), reshape=True, order=1, mode="constant", cval=0.0)
    L = v.sum(axis=0) * sp[0]                      # along R: image indexed (A', S)
    im = L.T[::-1, ::-1]                           # rows: S top; anterior on the left (as hero drr_lat)
    w0 = mu.shape[1]; c = (im.shape[1] - w0) // 2; im = im[:, c:c + w0]   # same width as the 90-degree image
    if win is None: win = np.percentile(im, [2, 99.7])
    a = np.clip((im - win[0]) / (win[1] - win[0]), 0, 1) ** 2.2
    Image.fromarray((a * 255).astype(np.uint8)).save(out / f"lat_{ang}.png")
    res[ang] = a
    print(ang, im.shape)
# how different are they? (normalised cross-correlation after cropping to a common width)
def ncc(a, b):
    w = min(a.shape[1], b.shape[1]); a = a[:, (a.shape[1] - w) // 2:][:, :w]; b = b[:, (b.shape[1] - w) // 2:][:, :w]
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
info = {"ncc_90_95": round(ncc(res[90], res[95]), 3), "ncc_90_100": round(ncc(res[90], res[100]), 3),
        "shapes": {k: list(v.shape) for k, v in res.items()}}
json.dump(info, open(out / "info.json", "w"), indent=1)
print(info)
