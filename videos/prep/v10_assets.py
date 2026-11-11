"""Video 10 assets: a 2D segmentation example, the encoder resolution pyramid, transposed-conv overlap
counts, and how much detail survives a down/up round trip of the hero occupancy grid.

Usage (from videos/): .venv/Scripts/python.exe -I prep/v10_assets.py
Reads data/raw/totalsegmentator_small/extracted/s1390 (same crop as prep/hero_assets.py) and public/data/hero/.
Writes public/data/v10/.
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from scipy import ndimage as nd

root = Path(__file__).resolve().parent.parent
subj = root.parent / "data/raw/totalsegmentator_small/extracted/s1390"
out = root / "public/data/v10"
out.mkdir(parents=True, exist_ok=True)
hero = root / "public/data/hero"

# ---------------- 2D segmentation example: coronal CT slice + pelvis mask (same crop as hero) ----------------
img = nib.load(subj / "ct.nii.gz")
ct = np.asanyarray(img.dataobj).astype(np.float32)
sp = np.array(img.header.get_zooms()[:3], float)
seg = subj / "segmentations"
load = lambda n: np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
names = ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3"]
parts = {n: load(n) for n in names}
pelvis = parts["hip_left"] | parts["hip_right"] | parts["sacrum"]
bone = np.zeros_like(pelvis)
for m in parts.values(): bone |= m
idx = np.argwhere(bone)
lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(ct.shape) - 1)
sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
ct, pelvis = ct[sl], pelvis[sl]
pc = np.argwhere(pelvis).mean(0)
ci = pc.astype(int)
W_LO, W_HI = -350, 1100
cor = ct[:, ci[1], :].T[::-1, ::-1]                       # rows: S top; cols: patient right on viewer's left
mcor = pelvis[:, ci[1], :].T[::-1, ::-1]
u8 = (np.clip((cor - W_LO) / (W_HI - W_LO), 0, 1) * 255).astype(np.uint8)
Image.fromarray(u8).save(out / "seg_ct.png")
rgba = np.zeros(mcor.shape + (4,), np.uint8)
rgba[mcor] = (255, 190, 90, 200)
Image.fromarray(rgba).save(out / "seg_mask.png")
seg_frac = float(mcor.mean())

# ---------------- encoder resolution pyramid of the AP DRR (pad to square, 128 -> 16) ----------------
ap = np.array(Image.open(hero / "drr_ap.png")).astype(np.float32)
H, W = ap.shape
S_ = max(H, W)
sq = np.zeros((S_, S_), np.float32)
sq[(S_ - H) // 2:(S_ - H) // 2 + H, (S_ - W) // 2:(S_ - W) // 2 + W] = ap
im128 = np.array(Image.fromarray(sq.astype(np.uint8)).resize((128, 128), Image.BILINEAR)).astype(np.float32)
pyr = {128: im128}
for n in (64, 32, 16):
    prev = pyr[n * 2]
    pyr[n] = prev.reshape(n, 2, n, 2).mean(axis=(1, 3))
for n, a in pyr.items():
    Image.fromarray(a.astype(np.uint8)).save(out / f"ap_{n}.png")
# the LAT DRR padded to a square and resized to the same 128x128 network input (display only)
lat = np.array(Image.open(hero / "drr_lat.png")).astype(np.float32)
Hl, Wl = lat.shape
sql = np.zeros((max(Hl, Wl),) * 2, np.float32)
sql[(sql.shape[0] - Hl) // 2:(sql.shape[0] - Hl) // 2 + Hl, (sql.shape[1] - Wl) // 2:(sql.shape[1] - Wl) // 2 + Wl] = lat
Image.fromarray(sql.astype(np.uint8)).resize((128, 128), Image.BILINEAR).save(out / "lat_128.png")

# ---------------- transposed convolution: how many kernel stamps land on each output pixel ----------------
def overlap(n_in, k, s, p=0):
    n_out = (n_in - 1) * s - 2 * p + k
    full = np.zeros(((n_in - 1) * s + k,) * 2, int)
    for i in range(n_in):
        for j in range(n_in):
            full[i * s:i * s + k, j * s:j * s + k] += 1
    return full[p:p + n_out, p:p + n_out]
ov_k3 = overlap(4, 3, 2)      # 9x9: counts 1, 2, 4 -> checkerboard
ov_k2 = overlap(4, 2, 2)      # 8x8: all ones

# ---------------- detail lost through a 16^3 / 32^3 bottleneck (no network involved) ----------------
occ = json.load(open(hero / "occ64.json"))
n = occ["n"]; cell = occ["cell"]
c = np.array(occ["centers"]).reshape(-1, 3)
ijk = np.round((c - c.min(0)) / cell).astype(int)
grid = np.zeros((n, n, n), np.float32)
off = (n - (ijk.max(0) + 1)) // 2                         # position inside the cube does not matter for Dice
ijk = ijk + off
grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
def dice(a, b): return float(2 * (a * b).sum() / (a.sum() + b.sum()))
roundtrip = {}
for m in (32, 16, 8):
    f = n // m
    small = grid.reshape(m, f, m, f, m, f).mean(axis=(1, 3, 5))
    up = nd.zoom(small, f, order=1, mode="nearest", grid_mode=True)
    rec = (up > 0.5).astype(np.float32)
    roundtrip[m] = round(dice(grid, rec), 3)
    if m == 16:   # the 16^3 round trip as scene-space voxel centres (same frame as occ64.json)
        on = np.argwhere(rec > 0) - off
        cen = c.min(0) + on * cell
        json.dump({"n": n, "cell": cell, "count": int(len(on)), "centers": np.round(cen, 4).flatten().tolist()},
                  open(out / "occ64_via16.json", "w"))
print("roundtrip dice", roundtrip)

json.dump({
    "seg_hw": list(mcor.shape), "seg_frac": seg_frac,
    "ov_k3": ov_k3.tolist(), "ov_k2": ov_k2.tolist(),
    "roundtrip_dice": roundtrip, "occ_count": int(grid.sum()), "n": n,
}, open(out / "v10.json", "w"), indent=1)
print("seg", mcor.shape, seg_frac, "ov_k3\n", ov_k3)
