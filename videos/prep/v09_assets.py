"""Video 09 assets: real 3x3 convolutions on our AP DRR.

Usage (from videos/): .venv/Scripts/python.exe -I prep/v09_assets.py
Writes public/data/v09/: conv.json (patch numbers, kernel, outputs, parameter counts) and feature-map PNGs.

Convention: deep-learning "convolution" = cross-correlation (no kernel flip), exactly what
torch.nn.Conv2d computes. We use scipy.ndimage.correlate with zero padding (padding=1).
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as nd

root = Path(__file__).resolve().parent.parent
out = root / "public/data/v09"
out.mkdir(parents=True, exist_ok=True)
ap = np.array(Image.open(root / "public/data/hero/drr_ap.png")).astype(np.int64)   # (235, 248), 0..255
H, W = ap.shape

K = {
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}

def correlate(img, k):
    return nd.correlate(img.astype(np.float64), k.astype(np.float64), mode="constant", cval=0.0)

# ---- the zoomed patch: a vertical bright line (pelvic brim) on the viewer's left side ----
cy, cx = 160, 82
P = ap[cy - 3:cy + 4, cx - 3:cx + 4]                     # 7x7
k = K["sobel_x"]
valid = np.array([[int((P[i:i + 3, j:j + 3] * k).sum()) for j in range(5)] for i in range(5)])
# check against scipy on the full image
full = correlate(ap, k)
assert np.allclose(valid, full[cy - 2:cy + 3, cx - 2:cx + 3]), "patch conv mismatch"
# a flat region for comparison (soft tissue)
fy, fx = 120, 30
flat = ap[fy - 1:fy + 2, fx - 1:fx + 2]

# ---- feature maps ----
maps = {}
def save_gray(a, name):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out / name)

def save_signed(a, name):
    """diverging map: negative -> blue, positive -> orange, zero -> black; scale by 99th pct of |a|"""
    s = np.percentile(np.abs(a[2:-2, 2:-2]), 99.5)   # interior only: zero padding makes false border edges
    v = np.clip(a / s, -1, 1)
    rgb = np.zeros(a.shape + (3,))
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    rgb[..., 0] = pos * 255 + neg * 70
    rgb[..., 1] = pos * 190 + neg * 170
    rgb[..., 2] = pos * 90 + neg * 255
    Image.fromarray(rgb.astype(np.uint8)).save(out / name)
    return float(s)

for name in ["sobel_x", "sobel_y"]:
    r = correlate(ap, K[name])
    maps[name] = {"min": float(r.min()), "max": float(r.max()), "scale": save_signed(r, f"fm_{name}.png")}
for name in ["blur", "sharpen"]:
    r = correlate(ap, K[name])
    maps[name] = {"min": float(r.min()), "max": float(r.max())}
    save_gray(r, f"fm_{name}.png")

# ---- 2x2 max-pool: a 4x4 patch with numbers, and the pooled image ----
py, px = 158, 80
P4 = ap[py:py + 4, px:px + 4]
pool4 = P4.reshape(2, 2, 2, 2).max(axis=(1, 3))
Hc, Wc = H // 2 * 2, W // 2 * 2
pooled = ap[:Hc, :Wc].reshape(Hc // 2, 2, Wc // 2, 2).max(axis=(1, 3))
save_gray(pooled, "pool_ap.png")

# ---- parameter counts (256x256 single-channel input) ----
n_in = 256 * 256
params = {
    "n_in": n_in,
    "fc_weights": n_in * n_in, "fc_bias": n_in, "fc_total": n_in * n_in + n_in,
    "fc_gb_float32": (n_in * n_in + n_in) * 4 / 1e9,
    "conv_1k": 3 * 3 + 1, "conv_64k": 64 * (3 * 3 + 1),
    "conv_16_32": 32 * (16 * 9 + 1),
    "conv3d_1_1": 27 + 1,
    "vox64": 64 ** 3, "pix64": 64 ** 2,
}
def out_size(n, k, p, s): return (n + 2 * p - k) // s + 1
sizes = {f"{n},{k},{p},{s}": out_size(n, k, p, s) for n, k, p, s in
         [(7, 3, 0, 1), (7, 3, 1, 1), (7, 3, 1, 2), (256, 3, 1, 1), (256, 3, 1, 2), (256, 3, 0, 1), (128, 3, 1, 2)]}

json.dump({
    "image_hw": [H, W], "patch_center_yx": [cy, cx], "patch": P.tolist(), "kernel": k.tolist(),
    "valid": valid.tolist(), "flat": flat.tolist(), "flat_center_yx": [fy, fx],
    "flat_out": int((flat * k).sum()),
    "maps": maps, "kernels": {n: v.tolist() for n, v in K.items()},
    "pool_patch": P4.tolist(), "pool_out": pool4.tolist(), "pool_patch_yx": [py, px],
    "pooled_hw": list(pooled.shape),
    "params": params, "sizes": sizes,
}, open(out / "conv.json", "w"), indent=1)
print("patch\n", P, "\nsobel_x valid\n", valid, "\nflat", flat.tolist(), (flat * k).sum())
print("pool", P4.tolist(), pool4.tolist())
print(params, sizes, maps)
