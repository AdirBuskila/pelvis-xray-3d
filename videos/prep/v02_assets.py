"""Video 2 assets ("where the CT numbers come from"): one real ray, the HU histogram with honest peaks,
per-tissue HU medians from the masks, a windowing slice, and line-integral statistics.

Usage: python v02_assets.py <subject_dir> <out_dir>      (subject_dir = .../extracted/s1390)
Same crop and the same HU->mu rule as hero_assets.py (mu_water = 0.02 / mm).
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage, signal
from PIL import Image

subj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seg = subj / "segmentations"
img = nib.load(subj / "ct.nii.gz")
sp = np.array(img.header.get_zooms()[:3], float)
ct_full = np.asanyarray(img.dataobj).astype(np.float32)
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
bone = pelvis | mk["femur_left"] | mk["femur_right"] | spine[sl]
I, J, K = ct.shape
MU_W = 0.02
mu = MU_W * (1 + np.clip(ct, -1000, None) / 1000.0)
info = {"mu_w_per_mm": MU_W, "spacing_mm": sp.tolist(), "crop_shape": [I, J, K]}

# ---- 1. one AP ray (along A) through the centre of the right femoral head ----
fr = mk["femur_right"]
ks = np.nonzero(fr.any((0, 1)))[0]
k_head = int(ks.max() - round(22 / sp[2]))                  # ~22 mm below the top of the femur
ii, jj = np.nonzero(fr[:, :, k_head])
i_ray = int(np.median(ii))
hu = ct[i_ray, :, k_head]                                   # j = 0 (posterior) ... J-1 (anterior)
hu_ray = hu[::-1]                                           # the X-ray travels anterior -> posterior (AP)
mu_ray = MU_W * (1 + np.clip(hu_ray, -1000, None) / 1000.0)
L_cum = np.cumsum(mu_ray) * sp[1]
bone_ray = bone[i_ray, ::-1, k_head]
info["ray"] = {
    "i": i_ray, "k": k_head, "ds_mm": float(sp[1]), "n": int(J),
    "hu": np.round(hu_ray).astype(int).tolist(), "mu": np.round(mu_ray, 5).tolist(),
    "L_cum": np.round(L_cum, 4).tolist(), "bone": bone_ray.astype(int).tolist(),
    "L_total": round(float(L_cum[-1]), 3), "T": float(np.exp(-L_cum[-1])),
    "ap_pixel_rc": [int(K - 1 - k_head), int(I - 1 - i_ray)],          # in drr_ap.png
    "bone_mm": round(float(bone_ray.sum() * sp[1]), 1),
    "scene_xy": [float((i_ray - np.argwhere(pelvis).mean(0)[0]) * sp[0] * 0.01),
                 float((k_head - np.argwhere(pelvis).mean(0)[2]) * sp[2] * 0.01)],
}
# where the ray enters / leaves the body (first/last voxel above -500 HU)
body = np.nonzero(hu_ray > -500)[0]
info["ray"]["body_in_out"] = [int(body.min()), int(body.max())]
info["ray"]["L_soft_equiv_mm"] = round(float(L_cum[-1] / MU_W), 1)

# ---- 2. HU histogram of the whole crop, honest peak finding ----
edges = np.arange(-1100, 1710, 10.0)
h, _ = np.histogram(ct, bins=edges)
hs = ndimage.gaussian_filter1d(h.astype(float), 1.5)
pk, prop = signal.find_peaks(hs, prominence=hs.max() * 0.02)
centers = (edges[:-1] + edges[1:]) / 2
peaks = [{"hu": float(centers[p]), "count": int(h[p]), "prom": float(prop["prominences"][n])} for n, p in enumerate(pk)]
info["hist"] = {"edges": edges.tolist(), "counts": h.astype(int).tolist(), "peaks": peaks, "total": int(ct.size)}
def frac(a, b): return float(((ct >= a) & (ct < b)).sum() / ct.size)
info["hist"]["frac"] = {"below_m1000": float((ct < -1000).sum() / ct.size), "air_lt_m900": frac(-1e9, -900),
                        "fat_m200_m30": frac(-200, -30), "soft_m30_150": frac(-30, 150),
                        "bone_gt_300": frac(300, 1e9), "gt_150": frac(150, 1e9)}
info["hist"]["below_m1000_count"] = int((ct < -1000).sum())
info["hist"]["min"] = float(ct.min())
info["hist"]["mode_m1024"] = int((ct == -1024).sum())

# ---- 3. median HU per tissue, from the masks of this same CT ----
tissues = {
    "muscle (gluteus maximus)": ["gluteus_maximus_left", "gluteus_maximus_right"],
    "bladder (urine)": ["urinary_bladder"],
    "iliac arteries (contrast)": ["iliac_artery_left", "iliac_artery_right"],
    "hip bone": ["hip_left", "hip_right"],
    "femur": ["femur_left", "femur_right"],
}
med = {}
for name, files in tissues.items():
    m = None
    for f in files:
        if (seg / f"{f}.nii.gz").exists():
            a = load(f); m = a if m is None else (m | a)
    if m is None or not m.any(): continue
    # erode 1 voxel so partial-volume edges do not dominate
    me = ndimage.binary_erosion(m) if m.sum() > 500 else m
    v = ct_full[me]
    med[name] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                 "n": int(me.sum())}
# outside-body air and subcutaneous fat (no masks): air = crop voxels < -900 that touch the border region;
# fat = voxels in [-200, -30] (the fat peak) -> report the median there only as a check of the peak position
air = ct[ct < -900]; med["air (outside body)"] = {"median": float(np.median(air)), "n": int(air.size)}
info["tissue_hu"] = med
for k, v in med.items(): v["mu"] = round(MU_W * (1 + max(v["median"], -1000) / 1000), 5)

# ---- 4. axial slice for windowing (linear HU encoding, decoded in the video) ----
ci = np.argwhere(pelvis).mean(0).astype(int)
ax = ct[:, :, ci[2]].T[::-1, ::-1]                           # rows: anterior top; patient right on viewer's left
ENC_LO, ENC_HI = -1024.0, 1526.0                             # 10 HU per grey level
Image.fromarray(np.round(np.clip((ax - ENC_LO) / (ENC_HI - ENC_LO), 0, 1) * 255).astype(np.uint8)).save(out / "axial_hu.png")
info["window_png"] = {"enc_lo": ENC_LO, "enc_hi": ENC_HI, "shape": list(ax.shape)}

# ---- 5. AP line integrals: why 1 - exp(-L) saturates ----
L_ap = mu.sum(axis=1) * sp[1]                                # (I, K)
bodypix = (ct > -500).any(axis=1)
Lb = L_ap[bodypix]
info["L_ap"] = {"p1": float(np.percentile(Lb, 1)), "p50": float(np.percentile(Lb, 50)), "p99": float(np.percentile(Lb, 99)),
                "max": float(Lb.max()), "min": float(Lb.min()),
                "T_p50": float(np.exp(-np.percentile(Lb, 50))), "sat_p1": float(1 - np.exp(-np.percentile(Lb, 1))),
                "sat_p50": float(1 - np.exp(-np.percentile(Lb, 50))), "sat_p99": float(1 - np.exp(-np.percentile(Lb, 99)))}
lh, le = np.histogram(Lb, bins=60, range=(0, 12))
info["L_ap"]["hist"] = {"counts": lh.tolist(), "edges": le.tolist()}

json.dump(info, open(out / "info.json", "w"))
show = {k: v for k, v in info.items() if k not in ("ray", "hist")}
show["L_ap"] = {k: v for k, v in info["L_ap"].items() if k != "hist"}
print(json.dumps(show, indent=1))
print("ray", {k: v for k, v in info["ray"].items() if k not in ("hu", "mu", "L_cum", "bone")})
print("peaks", peaks, info["hist"]["frac"], info["hist"]["below_m1000_count"], info["hist"]["min"], info["hist"]["mode_m1024"])
print("ray hu", info["ray"]["hu"])
