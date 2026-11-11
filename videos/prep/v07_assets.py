"""Video 7 assets: a REAL visual hull of the hero pelvis (s1390) from its AP + LAT silhouettes,
real Dice/HD95 against the true 64^3 mask, multi-view hulls, back-projection and CT (FBP) demos.

Usage: python -I -X utf8 prep/v07_assets.py      (writes public/data/v07/)
Conventions: see v07_common.py (64^3 cube of 352 mm centred on the pelvis centroid, 5.5 mm cells).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))      # -I drops the script dir from sys.path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import radon, iradon
from v07_common import (N, CELL, CUBE_MM, good_subjects, load_subject, subject_cached, silhouettes,
                        visual_hull, multiview_hull, dice, hd95, to_display, to_scene_tex, MU_W)

OUT = Path(__file__).resolve().parents[1] / "public" / "data" / "v07"
OUT.mkdir(parents=True, exist_ok=True)
HERO = "s1390"
d = subject_cached(HERO)
g = d["mask64"]                                   # (R, A, S) bool
ap, lat = silhouettes(g)
hull = visual_hull(ap, lat)
extra = hull & ~g
assert not (g & ~hull).any(), "the hull must contain the truth"

def flat(m):
    """grid indices (r, a, s) of the set voxels, flattened"""
    return np.argwhere(m).astype(int).ravel().tolist()

metrics = {"subject": HERO, "n": N, "cell_mm": CELL, "cube_mm": CUBE_MM,
           "true_vox": int(g.sum()), "hull_vox": int(hull.sum()), "extra_vox": int(extra.sum()),
           "missed_vox": int((g & ~hull).sum()),
           "hull_dice": round(dice(hull, g), 4), "hull_hd95_mm": round(hd95(hull, g), 2)}

# ---- multi-view hulls (views around the S axis, rays horizontal) ----
views = [1, 2, 3, 4, 6, 8, 12, 18, 36, 90, 180]
curve, mv = [], {}
for n in views:
    h = multiview_hull(g, n)
    assert not (g & ~h).any()
    if n == 2:
        assert (h == hull).all()
    mv[n] = h
    curve.append({"views": n, "dice": round(dice(h, g), 4), "hd95_mm": round(hd95(h, g), 2), "vox": int(h.sum())})
    print("views", curve[-1])
metrics["multiview"] = curve

# ---- a different shape with EXACTLY the same two silhouettes: per axial slice a diagonal that
# covers every AP row and every LAT column (any such set lies inside the hull and casts the same shadows)
alt = np.zeros_like(g)
for k in range(N):
    R = np.flatnonzero(ap[:, k]); A = np.flatnonzero(lat[:, k])
    if len(R) == 0:
        continue
    m = max(len(R), len(A))
    for i in range(m):
        alt[R[round(i * (len(R) - 1) / max(m - 1, 1))], A[round(i * (len(A) - 1) / max(m - 1, 1))], k] = True
a2, l2 = silhouettes(alt)
assert (a2 == ap).all() and (l2 == lat).all()
metrics["alt_vox"] = int(alt.sum()); metrics["alt_dice"] = round(dice(alt, g), 4)

# ---- obturator foramina: enclosed holes of the AP silhouette (see-through from the front) ----
holes, nh = ndimage.label(~ap)
border = set(np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
comps = sorted([(int((holes == i).sum()), i) for i in range(1, nh + 1) if i not in border], reverse=True)
foramina = []
for size, i in comps[:2]:
    r, s = np.argwhere(holes == i).mean(0)
    mm_r, mm_s = (r + 0.5) * CELL - CUBE_MM / 2, (s + 0.5) * CELL - CUBE_MM / 2
    foramina.append({"cells": size, "x": round(mm_r * 0.01, 3), "y": round(mm_s * 0.01, 3)})
    # the hull is empty along the whole A axis through the hole
    assert not hull.transpose(0, 2, 1)[holes == i].any()
metrics["ap_holes"] = foramina
print("AP holes", comps[:4], foramina)

# ---- all 27 'good' subjects: the same 2-view hull, for the small results table ----
per = []
for s in good_subjects():
    m = subject_cached(s)["mask64"]
    h = visual_hull(*silhouettes(m))
    per.append({"subject": s, "dice": round(dice(h, m), 4), "hd95_mm": round(hd95(h, m), 2)})
dd = np.array([p["dice"] for p in per]); hh = np.array([p["hd95_mm"] for p in per])
metrics["all27"] = {"n": len(per), "dice_mean": round(dd.mean(), 4), "dice_std": round(dd.std(), 4),
                    "dice_min": round(dd.min(), 4), "dice_max": round(dd.max(), 4),
                    "hd95_mean": round(hh.mean(), 2), "hd95_std": round(hh.std(), 2), "per": per}
print("27:", metrics["all27"]["dice_mean"], metrics["all27"]["hd95_mean"])

# ---- voxel index lists for the 3D scenes ----
json.dump({"n": N, "cell": CELL * 0.01, "cube": CUBE_MM * 0.01,
           "truth": flat(g), "extra": flat(extra), "alt": flat(alt),
           "extra4": flat(mv[4] & ~g), "extra180": flat(mv[180] & ~g),
           "ap": ap.astype(int).ravel().tolist(), "lat": lat.astype(int).ravel().tolist()},  # (u, s) C-order
          open(OUT / "vox.json", "w"), separators=(",", ":"))

# ---- 2D images ----
BONE, RED, DARK = (239, 230, 210), (255, 110, 120), (10, 16, 26)
def sil_png(sil, name, scale=6):
    img = (sil.T[::-1, ::-1] * 255).astype(np.uint8)          # radiological: S top, col 0 = max u
    Image.fromarray(img).resize((N * scale, N * scale), Image.NEAREST).save(OUT / name)
sil_png(ap, "sil_ap.png"); sil_png(lat, "sil_lat.png")

def axial_png(true_sl, extra_sl, name, scale=6):
    """axial cell grid seen from ABOVE with the patient facing the viewer (matches the 3D top views):
    posterior at the top, anterior at the bottom, patient's right on the viewer's left"""
    rgb = np.zeros((N, N, 3), np.uint8); rgb[:] = DARK
    rgb[extra_sl] = RED; rgb[true_sl] = BONE
    im = rgb.transpose(1, 0, 2)[:, ::-1]                       # rows: A ascending; cols: R descending
    Image.fromarray(np.ascontiguousarray(im)).resize((N * scale, N * scale), Image.NEAREST).save(OUT / name)

SLICES = {"wings": 36, "acetab": 26, "pubis": 16}
slices = {}
for key, k in SLICES.items():
    axial_png(g[:, :, k], extra[:, :, k], f"axial_{key}.png")
    axial_png(g[:, :, k], np.zeros_like(g[:, :, k]), f"axial_{key}_true.png")
    for n in (4, 180):
        axial_png(g[:, :, k], (mv[n] & ~g)[:, :, k], f"axial_{key}_v{n}.png")
    slices[key] = {"k": k, "y": round(((k + 0.5) * CELL - CUBE_MM / 2) * 0.01, 4),
                   "ap_row": ap[:, k].astype(int).tolist(), "lat_row": lat[:, k].astype(int).tolist(),
                   "true": int(g[:, :, k].sum()), "hull": int(hull[:, :, k].sum())}
metrics["slices"] = slices

# hero DRRs in the cube frame: display (radiological) + scene-oriented textures for the 3D smears
dap, dlat = d["drr_ap"], d["drr_lat"]
for name, a in [("drr_ap", dap), ("drr_lat", dlat)]:
    Image.fromarray(to_display(a)).save(OUT / f"{name}.png")
    disp = to_display(a)[:, ::-1]                                # col 0 = min u: matches a plane's +u
    Image.fromarray(np.ascontiguousarray(disp)).save(OUT / f"{name}_scene.png")

# ---- one real CT axial slice (wing level): 2-view back-projection and filtered back-projection ----
ct, m_full, sp = load_subject(HERO)
c = d["c_vox"]
yk = SLICES["wings"]
z_mm = (yk + 0.5) * CELL - CUBE_MM / 2
zi = int(round(c[2] + z_mm / sp[2]))
half = int(round(CUBE_MM / 2 / sp[0]))
r0, a0 = int(round(c[0])) - half, int(round(c[1])) - half
mu = MU_W * (1 + np.clip(ct[:, :, zi], -1000, None) / 1000.0)
pad = np.pad(mu, half + 2)
sl = pad[r0 + half + 2: r0 + 3 * half + 2, a0 + half + 2: a0 + 3 * half + 2]   # (R, A), 2*half square
img = sl.T[:, ::-1]                                    # seen from above: anterior at the bottom, patient right left
size = img.shape[0]
def png(a, name, lo=None, hi=None, gamma=1.0):
    lo = np.percentile(a, 1) if lo is None else lo; hi = np.percentile(a, 99.8) if hi is None else hi
    v = np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) ** gamma
    Image.fromarray((v * 255).astype(np.uint8)).resize((300, 300), Image.BILINEAR).save(OUT / name)
png(img, "ct_slice.png", lo=0.0, hi=MU_W * 2.2)
# plain back-projection of the two real ray-sum profiles (AP: sum over A; LAT: sum over R).
# For clarity only the BONE part is projected: attenuation above that of soft tissue (HU > 150).
hu = np.pad(ct[:, :, zi], half + 2, constant_values=-1000)[r0 + half + 2: r0 + 3 * half + 2, a0 + half + 2: a0 + 3 * half + 2]
bone = (MU_W * np.clip(hu - 150, 0, None) / 1000.0).T[:, ::-1]
png(bone, "ct_slice_bone.png", lo=0.0, hi=np.percentile(bone[bone > 0], 99))
p_ap = bone.sum(axis=0)           # one value per column (R), the AP ray sums
p_lat = bone.sum(axis=1)          # one value per row (A), the LAT ray sums
bp_ap = np.tile(p_ap[None, :], (size, 1)); bp_lat = np.tile(p_lat[:, None], (1, size))
hi = (bp_ap + bp_lat).max()
png(bp_ap, "bp_ap.png", 0, p_ap.max()); png(bp_lat, "bp_lat.png", 0, p_lat.max()); png(bp_ap + bp_lat, "bp_sum.png", 0, hi)
json.dump({"ap": np.round(p_ap / p_ap.max(), 3).tolist()[::4], "lat": np.round(p_lat / p_lat.max(), 3).tolist()[::4]},
          open(OUT / "profiles.json", "w"))
# filtered back-projection (what CT does) from n views over 180 deg
fbp_err = {}
for n in (2, 8, 32, 180):
    th = np.arange(n) * 180.0 / n
    sino = radon(img, theta=th, circle=False)
    rec = iradon(sino, theta=th, filter_name="ramp", circle=False, output_size=size)
    fbp_err[n] = round(float(np.sqrt(((rec - img) ** 2).mean()) / img.std()), 3)
    png(rec, f"fbp_{n}.png", lo=0.0, hi=MU_W * 2.2)
metrics["fbp_rel_rmse"] = fbp_err
print("fbp rel rmse", fbp_err)

json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
print({k: v for k, v in metrics.items() if k not in ("all27", "slices", "multiview")})
