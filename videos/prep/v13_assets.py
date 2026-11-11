"""Video 13 assets: a real DeepFluoro fluoroscopy frame next to our DRR of the same cadaver's CT.

Usage: python -I v13_assets.py <deepfluoro_subject_dir> <hero_dir> <out_dir>

DeepFluoro is CC BY-NC-ND 4.0 (Grupp et al. 2020): everything this script writes goes to
videos/public/data/df_v13/ (gitignored) and nowhere else. Video 13 renders locally only.

What is computed (all real computations unless marked ILLUSTRATIVE):
- real.png: frame <FRAME> of subject01, log-transformed and inverted so bone is bright.
- drr_ap.png: our parallel-beam AP DRR of the same subject's CT (same recipe as hero_assets.py).
- the field of view / magnification: the real frame is matched to the DRR by template matching
  over scale and shift (both left-right flips tried); the best match gives the real frame's
  footprint on the DRR (fov box) and its pixel size at the patient -> magnification.
- noise: std of a high-pass residual inside the bone-free region of both images.
- realism chain on the DRR (scatter haze, quantum noise, contrast curve): our own processing.
- ILLUSTRATIVE (hero CT, TotalSegmentator): a deformed "prediction" of the hero pelvis, its mesh,
  its AP silhouette vs the GT silhouette; and an MC-dropout-style variance map from perturbed copies.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, transform, feature

subj, hero, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert out.name == "df_v13", "DeepFluoro-derived assets may only go to public/data/df_v13"
out.mkdir(parents=True, exist_ok=True)
FRAME = "060"
info = {}

def save(a, name, size=None):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if size: im = im.resize(size, Image.LANCZOS)
    im.save(out / name)

def win(a, p=(1, 99.5), gamma=1.0, mask=None):
    ref = a[mask] if mask is not None else a
    lo, hi = np.percentile(ref, p)
    return np.clip((a - lo) / (hi - lo), 0, 1) ** gamma

# ---------------- the real frame ----------------
d = pydicom.dcmread(subj / "xrays" / f"{FRAME}.dcm")
raw = d.pixel_array.astype(np.float32)
sdd = float(d.DistanceSourceToDetector); px = float(d.PixelSpacing[0])
L_real = -np.log(np.maximum(raw, 1) / raw.max())        # attenuation-like: bone bright
c0 = slice(300, 1236)                                    # window on the central region (the border is vignetted)
lo, hi = np.percentile(L_real[c0, c0], [0.5, 99.8])
real = np.clip((L_real - lo) / (hi - lo), 0, 1) ** 0.85
save(real, "real.png", (768, 768))
info.update(frame=FRAME, sdd_mm=sdd, det_px_mm=px, det_px=int(raw.shape[0]), det_mm=px * raw.shape[0])

# ---------------- our DRR of the same CT (parallel beam, AP) ----------------
img = nib.load(subj / "volume.nii.gz")
assert nib.aff2axcodes(img.affine) == ("L", "P", "S")
ct = np.asanyarray(img.dataobj).astype(np.float32)
sp = np.array(img.header.get_zooms()[:3], float)
MU_W = 0.02
mu = MU_W * (1 + np.clip(ct, -1000, None) / 1000.0)
L_ap = mu.sum(axis=1) * sp[1]                            # (L, S)
# rows: S at the top; columns: index along L increases to the right = patient's right on the viewer's left
drr_L = L_ap.T[::-1, :]
drr = win(drr_L, (2, 99.7), 2.2)
save(drr, "drr_ap.png")
info["drr_shape"] = list(drr.shape)                      # 1 px = 1 mm

# ---------------- match the real frame onto the DRR: scale + shift (+ flip) ----------------
def dog(a, s1=1.5, s2=12):  # band-pass: keeps bone edges and texture, drops contrast/vignetting differences
    return filters.gaussian(a, s1) - filters.gaussian(a, s2)
D = dog(drr_L)
MARGIN = 0.15                                            # ignore the vignetted border of the real frame
best = None
for flip in (False, True):
    R0 = L_real[:, ::-1] if flip else L_real
    for mmpx in np.linspace(0.08, 0.26, 37):            # real pixel size at the patient (mm)
        n = int(round(raw.shape[0] * mmpx))              # template size in DRR pixels (1 mm)
        if n >= min(D.shape) - 2: continue
        T = dog(transform.resize(R0, (n, n), anti_aliasing=True))
        k = int(n * MARGIN); T = T[k:n - k, k:n - k]
        r = feature.match_template(D, T)
        ij = np.unravel_index(np.argmax(r), r.shape)
        if best is None or r[ij] > best[0]:
            best = (float(r[ij]), flip, float(mmpx), n, [int(ij[0]) - k, int(ij[1]) - k])
score, flip, mmpx, n, (y0, x0) = best
info.update(match_ncc=round(score, 3), match_flip=flip, mm_per_px_at_patient=round(mmpx, 4),
            fov_mm=round(n, 1), magnification=round(px / mmpx, 3),
            fov_box=[x0 / drr.shape[1], y0 / drr.shape[0], n / drr.shape[1], n / drr.shape[0]])
print("match", best)
if flip:  # keep real.png in the radiology convention we used for the DRR
    save(real[:, ::-1], "real.png", (768, 768))
    real = real[:, ::-1]
crop = drr[y0:y0 + n, x0:x0 + n]
save(crop, "drr_crop.png", (768, 768))

# ---------------- noise and contrast: real numbers ----------------
rs = transform.resize(real, (n, n), anti_aliasing=False)
def hp_std(a):  # high-frequency residual std (noise + fine texture), on a 0..1 display image
    return float(np.std(a - filters.gaussian(a, 1.5)))
info["hp_std_real"] = round(hp_std(rs), 4)
info["hp_std_drr"] = round(hp_std(crop), 4)
# a horizontal line profile through the middle of the matched region (same anatomy in both)
row = int(n * 0.62)
info["profile_row"] = row / n
info["profile_real"] = np.round(rs[row, ::max(1, n // 160)], 3).tolist()
info["profile_drr"] = np.round(crop[row, ::max(1, n // 160)], 3).tolist()
hr, _ = np.histogram(rs, 64, (0, 1)); hd, _ = np.histogram(crop, 64, (0, 1))
info["hist_real"] = (hr / hr.max()).round(3).tolist(); info["hist_drr"] = (hd / hd.max()).round(3).tolist()
# zoom tiles (same anatomy) to show noise/grain
cy, cx, z = int(n * 0.55), int(n * 0.30), int(n * 0.16)
F = raw.shape[0] / n                                      # full-res real pixels per DRR pixel
zr = real[int(cy * F):int((cy + z) * F), int(cx * F):int((cx + z) * F)]
save(transform.resize(zr, (384, 384), anti_aliasing=True), "zoom_real.png")     # full detector resolution
save(transform.resize(crop[cy:cy + z, cx:cx + z], (384, 384), order=0), "zoom_drr.png")  # 1 mm CT grid
info["zoom_mm"] = z
info["zoom_box"] = [cx / n, cy / n, z / n, z / n]

# ---------------- realism chain on the DRR crop (our processing, for teaching) ----------------
Lc = drr_L[y0:y0 + n, x0:x0 + n]
I0 = 8000.0
I = I0 * np.exp(-Lc / Lc.max() * 4.0)                   # detector counts behind the body
scatter = 0.25 * filters.gaussian(I, 30) + 0.01 * I0   # scattered photons: a smooth haze
rng = np.random.default_rng(0)
noisy = rng.poisson(I + scatter).astype(np.float32)     # quantum (Poisson) noise
blurred = filters.gaussian(noisy, 1.2)                   # detector blur
Lref = -np.log(I / I0); lo, hi = np.percentile(Lref, [0.5, 99.7])   # ONE fixed window for every step
for a, name in [(I, "chain0"), (I + scatter, "chain1"), (noisy, "chain2"), (blurred, "chain3")]:
    save(np.clip((-np.log(np.maximum(a, 1) / I0) - lo) / (hi - lo), 0, 1), f"{name}.png", (512, 512))

# ---------------- ILLUSTRATIVE: re-projection and MC dropout on the hero pelvis ----------------
meta = json.load(open(hero / "meta.json"))
dims = meta["vol_dims_xyz"]                              # (R, A, S) at 3 mm
lab = np.fromfile(hero / "labels_u8.raw", np.uint8).reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0)
gt = np.isin(lab, [1, 2, 3])
vsp = np.array(meta["vol_spacing_mm"])
# a deformed "prediction": smooth random displacement + the tip of one iliac wing missing
g = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]].astype(np.float32)
f = [filters.gaussian(np.random.default_rng(s).standard_normal(dims), 6) for s in (1, 2, 3)]
f = [3.0 * a / np.abs(a).max() for a in f]
pred = ndimage.map_coordinates(gt.astype(np.float32), [g[i] + f[i] for i in range(3)], order=1) > 0.5
idx = np.argwhere(gt); top = idx[:, 2].max()
wing = (g[0] < dims[0] * 0.38) & (g[2] > top - 10)        # the top of one iliac wing
pred &= ~wing
low = np.argwhere(gt)[:, 2].min()
grow = np.roll(gt, -3, axis=2) & (g[2] < low + 14) & (g[0] > dims[0] * 0.5)   # one ischium drawn too long
pred |= grow
pred = ndimage.binary_opening(pred, iterations=1)
def sil(m):  # AP silhouette: rows S top, columns patient right on the viewer's left (same as hero DRR)
    return m.any(axis=1).T[::-1, ::-1]
sg, sp_ = sil(gt), sil(pred)
inter, union = (sg & sp_).sum(), (sg | sp_).sum()
info["illus_sil_iou"] = round(float(inter / union), 3)
info["illus_dice3d"] = round(float(2 * (gt & pred).sum() / (gt.sum() + pred.sum())), 3)
rgb = np.zeros(sg.shape + (4,), np.uint8)
rgb[sg & sp_] = (120, 200, 255, 200)                      # agree
rgb[sg & ~sp_] = (127, 208, 255, 255); rgb[sg & ~sp_, :3] = (90, 255, 170)   # GT only (missed)
rgb[~sg & sp_] = (255, 120, 130, 255)                     # predicted only (extra)
Image.fromarray(rgb).resize((sg.shape[1] * 4, sg.shape[0] * 4), Image.NEAREST).save(out / "reproj_diff.png")
for m, name in [(sg, "sil_gt.png"), (sp_, "sil_pred.png")]:
    Image.fromarray((m * 255).astype(np.uint8)).resize((m.shape[1] * 4, m.shape[0] * 4), Image.NEAREST).save(out / name)
info["sil_shape"] = list(sg.shape)

# mesh of the deformed shape in hero scene coordinates (x=R, y=S, z=P, centred on the hero pelvis centroid)
pc = np.argwhere(gt).mean(0) * vsp
v, fc, nn, _ = measure.marching_cubes(filters.gaussian(pred.astype(np.float32), 0.8), 0.5, spacing=tuple(vsp))
def to_scene(p):
    q = p - pc
    return np.stack([q[:, 0], q[:, 2], -q[:, 1]], 1) * 0.01
# note: the hero meshes were built from the 1.5 mm crop with its own centroid; this mesh is
# centred on the same pelvis (3 mm grid), so it lines up to within a few mm
xyz = to_scene(v); nrm = np.stack([nn[:, 0], nn[:, 2], -nn[:, 1]], 1)
with open(out / "pred_illus.obj", "w") as fo:
    fo.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in xyz)
    fo.writelines(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in nrm)
    fo.writelines(f"f {a+1}//{a+1} {c+1}//{c+1} {b+1}//{b+1}\n" for a, b, c in fc)
info["pred_verts"] = len(xyz)

# MC-dropout-style passes on one coronal slice: K perturbed copies -> mean, variance, error
ci = int(np.argwhere(gt)[:, 1].mean())
sl_gt = gt[:, ci, :].T[::-1, ::-1].astype(np.float32)
H, W = sl_gt.shape
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = filters.gaussian(sl_gt, 1.0)
# a systematic "model error": one region shifted, so the mean prediction is wrong there too
bias = filters.gaussian(np.random.default_rng(7).standard_normal((H, W)), 5); bias *= 5.0 / np.abs(bias).max()
passes = []
for k in range(6):
    r = np.random.default_rng(100 + k)
    dx = filters.gaussian(r.standard_normal((H, W)), 3); dy = filters.gaussian(r.standard_normal((H, W)), 3)
    dx *= 2.0 / np.abs(dx).max(); dy *= 2.0 / np.abs(dy).max()
    p = ndimage.map_coordinates(base, [yy + dy + bias, xx + dx], order=1)
    passes.append(np.clip((p - 0.5) * 3 + 0.5, 0, 1))
P = np.stack(passes); mean, var = P.mean(0), P.var(0)
err = np.abs(mean - sl_gt)                                # soft error of the mean prediction
def heat(a, name, col):
    a = a / max(a.max(), 1e-6)
    rgb = np.zeros((H, W, 3));
    for i in range(3): rgb[..., i] = a * col[i]
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((W * 4, H * 4), Image.BILINEAR).save(out / name)
for k in range(4):
    Image.fromarray((passes[k] * 255).astype(np.uint8)).resize((W * 4, H * 4), Image.BILINEAR).save(out / f"mc_pass{k}.png")
heat(var, "mc_var.png", (1.0, 0.8, 0.45))
heat(err, "mc_err.png", (1.0, 0.45, 0.5))
Image.fromarray((sl_gt * 255).astype(np.uint8)).resize((W * 4, H * 4), Image.NEAREST).save(out / "mc_gt.png")
info["mc_shape"] = [H, W]

# a few more frames of the same cadaver (thumbnails, same display recipe and orientation)
for k, fr in enumerate(["000", "030", "044", "090"]):
    r = pydicom.dcmread(subj / "xrays" / f"{fr}.dcm").pixel_array.astype(np.float32)
    Lr = -np.log(np.maximum(r, 1) / r.max())
    lo_, hi_ = np.percentile(Lr[c0, c0], [0.5, 99.8])
    im = np.clip((Lr - lo_) / (hi_ - lo_), 0, 1) ** 0.85
    save(im[:, ::-1] if flip else im, f"thumb{k}.png", (256, 256))
json.dump(info, open(out / "info.json", "w"), indent=1)
print({k: v for k, v in info.items() if not isinstance(v, list) or len(v) < 6})
