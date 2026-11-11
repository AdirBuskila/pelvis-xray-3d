"""Video 4 assets: parallel vs perspective DRRs of the hero CT (s1390), computed here with NumPy.

Usage: python v04_assets.py <subject_dir> <out_dir>

Same crop, centroid and scene convention as hero_assets.py (x = R, y = S, z = P, centred on the
pelvis centroid). The perspective DRR is real ray marching: one ray per pixel from a point source,
sampled every half voxel with trilinear interpolation (scipy map_coordinates, order=1), and the
samples times the step length summed into the line integral L = sum(mu * ds).

Geometry (our choice, written to v04/geom.json): source-to-detector SDD = 1000 mm, source-to-pelvis
centroid SOD = 700 mm. The perspective images are resampled on the SAME grid as the parallel ones
at the plane through the centroid ("isocentre" plane, 1.5 mm), so the two can be overlaid pixel by
pixel; on the physical detector each pixel is M = SDD/SOD times larger.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from PIL import Image

subj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seg = subj / "segmentations"
img = nib.load(subj / "ct.nii.gz")
assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
sp = np.array(img.header.get_zooms()[:3], dtype=float)
ct = np.asanyarray(img.dataobj).astype(np.float32)
load = lambda n: np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
parts = {n: load(n) for n in ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]}
parts["spine"] = load("vertebrae_L5") | load("vertebrae_L4") | load("vertebrae_L3")
pelvis = parts["hip_left"] | parts["hip_right"] | parts["sacrum"]
allm = np.zeros_like(pelvis)
for m in parts.values(): allm |= m
idx = np.argwhere(allm)
lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(ct.shape) - 1)
sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
ct, pelvis = ct[sl], pelvis[sl]
print("crop", ct.shape)                                   # must match hero meta crop_shape
pc = np.argwhere(pelvis).mean(0) * sp                     # mm, voxel-space (R, A, S)
s0 = sp[0]
assert np.allclose(sp, s0)

MU_W = 0.02                                               # 1/mm, as in hero_assets.py
mu = (MU_W * (1 + np.clip(ct, -1000, None) / 1000.0)).astype(np.float32)
SDD, SOD = 1000.0, 700.0
M = SDD / SOD

def par(axis):
    return mu.sum(axis=axis) * s0                         # (R,S) for axis 1, (A,S) for axis 0

def persp(view):
    """ray-march from a point source through the isocentre-plane grid (same grid as par)"""
    nR, nA, nS = mu.shape
    pcv = pc / s0                                         # centroid in voxel units
    if view == "ap":                                      # source anterior: A index = pcA + SOD
        src = np.array([pcv[0], pcv[1] + SOD / s0, pcv[2]])
        u, w = np.meshgrid(np.arange(nR), np.arange(nS), indexing="ij")   # iso points (i, pcA, k)
        iso = np.stack([u, np.full_like(u, pcv[1], dtype=float), w], -1).astype(np.float64)
        ax = 1
    else:                                                 # LAT: source on the patient's right (+R)
        src = np.array([pcv[0] + SOD / s0, pcv[1], pcv[2]])
        u, w = np.meshgrid(np.arange(nA), np.arange(nS), indexing="ij")
        iso = np.stack([np.full_like(u, pcv[0], dtype=float), u, w], -1).astype(np.float64)
        ax = 0
    d = iso - src                                          # ray direction (voxel units), per pixel
    n_ax = mu.shape[ax]
    planes = np.arange(-0.5, n_ax - 0.5 + 1e-6, 0.5)       # sample every half voxel along the main axis
    L = np.zeros(iso.shape[:2], np.float64)
    cov = np.ones(iso.shape[:2], bool)                     # ray stays inside the CT crop all the way
    lim = np.array(mu.shape) - 1
    seglen = np.linalg.norm(d, axis=-1) / np.abs(d[..., ax]) * 0.5 * s0   # mm per half-voxel step
    for p in planes:
        s = (p - src[ax]) / d[..., ax]
        pts = src + d * s[..., None]
        if 0 <= p <= lim[ax]:
            cov &= ((pts >= 0) & (pts <= lim)).all(-1)
        L += ndimage.map_coordinates(mu, pts.reshape(-1, 3).T, order=1, mode="constant", cval=0.0).reshape(L.shape)
    COV[view] = cov
    return L * seglen

COV = {}
L = {"par_ap": par(1), "par_lat": par(0), "persp_ap": persp("ap"), "persp_lat": persp("lat")}
for k, v in L.items(): print(k, v.shape, float(v.min()), float(v.max()))

# the CT is a crop: diverging rays near the edges leave it through its sides, so both images are cut
# to the rectangle where every perspective ray stays inside the crop (separable in the two axes)
CROP = {}
for v in ("ap", "lat"):
    c = COV[v]; a_ = np.where(c.any(1))[0]; b_ = np.where(c.any(0))[0]
    CROP[v] = (slice(a_[0], a_[-1] + 1), slice(b_[0], b_[-1] + 1))
    assert c[CROP[v]].all()
print("crop rect", {v: [(x.start, x.stop) for x in CROP[v]] for v in CROP})
Lc = {k: v[CROP[k.split("_")[1]]] for k, v in L.items()}
# display: one window per view (from its parallel image), applied to parallel and perspective alike
WIN = {v: np.percentile(Lc[f"par_{v}"], [2, 99.7]) for v in ("ap", "lat")}
lo_, hi_ = WIN["ap"]
GAMMA = 2.2
disp = lambda a, v="ap": np.clip((a - WIN[v][0]) / (WIN[v][1] - WIN[v][0]), 0, 1) ** GAMMA
orient = lambda a: a.T[::-1, ::-1]                         # rows: S on top; AP: patient R on viewer's left; LAT: anterior on the left
imgs = {k: orient(disp(v, k.split("_")[1])) for k, v in Lc.items()}
for k, a in imgs.items():
    Image.fromarray((a * 255).astype(np.uint8)).save(out / f"{k}.png")

# overlay: parallel in cyan, perspective in warm; where they agree the result is near white/grey
for v in ("ap", "lat"):
    p, q = imgs[f"par_{v}"], imgs[f"persp_{v}"]
    rgb = np.stack([q, 0.55 * p + 0.45 * q, p], -1)
    rgb = np.clip(rgb * 1.15, 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out / f"overlay_{v}.png")

# the saturated / exp version and the intensity mapping: histogram of L (AP parallel)
h, e = np.histogram(Lc["par_ap"], bins=80, range=(0, float(Lc["par_ap"].max())))
# pixel spacing: the same parallel AP with 6 mm and 12 mm pixels (block means of the line integral)
def binned(a, f):
    nx, ny = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:nx, :ny].reshape(nx // f, f, ny // f, f).mean((1, 3))
for f in (4, 8):
    Image.fromarray((orient(disp(binned(Lc["par_ap"], f))) * 255).astype(np.uint8)).save(out / f"par_ap_{int(f * s0)}mm.png")

# a single real ray (perspective AP) through the right acetabulum / femoral head region, for the profile plot
fr = np.argwhere(np.asanyarray(nib.load(seg / "femur_right.nii.gz").dataobj)[sl] > 0)
head = fr[fr[:, 2] > np.percentile(fr[:, 2], 92)].mean(0)    # top of the right femur ~ femoral head (voxel idx)
pcv = pc / s0
src = np.array([pcv[0], pcv[1] + SOD / s0, pcv[2]])
iso = np.array([head[0], pcv[1], head[2]])
d = iso - src
ts = np.linspace((mu.shape[1] - 1 - src[1]) / d[1], (0 - src[1]) / d[1], 600)   # A from front to back of the crop
pts = src + d * ts[:, None]
vals = ndimage.map_coordinates(mu, pts.T, order=1, mode="constant")
step = np.linalg.norm(d) * abs(ts[1] - ts[0]) * s0
s_mm = (ts - ts[0]) * np.linalg.norm(d) * s0
Lray = float(vals.sum() * step)
# where that pixel sits in the displayed perspective AP image (row, col)
r_img, c_img = CROP["ap"][1].stop - 1 - head[2], CROP["ap"][0].stop - 1 - head[0]
to_scene = lambda p_vox: [float((p_vox[0] * s0 - pc[0]) * 0.01), float((p_vox[2] * s0 - pc[2]) * 0.01), float(-(p_vox[1] * s0 - pc[1]) * 0.01)]

# coarse axial slice of mu (for the "sum down each column" diagram): 9 mm cells, at the femoral-head height
k = int(round(head[2]))
cell = 6
sl2 = mu[:, :, k]
nx, ny = (sl2.shape[0] // cell) * cell, (sl2.shape[1] // cell) * cell
coarse = sl2[:nx, :ny].reshape(nx // cell, cell, ny // cell, cell).mean((1, 3))   # (R, A) cells
row_par = L["par_ap"][:, k]  # full crop width, R index order                                                         # that AP image row (R)

json.dump({
    "sdd_mm": SDD, "sod_mm": SOD, "M_iso": M, "iso_spacing_mm": s0, "det_spacing_mm": s0 * M,
    "mu_water_per_mm": MU_W, "window_L": [float(lo_), float(hi_)], "gamma": GAMMA,
    "img_ap": list(imgs["par_ap"].shape), "img_lat": list(imgs["par_lat"].shape),
    "box_z_mm": [float(-(mu.shape[1] - 1) * s0 + pc[1]), float(pc[1])],
    "L_hist": {"counts": h.tolist(), "edges": e.tolist()},
    "L_range": [float(Lc["par_ap"].min()), float(Lc["par_ap"].max())],
    "crop_rect": {v: [[int(x.start), int(x.stop)] for x in CROP[v]] for v in CROP},
    "ray": {"s_mm": np.round(s_mm, 2).tolist()[::4], "mu": np.round(vals, 5).tolist()[::4], "L": Lray,
            "src_scene": to_scene(src), "iso_scene": to_scene(iso), "rowcol": [float(r_img), float(c_img)],
            "pixel_value_persp": float(L["persp_ap"][int(round(head[0])), int(round(head[2]))])},
    "axial": {"slice_k": k, "cell_mm": cell * s0, "grid_RA": np.round(coarse, 5).tolist(),
              "row_L": np.round(row_par, 3).tolist()},
}, open(out / "v04.json", "w"))
print("ray L", Lray, "pixel", float(L["persp_ap"][int(round(head[0])), int(round(head[2]))]), "M", M)

# ---- landmark positions in the displayed images (parallel vs perspective), for on-screen arrows ----
lm = json.load(open(out.parent / "v03" / "points.json"))["landmarks_mm"] if (out.parent / "v03" / "points.json").exists() else {}
def img_pos(p_scene_mm, view, kind):
    x, y, z = p_scene_mm                                   # scene mm (x = R, y = S, z = P) relative to the centroid
    if view == "ap":
        k = 1.0 if kind == "par" else SOD / (SOD + z)        # source at z = -SOD
        u, w = x * k, y * k                                  # iso-plane (R, S) mm
        i, kk = (u + pc[0]) / s0, (w + pc[2]) / s0
        cr = CROP["ap"]; return [float(cr[1].stop - 1 - kk), float(cr[0].stop - 1 - i)]   # (row, col)
    k = 1.0 if kind == "par" else SOD / (SOD - x)            # LAT source at x = +SOD
    a_mm, w = -z * k, y * k                                  # A = -z
    j, kk = (a_mm + pc[1]) / s0, (w + pc[2]) / s0
    cr = CROP["lat"]; return [float(cr[1].stop - 1 - kk), float(cr[0].stop - 1 - j)]
heads = {}
for side in ("left", "right"):
    f = np.argwhere(np.asanyarray(nib.load(seg / f"femur_{side}.nii.gz").dataobj)[sl] > 0)
    hv = f[f[:, 2] > np.percentile(f[:, 2], 92)].mean(0) * s0 - pc
    heads[f"femhead_{side}"] = [float(hv[0]), float(hv[2]), float(-hv[1])]
pts_ = {**{k: v for k, v in lm.items()}, **heads}
marks = {k: {f"{kind}_{v}": img_pos(p, v, kind) for v in ("ap", "lat") for kind in ("par", "persp")} for k, p in pts_.items()}
# extent of the AP image on the iso plane, in scene units [xmin, xmax, ymin, ymax] (x = R)
cr = CROP["ap"]
ext = [float((cr[0].start * s0 - pc[0]) * 0.01), float(((cr[0].stop - 1) * s0 - pc[0]) * 0.01),
       float((cr[1].start * s0 - pc[2]) * 0.01), float(((cr[1].stop - 1) * s0 - pc[2]) * 0.01)]
crl = CROP["lat"]
ext_lat = [float(-((crl[0].stop - 1) * s0 - pc[1]) * 0.01), float(-(crl[0].start * s0 - pc[1]) * 0.01),
           float((crl[1].start * s0 - pc[2]) * 0.01), float(((crl[1].stop - 1) * s0 - pc[2]) * 0.01)]  # [zmin, zmax, ymin, ymax]
j = json.load(open(out / "v04.json"))
j.update({"marks": marks, "landmarks_scene_mm": pts_, "ap_iso_extent": ext, "lat_iso_extent": ext_lat})
json.dump(j, open(out / "v04.json", "w"))
print("marks", json.dumps(marks, indent=0)[:900], "ext", ext, ext_lat)
