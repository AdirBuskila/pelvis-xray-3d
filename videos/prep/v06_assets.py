"""Video 6 assets ("measuring a reconstruction"): real metric numbers between real masks.

Usage: python -I v06_assets.py <extracted_dir> <out_dir>

Ground truth (GT) = the hero pelvis s1390 (hip_left | hip_right | sacrum) on the same 64^3 cube as the
hero occupancy grid (hero_assets.py). "Predictions" are real masks changed in known ways:
  other  : another person's pelvis (s1361), centroid-aligned, on the same cube
  wing   : GT with the top of the left iliac wing removed
  shift  : GT moved 2 cells (10.3 mm) superior
  rim    : GT with the lateral rim of the left acetabulum shaved off (changes the CE angle)
  empty  : nothing predicted
Every metric is computed on the binary 64^3 grids with the cell size in mm (as the project will),
surface distances via Euclidean distance transforms (scipy). Chamfer uses points sampled on the
marching-cubes meshes. Writes metrics.json, meshes (mesh_*.json with per-vertex distance), overlap
voxels (ovl_*.json), a coronal layer for the 2D diagrams (slice.json), CE-angle slices (ce_*.png).
Scene convention as hero: x = patient Right, y = Superior, z = Posterior, 0.01 units/mm.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure, filters
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
SCALE, N = 0.01, 64

def load_subject(s):
    seg = root / s / "segmentations"
    img = nib.load(seg / "hip_left.nii.gz")
    assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    L = lambda n: np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
    return {k: L(k) for k in ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]}, \
        np.array(img.header.get_zooms()[:3], float)

H, sp = load_subject("s1390")
gt_full = H["hip_left"] | H["hip_right"] | H["sacrum"]
pc = np.argwhere(gt_full).mean(0) * sp
idx = np.argwhere(gt_full); plo, phi = idx.min(0), idx.max(0)
cube_mm = ((phi - plo + 1) * sp).max() * 1.06              # identical to hero_assets.py
c_mm = (plo + phi + 1) / 2 * sp
cell = cube_mm / N
g = (np.arange(N) + 0.5) / N * cube_mm - cube_mm / 2

def to_scene(p_mm):
    q = p_mm - pc
    return np.stack([q[:, 0], q[:, 2], -q[:, 1]], axis=1) * SCALE

def to_grid(mask, spacing, centre_mm):
    """soft occupancy on the 64^3 cube centred at centre_mm (same sampling as the hero occ grids)"""
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    vox = (np.stack([X, Y, Z], -1).reshape(-1, 3) + centre_mm) / spacing
    b = filters.gaussian(mask.astype(np.float32), sigma=0.6 * cell / spacing[0])
    return ndimage.map_coordinates(b, vox.T, order=1, mode="constant").reshape(N, N, N)

# ---------------- the cases ----------------
soft = {"gt": to_grid(gt_full, sp, c_mm)}
# another person, centroid-aligned
O, sp_o = load_subject("s1361")
other = O["hip_left"] | O["hip_right"] | O["sacrum"]
pc_o = np.argwhere(other).mean(0) * sp_o
soft["other"] = to_grid(other, sp_o, c_mm - pc + pc_o)
# the top of the left iliac wing removed
import os
WING_MM = float(os.environ.get("WING_MM", 75.0))
hl = H["hip_left"].copy()
top = np.argwhere(hl)[:, 2].max()
cut = hl.copy(); cut[:, :, : top - int(WING_MM / sp[2]) + 1] = False
soft["wing"] = to_grid((hl & ~cut) | H["hip_right"] | H["sacrum"], sp, c_mm)
wing_removed_cm3 = float(cut.sum() * sp.prod() / 1000)
# shifted 2 cells superior (S is grid axis 2)
SHIFT = 2
sh = np.zeros_like(soft["gt"]); sh[:, :, SHIFT:] = soft["gt"][:, :, :-SHIFT]
soft["shift"] = sh

# ---------------- CE (Wiberg) angle, simplified, on the coronal slice through the femoral head centre ----------------
def head_centre(f):
    t = np.argwhere(f)[:, 2].max(); sub = f.copy(); sub[:, :, : t - int(60 / sp[2])] = False
    d = ndimage.distance_transform_edt(sub, sampling=sp)
    return np.array(np.unravel_index(np.argmax(d), d.shape)), float(d.max())

def ce_angle(hip, c, r, lateral_sign):
    """angle between the vertical through the head centre and the line to the lateral acetabular edge:
    the most lateral hip-bone point above the centre in a band just outside the head surface"""
    pts = np.argwhere(hip[:, c[1], :])
    dx = (pts[:, 0] - c[0]) * sp[0] * lateral_sign; dz = (pts[:, 1] - c[2]) * sp[2]
    dist = np.hypot(dx, dz)
    ok = (dz >= 0) & (dist >= r - 3) & (dist <= r + 8)
    ang = np.degrees(np.arctan2(dx[ok], dz[ok]))
    k = int(np.argmax(ang))
    return float(ang[k]), pts[ok][k]

cL, rL = head_centre(H["femur_left"])
ce_gt, edge_gt = ce_angle(H["hip_left"], cL, rL, -1)          # left: lateral = decreasing R index
# shave the lateral rim: hip_left voxels near the head whose lateral angle exceeds (CE - RIM_DEG)
RIM_DEG = 10.0
ii, jj, kk = np.nonzero(H["hip_left"])
dx = (ii - cL[0]) * sp[0] * -1; dy = (jj - cL[1]) * sp[1]; dz = (kk - cL[2]) * sp[2]
ang = np.degrees(np.arctan2(dx, dz))
shave = (ang > ce_gt - RIM_DEG) & (dz > -5) & (np.sqrt(dx**2 + dy**2 + dz**2) < rL + 12) & (dx > 0)
hl_rim = H["hip_left"].copy(); hl_rim[ii[shave], jj[shave], kk[shave]] = False
ce_rim, edge_rim = ce_angle(hl_rim, cL, rL, -1)
rim_removed_cm3 = float(shave.sum() * sp.prod() / 1000)
soft["rim"] = to_grid(hl_rim | H["hip_right"] | H["sacrum"], sp, c_mm)

# CE slice pictures: coronal slice through the head centre, cropped around the left hip joint
def ce_png(hip, removed, name):
    sl = hip[:, cL[1], :]; fs = H["femur_left"][:, cL[1], :]
    img = np.zeros(sl.shape + (3,), np.uint8)
    img[fs] = (70, 110, 190); img[sl] = (236, 228, 208)
    if removed is not None: img[removed[:, cL[1], :]] = (255, 110, 120)
    img = img.transpose(1, 0, 2)[::-1, ::-1]                    # rows: S top; cols: patient right first
    W_, H_ = sl.shape
    cx, cy = W_ - 1 - cL[0], H_ - 1 - cL[2]
    half = int(75 / sp[0])
    box = (cx - half, cy - int(1.1 * half), cx + half, cy + int(0.9 * half))
    Image.fromarray(img).crop(box).resize((2 * half * 8, 2 * half * 8), Image.NEAREST).save(out / name)
    return box
rim_mask = H["hip_left"] & ~hl_rim
box = ce_png(H["hip_left"], None, "ce_gt.png")
ce_png(hl_rim, rim_mask, "ce_rim.png")
def px(p):  # voxel (i, k) on the slice -> fraction (0..1) inside the cropped picture
    W_, H_ = H["hip_left"].shape[0], H["hip_left"].shape[2]
    return [float((W_ - 1 - p[0] - box[0] + 0.5) / (box[2] - box[0])), float((H_ - 1 - p[1] - box[1] + 0.5) / (box[3] - box[1]))]
ce = {"gt": ce_gt, "rim": ce_rim, "head_r_mm": rL, "rim_deg": RIM_DEG, "rim_removed_cm3": rim_removed_cm3,
      "centre_px": px([cL[0], cL[2]]), "edge_gt_px": px(edge_gt), "edge_rim_px": px(edge_rim),
      "r_px": float(rL / sp[0] / (box[2] - box[0])),
      "centre_scene": to_scene((cL * sp)[None])[0].tolist(), "edge_gt_scene": to_scene((np.array([edge_gt[0], cL[1], edge_gt[1]]) * sp)[None])[0].tolist(),
      "edge_rim_scene": to_scene((np.array([edge_rim[0], cL[1], edge_rim[1]]) * sp)[None])[0].tolist()}
print("CE", ce)

# ---------------- metrics ----------------
def surface(m):
    return m & ~ndimage.binary_erosion(m)

def surf_dists(a, b):
    """distances (mm) from each surface voxel of a to the nearest surface voxel of b"""
    sa, sb = surface(a), surface(b)
    return ndimage.distance_transform_edt(~sb, sampling=cell)[sa]

def metrics(a, b, tau=cell):
    """a = ground truth, b = prediction (binary grids)"""
    inter, sa, sb = (a & b).sum(), a.sum(), b.sum()
    m = {"dice": float(2 * inter / (sa + sb)) if sa + sb else 1.0,
         "iou": float(inter / (a | b).sum()) if (a | b).any() else 1.0,
         "tp": int(inter), "fp": int((b & ~a).sum()), "fn": int((a & ~b).sum()), "vol_gt": int(sa), "vol_pred": int(sb)}
    if not a.any() or not b.any():
        m.update({"hd": None, "hd95": None, "assd": None, "nsd": 0.0})   # undefined: one surface is missing
        return m
    dab, dba = surf_dists(a, b), surf_dists(b, a)
    m.update({"hd": float(max(dab.max(), dba.max())),
              "hd95": float(max(np.percentile(dab, 95), np.percentile(dba, 95))),
              "assd": float((dab.sum() + dba.sum()) / (len(dab) + len(dba))),
              "nsd": float(((dab <= tau).sum() + (dba <= tau).sum()) / (len(dab) + len(dba)))})
    return m

def mesh_of(sv):
    v, f, nrm, _ = measure.marching_cubes(np.pad(sv, 1), 0.5, spacing=(cell,) * 3)
    v = v - cell + c_mm - cube_mm / 2 + cell / 2
    return v, f, nrm

def sample_surface(v, f, n, seed):
    tri = v[f]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), n, p=area / area.sum()); r1, r2 = rng.random(n), rng.random(n); s1 = np.sqrt(r1)
    bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    return (tri[pick] * bary[:, :, None]).sum(1)

bins = {k: v > 0.5 for k, v in soft.items()}
bins["empty"] = np.zeros_like(bins["gt"])
meshes = {k: mesh_of(v) for k, v in soft.items()}
NPC = 4000
clouds = {k: sample_surface(v, f, NPC, 1 + i) for i, (k, (v, f, _)) in enumerate(meshes.items())}
trees = {k: cKDTree(p) for k, p in clouds.items()}

dense = {k: cKDTree(sample_surface(v, f, 60000, 50 + i)) for i, (k, (v, f, _)) in enumerate(meshes.items())}
def chamfer(a, b):
    """symmetric Chamfer (mm): mean nearest-neighbour distance a->b and b->a, averaged. The NPC points
    of one cloud are matched against a dense sampling of the other surface (sampling floor ~0.5 mm);
    identical clouds give exactly 0 (the sanity test)."""
    if a == b: return float(trees[a].query(clouds[a])[0].mean())
    return float((dense[b].query(clouds[a])[0].mean() + dense[a].query(clouds[b])[0].mean()) / 2)

M = {}
for k in ["gt", "other", "wing", "shift", "rim", "empty"]:
    M[k] = metrics(bins["gt"], bins[k])
    M[k]["chamfer"] = chamfer("gt", k) if k in clouds else None
M["gt_self_chamfer_resampled"] = float((dense["gt"].query(clouds["gt"])[0].mean()))   # the sampling floor
# a hand-checkable unit test: two 4x4x4 cubes, one shifted by one cell -> Dice = 2*48/(64+64) = 0.75
A = np.zeros((10, 10, 10), bool); A[2:6, 2:6, 2:6] = True
B = np.roll(A, 1, axis=0)
M["cubes"] = {"dice": float(2 * (A & B).sum() / (A.sum() + B.sum())), "inter": int((A & B).sum())}
# the distance histogram (both directions) for the "other" case
d_all = np.concatenate([surf_dists(bins["gt"], bins["other"]), surf_dists(bins["other"], bins["gt"])])
hist, edges = np.histogram(d_all, bins=24, range=(0, 60))
M["other_hist"] = {"counts": hist.tolist(), "edges": edges.tolist(), "overflow": int((d_all > 60).sum()),
                   "p95": float(np.percentile(d_all, 95))}
# how much each change removes / moves
M["info"] = {"cell_mm": cell, "cube_mm": cube_mm, "wing_mm": WING_MM, "wing_removed_cm3": wing_removed_cm3,
             "gt_cm3": float(gt_full.sum() * sp.prod() / 1000), "shift_mm": SHIFT * cell, "nsd_tau_mm": cell,
             "chamfer_points": NPC, "ce": ce}
json.dump(M, open(out / "metrics.json", "w"), indent=1)
for k in ["gt", "other", "wing", "shift", "rim", "empty"]:
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in M[k].items()})
print("info", M["info"]["wing_removed_cm3"], M["info"]["gt_cm3"], "cubes", M["cubes"], "selfCD", M["gt_self_chamfer_resampled"])

# ---------------- meshes with per-vertex distance to the other surface (dense samples) ----------------
def write_mesh(name, k, dist_to):
    v, f, nrm = meshes[k]
    d = {t: np.round(dense[t].query(v)[0], 1).tolist() for t in dist_to}
    xyz = to_scene(v); nn = np.stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]], 1)
    json.dump({"v": np.round(xyz, 4).flatten().tolist(), "n": np.round(nn, 3).flatten().tolist(),
               "f": f[:, [0, 2, 1]].flatten().tolist(), "d": d}, open(out / f"mesh_{name}.json", "w"))
    return len(v)
print("mesh verts", {
    "gt": write_mesh("gt", "gt", ["other", "wing", "shift"]),
    "other": write_mesh("other", "other", ["gt"]),
    "shift": write_mesh("shift", "shift", ["gt"]),
})

# ---------------- overlap voxels (TP / FP / FN centres) ----------------
centres = lambda m: np.round(to_scene((np.argwhere(m) + 0.5) / N * cube_mm - cube_mm / 2 + c_mm), 4).flatten().tolist()
for k in ["other"]:
    a, b = bins["gt"], bins[k]
    json.dump({"cell": cell * SCALE, "tp": centres(a & b), "fp": centres(b & ~a), "fn": centres(a & ~b)},
              open(out / f"ovl_{k}.json", "w"))

# ---------------- one coronal layer for the 2D diagrams ----------------
j = int(np.argmax(bins["gt"].sum((0, 2))))
flip = lambda a2: a2.T[::-1, ::-1]                      # (R, S) -> rows S top, cols patient right first
a2, b2 = flip(bins["gt"][:, j, :]), flip(bins["other"][:, j, :])
code = (a2 & b2) * 1 + (b2 & ~a2) * 2 + (a2 & ~b2) * 3
edt_gt = ndimage.distance_transform_edt(~surface(bins["gt"]), sampling=cell)
json.dump({"j": j, "code": code.astype(int).tolist(),
           "surf_gt": flip(surface(bins["gt"])[:, j, :]).astype(int).tolist(),
           "surf_pred": flip(surface(bins["other"])[:, j, :]).astype(int).tolist(),
           "edt": np.round(flip(edt_gt[:, j, :]), 1).tolist(),
           "counts": {"tp": int((code == 1).sum()), "fp": int((code == 2).sum()), "fn": int((code == 3).sum())}},
          open(out / "slice.json", "w"))
print("slice", j, {"tp": int((code == 1).sum()), "fp": int((code == 2).sum()), "fn": int((code == 3).sum())})
