"""Video 5 assets ("representing 3D shape"), all from the hero pelvis s1390.

Usage: python -I v05_assets.py <extracted_dir> <out_dir>

Writes into <out_dir> (videos/public/data/v05):
  points.json        surface point cloud (area-weighted samples of the marching-cubes surface)
  sdf_coronal.png    signed distance field on the coronal plane through the pelvis centroid, uint8:
                     value = (clip(sdf_mm, -SDF_R, SDF_R) + SDF_R) / (2 SDF_R) * 255, negative = inside
  mc16/32/64/128.obj marching cubes of the n^3 occupancy grid (same cube as hero occ*.json),
                     faces sorted bottom-up so a draw range reveals the mesh like a sweep
  grid64_coronal.json the 64x64 coronal layer of the soft occupancy (for marching squares)
  mc_cases.json      triangles of a few single-cube cases (skimage marching cubes on a 2x2x2 cube)
  ct_coronal_full.png the full-FOV coronal CT slice; ct64_coronal.png / mask64_coronal.png resampled
  stats.json         every number shown in the video
Scene convention as in hero_assets.py: x = patient Right, y = Superior, z = Posterior, 0.01 units/mm,
centred on the hero pelvis centroid.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure, filters
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
S = "s1390"
SCALE = 0.01
seg = root / S / "segmentations"
img = nib.load(root / S / "ct.nii.gz")
assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
sp = np.array(img.header.get_zooms()[:3], float)
ct = np.asanyarray(img.dataobj).astype(np.float32)
load = lambda n: np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
pelvis = load("hip_left") | load("hip_right") | load("sacrum")
pc = np.argwhere(pelvis).mean(0) * sp          # centroid in full-volume mm (same scene origin as hero)
stats = {"subject": S, "ct_shape": list(map(int, ct.shape)), "spacing_mm": sp.tolist(),
         "ct_voxels": int(np.prod(ct.shape))}

def to_scene(p_mm):
    q = p_mm - pc
    return np.stack([q[:, 0], q[:, 2], -q[:, 1]], axis=1) * SCALE

def write_obj(path, v_mm, f, nrm, sort_y=False):
    xyz = to_scene(v_mm)
    nn = np.stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]], axis=1)
    if sort_y:
        f = f[np.argsort(xyz[f].mean(1)[:, 1], kind="stable")]
    with open(path, "w") as fo:
        fo.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in xyz)
        fo.writelines(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in nn)
        fo.writelines(f"f {a+1}//{a+1} {c+1}//{c+1} {b+1}//{b+1}\n" for a, b, c in f)   # z flip -> reverse winding
    return len(xyz), len(f)

# ---------------- crop around the pelvis (with margin) for the heavy work ----------------
idx = np.argwhere(pelvis)
plo, phi = idx.min(0), idx.max(0)
lo = np.maximum(plo - 20, 0); hi = np.minimum(phi + 21, np.array(pelvis.shape))
crop = pelvis[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
off = lo * sp                                    # mm offset of the crop

from scipy.spatial import cKDTree
# reference surface: full-res marching cubes (step 1), as dense points
vr, _, _, _ = measure.marching_cubes(filters.gaussian(crop.astype(np.float32), 1.1), 0.5, spacing=tuple(sp))
ref_tree = cKDTree(vr + off)

# ---------------- point cloud ----------------
v, f, n_, _ = measure.marching_cubes(filters.gaussian(crop.astype(np.float32), 1.1), 0.5, spacing=tuple(sp), step_size=2)
v = v + off
tri = v[f]
area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
rng = np.random.default_rng(0)
NPTS = 7000
pick = rng.choice(len(f), NPTS, p=area / area.sum())
r1, r2 = rng.random(NPTS), rng.random(NPTS)
s1 = np.sqrt(r1)
bary = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
pts = (tri[pick] * bary[:, :, None]).sum(1)
json.dump({"n": NPTS, "p": np.round(to_scene(pts), 4).flatten().tolist()}, open(out / "points.json", "w"))
stats["points"] = NPTS
stats["surface_area_cm2"] = float(area.sum() / 100)

# ---------------- signed distance field, coronal plane through the centroid ----------------
SDF_R = 30.0
d_out = ndimage.distance_transform_edt(~crop, sampling=sp)
d_in = ndimage.distance_transform_edt(crop, sampling=sp)
sdf = np.where(crop, -(d_in - 0.5 * sp[0]), d_out - 0.5 * sp[0])    # ~0 on the boundary between voxels
ja = int(round(pc[1] / sp[1])) - lo[1]
sl = sdf[:, ja, :]                                                  # (R, S)
u8 = ((np.clip(sl, -SDF_R, SDF_R) + SDF_R) / (2 * SDF_R) * 255).round().astype(np.uint8)
Image.fromarray(u8.T[::-1, ::-1]).save(out / "sdf_coronal.png")    # rows: S top; col 0 = patient right
# plane placement in scene units: centre + size (x = R, y = S) and its z (A index -> scene z)
p0 = to_scene(np.array([off + np.array([0, ja * sp[1], 0])]))[0]
p1 = to_scene(np.array([off + np.array([crop.shape[0] * sp[0], ja * sp[1], crop.shape[2] * sp[2]])]))[0]
stats["sdf_plane"] = {"cx": float((p0[0] + p1[0]) / 2), "cy": float((p0[1] + p1[1]) / 2), "z": float(p0[2]),
                      "w": float(abs(p1[0] - p0[0])), "h": float(abs(p1[1] - p0[1])), "range_mm": SDF_R}
stats["sdf_min_mm"] = float(sdf.min())

# ---------------- occupancy grids (identical cube + sampling to hero_assets.py) ----------------
ext_mm = (phi - plo + 1) * sp
cube_mm = ext_mm.max() * 1.06
c_mm = (plo + phi + 1) / 2 * sp
stats["cube_mm"] = float(cube_mm)
stats["cube_center_scene"] = to_scene(c_mm[None])[0].tolist()
stats["pelvis_extent_mm"] = ext_mm.tolist()
grids, res = {}, {}
for n in (16, 32, 64, 128):
    g = (np.arange(n) + 0.5) / n * cube_mm - cube_mm / 2
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    vox = (np.stack([X, Y, Z], -1).reshape(-1, 3) + c_mm) / sp
    blurred = filters.gaussian(pelvis.astype(np.float32), sigma=0.6 * (cube_mm / n) / sp[0])
    val = ndimage.map_coordinates(blurred, vox.T, order=1, mode="constant").reshape(n, n, n)
    grids[n] = val
    cell = cube_mm / n
    v, f, nrm, _ = measure.marching_cubes(np.pad(val, 1), 0.5, spacing=(cell,) * 3)
    v = v - cell + c_mm - cube_mm / 2 + cell / 2                      # padded index -> mm (cell centres)
    nv, nf = write_obj(out / f"mc{n}.obj", v, f, nrm, sort_y=True)   # faces bottom-up: drawRange = sweep
    # symmetric mean distance between this mesh and the full-res reference surface (both directions)
    d1 = ref_tree.query(v)[0]
    d2 = cKDTree(v).query(ref_tree.data)[0]
    dv = np.concatenate([d1, d2])
    res[n] = {"n": n, "cells": n ** 3, "occupied": int((val > 0.5).sum()), "cell_mm": cell,
              "mb_float32": n ** 3 * 4 / 2 ** 20, "verts": nv, "faces": nf,
              "mesh_err_mm": float(dv.mean()), "mesh_err_p95_mm": float(np.percentile(dv, 95))}
stats["res"] = res
print("resolutions", json.dumps(res, indent=1))

# ---------------- 64x64 coronal layer of the soft grid (marching squares) ----------------
g64 = grids[64]
j64 = int(np.argmax((g64 > 0.5).sum((0, 2))))          # the coronal layer with the most bone
lay = g64[:, j64, :]                                     # (R, S)
json.dump({"n": 64, "j": j64, "v": np.round(lay.T[::-1, ::-1], 3).tolist()},  # [row=S top][col=patient right first]
          open(out / "grid64_coronal.json", "w"))
stats["grid64_layer"] = j64

# ---------------- single-cube marching cubes cases ----------------
cases = {"one": [(0, 0, 0)], "edge": [(0, 0, 0), (1, 0, 0)], "three": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
         "face": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], "diag": [(0, 0, 0), (1, 1, 1)]}
mc = {}
for name, cs in cases.items():
    a = np.zeros((2, 2, 2), np.float32)
    for c in cs: a[c] = 1.0
    v, f, _, _ = measure.marching_cubes(a, 0.5, method="lewiner")
    mc[name] = {"corners": [list(c) for c in cs], "v": np.round(v, 4).flatten().tolist(), "f": f.flatten().tolist()}
json.dump(mc, open(out / "mc_cases.json", "w"))

# ---------------- crop + resample: full-FOV coronal CT, the cube, the 64^2 result ----------------
W_LO, W_HI = -350, 1100
win = lambda a: (np.clip((a - W_LO) / (W_HI - W_LO), 0, 1) * 255).astype(np.uint8)
g = (np.arange(64) + 0.5) / 64 * cube_mm - cube_mm / 2
a_mm = c_mm[1] + g[j64]                                  # the same coronal layer as grid64_coronal.json
jfull = int(round(a_mm / sp[1]))
Image.fromarray(win(ct[:, jfull, :]).T[::-1, ::-1]).save(out / "ct_coronal_full.png")
X, Z = np.meshgrid(g, g, indexing="ij")
vox = np.stack([X + c_mm[0], np.full_like(X, a_mm), Z + c_mm[2]], -1).reshape(-1, 3) / sp
ct64 = ndimage.map_coordinates(ct, vox.T, order=1, mode="constant", cval=-1000).reshape(64, 64)
Image.fromarray(win(ct64).T[::-1, ::-1]).save(out / "ct64_coronal.png")
m64 = ndimage.map_coordinates(filters.gaussian(pelvis.astype(np.float32), sigma=0.6 * (cube_mm / 64) / sp[0]),
                              vox.T, order=1, mode="constant").reshape(64, 64) > 0.5
Image.fromarray((m64 * 255).astype(np.uint8).T[::-1, ::-1]).save(out / "mask64_coronal.png")
# pixel geometry of the full slice: width = R voxels, height = S voxels (display flips both)
fov = np.array(ct.shape) * sp
stats["crop"] = {
    "fov_mm": fov.tolist(), "full_px": [int(ct.shape[0]), int(ct.shape[2])],
    # bbox and cube in display fractions (0..1, x from the left = patient right, y from the top = superior)
    "bbox": [float(1 - (phi[0] + 1) * sp[0] / fov[0]), float(1 - (phi[2] + 1) * sp[2] / fov[2]),
             float(ext_mm[0] / fov[0]), float(ext_mm[2] / fov[2])],
    "cube": [float(1 - (c_mm[0] + cube_mm / 2) / fov[0]), float(1 - (c_mm[2] + cube_mm / 2) / fov[2]),
             float(cube_mm / fov[0]), float(cube_mm / fov[2])],
}
json.dump(stats, open(out / "stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "res"}, indent=1))
