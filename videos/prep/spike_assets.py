"""Spike asset prep: CT -> bone meshes (OBJ) + parallel-beam DRRs (PNG) for the 3D test shot.

Throwaway. Usage: python spike_assets.py <subject_dir> <out_dir>
Volume axes are (L, P, S) at 1 mm. Scene coordinates: x = left-right, y = up (S), z = front-back (P).
"""
import sys, json
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import measure, filters
from PIL import Image

subj, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

ct = np.asanyarray(nib.load(subj / "volume.nii.gz").dataobj).astype(np.float32)
mask = np.asanyarray(nib.load(subj / "mask.nii.gz").dataobj)
center = np.array(ct.shape) / 2.0
SCALE = 0.01  # 1 mm -> 0.01 scene units, so the pelvis is ~3-4 units wide

# ---- meshes: one OBJ per bone group ----
groups = {"pelvis": [1, 2, 4, 7], "femurs": [5, 6], "spine": [3]}
for name, labels in groups.items():
    # blur the binary mask before marching cubes so the surface is smooth, not voxel stair-steps
    m = filters.gaussian(np.isin(mask, labels).astype(np.float32), sigma=1.6)
    verts, faces, normals, _ = measure.marching_cubes(m, level=0.5, step_size=2)
    v = (verts - center) * SCALE
    # (L, P, S) -> (x, y, z) = (-L, S, P): patient's left on screen right as in radiology AP view
    xyz = np.stack([-v[:, 0], v[:, 2], v[:, 1]], axis=1)
    nrm = np.stack([-normals[:, 0], normals[:, 2], normals[:, 1]], axis=1)
    with open(out / f"{name}.obj", "w") as f:
        f.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in xyz)
        f.writelines(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in nrm)
        # x was mirrored, so flip winding to keep faces pointing outwards
        f.writelines(f"f {a+1}//{a+1} {c+1}//{c+1} {b+1}//{b+1}\n" for a, b, c in faces)
    print(name, len(xyz), "verts")

# ---- DRR: HU -> mu, integrate along a ray direction (parallel beam) ----
MU_WATER = 0.02  # 1/mm, roughly at diagnostic energies
mu = MU_WATER * (1 + np.clip(ct, -1000, None) / 1000.0)

def drr(axis):
    line_integral = mu.sum(axis=axis)        # sum of mu * 1 mm along each ray
    # 1 - exp(-L) saturates through ~20 cm of soft tissue, so display the line integral
    # itself with percentile windowing (what a radiograph's log-contrast effectively shows)
    lo, hi = np.percentile(line_integral, [2, 99.7])
    return np.clip((line_integral - lo) / (hi - lo), 0, 1)

ap = drr(axis=1).T[::-1, ::-1]   # rays along P -> image (S, L), S up, patient left on right
lat = drr(axis=0).T[::-1, :]     # rays along L -> image (S, P)
for name, img in [("drr_ap", ap), ("drr_lat", lat)]:
    Image.fromarray((np.clip(img, 0, 1) ** 2.2 * 255).astype(np.uint8)).save(out / f"{name}.png")

json.dump({"shape_mm": list(map(int, ct.shape)), "scale": SCALE,
           "ap_size": [ap.shape[1] * SCALE, ap.shape[0] * SCALE],
           "lat_size": [lat.shape[1] * SCALE, lat.shape[0] * SCALE]},
          open(out / "meta.json", "w"), indent=1)
print("done", ap.shape, lat.shape)
