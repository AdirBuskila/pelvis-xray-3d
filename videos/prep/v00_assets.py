"""Video 0 assets: a few extra pelvises (the "prior" scene) and the bicylinder (ambiguity scene).
Usage: python v00_assets.py <extracted_dir> <out_dir> <subject> [<subject> ...]"""
import sys
from pathlib import Path
import numpy as np, nibabel as nib
from skimage import measure, filters

root, out, subs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)

def write_obj(path, verts, faces, normals, flip=False):
    with open(path, "w") as fo:
        fo.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in verts)
        fo.writelines(f"vn {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in normals)
        for a, b, c in faces:
            fo.write(f"f {a+1}//{a+1} {c+1}//{c+1} {b+1}//{b+1}\n" if flip else f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")

# --- extra pelvises, same scene convention as hero_assets.py (x=R, y=S, z=P, centred, 0.01/mm) ---
for s in subs:
    seg = root / s / "segmentations"
    m = None
    for n in ["hip_left", "hip_right", "sacrum"]:
        a = np.asanyarray(nib.load(seg / f"{n}.nii.gz").dataobj) > 0
        m = a if m is None else (m | a)
    sp = np.array(nib.load(seg / "hip_left.nii.gz").header.get_zooms()[:3], float)
    idx = np.argwhere(m); lo, hi = idx.min(0) - 4, idx.max(0) + 5
    m = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    v, f, nrm, _ = measure.marching_cubes(filters.gaussian(m.astype(np.float32), 1.2), 0.5, spacing=tuple(sp), step_size=2)
    v = v - np.argwhere(m).mean(0) * sp
    xyz = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1) * 0.01
    write_obj(out / f"pelvis_{s}.obj", xyz, f, np.stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]], 1), flip=True)
    print("pelvis", s, len(xyz))

# --- bicylinder (Steinmetz solid): x^2+y^2<=1 (axis z) AND y^2+z^2<=1 (axis x) ---
# silhouette along z is the disk x^2+y^2<=1 and along x the disk y^2+z^2<=1: same shadows as a sphere
n = 160; g = np.linspace(-1.15, 1.15, n); X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
sdf = np.maximum(np.sqrt(X**2 + Y**2), np.sqrt(Y**2 + Z**2)) - 1.0
v, f, nrm, _ = measure.marching_cubes(sdf, 0.0, spacing=(g[1] - g[0],) * 3)
v = v + g[0]
write_obj(out / "bicylinder.obj", v, f, -nrm)   # sdf gradient points outward; marching cubes normals point to lower values
print("bicylinder", len(v))
