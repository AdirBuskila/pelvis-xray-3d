"""Video 3 assets: a subsampled point cloud of the hero pelvis and three landmarks, for projection demos.

Usage: python v03_assets.py <hero_dir> <out_dir>
Reads the hero meshes (scene units: x = R, y = S, z = P, 0.01 per mm, centred on the pelvis centroid)
and writes v03/points.json: points (scene units) and landmarks in mm, plus the worked K[R|t] example
computed here, so every number the video shows is computed, not typed.
"""
import json, sys
from pathlib import Path
import numpy as np

hero, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def verts(name):
    v = [l.split()[1:4] for l in open(hero / f"{name}.obj") if l.startswith("v ")]
    return np.array(v, float)

pel = verts("pelvis")
hr = verts("hip_right")
sac = verts("sacrum")
rng = np.random.default_rng(3)
pts = pel[rng.choice(len(pel), 3000, replace=False)]

mm = lambda p: np.round(np.asarray(p) / 0.01).astype(int)   # scene units -> mm
up = hr[hr[:, 1] > np.percentile(hr[:, 1], 55)]
asis = up[np.argmin(up[:, 2])]                               # most anterior point of the upper right hip bone ~ right ASIS
low = pel[pel[:, 1] < np.percentile(pel[:, 1], 25)]
near_mid = low[np.abs(low[:, 0]) < 0.15]
pubis = near_mid[np.argmin(near_mid[:, 2])]                  # most anterior low midline point ~ pubic symphysis
sacp = sac[np.argmax(sac[:, 2])]                             # most posterior sacrum point
L = {"asis_r": mm(asis).tolist(), "pubis": mm(pubis).tolist(), "sacrum_post": mm(sacp).tolist()}

# ---- the worked example: an AP X-ray "camera" ----
SDD, SOD, PIX, N = 1000.0, 700.0, 2.0, 300                  # mm, mm, mm/pixel, pixels per side
f = SDD / PIX                                                # focal length in pixels
cx = cy = N / 2
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
# camera at the source (anterior, scene z = -SOD), looking +z (posterior).
# image x to the viewer's right = patient's left (-x); image y down = inferior (-y)
R = np.diag([-1.0, -1.0, 1.0])
C = np.array([0, 0, -SOD])
t = -R @ C
P = K @ np.hstack([R, t[:, None]])
X = np.array(L["asis_r"], float)
Xc = R @ X + t
uvw = P @ np.append(X, 1)
uv = uvw[:2] / uvw[2]
# parallel (orthographic) projection of the same point at the scale of the isocentre plane, for comparison
M_iso = SDD / SOD
uv_par = np.array([cx - M_iso * X[0] / PIX, cy - M_iso * X[1] / PIX])
ex = {"SDD": SDD, "SOD": SOD, "pix_mm": PIX, "N": N, "f_px": f, "c": [cx, cy], "K": K.tolist(), "R": R.tolist(),
      "t": t.tolist(), "P": P.tolist(), "X": X.tolist(), "Xc": Xc.tolist(), "uvw": uvw.tolist(), "uv": uv.tolist(),
      "uv_parallel_iso": uv_par.tolist(), "mag_point": SDD / Xc[2]}
json.dump({"points": np.round(pts, 4).flatten().tolist(), "landmarks_mm": L, "example": ex},
          open(out / "points.json", "w"))
print(json.dumps({"landmarks": L, "example": ex}, indent=1))
