"""Video 8 assets: a REAL (small) nearest-neighbour baseline on the 27 'good' subjects.

For every subject: parallel AP + LAT DRRs (whole-CT line integrals, detector cropped to a 352 mm square
around the pelvis centroid) and a 64^3 pelvis mask centred on the same centroid (v07_common.py).
Query = hero s1390, database = the other 26. Distance = L2 between the concatenated 32x32 AP and LAT
line-integral images. Dice / HD95 of the retrieved pelvis vs the hero's true pelvis, centroid-aligned.
Also leave-one-out over all 27 for the first results table (random / NN / oracle retrieval / visual hull).

Usage: python -I -X utf8 prep/v08_assets.py      (writes public/data/v08/)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from skimage import measure, filters
from v07_common import (good_subjects, subject_cached, load_subject, centroid_vox, silhouettes, visual_hull,
                        dice, hd95, N, CELL, CUBE_MM)

OUT = Path(__file__).resolve().parents[1] / "public" / "data" / "v08"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "g").mkdir(exist_ok=True)
HERO = "s1390"
S = good_subjects()
assert S[0] == HERO and len(S) == 27
D = {s: subject_cached(s) for s in S}

def down32(a):
    return a.reshape(32, 4, 32, 4).mean((1, 3))

F = np.stack([np.concatenate([down32(D[s]["drr_ap"]).ravel(), down32(D[s]["drr_lat"]).ravel()]) for s in S])
M = [D[s]["mask64"] for s in S]
n = len(S)
dist = np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))
DI = np.array([[dice(M[i], M[j]) for j in range(n)] for i in range(n)])
HD = np.array([[hd95(M[j], M[i]) if i != j else 0.0 for j in range(n)] for i in range(n)])

# ---- leave-one-out over all 27 ----
rows = {"random": [], "nn": [], "oracle": [], "hull": []}
loo = []
for i in range(n):
    o = [k for k in range(n) if k != i]
    j = o[int(np.argmin(dist[i, o]))]
    b = o[int(np.argmax(DI[i, o]))]
    h = visual_hull(*silhouettes(M[i]))
    rows["random"].append((np.mean(DI[i, o]), np.mean(HD[i, o])))
    rows["nn"].append((DI[i, j], HD[i, j]))
    rows["oracle"].append((DI[i, b], HD[i, b]))
    rows["hull"].append((dice(h, M[i]), hd95(h, M[i])))
    loo.append({"query": S[i], "nn": S[j], "nn_dice": round(DI[i, j], 4), "oracle": S[b], "oracle_dice": round(DI[i, b], 4)})
table = {}
for k, v in rows.items():
    a = np.array(v)
    table[k] = {"dice_mean": round(a[:, 0].mean(), 4), "dice_std": round(a[:, 0].std(), 4),
                "hd95_mean": round(a[:, 1].mean(), 2), "hd95_std": round(a[:, 1].std(), 2)}
print("table", table)
nn_is_oracle = sum(r["nn"] == r["oracle"] for r in loo)

# ---- the hero query ----
o = list(range(1, n))
order = sorted(o, key=lambda k: dist[0, k])
j, b = order[0], max(o, key=lambda k: DI[0, k])
hero = {"query": HERO, "nn": S[j], "nn_dist": round(float(dist[0, j]), 2),
        "nn_dice": round(DI[0, j], 4), "nn_hd95_mm": round(HD[0, j], 2),
        "oracle": S[b], "oracle_dice": round(DI[0, b], 4), "oracle_hd95_mm": round(HD[0, b], 2),
        "oracle_rank": order.index(b) + 1,
        "ranked": [{"s": S[k], "dist": round(float(dist[0, k]), 2), "dice": round(DI[0, k], 4),
                    "hd95": round(HD[0, k], 2)} for k in order]}
from scipy.stats import spearmanr
rho = spearmanr(dist[0, o], DI[0, o])
hero["spearman_dist_dice"] = round(float(rho.statistic), 3)
hero["spearman_p"] = round(float(rho.pvalue), 3)
print("hero", {k: v for k, v in hero.items() if k != "ranked"})

g, r = M[0], M[j]
json.dump({"n": N, "cell": CELL * 0.01, "cube": CUBE_MM * 0.01,
           "tp": np.argwhere(g & r).ravel().tolist(), "fp": np.argwhere(r & ~g).ravel().tolist(),
           "fn": np.argwhere(g & ~r).ravel().tolist()},
          open(OUT / "vox.json", "w"), separators=(",", ":"))

# ---- gallery images: one global window over all subjects (so brightness differences are real) ----
allap = np.stack([D[s]["drr_ap"] for s in S]); alllat = np.stack([D[s]["drr_lat"] for s in S])
def disp(a, lo, hi, gamma=2.2):
    v = np.clip((a - lo) / (hi - lo), 0, 1) ** gamma
    return (v.T[::-1, ::-1] * 255).astype(np.uint8)    # superior at the top; AP: patient's right on the left
wap = np.percentile(allap, [2, 99.7]); wlat = np.percentile(alllat, [2, 99.7])
for s in S:
    for name, a, w in [("ap", D[s]["drr_ap"], wap), ("lat", D[s]["drr_lat"], wlat)]:
        Image.fromarray(disp(a, *w)).save(OUT / "g" / f"{name}_{s}.png")
        Image.fromarray(disp(down32(a), *w)).save(OUT / "g" / f"{name}32_{s}.png")

# ---- retrieved pelvis mesh (same scene convention as hero_assets: x=R, y=S, z=P, centroid origin) ----
_, m, sp = load_subject(S[j], with_ct=False)
idx = np.argwhere(m); lo, hi = idx.min(0) - 4, idx.max(0) + 5
c = centroid_vox(m) - lo
mm = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
v, f, nrm, _ = measure.marching_cubes(filters.gaussian(mm.astype(np.float32), 1.2), 0.5, spacing=tuple(sp), step_size=2)
v = v - c * sp
xyz = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1) * 0.01
nn3 = np.stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]], 1)
with open(OUT / "pelvis_nn.obj", "w") as fo:
    fo.writelines(f"v {a:.4f} {b_:.4f} {c_:.4f}\n" for a, b_, c_ in xyz)
    fo.writelines(f"vn {a:.3f} {b_:.3f} {c_:.3f}\n" for a, b_, c_ in nn3)
    fo.writelines(f"f {a+1}//{a+1} {c_+1}//{c_+1} {b_+1}//{b_+1}\n" for a, b_, c_ in f)
print("mesh", len(xyz))

json.dump({"subjects": S, "n": n, "feature": "AP+LAT line integrals, 32x32 each (11 mm pixels), L2",
           "cube_mm": CUBE_MM, "cell_mm": CELL, "hero": hero, "table": table,
           "loo_nn_equals_oracle": nn_is_oracle, "loo": loo},
          open(OUT / "nn.json", "w"), indent=1)

# ---- |query - nearest neighbour| on the 32x32 images (what the L2 distance sums, squared) ----
for name in ("ap", "lat"):
    q, r_ = down32(D[HERO][f"drr_{name}"]), down32(D[S[j]][f"drr_{name}"])
    dif = np.abs(q - r_)
    v = np.clip(dif / np.percentile(np.abs(down32(D[HERO][f"drr_{name}"]) - down32(D[S[order[-1]]][f"drr_{name}"])), 99), 0, 1)
    Image.fromarray((v.T[::-1, ::-1] * 255).astype(np.uint8)).save(OUT / f"diff_{name}.png")
print("diff images written (scale: 99th pct of |query - farthest|)")
