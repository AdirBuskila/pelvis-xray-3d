"""Find TotalSegmentator subjects with a COMPLETE pelvis (not cut by the scan's field of view) and no metal.
Stage 1 uses only the small mask files (fetched remotely); stage 2 fetches CTs of the best candidates.
Usage: python screen_pelvis.py <zip_url> <work_dir> <n_ct>"""
import io, json, os, sys, subprocess
import numpy as np, nibabel as nib

url, work, n_ct = sys.argv[1], sys.argv[2], int(sys.argv[3])
here = os.path.dirname(os.path.abspath(__file__))
rz = lambda *a: subprocess.run([sys.executable, "-I", os.path.join(here, "remote_zip.py"), url, *a],
                               capture_output=True, text=True).stdout
hips = [l.split()[0].split("/")[0] for l in rz("list", "*/segmentations/hip_left.nii.gz").splitlines()]
res = {}
for s in hips:
    r = {}
    for n in ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]:
        p = f"{work}/{s}/segmentations/{n}.nii.gz"
        if not os.path.exists(p): rz("get", f"{s}/segmentations/{n}.nii.gz", work)
        m = np.asanyarray(nib.load(p).dataobj) > 0
        z = np.argwhere(m)[:, 2] if m.any() else np.array([-1])
        r[n] = (int(m.sum()), int(z.min()), int(z.max()), int(m.shape[2]))
    # complete = hips and sacrum present and not touching the bottom/top slice; femurs reasonably long
    ok = all(r[k][0] > 1000 and r[k][1] > 2 and r[k][2] < r[k][3] - 3 for k in ["hip_left", "hip_right", "sacrum"])
    fem = min(r["femur_left"][0], r["femur_right"][0])
    res[s] = {"complete": ok, "femur_vox": fem, "r": r}
    print(s, "complete" if ok else "-", fem, flush=True)
cands = sorted([s for s in res if res[s]["complete"]], key=lambda s: -res[s]["femur_vox"])
print("complete pelvis:", len(cands), "of", len(hips), cands)
for s in cands[:n_ct]:
    p = f"{work}/{s}/ct.nii.gz"
    if not os.path.exists(p): rz("get", f"{s}/ct.nii.gz", work)
    ct = np.asanyarray(nib.load(p).dataobj)
    zl, zh = res[s]["r"]["hip_left"][1], res[s]["r"]["hip_left"][2]
    metal = int((ct[:, :, max(0, zl - 40):zh + 10] > 2500).sum())
    res[s]["metal_vox"] = metal
    print("CT", s, "metal voxels:", metal, flush=True)
json.dump({"candidates": cands, "results": res}, open(f"{work}/screen.json", "w"), indent=1)
