"""Screen TotalSegmentator subjects: complete pelvis inside the scan, femur coverage, metal near the pelvis.
Usage: python screen_local.py <extracted_dir> <out_json>"""
import json, os, sys
import numpy as np, nibabel as nib
root, out = sys.argv[1], sys.argv[2]
res = {}
for s in sorted(d for d in os.listdir(root) if d.startswith("s")):
    seg = f"{root}/{s}/segmentations"
    r = {}
    for n in ["hip_left", "hip_right", "sacrum", "femur_left", "femur_right"]:
        m = np.asanyarray(nib.load(f"{seg}/{n}.nii.gz").dataobj) > 0
        z = np.argwhere(m)[:, 2] if m.any() else np.array([-1])
        r[n] = [int(m.sum()), int(z.min()), int(z.max()), int(m.shape[2])]
    complete = all(r[k][0] > 1000 and r[k][1] > 2 and r[k][2] < r[k][3] - 3 for k in ["hip_left", "hip_right", "sacrum"])
    item = {"complete": complete, "femur_vox": min(r["femur_left"][0], r["femur_right"][0]), "r": r}
    if complete:
        ct = np.asanyarray(nib.load(f"{root}/{s}/ct.nii.gz").dataobj)
        zl, zh = r["hip_left"][1], r["hip_left"][2]
        item["metal_vox"] = int((ct[:, :, max(0, zl - 40):zh + 10] > 2500).sum())
        item["spacing"] = [float(x) for x in nib.load(f"{root}/{s}/ct.nii.gz").header.get_zooms()]
    res[s] = item
good = sorted([s for s in res if res[s]["complete"] and res[s].get("metal_vox", 1) < 50], key=lambda s: -res[s]["femur_vox"])
json.dump({"good": good, "results": res}, open(out, "w"), indent=1)
print("with hips:", sum(1 for s in res if res[s]["r"]["hip_left"][0] > 1000), "| complete:", sum(res[s]["complete"] for s in res),
      "| complete & metal-free:", len(good))
for s in good[:10]: print(s, res[s]["femur_vox"], res[s]["metal_vox"], res[s]["spacing"])
