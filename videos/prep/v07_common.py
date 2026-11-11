"""Shared helpers for videos 7 and 8 (visual hull, nearest-neighbour baseline).

Conventions (same as hero_assets.py): volumes are RAS (axis 0 = Right, 1 = Anterior, 2 = Superior),
scene x = R, y = S, z = -A (posterior), 1 mm = 0.01 units, origin = the pelvis centroid.

The 64^3 grid: a fixed CUBE_MM cube centred on each subject's pelvis centroid (so two subjects'
grids are centroid-aligned by construction), cell = CUBE_MM / 64.
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "totalsegmentator_small"
N = 64
CUBE_MM = 352.0
CELL = CUBE_MM / N           # 5.5 mm
MU_W = 0.02                  # 1/mm, as hero_assets.py
SCALE = 0.01


def good_subjects():
    return json.load(open(ROOT / "screen.json"))["good"]


def load_subject(s, with_ct=True):
    d = ROOT / "extracted" / s
    seg = d / "segmentations"
    m = None
    for n in ("hip_left", "hip_right", "sacrum"):
        im = nib.load(seg / f"{n}.nii.gz")
        assert nib.aff2axcodes(im.affine) == ("R", "A", "S"), (s, nib.aff2axcodes(im.affine))
        a = np.asanyarray(im.dataobj) > 0
        m = a if m is None else (m | a)
    sp = np.array(im.header.get_zooms()[:3], float)
    ct = None
    if with_ct:
        ci = nib.load(d / "ct.nii.gz")
        assert nib.aff2axcodes(ci.affine) == ("R", "A", "S")
        ct = np.asanyarray(ci.dataobj).astype(np.float32)
    return ct, m, sp


def centroid_vox(mask):
    return np.argwhere(mask).mean(0)


def _sub_axis(sub):
    """cell-centre offsets (mm) with `sub` sub-samples per cell, relative to the cube centre"""
    g = ((np.arange(N) + 0.5) / N - 0.5) * CUBE_MM
    o = ((np.arange(sub) + 0.5) / sub - 0.5) * CELL
    return (g[:, None] + o[None, :]).ravel()


def mask64(mask, sp, c_vox, sub=3):
    """fraction of each 5.5 mm cell inside the mask (trilinear sub-sampling) > 0.5"""
    ax = _sub_axis(sub)
    f = mask.astype(np.float32)
    out = np.zeros((N * sub, N * sub, N * sub), np.float32)
    X = c_vox[0] + ax / sp[0]
    Y = c_vox[1] + ax / sp[1]
    for k, z in enumerate(c_vox[2] + ax / sp[2]):          # slab by slab (memory)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        out[:, :, k] = ndimage.map_coordinates(f, [XX.ravel(), YY.ravel(), np.full(XX.size, z)],
                                               order=1, mode="constant").reshape(XX.shape)
    frac = out.reshape(N, sub, N, sub, N, sub).mean((1, 3, 5))
    return frac > 0.5


def drr_pair(ct, sp, c_vox, res=128, sub=2):
    """parallel-beam line integrals through the WHOLE CT, detector cropped to the cube around the
    pelvis centroid. Returns AP (R, S) and LAT (A, S) arrays of shape (res, res), units mu*mm."""
    mu = MU_W * (1 + np.clip(ct, -1000, None) / 1000.0)
    L_ap = mu.sum(axis=1) * sp[1]        # (R, S)
    L_lat = mu.sum(axis=0) * sp[0]       # (A, S)
    g = ((np.arange(res) + 0.5) / res - 0.5) * CUBE_MM
    o = ((np.arange(sub) + 0.5) / sub - 0.5) * (CUBE_MM / res)
    ax = (g[:, None] + o[None, :]).ravel()

    def samp(L, i0, i1):
        U, V = np.meshgrid(c_vox[i0] + ax / sp[i0], c_vox[i1] + ax / sp[i1], indexing="ij")
        v = ndimage.map_coordinates(L, [U.ravel(), V.ravel()], order=1, mode="constant", cval=0.0)
        return v.reshape(res, sub, res, sub).mean((1, 3))
    return samp(L_ap, 0, 2), samp(L_lat, 1, 2)


def to_display(a, p=(2, 99.7), gamma=2.2, lo_hi=None):
    """(u, S) array -> uint8 image in radiological orientation: superior at the top, and col 0 =
    max u (AP: patient's right on the viewer's left; LAT: anterior on the left)"""
    lo, hi = lo_hi if lo_hi is not None else np.percentile(a, p)
    v = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) ** gamma
    return (v.T[::-1, ::-1] * 255).astype(np.uint8)


def to_scene_tex(a):
    """(u, S) array -> image for a three.js plane whose texture u runs along +u: S at the top"""
    return a.T[::-1, :]


def silhouettes(m):
    return m.any(axis=1), m.any(axis=0)          # AP (R, S), LAT (A, S)


def visual_hull(ap_sil, lat_sil):
    """voxel (r, a, s) is kept iff its AP ray (r, s) and its LAT ray (a, s) both hit the silhouette"""
    return ap_sil[:, None, :] & lat_sil[None, :, :]


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def _surface(m):
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1))


def hd95(a, b, spacing=CELL):
    """95th percentile of the pooled symmetric surface distances (MedPy convention), mm"""
    sa, sb = _surface(a), _surface(b)
    if not sa.any() or not sb.any():
        return float("inf")
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.hstack([da, db]), 95))


def grid_scene(idx):
    """(k, 3) integer grid indices (R, A, S) -> scene coordinates (x=R, y=S, z=-A)"""
    mm = (np.asarray(idx) + 0.5) * CELL - CUBE_MM / 2
    return np.stack([mm[:, 0], mm[:, 2], -mm[:, 1]], 1) * SCALE


def multiview_hull(m, n_views):
    """visual hull from n parallel views spread over 180 deg around the S axis (all rays horizontal,
    like our AP/LAT pair and like a CT gantry). Per axial slice, a test cell survives iff its centre
    projects inside the union of the true cells' projected intervals for every view. For 0/90 deg
    this is exactly visual_hull(silhouettes)."""
    c = (np.arange(N) + 0.5) * CELL
    R, A = np.meshgrid(c, c, indexing="ij")
    out = np.zeros_like(m)
    thetas = np.arange(n_views) * np.pi / n_views
    for k in range(N):
        sl = m[:, :, k]
        if not sl.any():
            continue
        keep = np.ones((N, N), bool)
        tr, ta = R[sl], A[sl]
        for th in thetas:
            cs, sn = np.cos(th), np.sin(th)
            h = CELL * (abs(cs) + abs(sn)) / 2 + 1e-9
            u = np.sort(tr * cs + ta * sn)
            lo, hi = u - h, u + h
            # merge intervals
            brk = np.r_[True, lo[1:] > np.maximum.accumulate(hi)[:-1]]
            starts = lo[brk]
            ends = np.maximum.accumulate(hi)[np.r_[brk[1:], True]]
            q = R * cs + A * sn
            j = np.searchsorted(starts, q, side="right") - 1
            inside = (j >= 0) & (q <= ends[np.clip(j, 0, None)])
            keep &= inside
        out[:, :, k] = keep
    return out


CACHE = Path(__file__).resolve().parents[1] / "out" / "v07_cache"


def subject_cached(s):
    """mask64 + 128px AP/LAT DRRs (cube around the pelvis centroid) for subject s, cached in out/"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{s}.npz"
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}
    ct, m, sp = load_subject(s)
    c = centroid_vox(m)
    g = mask64(m, sp, c)
    ap, lat = drr_pair(ct, sp, c)
    d = {"mask64": g, "drr_ap": ap.astype(np.float32), "drr_lat": lat.astype(np.float32), "c_vox": c,
         "sp": sp, "n_vox_full": np.array(m.sum()), "ct_shape": np.array(ct.shape)}
    np.savez_compressed(f, **d)
    return d
