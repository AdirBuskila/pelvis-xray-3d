"""Building the processed dataset v1 (unit 02, part C): screening, the canonical cube, the DRR pair.

Written by Claude (infrastructure), but the parts that carry ideas are YOUR functions, imported here:
    xray3d.volume.touches_scan_edge                  (unit 01)  is the pelvis cut by the scan?
    xray3d.physics.hu_to_mu                          (unit 01)  HU -> attenuation
    xray3d.shape.bbox_center_mm, cube_grid,          (unit 02)  where the cube goes, its voxel centres,
                 resample, occupancy                            sampling a volume there, the 64^3 occupancy
So the dataset is built with your code. If one of them is wrong, the dataset is wrong: fix the function,
then rebuild.

What one subject becomes (see dataset.py for the file format):
  1. screening: all 3 pelvis bones present (> 1000 voxels), none touching the first/last 3 slices along the
     head-feet axis, no metal (< 50 voxels above 2500 HU near the hips: the rule of data/scripts/screen_local.py),
     and the pelvis fits inside the 320 mm cube (the occupancy doesn't touch the cube's faces).
  2. the cube: centred on the centre of the pelvis's bounding box, 64^3 voxels of 5 mm, RAS axes.
  3. the X-rays: the CT is resampled into a 480 x 480 x 400 mm box (2 mm voxels, RAS, air outside the scan)
     around the cube centre, turned into mu with hu_to_mu, and rendered by DiffDRR as a perspective AP and
     LAT image (128 x 128 pixels of 4 mm, source 750 mm from the cube centre, detector 1020 mm from the source).
     The same rendering of mu * pelvis gives the pelvis-only images ap_bone and lat_bone.
  4. extras: the centre and radius of each femoral head (for the CE angle in unit 02/07).

The full dataset (1,228 CTs in a 23.6 GB zip on Zenodo) is never downloaded as a whole: `build_remote` reads
the zip's table of contents once, then fetches only the files it needs, subject by subject, with several HTTP
range requests in parallel. Rejected and finished subjects are recorded in screening.json, so a restart
skips them (resumable).
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime
import gzip
import json
import re
import threading
import time
import urllib.request
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

from . import data, dataset

G = dataset.GEOMETRY
MIN_BONE_VOXELS = 1000      # fewer than this: the bone is missing (or a few stray voxels)
EDGE_MARGIN = 3             # slices; same as data/scripts/screen_local.py
METAL_HU = 2500             # denser than any bone
METAL_MAX_VOXELS = 50       # more voxels than this above METAL_HU near the hips -> metal (prosthesis, screws)


# ---------------------------------------------------------------- small helpers

def pelvis_union(masks: dict) -> np.ndarray:
    """hip_left | hip_right | sacrum: the target of the whole project."""
    return masks["hip_left"] | masks["hip_right"] | masks["sacrum"]


def head_feet_axis(affine) -> tuple[int, str]:
    """(array axis that runs head-feet, its letter 'S' or 'I')."""
    codes = nib.aff2axcodes(affine)
    ax = [c in "SI" for c in codes].index(True)
    return ax, codes[ax]


def _student(module: str, name: str):
    """Fetch one of the students' functions, with a clear message if it isn't moved into src/xray3d yet."""
    import importlib

    mod = importlib.import_module(f"xray3d.{module}")
    fn = getattr(mod, name, None)
    if fn is None:
        raise RuntimeError(f"xray3d.{module}.{name} is missing: move your function into src/xray3d/{module}.py first.")
    return fn


# ---------------------------------------------------------------- 1. screening

def screen_masks(masks: dict, affine) -> dict:
    """Is the pelvis there and complete? Uses YOUR touches_scan_edge along the head-feet axis."""
    touches_scan_edge = _student("volume", "touches_scan_edge")
    counts = {b: int(masks[b].sum()) for b in data.PELVIS}
    if any(c <= MIN_BONE_VOXELS for c in counts.values()):
        return {"ok": False, "reason": "bone missing", "voxels": counts}
    ax, _ = head_feet_axis(affine)
    cut = [b for b in data.PELVIS if touches_scan_edge(masks[b], ax, margin=EDGE_MARGIN)]
    if cut:
        return {"ok": False, "reason": "cut by the scan: " + ",".join(cut), "voxels": counts}
    return {"ok": True, "reason": "", "voxels": counts}


def metal_voxels(hu: np.ndarray, hip_left: np.ndarray, affine) -> int:
    """Voxels above METAL_HU in the slab of slices around the hips (screen_local.py's rule).

    The slab runs from 40 slices below the hip bone to 10 slices above it (60 mm / 15 mm at 1.5 mm slices),
    which covers both hip joints and the top of the femurs, where prostheses sit.
    """
    ax, letter = head_feet_axis(affine)
    idx = np.flatnonzero(hip_left.any(axis=tuple(a for a in range(3) if a != ax)))
    lo, hi = int(idx[0]), int(idx[-1])
    if letter == "S":       # index grows towards the head: "below" = smaller index
        a, b = lo - 40, hi + 10
    else:                   # index grows towards the feet
        a, b = lo - 10, hi + 40
    sl = [slice(None)] * 3
    sl[ax] = slice(max(a, 0), b + 1)
    return int((hu[tuple(sl)] > METAL_HU).sum())


# ---------------------------------------------------------------- 2. the cube and the box

def femoral_head(femur: np.ndarray, affine) -> tuple[np.ndarray, float]:
    """Centre (world mm) and radius (mm) of the femoral head: the deepest point inside the top 60 mm of the femur.

    'Deepest' = largest distance to the femur's surface (a distance transform); for a ball that's its centre,
    and the distance is its radius.
    """
    from scipy import ndimage

    if femur.sum() <= MIN_BONE_VOXELS:
        return np.full(3, np.nan), float("nan")
    aff = np.asarray(affine, float)
    idx = np.argwhere(femur)
    z = idx @ aff[2, :3] + aff[2, 3]                      # world S coordinate of every femur voxel
    top = idx[z >= z.max() - 60.0]
    sub = np.zeros(femur.shape, bool)
    sub[tuple(top.T)] = True
    d = ndimage.distance_transform_edt(sub, sampling=np.linalg.norm(aff[:3, :3], axis=0))
    ijk = np.array(np.unravel_index(np.argmax(d), d.shape), float)
    return aff[:3, :3] @ ijk + aff[:3, 3], float(d.max())


def build_box(hu: np.ndarray, pelvis: np.ndarray, affine, center_mm) -> tuple[np.ndarray, np.ndarray]:
    """mu of the whole CT and of the pelvis only, on the 480 x 480 x 400 mm box (2 mm voxels, RAS) around the cube.

    Uses YOUR cube_grid (the box is just a bigger, coarser 'cube'), resample (linear for the CT, air (-1000 HU)
    outside the scan; the mask as a float, thresholded at 0.5) and hu_to_mu. Done in slabs to save memory.
    """
    cube_grid, resample = _student("shape", "cube_grid"), _student("shape", "resample")
    hu_to_mu = _student("physics", "hu_to_mu")
    size = np.array(G["box_mm"], float)
    n = np.round(size / G["box_voxel_mm"]).astype(int)
    grid = cube_grid(np.asarray(center_mm, float), size, n).astype(np.float32)   # (240, 240, 200, 3)
    mu = np.empty(tuple(n), np.float32)
    bone = np.empty(tuple(n), np.float32)
    pel = pelvis.astype(np.float32)
    for i0 in range(0, n[0], 40):
        g = grid[i0:i0 + 40]
        h = resample(hu, affine, g, order=1, fill=-1000.0)
        m = resample(pel, affine, g, order=1, fill=0.0) >= 0.5
        mu[i0:i0 + 40] = hu_to_mu(h, G["mu_water"])
        bone[i0:i0 + 40] = mu[i0:i0 + 40] * m
    return mu, bone


# ---------------------------------------------------------------- 3. the X-rays (DiffDRR)

_RENDER_LOCK = threading.Lock()   # one rendering at a time (GPU memory; DiffDRR isn't thread-safe)


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_views(mu_box: np.ndarray, voxel_mm: float | None = None, views=dataset.VIEWS) -> dict[str, np.ndarray]:
    """Perspective line-integral images p of a mu volume centred on the cube centre: {"ap": (128,128), "lat": ...}.

    How DiffDRR is driven (found empirically with marker voxels, see `check_geometry`):
      * the volume is given in RAS with an affine that puts its centre at world (0, 0, 0) (read(...,
        center_volume=True) would do the same); so DiffDRR's world frame IS our cube frame.
      * read(..., orientation="AP"), source 750 mm from the origin: rotation (0, 0, 0) puts the source in front
        of the patient (+A), rotation (pi/2, 0, 0) (Z, X, Y euler) puts it on the patient's left (-R).
      * DiffDRR's own HU -> density conversion is replaced by OUR mu (1/mm), so the output is the physical line
        integral p = sum(mu * length in mm), the same quantity as unit 01's parallel_drr.
      * its images come out exactly in our orientation (rows head -> feet; AP columns R -> L; LAT columns A -> P):
        no flips needed.
    """
    import torch
    from diffdrr.data import read
    from diffdrr.drr import DRR
    from torchio import ScalarImage

    voxel_mm = voxel_mm or G["box_voxel_mm"]
    shape = np.array(mu_box.shape)
    A = np.diag([voxel_mm] * 3 + [1.0])
    A[:3, 3] = -voxel_mm * (shape - 1) / 2.0                 # voxel centre (n-1)/2 -> world origin
    t = torch.from_numpy(np.ascontiguousarray(mu_box, dtype=np.float32))[None]
    with _RENDER_LOCK:
        subject = read(ScalarImage(tensor=t, affine=A), orientation="AP", center_volume=True)
        subject.density = ScalarImage(tensor=t, affine=subject.volume.affine)   # our mu, not DiffDRR's density
        dev = _device()
        drr = DRR(subject, sdd=G["sdd_mm"], height=G["detector_px"], delx=G["pixel_mm"]).to(dev)
        out = {}
        with torch.no_grad():
            for v in views:
                pose = G["diffdrr"][v]
                img = drr(torch.tensor([pose["rotation"]], dtype=torch.float32, device=dev),
                          torch.tensor([pose["translation"]], dtype=torch.float32, device=dev),
                          parameterization="euler_angles", convention="ZXY")
                out[v] = img[0, 0].cpu().numpy().astype(np.float32)
        del drr
    return out


def check_geometry(project_pinhole=None, verbose: bool = True) -> float:
    """Render single bright voxels at known cube-frame positions and compare with dataset.camera.

    Returns the largest error in pixels (should be well below 1). `project_pinhole` defaults to YOUR
    xray3d.geometry.project_pinhole (unit 01).
    """
    from scipy import ndimage

    project_pinhole = project_pinhole or _student("geometry", "project_pinhole")
    s = 4.0
    shape = np.array([121, 121, 101])                     # odd: voxel centres sit on round mm values
    marks = np.array([[100.0, 80, 0], [-100, -80, 40], [0, 0, 120], [60, -100, -80]])
    worst = 0.0
    for p in marks:
        mu = np.zeros(tuple(shape), np.float32)
        mu[tuple(np.round(p / s + (shape - 1) / 2).astype(int))] = 1.0
        imgs = render_views(mu, s)
        for v in dataset.VIEWS:
            row, col = ndimage.center_of_mass(imgs[v])
            K, R, t = dataset.camera(v)
            u, w = project_pinhole(p[None], K, R, t)[0]
            err = float(np.hypot(u - col, w - row))
            worst = max(worst, err)
            if verbose:
                print(f"  marker {p.astype(int).tolist()!s:<16} {v:<3}: rendered (col {col:6.2f}, row {row:6.2f})"
                      f"   camera() + project_pinhole (col {u:6.2f}, row {w:6.2f})   error {err:.2f} px")
    return worst


# ---------------------------------------------------------------- 4. one subject

def build_subject(hu: np.ndarray, affine, masks: dict, render: bool = True) -> tuple[dict | None, dict]:
    """(arrays, info) for one subject that passed screen_masks. arrays is None if the pelvis doesn't fit the cube.

    masks must hold hip_left, hip_right, sacrum and (for the CE angle extras) femur_left, femur_right.
    """
    bbox_center_mm, occupancy = _student("shape", "bbox_center_mm"), _student("shape", "occupancy")
    touches_scan_edge = _student("volume", "touches_scan_edge")
    t0 = time.time()
    pelvis = pelvis_union(masks)
    center = np.asarray(bbox_center_mm(pelvis, affine), float)
    occ = np.asarray(occupancy(pelvis, affine, center, G["cube_mm"], G["cube_n"])).astype(np.uint8)
    info = {"occ_voxels": int(occ.sum()), "spacing_mm": np.linalg.norm(np.asarray(affine)[:3, :3], axis=0).round(3).tolist()}
    # the pelvis must fit inside the 320 mm cube: no occupied voxel on any face of the cube
    if any(touches_scan_edge(occ.astype(bool), a, margin=1) for a in range(3)):
        info["reason"] = "pelvis larger than the cube"
        return None, info
    heads = np.full((2, 4), np.nan, np.float32)
    for r, side in enumerate(("left", "right")):
        f = masks.get(f"femur_{side}")
        if f is not None:
            c, rad = femoral_head(f, affine)
            heads[r, :3], heads[r, 3] = c - center, rad
    arrays = {"occ": occ, "center_mm": center.astype(np.float64), "femoral_heads": heads}
    if render:
        mu, bone = build_box(hu, pelvis, affine, center)
        whole, only = render_views(mu), render_views(bone)
        arrays.update(ap=whole["ap"], lat=whole["lat"], ap_bone=only["ap"], lat_bone=only["lat"])
    info["seconds"] = round(time.time() - t0, 1)
    return arrays, info


def save_subject(out_dir, sid: str, arrays: dict) -> Path:
    """Write <sid>.npz (compressed), atomically: a half-written file never looks finished."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{sid}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    final = out_dir / f"{sid}.npz"
    tmp.replace(final)
    return final


def write_index(out_dir, source: str, screening: dict | None = None) -> dict:
    """index.json listing every <sid>.npz in out_dir, with the geometry and provenance."""
    out_dir = Path(out_dir)
    ids = sorted(p.stem for p in out_dir.glob("s*.npz") if ".tmp" not in p.name)
    idx = {"version": dataset.VERSION, "subjects": ids, "geometry": G,
           "created": datetime.datetime.now().isoformat(timespec="seconds"), "source": source,
           "occupancy": "hip_left | hip_right | sacrum, resampled linearly as float, >= 0.5",
           "license": "derived from TotalSegmentator v2.0.1 (CC BY 4.0, Wasserthal et al. 2023)"}
    if screening is not None:
        idx["screening_counts"] = count_reasons(screening)
    (out_dir / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def count_reasons(screening: dict) -> dict:
    out: dict[str, int] = {}
    for r in screening.values():
        key = "built" if r.get("built") else (r.get("reason") or "?").split(":")[0]
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------- the local build (small subset on disk)

def build_local(out_dir, ids=data.GOOD_SUBJECTS, render: bool = True, skip_existing: bool = True) -> dict:
    """Build v1 from subjects already unzipped under data.subjects_dir(). Returns the screening record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "screening.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    for n, sid in enumerate(ids, 1):
        if skip_existing and (out_dir / f"{sid}.npz").exists() and rec.get(sid, {}).get("built"):
            continue
        hu, affine = data.load_ct(sid)
        masks = data.load_mask(sid, data.PELVIS + data.FEMURS)
        r = screen_masks(masks, affine)
        if r["ok"]:
            r["metal_voxels"] = metal_voxels(hu, masks["hip_left"], affine)
            if r["metal_voxels"] >= METAL_MAX_VOXELS:
                r.update(ok=False, reason="metal")
        if r["ok"]:
            arrays, info = build_subject(hu, affine, masks, render=render)
            r.update(info)
            if arrays is None:
                r["ok"] = False
            else:
                save_subject(out_dir, sid, arrays)
                r["built"] = True
        rec[sid] = r
        rec_path.write_text(json.dumps(rec, indent=1))
        print(f"[{n}/{len(ids)}] {sid}: " + ("built" if r.get("built") else "rejected (" + r["reason"] + ")")
              + (f", {r['occ_voxels']} occupied voxels, {r.get('seconds', 0)} s" if r.get("built") else ""))
    write_index(out_dir, "TotalSegmentator v2.0.1 small subset (Zenodo 10047263), local", rec)
    return rec


# ---------------------------------------------------------------- the remote build (full dataset, streamed)

MEMBER_RE = re.compile(r"(?:^|/)(s\d{4})/(ct|segmentations/([A-Za-z0-9_]+))\.nii\.gz$")


def _get_range(url: str, start: int, end: int, tries: int = 8) -> bytes:
    """Bytes start..end (inclusive) of a remote file, retrying with backoff (servers drop connections)."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data_ = r.read()
            if len(data_) == end - start + 1:
                return data_
            last = IOError(f"short read {len(data_)} of {end - start + 1}")
        except Exception as e:  # noqa: BLE001 (network errors come in many types)
            last = e
        time.sleep(2 * (attempt + 1))
    raise IOError(f"range {start}-{end} failed {tries} times: {last}")


class RemoteSubjects:
    """The full TotalSegmentator zip on Zenodo, as {subject: {file name: ZipInfo}}, read with parallel ranges.

    Opening reads only the zip's table of contents (xray3d.remote_zip); `fetch` then downloads each wanted
    member with its own range request(s), big members split into chunks fetched in parallel.
    """

    def __init__(self, url: str = data.FULL_DATASET_URL, workers: int = 8, chunk_mb: int = 16):
        from .remote_zip import open_remote_zip

        t0 = time.time()
        z = open_remote_zip(url)
        self.url = z.fp.raw.url if hasattr(z.fp, "raw") else url   # the final URL after redirects
        self.size = z.fp.raw.size if hasattr(z.fp, "raw") else None
        self.members: dict[str, dict] = {}
        for info in z.infolist():
            m = MEMBER_RE.search(info.filename)
            if m:
                name = "ct" if m.group(2) == "ct" else m.group(3)
                self.members.setdefault(m.group(1), {})[name] = info
        self.pool = cf.ThreadPoolExecutor(workers)
        self.chunk = chunk_mb * 2**20
        print(f"remote zip: {len(self.members)} subjects listed in {time.time() - t0:.0f} s")

    def subjects(self) -> list[str]:
        return sorted(self.members)

    def _member_bytes(self, info) -> bytes:
        # local file header: 30 fixed bytes, then the name and an "extra" field whose lengths are in the header
        head = _get_range(self.url, info.header_offset, info.header_offset + 29)
        n_name, n_extra = int.from_bytes(head[26:28], "little"), int.from_bytes(head[28:30], "little")
        start = info.header_offset + 30 + n_name + n_extra
        size = info.compress_size
        pieces = [(a, min(a + self.chunk, size) - 1) for a in range(0, size, self.chunk)] or [(0, -1)]
        futs = [self.pool.submit(_get_range, self.url, start + a, start + b) for a, b in pieces if b >= a]
        raw = b"".join(f.result() for f in futs)
        if info.compress_type == 8:            # deflate
            out = zlib.decompressobj(-15).decompress(raw)
        elif info.compress_type == 0:          # stored
            out = raw
        else:
            raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
        if zlib.crc32(out) & 0xFFFFFFFF != info.CRC:
            raise IOError(f"CRC mismatch for {info.filename}")
        return out

    def fetch(self, sid: str, names) -> dict[str, nib.Nifti1Image]:
        """{name: NIfTI image} for e.g. names=("hip_left", "ct"), all requests in parallel, nothing written to disk."""
        infos = {n: self.members[sid][n] for n in names}
        with cf.ThreadPoolExecutor(len(infos)) as outer:
            blobs = dict(zip(infos, outer.map(self._member_bytes, infos.values())))
        return {n: nib.Nifti1Image.from_bytes(gzip.decompress(b)) for n, b in blobs.items()}


def build_remote(out_dir, url: str = data.FULL_DATASET_URL, ids=None, limit: int | None = None,
                 render: bool = True, subject_workers: int = 3, range_workers: int = 12) -> dict:
    """Build v1 from the full dataset, streamed subject by subject. Resumable: rerun after a crash/timeout.

    Per subject: fetch the 3 pelvis masks (a few hundred kB) -> screen; only if they pass, fetch the CT and the
    femurs (~30-150 MB) -> metal rule -> build -> save. screening.json records every decision.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "screening.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    lock = threading.Lock()
    remote = RemoteSubjects(url, workers=range_workers)
    todo = [s for s in (ids or remote.subjects()) if s not in rec]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(rec)} subjects already done, {len(todo)} to go")
    t_start, done = time.time(), [0]

    def one(sid):
        have = remote.members.get(sid, {})
        if not all(b in have for b in data.PELVIS) or "ct" not in have:
            return sid, {"ok": False, "reason": "files missing in the zip"}
        imgs = remote.fetch(sid, data.PELVIS)
        affine = imgs["hip_left"].affine
        masks = {b: np.asanyarray(imgs[b].dataobj) > 0 for b in data.PELVIS}
        r = screen_masks(masks, affine)
        if not r["ok"]:
            return sid, r
        extra = ["ct"] + [f for f in data.FEMURS if f in have]
        imgs = remote.fetch(sid, extra)
        hu = np.asanyarray(imgs["ct"].dataobj)
        for f in data.FEMURS:
            if f in imgs:
                masks[f] = np.asanyarray(imgs[f].dataobj) > 0
        r["metal_voxels"] = metal_voxels(hu, masks["hip_left"], affine)
        if r["metal_voxels"] >= METAL_MAX_VOXELS:
            r.update(ok=False, reason="metal")
            return sid, r
        arrays, info = build_subject(hu, affine, masks, render=render)
        r.update(info)
        if arrays is None:
            r["ok"] = False
            return sid, r
        save_subject(out_dir, sid, arrays)
        r["built"] = True
        return sid, r

    def safe(sid):
        try:
            return one(sid)
        except Exception as e:  # noqa: BLE001 (record and move on; a rerun retries nothing recorded as error)
            return sid, {"ok": False, "reason": f"error: {type(e).__name__}: {str(e)[:200]}"}

    with cf.ThreadPoolExecutor(subject_workers) as pool:
        for sid, r in pool.map(safe, todo):
            with lock:
                if not r["reason"].startswith("error"):     # errors are NOT recorded, so a rerun retries them
                    rec[sid] = r
                    rec_path.write_text(json.dumps(rec, indent=1))
                done[0] += 1
                el = time.time() - t_start
                n_built = sum(1 for x in rec.values() if x.get("built"))
                print(f"[{done[0]}/{len(todo)}] {sid}: {'built' if r.get('built') else r['reason']}"
                      f"   (built so far {n_built}, {el / done[0]:.0f} s/subject, ~{el / done[0] * (len(todo) - done[0]) / 60:.0f} min left)")
    write_index(out_dir, f"TotalSegmentator v2.0.1 full (Zenodo 10047292), streamed from {url}", rec)
    return rec


# ---------------------------------------------------------------- checks used by the notebooks

def project_occupancy(occ: np.ndarray, view: str, project_pinhole=None, sub: int = 3) -> np.ndarray:
    """Silhouette of the occupied voxels in one view: project `sub`^3 points per voxel, mark their pixels.

    Uses dataset.camera(view) and YOUR project_pinhole (unit 01). The result can be compared with ap_bone > 0.
    """
    project_pinhole = project_pinhole or _student("geometry", "project_pinhole")
    cell = G["cube_mm"] / G["cube_n"]
    centres = dataset.voxel_centers_mm()[np.asarray(occ, bool)]                   # (N, 3)
    offs = (np.arange(sub) + 0.5) / sub * cell - cell / 2
    o = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
    pts = (centres[:, None, :] + o[None]).reshape(-1, 3)
    K, R, t = dataset.camera(view)
    uv = np.round(project_pinhole(pts, K, R, t)).astype(int)
    n = G["detector_px"]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < n) & (uv[:, 1] >= 0) & (uv[:, 1] < n)
    img = np.zeros((n, n), bool)
    img[uv[ok, 1], uv[ok, 0]] = True        # row = v, column = u
    return img


def silhouette_iou(d: dict, view: str, project_pinhole=None, threshold: float = 1e-3) -> float:
    """IoU between the projected occupancy and the rendered pelvis-only image (> threshold) of one view."""
    a = project_occupancy(d["occ"], view, project_pinhole)
    b = d[f"{view}_bone"] > threshold
    return float((a & b).sum() / max((a | b).sum(), 1))


# ---------------------------------------------------------------- display and export helpers (unit 02 notebooks)

def write_obj(path, verts: np.ndarray, faces: np.ndarray) -> Path:
    """Save a triangle mesh as a Wavefront .obj (opens in Blender, MeshLab, Windows 3D Viewer, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fo:
        fo.writelines(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in verts)
        fo.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)   # .obj counts vertices from 1
    return path


def show_mesh(verts: np.ndarray, faces: np.ndarray, title: str = "", ax=None, view=(10, -70), color="#e8dcc0"):
    """Draw a mesh with matplotlib (slow-ish above ~100k triangles, fine for 64^3 meshes).

    The axes are the cube frame: x = patient's Right, y = Anterior, z = Superior (mm).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    own = ax is None
    if own:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="3d")
    tri = verts[faces]
    # simple shading: brighter where the triangle faces the viewer's light
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])
    shade = 0.35 + 0.65 * np.abs(n @ light)
    base = np.array(matplotlib_color(color))
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(shade[:, None] * base, 0, 1), linewidths=0))
    lo, hi = verts.min(0), verts.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.set_xlabel("x (R) mm")
    ax.set_ylabel("y (A) mm")
    ax.set_zlabel("z (S) mm")
    ax.set_title(title)
    if own:
        plt.show()
    return ax


def matplotlib_color(c):
    import matplotlib.colors as mcolors

    return mcolors.to_rgb(c)


def overlap_image(gt2d: np.ndarray, pred2d: np.ndarray) -> np.ndarray:
    """RGB picture of two 2D masks: TP white-ish, FP (predicted, not true) red, FN (missed) blue."""
    g, p = np.asarray(gt2d, bool), np.asarray(pred2d, bool)
    img = np.zeros(g.shape + (3,))
    img[g & p] = (0.92, 0.89, 0.82)
    img[p & ~g] = (0.90, 0.30, 0.30)
    img[g & ~p] = (0.30, 0.50, 0.95)
    return img


def coronal(vol3d: np.ndarray, j: int | None = None) -> np.ndarray:
    """A coronal layer of a cube (RAS axes) for display: rows head -> feet, columns patient's Right -> Left.

    j = the A index of the layer; default = the layer with the most occupied voxels.
    """
    v = np.asarray(vol3d)
    if j is None:
        j = int(np.argmax((v > 0).sum(axis=(0, 2))))
    return v[:, j, :].T[::-1, ::-1]       # (R, S) -> rows S (top) .. I, columns R (left) .. L
