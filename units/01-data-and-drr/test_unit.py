"""Checks for unit 01 (data and DRR).

Every check_* takes YOUR function and tests it on small cases whose answer was worked out by
hand. There is no reference solution in here: the expected values are numbers, not code.

  notebook:  import test_unit as T;  T.check_voxel_spacing(voxel_spacing)
  pytest:    pytest units/01-data-and-drr   (tests the copies you moved into src/xray3d;
                                             functions not moved yet are skipped)
"""
from __future__ import annotations

import numpy as np

from xray3d import data, drr, geometry, physics, split, volume
from xray3d.checks import call, close, run, skip_if_missing

# s1390's real affine (1.5 mm voxels, axes R, A, S)
AFF_S1390 = np.array([[1.5, 0, 0, -228.58], [0, 1.5, 0, -60.58], [0, 0, 1.5, -12.7], [0, 0, 0, 1]])


def _have_hero() -> bool:
    return data.ct_path(data.HERO).exists()


# ---------------------------------------------------------------- part A: images and volumes

def check_voxel_spacing(fn):
    def diagonal():
        close(call(fn, np.diag([1.5, 1.5, 3.0, 1.0])), [1.5, 1.5, 3.0], "diag(1.5, 1.5, 3)")

    def flipped_axes():
        # negative entries mean the axis runs the other way (e.g. towards Left instead of Right).
        # A spacing is a length, so it's still positive.
        close(call(fn, np.diag([-0.8, -0.8, 2.5, 1.0])), [0.8, 0.8, 2.5], "diag(-0.8, -0.8, 2.5)")

    def rotated():
        # the same voxels, tilted 30 degrees: the matrix isn't diagonal any more, the spacing didn't change
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        a = np.eye(4)
        a[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([0.7, 0.7, 2.0])
        a[:3, 3] = [10, 20, 30]
        close(call(fn, a), [0.7, 0.7, 2.0], "a rotated affine (30 deg about z)")

    def real_ct():
        if not _have_hero():
            print("  (hero CT not on disk: real-CT check skipped)")
            return
        img = data.load_nifti(data.ct_path(data.HERO))
        close(call(fn, img.affine), img.header.get_zooms()[:3], "s1390 vs its NIfTI header", atol=1e-4)

    run("voxel_spacing", diagonal, flipped_axes, rotated, real_ct)


def check_voxel_to_mm(fn):
    def s1390():
        got = call(fn, AFF_S1390, np.array([[0, 0, 0], [10, 20, 30]]))
        close(got, [[-228.58, -60.58, -12.7], [-213.58, -30.58, 32.3]], "s1390 affine, voxels (0,0,0) and (10,20,30)")

    def flipped():
        a = np.array([[-0.8, 0, 0, 100], [0, -0.8, 0, 50], [0, 0, 2.5, -300], [0, 0, 0, 1]])
        close(call(fn, a, np.array([[100, 200, 10]])), [[20, -110, -275]], "flipped axes, voxel (100,200,10)")

    def swapped():
        # array axis 0 runs along world y and axis 1 along world x
        a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        close(call(fn, a, np.array([[2, 3, 4]])), [[3, 2, 4]], "swapped axes, voxel (2,3,4)")

    def rotated():
        # axis 0 runs along world y, axis 1 along world -x: the 3x3 part is not symmetric, so A and A.T differ
        a = np.array([[0, -1, 0, 5], [1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1.0]])
        close(call(fn, a, np.array([[2, 3, 4]])), [[2, 2, 8]], "rotated axes, voxel (2,3,4) (affine @ [i, j, k, 1], not the other way round)")

    def shape():
        got = np.asarray(call(fn, AFF_S1390, np.zeros((7, 3))))
        assert got.shape == (7, 3), f"7 voxels in should give shape (7, 3) out, got {got.shape}"

    run("voxel_to_mm", s1390, flipped, swapped, rotated, shape)


def _toy_volume():
    # every voxel holds its own index as digits: vol[i, j, k] = 100 i + 10 j + k
    i, j, k = np.meshgrid(np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    return 100 * i + 10 * j + k


def check_orthogonal_planes(fn):
    vol = _toy_volume()

    def planes_are(codes, want):
        got = call(fn, vol, (1, 2, 3), codes)
        assert set(got) == {"axial", "coronal", "sagittal"}, f"keys should be axial/coronal/sagittal, got {sorted(got)}"
        for name, w in want.items():
            g = np.asarray(got[name])
            assert g.shape == w.shape, f"codes {codes}: {name} plane has shape {g.shape}, expected {w.shape}"
            assert np.array_equal(g, w), f"codes {codes}: {name} plane holds the wrong voxels (first row {g[0].tolist()})"

    def ras():
        planes_are(("R", "A", "S"), {"axial": vol[:, :, 3], "coronal": vol[:, 2, :], "sagittal": vol[1, :, :]})

    def lps():
        # L/P/I run along the same lines as R/A/S, just the other way: same planes
        planes_are(("L", "P", "S"), {"axial": vol[:, :, 3], "coronal": vol[:, 2, :], "sagittal": vol[1, :, :]})

    def permuted():
        # here array axis 0 is the head-feet axis
        planes_are(("I", "L", "P"), {"axial": vol[1, :, :], "coronal": vol[:, :, 3], "sagittal": vol[:, 2, :]})

    run("orthogonal_planes", ras, lps, permuted)


def check_mask_volume_cm3(fn):
    def unit_cube():
        close(call(fn, np.ones((10, 10, 10), bool), (1.0, 1.0, 1.0)), 1.0, "1000 voxels of 1 mm^3")

    def our_spacing():
        close(call(fn, np.ones((10, 10, 10), bool), (1.5, 1.5, 1.5)), 3.375, "1000 voxels of 1.5 mm")

    def anisotropic_and_partial():
        m = np.zeros((20, 20, 20), bool)
        m[:5, :4, :2] = True  # 40 voxels
        close(call(fn, m, (0.8, 0.8, 2.5)), 0.064, "40 voxels of 0.8 x 0.8 x 2.5 mm")

    def real_pelvis():
        if not _have_hero():
            print("  (hero masks not on disk: real-pelvis check skipped)")
            return
        masks = data.load_mask(data.HERO, data.PELVIS)
        pelvis = masks["hip_left"] | masks["hip_right"] | masks["sacrum"]
        v = float(call(fn, pelvis, (1.5, 1.5, 1.5)))
        assert 300 < v < 3000, f"s1390 pelvis came out {v:.0f} cm^3; a solid adult pelvis mask is roughly 1,000-1,500 cm^3"

    run("mask_volume_cm3", unit_cube, our_spacing, anisotropic_and_partial, real_pelvis)


def check_touches_scan_edge(fn):
    def inside():
        m = np.zeros((10, 10, 30), bool)
        m[3:6, 3:6, 10:20] = True
        assert not bool(call(fn, m, 2)), "a blob in the middle doesn't touch the edge"

    def cut_at_bottom():
        m = np.zeros((10, 10, 30), bool)
        m[3:6, 3:6, 0:12] = True
        assert bool(call(fn, m, 2)), "a mask starting at slice 0 is cut by the scan"

    def cut_at_top_within_margin():
        m = np.zeros((10, 10, 30), bool)
        m[3:6, 3:6, 15:28] = True  # last voxel at slice 27; slices 27, 28, 29 are within 3 of the end
        assert bool(call(fn, m, 2, margin=3)), "slice 27 of 30 is within 3 slices of the top"
        assert not bool(call(fn, m, 2, margin=1)), "with margin=1 only slice 29 counts as the edge"

    def other_axis():
        m = np.zeros((10, 10, 30), bool)
        m[0:4, 3:6, 10:20] = True
        assert bool(call(fn, m, 0)), "along axis 0 this mask starts at index 0"
        assert not bool(call(fn, m, 2)), "along axis 2 it's well inside"

    def empty():
        assert not bool(call(fn, np.zeros((5, 5, 5), bool), 2)), "an empty mask touches nothing"

    run("touches_scan_edge", inside, cut_at_bottom, cut_at_top_within_margin, other_axis, empty)


# ---------------------------------------------------------------- part B: HU and attenuation

def check_hu_to_mu(fn):
    def reference_points():
        hu = np.array([0, -1000, 1000, 3000, 500])
        close(call(fn, hu), [0.02, 0.0, 0.04, 0.08, 0.03], "water, air, +1000, +3000, +500 HU")

    def below_air():
        # CT padding is often -1024 or -2048. Attenuation can't be negative.
        close(call(fn, np.array([-1024, -2048, -1200])), [0, 0, 0], "values below -1000 HU", atol=1e-9)

    def other_mu_water():
        close(call(fn, np.array([0, 1000]), mu_water=0.019), [0.019, 0.038], "mu_water=0.019")

    def int16_volume():
        hu = np.full((3, 4, 5), 2000, dtype=np.int16)  # CT files store int16
        mu = np.asarray(call(fn, hu))
        assert mu.shape == (3, 4, 5), f"shape should be kept, got {mu.shape}"
        assert np.issubdtype(mu.dtype, np.floating), f"attenuation needs a float dtype, got {mu.dtype}"
        close(mu.ravel()[:3], [0.06, 0.06, 0.06], "int16 input of 2000 HU")

    run("hu_to_mu", reference_points, below_air, other_mu_water, int16_volume)


def check_window(fn):
    def soft_tissue():
        hu = np.array([-160, 40, 240, -1000, 3000, 140])
        close(call(fn, hu, 40, 400), [0, 0.5, 1, 0, 1, 0.75], "soft-tissue window (center 40, width 400)")

    def bone():
        close(call(fn, np.array([400, 1400, 900]), 400, 2000), [0.5, 1.0, 0.75], "bone window (center 400, width 2000)")

    run("window", soft_tissue, bone)


# ---------------------------------------------------------------- part C: cameras

K = np.array([[1000.0, 0, 320], [0, 1000.0, 240], [0, 0, 1]])
ROT_Y90 = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # rotation by +90 degrees about the y axis


def check_project_pinhole(fn):
    I, z = np.eye(3), np.zeros(3)

    def on_axis():
        close(call(fn, np.array([[0, 0, 500.0]]), K, I, z), [[320, 240]], "a point on the optical axis lands on the principal point")

    def off_axis():
        close(call(fn, np.array([[50, 0, 500.0], [0, -30, 500.0]]), K, I, z), [[420, 240], [320, 180]], "points 50 mm right / 30 mm up at depth 500")

    def twice_as_far():
        close(call(fn, np.array([[50, 0, 1000.0]]), K, I, z), [[370, 240]], "the same offset at depth 1000 (perspective: half the shift)")

    def translated():
        close(call(fn, np.array([[10, 0, 400.0]]), K, I, np.array([0, 0, 100.0])), [[340, 240]], "t = (0, 0, 100)")

    def rotated():
        close(call(fn, np.array([[-500, 20, 0.0]]), K, ROT_Y90, z), [[320, 280]], "R = 90 deg about y")

    run("project_pinhole", on_axis, off_axis, twice_as_far, translated, rotated)


def check_project_parallel(fn):
    I, z, c = np.eye(3), np.zeros(3), np.array([100.0, 100.0])

    def on_axis():
        close(call(fn, np.array([[0, 0, 500.0]]), I, z, 2.0, c), [[100, 100]], "a point on the axis lands on the centre")

    def off_axis():
        close(call(fn, np.array([[50, 0, 500.0], [0, -30, 500.0]]), I, z, 2.0, c), [[125, 100], [100, 85]], "50 mm right / 30 mm up, 2 mm pixels")

    def depth_does_not_matter():
        close(call(fn, np.array([[50, 0, 1000.0], [50, 0, 3.0]]), I, z, 2.0, c), [[125, 100], [125, 100]], "the same point at depth 1000 and 3 (parallel: no shrinking)")

    def rotated():
        close(call(fn, np.array([[-500, 20, 0.0]]), ROT_Y90, z, 2.0, c), [[100, 110]], "R = 90 deg about y")
        close(call(fn, np.array([[0, 0, 500.0]]), ROT_Y90, z, 2.0, c), [[350, 100]], "R = 90 deg about y, a point on the world z axis")

    def translated():
        close(call(fn, np.array([[10, 0, 400.0]]), I, np.array([4, 6, 100.0]), 2.0, c), [[107, 103]], "t = (4, 6, 100)")

    run("project_parallel", on_axis, off_axis, depth_does_not_matter, rotated, translated)


# ---------------------------------------------------------------- part D: the DRR

def check_parallel_drr(fn):
    mu = np.zeros((20, 20, 20))
    mu[5:15, 5:15, 5:15] = 0.02  # a 10-voxel cube of water

    def cube_1mm():
        p = np.asarray(call(fn, mu, (1.0, 1.0, 1.0), 0))
        assert p.shape == (20, 20), f"summing a 20^3 volume along one axis gives a 20x20 image, got {p.shape}"
        close([p[10, 10], p[0, 0]], [0.2, 0.0], "ray through the cube (10 mm of water) / ray that misses")

    def uses_the_right_spacing():
        p2 = np.asarray(call(fn, mu, (1.0, 1.0, 2.5), 2))
        close(p2[10, 10], 0.5, "along axis 2 with 2.5 mm voxels the cube is 25 mm thick")
        p0 = np.asarray(call(fn, mu, (1.0, 1.0, 2.5), 0))
        close(p0[10, 10], 0.2, "along axis 0 the voxels are 1 mm, so still 10 mm")

    def twenty_cm_of_water():
        w = np.full((4, 4, 200), 0.02)
        close(np.asarray(call(fn, w, (1.0, 1.0, 1.0), 2))[0, 0], 4.0, "200 mm of water")

    run("parallel_drr", cube_1mm, uses_the_right_spacing, twenty_cm_of_water)


def _marker_volumes():
    """The same tiny 'patient' stored in three different array orientations.

    RAS version: shape (4, 5, 6), one dense voxel at the patient's far Right (i=3), far
    Posterior (j=0) and top (Superior, k=5). Its AP image is 6 rows (S->I) by 4 columns (R->L);
    its LAT image is 6 rows (S->I) by 5 columns (A->P). Both put the voxel on the top row.
    """
    ras = np.zeros((4, 5, 6))
    ras[3, 0, 5] = 1.0
    sp_ras = (1.0, 2.0, 3.0)
    lps = ras[::-1, ::-1, :]  # flip R->L and A->P
    psr = np.transpose(ras, (1, 2, 0))[::-1, :, :]  # axes become (A, S, R), then flip A->P
    return [
        (("R", "A", "S"), ras, sp_ras),
        (("L", "P", "S"), lps, sp_ras),
        (("P", "S", "R"), psr, (sp_ras[1], sp_ras[2], sp_ras[0])),
    ]


def check_ap_lat_views(fn):
    def view(codes, vol, sp):
        ap, lat = (np.asarray(x) for x in call(fn, vol, sp, codes))
        assert ap.shape == (6, 4), f"codes {codes}: AP should be 6 rows (S->I) x 4 cols (R->L), got {ap.shape}"
        assert lat.shape == (6, 5), f"codes {codes}: LAT should be 6 rows (S->I) x 5 cols (A->P), got {lat.shape}"
        r, c = np.unravel_index(np.argmax(ap), ap.shape)
        assert (r, c) == (0, 0), f"codes {codes}: in AP the marker (top, patient's Right) should be at row 0, col 0, found ({r}, {c})"
        close(ap[0, 0], 2.0, f"codes {codes}: AP ray crosses the marker along A-P, voxel length 2 mm")
        r, c = np.unravel_index(np.argmax(lat), lat.shape)
        assert (r, c) == (0, 4), f"codes {codes}: in LAT the marker (top, Posterior) should be at row 0, col 4, found ({r}, {c})"
        close(lat[0, 4], 1.0, f"codes {codes}: LAT ray crosses the marker along R-L, voxel length 1 mm")

    cases = [lambda c=c, v=v, s=s: view(c, v, s) for c, v, s in _marker_volumes()]
    run("ap_lat_views", *cases)


# ---------------------------------------------------------------- part E: the split

def check_split_subjects(fn):
    ids = [f"s{i:04d}" for i in range(1000)]

    def partition():
        sp = call(fn, ids)
        assert set(sp) == {"train", "val", "test"}, f"keys should be train/val/test, got {sorted(sp)}"
        tr, va, te = (set(sp[k]) for k in ("train", "val", "test"))
        assert not (tr & va or tr & te or va & te), "a subject is in two parts: that's leakage"
        assert tr | va | te == set(ids), "some subjects went missing"
        assert sum(len(sp[k]) for k in sp) == len(ids), "a subject appears twice inside one part"

    def sizes():
        sp = call(fn, ids)
        got = [len(sp[k]) for k in ("train", "val", "test")]
        assert all(abs(g - w) <= 1 for g, w in zip(got, [700, 100, 200])), f"sizes {got}, expected about 700/100/200"

    def reproducible():
        assert call(fn, ids) == call(fn, ids), "two calls with the same seed must give the same split"

    def order_does_not_matter():
        a = call(fn, ids)
        b = call(fn, list(reversed(ids)))
        assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}, (
            "the split changed when the input list was reversed; the same subjects must always land in the same part")

    def seed_matters():
        assert sorted(call(fn, ids, seed=0)["test"]) != sorted(call(fn, ids, seed=1)["test"]), "a different seed should give a different split"

    run("split_subjects", partition, sizes, reproducible, order_does_not_matter, seed_matters)


# ---------------------------------------------------------------- pytest: the copies in src/xray3d

def test_voxel_spacing():
    check_voxel_spacing(skip_if_missing(volume, "voxel_spacing"))


def test_voxel_to_mm():
    check_voxel_to_mm(skip_if_missing(volume, "voxel_to_mm"))


def test_orthogonal_planes():
    check_orthogonal_planes(skip_if_missing(volume, "orthogonal_planes"))


def test_mask_volume_cm3():
    check_mask_volume_cm3(skip_if_missing(volume, "mask_volume_cm3"))


def test_touches_scan_edge():
    check_touches_scan_edge(skip_if_missing(volume, "touches_scan_edge"))


def test_hu_to_mu():
    check_hu_to_mu(skip_if_missing(physics, "hu_to_mu"))


def test_window():
    check_window(skip_if_missing(physics, "window"))


def test_project_pinhole():
    check_project_pinhole(skip_if_missing(geometry, "project_pinhole"))


def test_project_parallel():
    check_project_parallel(skip_if_missing(geometry, "project_parallel"))


def test_parallel_drr():
    check_parallel_drr(skip_if_missing(drr, "parallel_drr"))


def test_ap_lat_views():
    check_ap_lat_views(skip_if_missing(drr, "ap_lat_views"))


def test_split_subjects():
    check_split_subjects(skip_if_missing(split, "split_subjects"))
