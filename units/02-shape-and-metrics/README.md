# Unit 02 · Shape and metrics

**Leads:** Adir on the cube and the dataset (A, C), Noa on the metrics (B), as in the proposal ·
**Videos:** 05, 06 · **Proposal week:** 2 · **Status:** ready; start with part A (after video 05)

**Needs:** unit 01 moved into `src/xray3d` (`volume.py`, `physics.py`, `geometry.py`).

## Goals
- Put every pelvis on the same 64³ grid (a fixed 320 mm cube, 5 mm voxels, RAS axes) and say why each choice was made.
- Go from voxels to a mesh in mm with marching cubes, and look at it.
- Score a reconstruction with overlap (Dice, IoU), surface distances (HD95, ASSD, NSD, Chamfer) and one clinical
  angle, and explain when they disagree.
- Build the processed dataset every later unit uses, with your own functions, and check its camera geometry.

## Videos
- 05 · Representing 3D shape (`G:\My Drive\x-ray-videos`). Optional: [Sebastian Lague, Marching Cubes](https://www.youtube.com/watch?v=M3iI2l0ltbE) (6 min).
- 06 · Measuring a reconstruction. Optional: [Geiger 8.4 Volumetric Fusion](https://www.youtube.com/watch?v=B2BTSKcYqtQ) (surfaces from volumes, 38 min).

## TODOs
| Part · video | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A · 05 | A1 | `bbox_center_mm` | `shape.py` | where the cube goes (box centre vs centroid) |
| | A2 | `cube_grid` | `shape.py` | voxel centres of a grid, in mm |
| | A3 | `resample` | `shape.py` | the inverse affine; interpolation order; what's outside |
| | A4 | `occupancy` | `shape.py` | a mask on another grid: float, linear, threshold 0.5 |
| | A5 | `mesh_from_occupancy` | `shape.py` | marching cubes, vertices in mm, a closed surface |
| B · 06 | B1 | `dice` | `metrics.py` | overlap |
| | B2 | `iou` | `metrics.py` | overlap, another denominator |
| | B3 | `surface_voxels` | `metrics.py` | a surface is what erosion removes |
| | B4 | `surface_distances` | `metrics.py` | distance transforms instead of all pairs |
| | B5 | `hd95` | `metrics.py` | the worst case, robustly, both directions |
| | B6 | `assd` | `metrics.py` | the average case, pooled |
| | B7 | `nsd` | `metrics.py` | a tolerance (1.5 mm and 5 mm: see the explainer) |
| | B8 | `chamfer` | `metrics.py` | the same idea on point clouds |
| | B9 | `ce_angle` | `metrics.py` | a clinical number from the shape |
| | B10 | `evaluate` | `metrics.py` | the one call later units use |
| C | – | `build_dataset.ipynb` (Claude's) | – | the dataset, built with your functions; the camera check |

## Done when
- every check is green, all functions are in `src/xray3d/`, and `pytest units` passes with nothing skipped,
- `build_dataset.ipynb` has run locally (27 subjects) and, on Kaggle, on the full dataset (saved as the Kaggle
  dataset `pelvis-xray3d-v1` and/or as a zip on Drive),
- the silhouette check prints IoU ≥ 0.9 for both views,
- `LOG.md` has the metric table of the test cases and the dataset numbers.
