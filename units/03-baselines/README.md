# Unit 03 · Baselines

**Leads:** Noa on the visual hull (A), Adir on nearest neighbour (B), as in the proposal ·
**Videos:** 07, 08 · **Proposal week:** 3 · **Status:** ready; start with part A (after video 07)

**Needs:** units 01–02 moved into `src/xray3d` (`geometry.project_pinhole`, `shape.mesh_from_occupancy`,
`metrics.dice`, `metrics.evaluate`) and the processed dataset v1 (unit 02, `build_dataset.ipynb`).

## Goals
- Go from a voxel to the pixel it lands on (rounding, the edge of the detector), and back: smear an image along its rays.
- Build the visual hull from two silhouettes and explain exactly what it gets wrong (concavities: the hollow of
  the pelvic ring) and why oracle silhouettes make it an upper bound for silhouette methods.
- Build nearest-neighbour retrieval on the X-rays, its ceiling (the oracle) and the laziest answer (the mean shape),
  and explain why retrieval is the floor a network must beat (Tatarchenko et al. 2019).
- Score all of them on the test split with your unit-02 metrics: the first results table, `results/baselines.csv`.

## Videos and reading
- 07 · From two views back to 3D (`G:\My Drive\x-ray-videos`). Optional: [Geiger 4.1 Stereo Preliminaries](https://www.youtube.com/watch?v=6hr6xpOU-uw) (44 min).
- 08 · The memorisation floor. Reading: [Tatarchenko et al. 2019, *What Do Single-view 3D Reconstruction Networks Learn?*](https://arxiv.org/abs/1905.03678), §1–4 (~30 min).

## TODOs
| Part · video | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A · 07 | A1 | `pixel_index` | `baselines.py` | voxel → pixel: rounding, on or off the detector |
| | A2 | `backproject` | `baselines.py` | smear an image back along its rays |
| | A3 | `visual_hull` | `baselines.py` | a voxel survives only if every silhouette sees it |
| B · 08 | B1 | `nn_features` | `baselines.py` | downsample + standardise: what "similar X-rays" means |
| | B2 | `nearest_neighbour` | `baselines.py` | L2 retrieval, all distances at once |
| | B3 | `oracle_neighbour` | `baselines.py` | the ceiling of retrieval (it cheats with the GT) |
| | B4 | `mean_shape` | `baselines.py` | the same answer for everyone |
| – | – | the results table (Claude's cell) | `results/baselines.csv` | every method × test subject × metric |

## Done when
- every check is green, all functions are in `src/xray3d/baselines.py`, and `pytest units` passes with nothing skipped,
- `results/baselines.csv` is written (45 rows locally: 5 methods × 9 test subjects) and committed,
- `LOG.md` has the table's means (Dice, HD95) and the explain-back answers.
