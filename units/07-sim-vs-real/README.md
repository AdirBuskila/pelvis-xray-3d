# Unit 07 · Simulated vs real

**Leads:** Adir on the gap (part A), Noa on the error analysis (part C); part B alternates · **Videos:** 13 ·
**Proposal week:** 7 · **Status:** ready (the real evaluation needs unit 06's trained models)

## Goals
- Turn real detector values into line integrals (Beer–Lambert, dark and flat levels) and say which errors matter.
- Bring a real C-arm frame into the network's camera: what can be matched exactly, what only approximately, what not at all.
- Measure the sim-to-real gap one difference at a time (field of view, geometry, appearance), with 1 and 2 views.
- Check a prediction without a 3D ground truth: re-projection against the frame's own labels, MC-dropout uncertainty.
- Find where the network fails on the simulated test set: Dice vs the clinical CE angle, the worst cases.

## Videos
- 13 · Simulated vs real.

## Data and licence
DeepFluoro (`data/raw/deepfluoro`, 6 cadavers, CC BY-NC-ND 4.0): already on the laptop. On Colab/Kaggle the notebook
downloads only what it needs (`deepfluoro.fetch()`, ~400 MB). **Nothing derived from it is committed**: clear the
notebook's outputs before saving (CI enforces it: `.github/workflows/licence.yml` fails any push whose unit 07
notebook has a cell output), and commit only `results/sim_vs_real_v1.csv` (aggregate numbers; with smoke models the
notebook writes `sim_vs_real_smoke.csv` next to the runs instead, outside git). The cached arrays live in
`deepfluoro.cache_dir()` under the git-ignored data folder (on Kaggle in `/tmp`, so they never become the
notebook's published Output).

All DeepFluoro frames are near-PA (source behind the cadaver; the largest angle between two frames of a specimen is
11–49°), so there is **no real lateral view**. The 2-view model gets (real AP + simulated LAT).

## Before the session
- Units 01–05 moved into `src/xray3d` (the notebook uses `hu_to_mu`, `project_pinhole`, `bbox_center_mm`,
  `occupancy`, `cube_grid`, `resample`, the metrics incl. `ce_angle`, `pixel_index`, the network and the training code).
- `pip install pydicom` and DiffDRR (`pip install -e ".[drr]"`).
- **The models:** download unit 06's Kaggle output (the `runs/` folder: `exp-ap-s0..2`, `exp-ap_lat-s0..2`) into
  `trainer.runs_root()` (the notebook prints it). Without them the notebook trains two tiny CPU models so it runs,
  and every number it prints is meaningless (the results file says so).
- First run: ~4 minutes to build the six specimens (cached afterwards); the rest runs on the CPU in a few minutes.

## TODOs
| Part | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A | A1 | `log_transform` | `src/xray3d/realxray.py` | detector intensity → line integral; dark vs flat |
| | A2 | `gap` | `realxray.py` | paired drop per specimen; sign conventions; n = 6 |
| B | B1 | `reprojection_iou` | `realxray.py` | the shadow of a prediction; points per voxel; field of view |
| | B2 | `mc_dropout_predict` | `realxray.py` | dropout on at test time, everything else eval |
| | B3 | `uncertainty_vs_error` | `realxray.py` | is the variance where the errors are? |
| C | C1 | `ce_error` | `realxray.py` | the clinical error; nan is not 0 |
| | C2 | `error_summary` | `realxray.py` | rank correlation, worst cases |

Claude's infrastructure used here: `xray3d/deepfluoro.py` (loading, frame choice, the real camera, the virtual AP
camera and the warp, the DRR at the real pose, the simulated pair, the cache), plus `prep`, `dataset`, `torchdata`,
`trainer`.

## What the geometry matching can and can't do
| | real frame | what we do | exact? |
|---|---|---|---|
| direction | a few degrees off the CT's A-P axis | turn the cube (and the ground truth) onto the source → pelvis line | yes (a homography) |
| side | source behind (PA) | mirror through the pelvis's coronal plane | no: depth magnification inverts (a few %) |
| distance | 650–830 mm source → pelvis | zoom by d / 750 | only at the centre's depth |
| field of view | ~200 mm at the pelvis (24–32 % of our image) | mark the rest invalid, fill with the mean | no |
| the CT | cropped to the cadaver (no table, little soft tissue) | nothing | no |

## Done when
- every check is green, the functions are in `src/xray3d/realxray.py`, and `pytest units/07-sim-vs-real` passes with nothing skipped,
- the notebook ran with unit 06's real models and `results/sim_vs_real_v1.csv` has `smoke_models_meaningless = False`,
- `LOG.md` has the gap per rung of the ladder (1 and 2 views) and the error-analysis numbers,
- the notebook is saved **without outputs**.
