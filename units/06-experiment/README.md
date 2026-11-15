# Unit 06 · A controlled experiment

**Leads:** Adir on 1 view vs 2 views (part B), Noa on training-set size (part C), as in the proposal ·
**Video:** 12 · **Proposal week:** 6 · **Status:** ready

**Needs:** units 02–05 moved into `src/xray3d` (`metrics.evaluate`, `nets.XrayTo3D`, `losses.bce_dice_loss`,
`training.train_step`, `training.validation_dice`), the processed dataset v1, `splits/split_v1.json`, and unit 03's
`results/baselines.csv`.

## Goals
- Design a comparison that changes **one** variable, and say what is held fixed (and what isn't quite: the extra
  LAT encoder).
- Explain why one run per arm proves nothing: 3 seeds per arm, mean ± sample std.
- Compare arms **patient by patient** (paired differences), with a bootstrap interval you wrote yourselves, and a
  decision rule that asks "is the difference bigger than the seed spread?".
- Run the real experiment on a Kaggle GPU (9 runs × 4,000 steps), and present an honest results table with the
  unit 03 baselines, including what you'd write if two views don't help.

## The design
| arm | views | training set | used in |
|---|---|---|---|
| `ap` | AP | 100 % | comparison 1 |
| `ap_lat` | AP + LAT | 100 % | comparisons 1 and 2 (the same runs serve both) |
| `ap_lat_f25` | AP + LAT | 25 % (nested subset) | comparison 2 |

Seeds 0, 1, 2 for every arm; everything else is `trainer.DEFAULTS` (4,000 steps, batch 8, Adam 1e-3, augmentation).
Best checkpoint by validation Dice, scored once on the test split with your `metrics.evaluate`.

## Video
- 12 · A controlled experiment (`G:\My Drive\x-ray-videos`).

## Before the session
- Units 02–05 in `src/xray3d`, committed **and pushed** (Kaggle clones the repo from GitHub).
- Locally the notebook runs a smoke test (2 seeds × 3 arms × 40 steps, about 5–8 minutes on the CPU): its numbers
  are meaningless; it only proves the pipeline end to end.
- The real run: this notebook on Kaggle (GPU T4, Internet on, dataset `pelvis-xray3d-v1` attached), *Save & Run All*.
  Estimated 1–2.5 hours for the 9 runs. Download `results/experiment_v1.csv` from the version's output.

## TODOs
| Part | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| B | B1 | `mean_std` | `src/xray3d/experiment.py` | sample std (n − 1); one run has no spread |
| | B2 | `seed_scores` | `experiment.py` | one number per run: the seed spread |
| | B3 | `per_subject_mean` | `experiment.py` | one number per patient, averaged over seeds |
| | B4 | `paired_differences` | `experiment.py` | match by patient, never by position |
| C | C1 | `bootstrap_ci` | `experiment.py` | resample patients with replacement; percentile interval |
| | C2 | `verdict` | `experiment.py` | real only if beyond both noises: patients (CI) and seeds (std) |
| | C3 | `results_table` | `experiment.py` | mean ± std over seeds; baselines have no ±; same patients in every row |

Claude's infrastructure used here: `run_experiment.py` (trains all arms × seeds through `xray3d.trainer`, resumable,
scores `best.pt` on the test split, writes `results/experiment_v1.csv`; also a command-line tool), and from unit 05
`xray3d/trainer.py` and `xray3d/torchdata.py`.

## Done when
- every check is green, the functions are in `src/xray3d/experiment.py`, and `pytest units/06-experiment` passes
  with nothing skipped,
- the real run is done on Kaggle, `results/experiment_v1.csv` and `results/experiment_v1_table.md` are committed,
- `LOG.md` has the table, the two verdicts, the time per run, and the explain-back answers.
