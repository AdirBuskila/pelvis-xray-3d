# Unit 08 · Putting it together

**Leads:** both (swap the driver on every TODO) · **Video:** 14 · **Proposal week:** 8 · **Status:** ready

**Needs:** units 02–05 moved into `src/xray3d`, the processed dataset v1, `splits/split_v1.json`, a trained model
(unit 06's best 2-view run from Kaggle, or unit 05's `first-2view`), and whichever results files exist:
`results/baselines.csv` (unit 03), `results/experiment_v1.csv` (unit 06), `results/sim_vs_real_v1.csv` (unit 07).

## Goals
- Take one patient's two X-rays to a pelvis mesh in millimetres, and say exactly what contract each step keeps
  (input standardisation, eval mode, the threshold, the cube frame).
- Show the method working on a typical test patient (the demo for class), and export the meshes.
- Tell the whole results story in one honest table: baselines → network (1 vs 2 views, training-set size) →
  simulated vs real; mean ± sd over **subjects**, with n.
- Write the report: the outline is in [`report_outline.md`](report_outline.md), with the licence section, the
  reproduce-everything checklist and the personal-contribution table built from the units' logs.

## Video
- 14 · Putting it together (`G:\My Drive\x-ray-videos`): the pipeline fly-through, the results story, the report, the demo.

## Before the session
- Download the chosen run from the Kaggle notebook's Output tab (`runs/<name>/best.pt`, plus `log.csv` and
  `summary.json`) into `trainer.runs_root()` (locally `data/processed/runs/<name>/`). Put its name in the
  notebook's `MODEL` parameter. For the class demo on Colab, also upload it to `MyDrive/pelvis-xray3d/runs/<name>/`.
- Fill in the `LOG.md` of every earlier unit (the contribution table is built from them).
- CPU only; the whole notebook takes a few minutes.

## TODOs
| Part | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A | A1 | `prepare_inputs` | `src/xray3d/pipeline.py` | the network's input contract: per-image standardisation, view order, shapes |
| | A2 | `predict_occupancy` | `pipeline.py` | logits → probability → mask; eval mode, no gradients |
| | A3 | `predict_from_xrays` | `pipeline.py` | the whole pipeline: two X-rays → occupancy → mesh in mm |
| – | – | the demo (Claude's cells) | | every test subject scored; the median one shown; OBJ export; the Colab cell for class |
| B | B1 | `best_and_worst` | `pipeline.py` | ranking by a metric: its direction, inf and NaN |
| | B2 | `story_table` | `pipeline.py` | mean ± sd over subjects (not rows), n, the non-finite values |
| C | – | the checklist, the contribution table (Claude's cells) | | where every number comes from; who did what |
| | – | the report | `report_outline.md` → your report | |

## Done when
- every check is green, the functions are in `src/xray3d/pipeline.py`, and `pytest units` passes with nothing skipped,
- the checklist cell shows ✔ on every line, and the story table has the rows of all three results files,
- the demo runs on Colab from Drive (try it before the class),
- the report is written from the outline, with the contribution table, and `LOG.md` is filled in.
