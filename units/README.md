# Learning units

Each unit is one part of the proposal's 8-week plan, done together in the weekly session.

| Unit | Topic | Videos (in `G:\My Drive\x-ray-videos`) | Leads |
|---|---|---|---|
| [00-setup](00-setup/) | environment (local, Colab, Kaggle) + the workflow on a toy | 00 | both |
| [01-data-and-drr](01-data-and-drr/) | CT files, masks, HU → μ, projection, the DRR, the split | 01, 02, 03, 04 | Noa data, Adir DRR |
| [02-shape-and-metrics](02-shape-and-metrics/) | the 64³ cube, marching cubes, Dice / HD95 / NSD / Chamfer, a clinical angle; building the dataset | 05, 06 | Noa metrics, Adir dataset |
| [03-baselines](03-baselines/) | visual hull and nearest neighbour, the first results table | 07, 08 | Noa hull, Adir NN |
| [04-network](04-network/) | convolution, encoder-decoder, lifting 2D features into 3D | 09, 10 | both |
| [05-training](05-training/) | Dice + BCE loss, the training step, overfit one batch, the first GPU run | 10, 11 | both |
| [06-experiment](06-experiment/) | 1 view vs 2, training-set size, seeds and honest tables | 12 | Adir 1 vs 2, Noa size |
| [07-sim-vs-real](07-sim-vs-real/) | real fluoroscopy (DeepFluoro), the gap, error analysis, uncertainty | 13 | Adir gap, Noa errors |
| [08-putting-it-together](08-putting-it-together/) | two X-rays → mesh end to end, the demo, the report | 14 | both |

## Every unit has the same files

| File | What it is | Who writes it |
|---|---|---|
| `README.md` | goals, videos, the TODO list, how to know you're done | Claude |
| `explainer.he.md` | short Hebrew recap to read **after** the video | Claude |
| `notebook.ipynb` | the guided notebook; the TODO cells are yours | Claude (scaffold) + you (TODOs) |
| `test_unit.py` | the checks your TODOs must pass | Claude |
| `LOG.md` | who drove each TODO, hints used, explain-back answers | you |

New units start from [`_template/`](_template/).

## The loop for every TODO

1. Read the TODO's text and its **nudge**. Talk about it before typing.
2. One of you types (the driver), the other explains what's being typed. Swap on the next TODO.
3. Run the check cell right below. Red → read the message, fix, rerun. Stuck → ask Claude for hint 2, then 3.
4. Green → the partner explains the block back in their own words.
5. When a part is done, move its functions into `src/xray3d/` and run `pytest units` (from the repo root).
