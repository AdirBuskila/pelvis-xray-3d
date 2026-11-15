# Unit 00 · Setup and the workflow

**Lead:** both, together · **Video:** 00 (the big picture) · **Time:** about 30 minutes, at the start of the first session · **Status:** ready

## Goals
- The notebook runs on the laptop, on Colab and on Kaggle, and you know what differs (GPU, where data lives).
- You've done the whole loop once on a toy: stub → your code → check → move into `src/xray3d` → `pytest` → `LOG.md`.
- You can say, with a 2D example, why one X-ray can't tell you the 3D shape.

## Before the session (once, on the laptop)
From the repo folder, in the Anaconda prompt:
```
python -m pip install -e .
jupyter lab units/00-setup/notebook.ipynb
```
(or open the notebook in VS Code and pick the Anaconda Python as the kernel).

## TODOs
| TODO | Function | Moves into | Concept |
|---|---|---|---|
| 0.1 | `shadow` | `src/xray3d/toy.py` | a projection adds up what each ray crosses (in mm) |
| 0.2 | `same_shadow` | `src/xray3d/toy.py` | one view loses depth: different shapes, same shadow |

## Done when
- both checks are green and `pytest` shows them **passed** (not skipped),
- the setup cell ran on Colab and on Kaggle, and `LOG.md` says what you saw,
- you've answered the two explain-back questions.
