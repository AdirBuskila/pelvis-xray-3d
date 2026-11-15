# Unit 05 · Training

**Leads:** both (swap the driver on every TODO) · **Videos:** 10 (end), 11 · **Proposal week:** 4–5 · **Status:** ready

## Goals
- Explain why the loss is BCE + soft Dice for a cube that is ~97 % empty, and write both.
- Write one training step and the validation Dice, and say what each line is for.
- Prove the network can learn (overfit one batch) before spending GPU time.
- Run the first real training on Colab/Kaggle, and read its curves (over/underfitting, best vs last checkpoint).

## Videos
- 11 · Training it (and the end of 10).

## Before the session
- Unit 04's classes in `src/xray3d/nets.py`; the processed dataset v1 (unit 02) on disk; `splits/split_v1.json`.
- Parts A–B run on the laptop (the overfit test takes 3–5 minutes on the CPU). Part C's real run needs a GPU:
  Colab (T4) or Kaggle (GPU T4, Internet on, dataset `pelvis-xray3d-v1` attached).

## TODOs
| Part | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A | A1 | `soft_dice_loss` | `src/xray3d/losses.py` | overlap with probabilities; per sample; eps |
| | A2 | `bce_dice_loss` | `losses.py` | per-voxel + per-shape loss; BCE from logits |
| B | B1 | `train_step` | `src/xray3d/training.py` | train mode, zero_grad, forward, backward, step |
| | B2 | `validation_dice` | `training.py` | eval mode, no_grad, the thresholded Dice per subject |
| | – | the overfit test (Claude's cell, your functions) | | can it memorise one batch? |
| C | – | the first real run (`xray3d.trainer`, Claude's) | | checkpoints, resume, seeds, curves |

Claude's infrastructure used here: `xray3d/torchdata.py` (dataset, standardisation, augmentation, loaders),
`xray3d/trainer.py` (the loop, checkpoints, log, resume, Drive) and `train.py` (the same from the command line).

## Done when
- every check is green, the functions are in `src/xray3d/`, and `pytest units/05-training` passes with nothing skipped,
- the overfit test passes (Dice > 0.9 on the batch),
- one real run (`first-2view`, 4,000 steps) has finished on a GPU, and `LOG.md` has its best val Dice.
