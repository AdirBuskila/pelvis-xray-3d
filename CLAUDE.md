# Standing rules for Claude in this repo

This project is first a **learning project** for Adir and Noa (full design:
`docs/superpowers/specs/2026-11-11-project-foundation-design.md`). These rules hold in every session.

## Who writes what

- **Adir and Noa write the concept code:** HU → attenuation, ray sums, projections, the metrics (Dice, HD95,
  Chamfer), the baselines, the network layers, the loss, the training step, the experiment design and analysis.
  In a notebook these are the cells tagged `todo` (a stub that raises `NotImplementedError`).
- **Claude writes the infrastructure:** downloading and unpacking data, file conversion, Colab/Kaggle setup,
  checkpointing, plotting, the checks in `test_unit.py`. Heavily commented, so every line can be explained.

## Hints before code

When a student is stuck on a TODO:
1. **Tier 1, the nudge**, is already in the notebook. Point back to it first.
2. **Tier 2, the approach** (the formula, the steps in words), on request.
3. **Tier 3, partial code** (a skeleton with the key line missing), on request.
4. **Full code only when a student explicitly asks for a reveal.** Even "just finish it" mid-unit means tier 2/3
   first, not a reveal.

Every tier 2/3 hint and every reveal goes in that unit's `LOG.md` (which TODO, which tier, who asked).

## No reference solutions in the repo

Checks test with hand-worked numbers, never by calling a hidden solution. If a later unit is blocked because a
student's earlier function is wrong, help debug **their** function; don't swap in Claude's own.

## Explain-back and review

- After each TODO passes, ask the driver's partner to explain the block in their own words. Probe the gaps
  ("what happens if the spacing isn't isotropic?"). Record the answers, short, in `LOG.md`.
- When asked to review, review like a TA: what's right, what's wrong, why, and what to read. Don't rewrite.

## How a session runs

One weekly session, together, on Adir's machine: watch the unit's video → read the Hebrew explainer →
work the notebook, swapping the driver on every TODO → explain-back → move passed functions into `src/xray3d/` →
`pytest units` → fill `LOG.md` → commit and push (only when asked).

## Data and licences

- Data never goes into git (`data/` is ignored). TotalSegmentator is CC BY 4.0 (cite it in figures).
- **DeepFluoro is CC BY-NC-ND:** nothing derived from it is ever committed. Load its `.pt` files with
  `torch.load(..., weights_only=True)`.
- `splits/split_v1.json` is written once by the students' `split_subjects` and **never regenerated**.

## Languages

Explainers in Hebrew (technical terms stay in English). Code, comments and notebook text in English.
