# Pelvis X-ray → 3D: project foundation and learning format

Date: 2026-11-11 · Team: Adir and Noa · Course: Intro to Computer Vision in Medical Imaging (HIT, 62217)

## Intent

Adir and Noa are doing their final project together: reconstructing 3D pelvis bone shape from two X-rays (AP + lateral) with a 2D-encoder / 3D-decoder network, and measuring the gap between simulated and real X-rays. The full research plan is in the proposal:
`C:\Users\Adir\Desktop\Adir\busi-notes\Year-3-Sem-A\Medical-CV\Project\הצעת פרויקט גמר - שחזור עצם תלת-ממדי משני צילומי רנטגן.md`
(an 8-week table, data sources, tools, metrics, risks). This document does not repeat it. It covers **how the work is organised**, not what the research is.

**Why this format:** the main goal is learning. The lecturer knows the field but is hard to follow, so this project is where they actually learn the material (DRR / projection geometry, 3D metrics, encoder-decoder networks, experiment design). They want to understand every line in the repo, but they do **not** want to hand-write the whole project, and they do not want Claude to write it all for them either.

## Goals

- A repo where every line is either written by Adir/Noa or commented well enough that they can explain it.
- The project runs as a sequence of **learning units**, one per part of the proposal (roughly one per week). Every unit follows the same loop:
  1. **Watch/read:** 1–3 resources (video first, then maybe one article). Claude checks that each link works before recommending it.
  2. **Explainer:** a short note in Hebrew from Claude: the concept in plain words, linked to the course lectures (weeks 9–12: camera geometry, projection, point clouds, meshes).
  3. **Guided notebook:** Claude writes the scaffold (imports, loading, plotting, plumbing). The cells that hold the idea are left as `TODO` blanks with a hint, plus a test that passes when the blank is filled in correctly.
  4. **Explain-back:** the student explains each block in their own words. Claude probes the gaps.
  5. **Review:** Claude reviews the student's code like a TA would: what's right, what's wrong, and why.
- A clear split of authorship:
  - **Students write** the concept-carrying code: HU→attenuation and ray summation, the metrics (Dice, HD95, Chamfer), the baselines, the network layers, the loss, the training step, the experiment design and its analysis.
  - **Claude writes** the infrastructure: dataset download/unpacking, file conversion, Colab/Kaggle setup, checkpointing, plotting helpers, the test harness. All of it is heavily commented.
- Each unit has a lead (per the proposal's table). After a unit is done, the lead explains it to the other partner. This doubles as the lecturer's required documentation of personal contribution.

## Non-goals

- No polished software product, packaging, CI or web demo.
- No complete code written by Claude for the concept parts, even when asked to "just finish it" mid-unit. Claude gives hints first, then partial code, and full code only when the student explicitly asks for it to be revealed. If that happens, it is recorded in the unit log.
- No new research directions here. Changes to research scope go back to the proposal.

## Anti-goals (failure even if it works)

- The pipeline runs, but Adir or Noa can't explain a block of it.
- TODOs so small they are fill-in-the-blank typing with no thinking (e.g. "type `np.exp`"). Each blank should require using the concept.
- TODOs so big that the student gets stuck for hours with nothing to go on. Hints are tiered (nudge → formula → partial code).
- Explainers that are just translated textbook text, or that are longer than the video.

## Constraints

- **Hardware:** Adir's laptop has AMD integrated graphics and 15 GB RAM, so there is no local GPU. Training and DiffDRR rendering at scale run on **Google Colab** (debugging) or **Kaggle** (long runs and the 23.6 GB unpack). Locally they run only light things: the NumPy DRR, metric tests and mesh visualisation.
- **Local Python:** Anaconda (`~/anaconda3`) with numpy/jupyter is available. `uv` is also installed.
- **Data never goes into git** (sizes: 3.2 GB / 23.6 GB / 4.84 GB; the DeepFluoro license is CC BY-NC-ND). The repo holds only code, notebooks, explainers and small derived artifacts (split file, metric CSVs, figures).
- **Languages:** explainers in Hebrew; code, identifiers, comments and notebook markdown in English. Technical terms stay in English inside the Hebrew text.
- **Same notebooks in three places:** they must run on Colab, Kaggle and locally. Shared code lives in an importable package, and each notebook starts with a setup cell that clones/installs the repo when it runs in the cloud.
- **Patient-level split:** the split is fixed in unit 1 and saved to a file that is never regenerated (see the proposal).

## Decided structure

```
pelvis-xray-3d/
  README.md                 # what the project is, how to run, unit index
  pyproject.toml            # installable package (pip install -e .)
  src/xray3d/               # shared code; student-written functions live here once a unit is done
  units/
    00-setup/               # environment check: local + Colab + Kaggle, the workflow itself
    01-data-and-drr/        # proposal week 1
    ...                     # one folder per unit, created when that unit starts
      README.md             # resources (links), goals, lead, status
      explainer.he.md       # Hebrew concept note
      notebook.ipynb        # guided notebook with TODO cells
      test_unit.py          # pytest checks for the TODOs
      LOG.md                # what was done, who did what, hints used, explain-back notes
  data/                     # gitignored; local data root
  docs/superpowers/specs/   # this document and future designs
```

How a TODO works: the notebook cell defines a function stub, with a docstring, shape annotations and a `raise NotImplementedError`, followed by a test cell that calls the unit's checks. Once the test passes, the student moves the function into `src/xray3d/` so later units import it. **Later units build on the students' own implementations.** A reference solution is not stored in the repo. If a later unit is blocked because an earlier function is broken, Claude helps debug that function rather than swapping in its own.

### Hints and standing rules for Claude (Claude's call)

- Each TODO has **one nudge** (tier 1) written in the notebook. Tiers 2 (formula/approach) and 3 (partial code) come from Claude **on request**. This keeps the deeper tiers one conversation away rather than one click away, and lets Claude log which hints were used.
- The repo root has a `CLAUDE.md` with the standing rules for every future session. It covers: hints before code; full code only when the student explicitly asks for a reveal, and every reveal logged in the unit's `LOG.md`; probing during explain-back; reviewing like a TA; and the student/Claude authorship split above. This is how the "don't just finish it" rule holds across sessions.

### How a week runs (decided by Adir)

Adir and Noa have **one weekly session together**. They watch the unit's video together and talk it through, then implement that unit's part of the project together with Claude, in Adir's Claude session on his machine. Noa doesn't need her own local environment or her own Claude setup.

Consequences (Claude's calls):
- **One notebook per unit**, worked on together; no per-person branches. Commits happen at the end of the session.
- The proposal's "lead" column becomes **who drives**: for each TODO, one partner types and the other explains, and they swap. `LOG.md` records the driver per TODO, the hints used and the explain-back notes. That is the documentation of personal contribution.
- The Hebrew explainer is a **short recap read after the video** (not homework before it). It covers what the video skipped and links the video to the project and to the syllabus week.
- Each session should take about 2–3 hours: video (≤30 min) + discussion + implementation. A unit too big for one session is split across weeks.

### Lecture material available

The local course folder (`C:\Users\Adir\Desktop\BSC\שנה ג\סמסטר א\מבוא לראייה ממוחשבת\`) has only the week-1 slides, a week-2 CNN doc and the syllabus (`Course info/Syllabus.md`, which lists topics for all 13 weeks). Weeks 9–12 haven't been taught yet. Until they are, explainers link to the **syllabus topic for that week** and say so; they don't claim to quote lecture content.

### Repo hosting (decided by Adir)

**Public GitHub repo** under `AdirBuskila`, with Noa (`noazzzzzz`) added as a collaborator. Because it's public, Colab and Kaggle can `git clone` it with no tokens. No data, credentials or Colab/Kaggle secrets ever go into it. The README credits both authors and the dataset licenses.

### Data on disk (downloaded 2026-11-11)

- `data/raw/totalsegmentator_small/Totalsegmentator_dataset_small_v201.zip`: 3.24 GB, Zenodo record 10047263, CC BY 4.0.
- `data/raw/deepfluoro/`: 4.05 GB, the `deepfluoro/` folder of the HF dataset `eigenvivek/xvr-data`, CC BY-NC-ND 4.0. The `.pt` files are PyTorch pickles: always load them with `torch.load(..., weights_only=True)`.
- The full TotalSegmentator (23.58 GB, Zenodo 10047292) is **not** downloaded locally. A Kaggle notebook downloads it, crops the pelvis to 64³ and saves only the result.

### Video path (decided by Adir: focus on computer vision and neural networks, not medicine; links checked 2026-11-11)

The main series is [Andreas Geiger's Computer Vision (Tübingen)](https://www.youtube.com/playlist?list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_). Basic CNNs are already covered by their Deep Learning course, so the path concentrates on what's new in this project.

| Unit | Topic | Video | Length |
|---|---|---|---|
| 01 | pinhole, perspective vs orthographic projection | [Geiger 2.2 Geometric Image Formation](https://www.youtube.com/watch?v=NIaICLR7D0Q) (optional, easier: [Nayar, Pinhole and Perspective Projection](https://www.youtube.com/watch?v=_EhY31MSbNM), 20 min) | 35 min |
| 02 | voxels → surfaces | [Sebastian Lague, Marching Cubes](https://www.youtube.com/watch?v=M3iI2l0ltbE) + [Geiger 8.4 Volumetric Fusion](https://www.youtube.com/watch?v=B2BTSKcYqtQ) | 6 + 38 min |
| 03 | why a second view gives depth; is reconstruction just retrieval? | [Geiger 4.1 Stereo Preliminaries](https://www.youtube.com/watch?v=6hr6xpOU-uw) + paper [Tatarchenko et al. 2019](https://arxiv.org/abs/1905.03678), §1–4 | 44 min + ~30 min reading |
| 04 | segmentation, U-Net, transposed conv | [Geiger 10.2 Semantic Segmentation](https://www.youtube.com/watch?v=NhSujgDL918) + [U-Net in 10 min](https://www.youtube.com/watch?v=NhdzGfB1q74) + [Transpose Conv vs Unpooling](https://www.youtube.com/watch?v=JnN_1Nh7w4M) | 16 + 10 + 6 min |
| 05 | networks that output 3D: voxels, points, meshes, occupancy | [Geiger 9.1 Implicit Neural Representations](https://www.youtube.com/watch?v=r6n3tZJoTdI) | 45 min |
| 07 | rendering as a differentiable operation (idea behind DiffDRR) | [Geiger 9.2 Differentiable Volumetric Rendering](https://www.youtube.com/watch?v=3e64ILZBJUs) | 28 min |

Medical background is optional only: [CT physics overview](https://www.youtube.com/watch?v=mvMO7CsU2To), 19 min. The unit 01 explainer covers the HU and Beer–Lambert parts the project needs. The Michigan EECS 498 lectures (Justin Johnson) were checked and are now **private** on YouTube, so they aren't used.

### Timeline (Claude's call; the deadline isn't known yet)

Unit 00 is kept to about 30 minutes and runs at the **start of the first session**, before unit 01. It doesn't take a week of its own.

## Scope of "the foundation" (this first build)

1. The repo skeleton above, `.gitignore`, `pyproject.toml` and a README.
2. The unit template, with tiered hints for TODOs.
3. **Unit 00 – setup:** check the local environment, open the same notebook on Colab and Kaggle, and practise the workflow once on a toy TODO (e.g. a 2D parallel projection of a synthetic square). Done together.
4. **Unit 01 – data and DRR** (proposal week 1), done together in one session or split across two if it's too long. Following the proposal's leads, Noa drives the data part and Adir drives the DRR part:
   - **Data:** download the 3.2 GB subset (Claude's script, run before the session), open NIfTI, check spacing, find the pelvis/femur label IDs, count the pelvis cases.
   - **DRR:** a parallel-beam NumPy DRR (HU→μ, sum along an axis, Beer–Lambert exp), then a first AP+LAT pair with DiffDRR on Colab.
   - **Split:** 70/10/20 by subject, saved to `splits/split_v1.json`.

Later units (metrics, baselines, U-Net, experiments, DeepFluoro) are designed when they start, using the same template.

## Left to the builder

- Exact dependency pins; the Colab/Kaggle setup-cell mechanics.
- The test style per TODO (tolerance-based numeric checks are fine).
- Which resources to recommend, provided they're checked to exist and are short.
- The toy exercise in unit 00.
