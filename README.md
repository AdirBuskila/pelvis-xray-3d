# pelvis-xray-3d

Reconstructing the 3D shape of the pelvis from two X-rays (AP + lateral) with a 2D-encoder / 3D-decoder network, and measuring how well it holds up on real X-rays rather than simulated ones.

Final project for *Intro to Computer Vision in Medical Imaging* (HIT, 2026–27), by Adir Buskila and Noa.

The project is built as a sequence of **learning units**. Each unit: watch a video together, implement that part of the pipeline in a guided notebook, explain it back. See [`docs/superpowers/specs/2026-11-11-project-foundation-design.md`](docs/superpowers/specs/2026-11-11-project-foundation-design.md) for how the work is organised.

> Status: foundation in progress. No code yet.

## Data (not in this repo)

| Dataset | Use | License |
|---|---|---|
| [TotalSegmentator v2.0.1](https://zenodo.org/records/10047292) (and its [small subset](https://zenodo.org/records/10047263)) | training CT + bone masks | CC BY 4.0 |
| [DeepFluoro](https://huggingface.co/datasets/eigenvivek/xvr-data) (via the `xvr-data` mirror) | evaluation on real fluoroscopy | CC BY-NC-ND 4.0 |

Datasets are downloaded into `data/` (gitignored) and are never committed.
