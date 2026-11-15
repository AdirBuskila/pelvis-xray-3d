# Unit 01 · Data and the DRR

**Leads:** Noa on the data parts (A, E), Adir on the DRR parts (B, C, D), as in the proposal ·
**Videos:** 01, 02, 03, 04 · **Proposal week:** 1 · **Status:** ready; start with part A (after video 01)

Big for one session, so it's split by video. Do each part right after its video; the notebook picks up where you left off.

## Goals
- Open a CT and its masks, and say what the shape, the affine and the axis codes mean.
- Count how many pelvises in the subset are usable, and why the rest aren't.
- Go from HU to attenuation, and from attenuation to a simulated X-ray pair (AP + LAT).
- Explain pinhole vs parallel projection, with numbers.
- Make the patient-level split, once.

## Data
The small TotalSegmentator subset (102 CTs) is already unzipped on the laptop in
`data/raw/totalsegmentator_small/extracted/`. On Colab/Kaggle the notebook downloads 3 subjects (~150 MB) by itself.

## TODOs
| Part · video | TODO | Function | Moves into | Concept |
|---|---|---|---|---|
| A · 01 | A1 | `voxel_spacing` | `volume.py` | what the affine's columns mean |
| | A2 | `voxel_to_mm` | `volume.py` | homogeneous coordinates |
| | A3 | `orthogonal_planes` | `volume.py` | which array axis is which anatomical direction |
| | A4 | `mask_volume_cm3` | `volume.py` | voxels have a physical size |
| | A5 | `touches_scan_edge` | `volume.py` | a cut pelvis can't be ground truth |
| B · 02 | B1 | `hu_to_mu` | `physics.py` | inverting the Hounsfield scale |
| | B2 | `window` | `physics.py` | windowing for display |
| C · 03 | C1 | `project_pinhole` | `geometry.py` | `K[R|t]` and the divide by depth |
| | C2 | `project_parallel` | `geometry.py` | parallel rays: no magnification |
| D · 04 | D1 | `parallel_drr` | `drr.py` | the line integral, in mm |
| | D2 | `ap_lat_views` | `drr.py` | beam axis + radiology orientation for any layout |
| | D3 | (DiffDRR, Claude's cell) | | perspective DRR; predict the LAT rotation |
| E | E1 | `split_subjects` | `split.py` | reproducible, patient-level split |

## Done when
- every check is green, all functions are in `src/xray3d/`, and `pytest units` passes with nothing skipped,
- `splits/split_v1.json` exists (written by **your** `split_subjects`) and is committed,
- `LOG.md` has the usable-pelvis count and the split sizes.
