"""xray3d: shared code for the pelvis X-ray -> 3D project.

Who wrote what (see CLAUDE.md for the rule):

  Infrastructure, written by Claude, heavily commented:
    env.py         where are we running (local / Colab / Kaggle), where is the data
    data.py        finding subjects, loading CT volumes and masks, fetching subjects in the cloud
    remote_zip.py  reading single files out of a remote zip without downloading all of it
    plotting.py    small display helpers
    checks.py      the pass/fail printing used by every unit's test_unit.py
    dataset.py     reading the processed dataset (unit 02) and each view's camera
    prep.py        building the processed dataset: the 64^3 cube + DiffDRR pairs
    torchdata.py   the PyTorch Dataset, input standardisation, augmentation (unit 05)
    trainer.py     the training loop around your train_step, checkpoints, logs (unit 05)
    deepfluoro.py  loading real fluoroscopy (unit 07; CC BY-NC-ND, nothing derived is committed)

  Concept code, written by Adir and Noa. A function moves here from a unit's notebook
  once its check passes, so later units import *your* implementation:
    volume.py      unit 01 part A: affine, voxel spacing, planes, masks
    physics.py     unit 01 part B: HU -> attenuation, windowing
    geometry.py    unit 01 part C: pinhole and parallel projection
    drr.py         unit 01 part D: the parallel-beam DRR
    split.py       unit 01 part E: the patient-level split
    toy.py         unit 00: the warm-up (shadows of a square)
    shape.py       unit 02: the canonical cube, resampling, meshes
    metrics.py     unit 02: Dice, IoU, surface distances, HD95, ASSD, NSD, Chamfer, a clinical angle
    baselines.py   unit 03: visual hull, nearest neighbour
    nets.py        unit 04: the network
    losses.py      unit 05: soft Dice, BCE + Dice
    training.py    unit 05: the training step
    experiment.py  unit 06: summary statistics, paired comparison
    realxray.py    unit 07: real X-ray preprocessing, the gap, uncertainty
    pipeline.py    unit 08: two X-rays -> mesh
"""

import os as _os

# Anaconda's NumPy (Intel MKL) and the pip build of PyTorch each ship an OpenMP runtime. On Windows, loading
# both aborts with "OMP: Error #15". Allowing the duplicate is the usual workaround for this exact pair; it has
# to be set before torch is imported, which is why it lives here (every notebook imports xray3d first).
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
