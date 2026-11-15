"""Written by Adir and Noa: unit 07 (after video 13): real X-rays, the gap, re-projection, uncertainty, errors.

Empty until you move your functions here from the unit's notebook (once their checks pass):
    part A (the gap):              log_transform, gap
    part B (checking a prediction): reprojection_iou, mc_dropout_predict, uncertainty_vs_error
    part C (error analysis):       ce_error, error_summary

Copy the whole function, docstring included, plus any imports it needs (e.g. `import numpy as np`,
`import torch`, `from xray3d.baselines import pixel_index`, `from xray3d.metrics import ce_angle`).
Then `pytest units` runs the same checks against this file. deepfluoro.load_specimen calls the log_transform here
when it isn't given one (unit 07's notebook passes its own).
"""
import numpy as np  # noqa: F401  (most of your functions will need it)
