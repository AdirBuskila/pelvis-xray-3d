"""Written by Adir and Noa: unit 02 part B (after video 06): how good is a reconstruction?

Empty until you move your functions here from the unit's notebook (once their checks pass):
    dice, iou, surface_voxels, surface_distances, hd95, assd, nsd, chamfer, ce_angle, evaluate

Copy the whole function, docstring included, plus any imports it needs (e.g. `import numpy as np`).
`ce_angle` and `evaluate` use your part-A functions (`cube_grid`, `mesh_from_occupancy`): add
`from xray3d import shape` and call `shape.cube_grid(...)` (the module, not the name, so a missing function
fails only the test that calls it).
Then `pytest units` runs the same checks against this file.
Units 03-08 score every method with these functions.
"""
import numpy as np  # noqa: F401  (most of your functions will need it)
