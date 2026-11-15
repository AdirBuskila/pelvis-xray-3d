"""Written by Adir and Noa: unit 03 (after videos 07 and 08): two baselines without a neural network.

Empty until you move your functions here from the unit's notebook (once their checks pass):
    part A (video 07, the visual hull):          pixel_index, backproject, visual_hull
    part B (video 08, the memorisation floor):   nn_features, nearest_neighbour, oracle_neighbour, mean_shape

Copy the whole function, docstring included, plus any imports it needs (e.g. `import numpy as np`,
`from xray3d import dataset, geometry, metrics`). Import the modules, not the names: then a unit-01/02 function
that hasn't been moved yet only fails the test that calls it, instead of every test in unit 03.
Then `pytest units` runs the same checks against this file.
Units 06 and 08 compare the network against these baselines.
"""
import numpy as np  # noqa: F401  (most of your functions will need it)
