"""Written by Adir and Noa: unit 04 (after videos 09 and 10): convolutions and the X-ray -> 3D network.

Empty until you move your functions and classes here from the unit's notebook (once their checks pass):
    conv_out_size, conv2d_naive, ConvBlock, Encoder2D, lift_view, Decoder3D, XrayTo3D

Copy the whole function or class, docstring included, plus any imports it needs.
Then `pytest units` runs the same checks against this file. Units 05-08 build `XrayTo3D` from here.
"""
import numpy as np  # noqa: F401
import torch  # noqa: F401
import torch.nn as nn  # noqa: F401
