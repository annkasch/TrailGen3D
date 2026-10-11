"""Stage 1: a learned stem map from aerial lidar.

A small U-Net reads a 22 m input window as a multi-channel top-down image (vertical profile
of returns, canopy height, penetration, ground hits, pulse density) and predicts, for every
0.25 m pixel, how likely a stem centre is there, the exact position inside the pixel, the
stem diameter and the tree height. Peaks in that map become the stem list of a Sky2Trail
output; ground and canopy stay as measured (same as v0).

numpy parts (no PyTorch needed): features, targets, augment, decode, dataset.
PyTorch parts: model, train, predict.
"""
