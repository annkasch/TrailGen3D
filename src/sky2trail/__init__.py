"""Sky2Trail: from aerial lidar to what a scanner on the ground would see.

Input: an aerial point cloud (USGS 3DEP). Output (see `output.Sky2TrailOutput`):
  - a completed point cloud, each point marked measured or generated,
  - a stem list (position, diameter, height, confidence),
  - a ground surface with a confidence layer.

`baseline.run_v0` fills these with fixed rules; trained models produce the same output.
"""

__version__ = "0.1.0"
