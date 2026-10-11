"""Patch tiling: every stem in exactly one core, cores fit inside field plots."""

import numpy as np

from sky2trail.config import load_config
from sky2trail.patches import assign_to_cores, coverage, cut_points, plot_patches, tile_patches
from sky2trail.synthetic import make_scene


def test_default_patch_sizes():
    p = load_config().patches
    assert p.core_size == 10.0 and p.margin == 6.0


def test_every_stem_in_exactly_one_core():
    scene = make_scene(size=60, n_trees=30, pulses_per_m2=2.0, seed=3)
    patches = tile_patches((0, 0, 60, 60))
    assert len(patches) == 36
    owner = assign_to_cores(scene.trees[:, 0], scene.trees[:, 1], patches)
    assert (owner >= 0).all()
    for k, p in enumerate(patches):
        inside = p.in_core(scene.trees[:, 0], scene.trees[:, 1])
        assert np.array_equal(inside, owner == k)


def test_shared_edge_goes_to_one_core():
    patches = tile_patches((0, 0, 20, 10))
    owner = assign_to_cores([10.0], [5.0], patches)
    assert owner[0] == 1  # half-open cores: x = 10 belongs to the right one


def test_input_window_and_coverage():
    scene = make_scene(size=60, n_trees=20, pulses_per_m2=4.0, seed=1)
    patches = tile_patches((0, 0, 60, 60))
    centre = next(p for p in patches if p.core_bounds[:2] == (20.0, 20.0))
    assert centre.input_size == 22.0
    sub = cut_points(scene.points, centre)
    assert sub.x.min() >= 14 and sub.x.max() < 36
    assert coverage(scene.points, centre) > 0.95
    corner = patches[0]  # half its input window lies outside the scene
    assert coverage(scene.points, corner) < 0.6


def test_cores_inside_field_plots():
    assert len(plot_patches(0, 0, "circle", 20)) == 1    # USGS Sierra plot
    assert len(plot_patches(0, 0, "square", 20)) == 4    # NEON 20 m plot
    assert len(plot_patches(0, 0, "square", 40)) == 16   # NEON 40 m plot
    for p in plot_patches(100, 50, "circle", 20):
        d = p.corners() - [100, 50]
        assert np.hypot(d[:, 0], d[:, 1]).max() <= 10
