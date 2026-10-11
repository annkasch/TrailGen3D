"""End-to-end checks on synthetic forests with known trees.

The synthetic forest is simple (round crowns, smooth terrain), so these thresholds are
regression guards, not claims about accuracy on real forests.
"""

import json

import numpy as np

from trailgen.config import load_config
from trailgen.evaluate import match_trunks
from trailgen.grid import Grid, bin_stat
from trailgen.io import write_nav2_map
from trailgen.pipeline import process, save
from trailgen.synthetic import make_scene


def _score(density, seed=0, max_dist=1.0):
    scene = make_scene(size=80, n_trees=25, pulses_per_m2=density, seed=seed)
    result = process(scene.points, load_config())
    pred = np.array([[t.x, t.y] for t in result.trees]).reshape(-1, 2)
    m = match_trunks(pred, np.zeros(len(pred)), scene.trees[:, :2], np.zeros(len(scene.trees)), max_dist)
    return scene, result, m


def test_grid_binning():
    g = Grid.from_bounds(0, 0, 10, 10, 1.0)
    assert g.shape == (10, 10)
    x = np.array([0.5, 0.5, 9.5])
    y = np.array([9.5, 9.5, 0.5])
    counts = bin_stat(g, x, y, None, "count")
    assert counts[0, 0] == 2 and counts[9, 9] == 1
    assert bin_stat(g, x, y, np.array([1.0, 3.0, 5.0]), "max")[0, 0] == 3.0


def test_ground_model_matches_truth():
    scene, result, _ = _score(density=2.0)
    t = result.terrain
    xs, ys = t.grid.cell_centers()
    inner = (slice(3, -3), slice(3, -3))
    err = t.dem[inner] - scene.ground_fn(xs[inner], ys[inner])
    assert np.abs(np.median(err)) < 0.15
    assert np.percentile(np.abs(err), 95) < 0.6


def test_trees_found_at_ql2():
    _, _, m = _score(density=2.0)
    assert m["recall"] >= 0.8
    assert m["precision"] >= 0.8


def test_trees_found_at_ql1():
    _, _, m = _score(density=8.0)
    assert m["recall"] >= 0.85
    assert m["precision"] >= 0.9


def test_trunks_block_and_open_ground_is_free():
    _, result, _ = _score(density=2.0)
    cm = result.costmap
    assert cm.blocked.any()
    assert np.median(cm.cost) < 0.5


def test_outputs_written(tmp_path):
    scene = make_scene(size=50, n_trees=8, pulses_per_m2=2.0, seed=1)
    result = process(scene.points, load_config())
    out = save(result, tmp_path / "out", preview=False)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trees"]["count"] > 0
    assert (out / "trees.csv").exists()
    assert (out / "nav2" / "map.pgm").exists()


def test_nav2_map_values(tmp_path):
    g = Grid.from_bounds(0, 0, 2, 1, 1.0)
    cost = np.array([[0.0, 1.0]])
    write_nav2_map(tmp_path, cost, g)
    data = (tmp_path / "map.pgm").read_bytes()
    pixels = data[-2:]
    assert pixels[0] == 254 and pixels[1] == 0  # free, blocked
