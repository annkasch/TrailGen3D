"""Sky2Trail v0 on synthetic forests with known stems.

The synthetic forest is simple (round crowns, smooth terrain), so thresholds are regression
guards, not claims about accuracy on real forests.
"""

import json

import numpy as np

from sky2trail.baseline import run_v0
from sky2trail.config import load_config
from sky2trail.evaluate import match_trunks
from sky2trail.grid import Grid, bin_stat
from sky2trail.output import (COMPONENT_STEM, SOURCE_GENERATED, SOURCE_MEASURED, load_completed_cloud,
                              load_stems)
from sky2trail.synthetic import make_scene


def _run(density, seed=0, size=80, n_trees=25):
    scene = make_scene(size=size, n_trees=n_trees, pulses_per_m2=density, seed=seed)
    return scene, run_v0(scene.points, load_config())


def _stem_score(scene, out, max_dist=1.0):
    pred = np.array([[s.x, s.y] for s in out.stems]).reshape(-1, 2)
    return match_trunks(pred, np.zeros(len(pred)), scene.trees[:, :2], np.zeros(len(scene.trees)), max_dist)


def test_grid_binning():
    g = Grid.from_bounds(0, 0, 10, 10, 1.0)
    assert g.shape == (10, 10)
    x = np.array([0.5, 0.5, 9.5])
    y = np.array([9.5, 9.5, 0.5])
    assert bin_stat(g, x, y, None, "count")[0, 0] == 2
    assert bin_stat(g, x, y, np.array([1.0, 3.0, 5.0]), "max")[0, 0] == 3.0


def test_ground_surface_matches_truth():
    scene, out = _run(2.0)
    g = out.ground
    xs, ys = g.grid.cell_centers()
    inner = (slice(3, -3), slice(3, -3))
    err = g.elevation[inner] - scene.ground_fn(xs[inner], ys[inner])
    assert abs(np.median(err)) < 0.15
    assert np.percentile(np.abs(err), 95) < 0.6
    assert 0 <= g.confidence.min() and g.confidence.max() <= 1


def test_stems_found_at_ql2_and_ql1():
    for density, min_recall, min_precision in [(2.0, 0.8, 0.8), (8.0, 0.85, 0.9)]:
        scene, out = _run(density)
        m = _stem_score(scene, out)
        assert m["recall"] >= min_recall, (density, m)
        assert m["precision"] >= min_precision, (density, m)


def test_completed_cloud_marks_measured_and_generated():
    scene, out = _run(2.0)
    p = out.points
    n_measured = int((p.source == SOURCE_MEASURED).sum())
    assert n_measured == len(scene.points)
    gen = p.source == SOURCE_GENERATED
    assert gen.any()
    assert (p.component[gen] == COMPONENT_STEM).all()
    # Generated stems start at the ground surface and stay below the tree tops.
    hag = p.z[gen] - out.ground.elevation_at(p.x[gen], p.y[gen])
    assert hag.min() > -0.3
    assert hag.max() < max(s.height for s in out.stems)


def test_output_roundtrip(tmp_path):
    scene, out = _run(2.0, size=50, n_trees=8, seed=1)
    d = out.save(tmp_path / "s2t")
    summary = json.loads((d / "summary.json").read_text())
    assert summary["method"] == "v0-rules"
    stems = load_stems(d / "stems.csv")
    assert len(stems) == len(out.stems)
    cloud = load_completed_cloud(d / "points")
    assert len(cloud) == len(out.points)
