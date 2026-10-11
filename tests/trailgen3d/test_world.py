"""TrailGen3D world building on top of a Sky2Trail output."""

import json

import numpy as np

from sky2trail.baseline import run_v0
from sky2trail.config import load_config as load_s2t
from sky2trail.grid import Grid
from sky2trail.synthetic import make_scene
from trailgen3d.config import load_config
from trailgen3d.export import write_nav2_map
from trailgen3d.pipeline import build_world, save_world


def _world(size=80, n_trees=25, seed=0):
    scene = make_scene(size=size, n_trees=n_trees, pulses_per_m2=2.0, seed=seed)
    s2t = run_v0(scene.points, load_s2t())
    return s2t, build_world(s2t, load_config())


def test_stems_block_and_open_ground_is_free():
    s2t, world = _world()
    cm = world.costmap
    assert cm.layers["stems"].any()
    assert np.median(cm.cost) < 0.5
    # Every stem centre is blocked.
    for s in s2t.stems:
        r, c, ok = cm.grid.index(np.array([s.x]), np.array([s.y]))
        if ok[0]:
            assert cm.blocked[r[0], c[0]]


def test_world_outputs(tmp_path):
    _, world = _world(size=50, n_trees=8, seed=1)
    out = save_world(world, tmp_path / "world", preview=False)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["sky2trail_method"] == "v0-rules"
    assert (out / "nav2" / "map.pgm").exists()


def test_nav2_map_values(tmp_path):
    g = Grid.from_bounds(0, 0, 2, 1, 1.0)
    write_nav2_map(tmp_path, np.array([[0.0, 1.0]]), g)
    pixels = (tmp_path / "map.pgm").read_bytes()[-2:]
    assert pixels[0] == 254 and pixels[1] == 0  # free, blocked
