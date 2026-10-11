"""Stage 1 numpy parts: input image, targets, augmentation, decoding, samples."""

import numpy as np

from sky2trail.config import load_config
from sky2trail.evaluate import match_trunks
from sky2trail.stage1.augment import transform
from sky2trail.stage1.dataset import StemSamples, save_samples, scene_samples
from sky2trail.stage1.decode import decode
from sky2trail.stage1.features import channel_names, rasterize, window_grid
from sky2trail.stage1.targets import make_targets, pixel_to_world, world_to_pixel
from sky2trail.synthetic import make_hard_scene

CFG = load_config()
S1 = CFG.stage1


def test_rasterize_channels():
    grid = window_grid((0, 0, 22, 22), S1.resolution)
    assert grid.shape == (88, 88)
    # One pulse at (5.1, 20.1): a return at 12 m and a ground return.
    x = np.array([5.1, 5.1]); y = np.array([20.1, 20.1])
    img = rasterize(x, y, np.array([12.0, 0.0]), np.array([False, True]), np.array([1, 2]), grid, S1)
    names = channel_names(S1)
    assert img.shape == (len(names), 88, 88)
    r, c = 7, 20  # (22 - 20.1) / 0.25 -> row 7; 5.1 / 0.25 -> col 20
    assert img[names.index("returns 10-20 m"), r, c] == 1.0
    assert np.isclose(img[names.index("max height"), r, c], 12.0 / S1.height_scale)
    assert img[names.index("ground hit"), r, c] == 1.0
    assert img[names.index("share not first return"), r, c] == 0.5
    assert img[names.index("empty"), r, c] == 0.0 and img[names.index("empty")].sum() == 88 * 88 - 1


def _stems():
    # v, u, dbh, height  (well separated)
    return np.array([[10.3, 12.7, 0.35, 21.0], [40.9, 60.2, 0.06, 5.0], [70.5, 20.5, 0.22, np.nan]])


def test_targets_decode_roundtrip():
    grid = window_grid((100, 200, 122, 222), S1.resolution)
    stems = _stems()
    t = make_targets(stems, grid.shape, np.ones(grid.shape, bool), True, S1)
    assert t["heat"][0].max() == 1.0 and t["heat"][1].max() == 1.0   # one small stem
    d = decode(t["heat"], t["offset"], t["size"], grid, S1)
    order = np.argsort(d["y"])[::-1]   # top row first, like the stems above
    x, y = pixel_to_world(stems[:, 0], stems[:, 1], grid)
    assert np.allclose(d["x"][order], x, atol=1e-5) and np.allclose(d["y"][order], y, atol=1e-5)
    assert np.allclose(d["dbh"][order][:2], stems[:2, 2], rtol=1e-5)
    assert np.isclose(d["height"][order][0], 21.0, rtol=1e-5)
    assert list(d["small"][order]) == [False, True, False]


def test_ignore_regions():
    shape = (88, 88)
    mask = np.zeros(shape, bool); mask[:, :44] = True
    t = make_targets(_stems(), shape, mask, small_labelled=False, cfg=S1)
    assert t["weight"][0, :, 44:].sum() == 0 and t["weight"][0, :, :44].all()
    assert t["weight"][1].sum() == 0          # small stems not mapped at this source
    assert t["reg_mask"].sum() == 2            # the stem at u = 60.2 lies outside the label area


def test_augmentation_moves_stems_with_the_image():
    stems = _stems()
    shape = (88, 88)
    full = np.ones(shape, bool)
    heat = make_targets(stems, shape, full, True, S1)["heat"]
    for k in range(4):
        for flip in (False, True):
            img_t, stems_t, _ = transform(heat, stems, [full], k, flip)
            heat_t = make_targets(stems_t, shape, full, True, S1)["heat"]
            assert np.allclose(img_t, heat_t), (k, flip)


def test_samples_cover_every_stem_once(tmp_path):
    scene = make_hard_scene(size=40, pulses_per_m2=4, seed=3)
    samples = scene_samples(scene, CFG, "test")
    assert len(samples) == 16
    n_core = 0
    for s in samples:
        r0, c0, r1, c1 = s["core_px"]
        st = s["stems"]
        inside = (st[:, 0] >= r0) & (st[:, 0] < r1) & (st[:, 1] >= c0) & (st[:, 1] < c1)
        n_core += int(inside.sum())
    assert n_core == len(scene.trees)
    path = save_samples(samples, tmp_path / "s.npz", S1)
    ds = StemSamples(path)
    assert len(ds) == 16 and ds.n_channels == len(channel_names(S1))
    item = ds.get(5, np.random.default_rng(0))
    assert item["image"].shape == (ds.n_channels, 88, 88) and item["heat"].shape == (2, 88, 88)


def test_perfect_maps_recover_all_stems():
    """Decoding the targets themselves gives back every stem: the upper bound for stage 1."""
    scene = make_hard_scene(size=40, pulses_per_m2=2, seed=4)
    samples = scene_samples(scene, CFG, "test")
    xs, ys = [], []
    for s in samples:
        grid = window_grid(s["bounds"], S1.resolution)
        t = make_targets(s["stems"], grid.shape, s["mask"], True, S1)
        r0, c0, r1, c1 = s["core_px"]
        x0, y1 = pixel_to_world(r0, c0, grid)
        x1, y0 = pixel_to_world(r1, c1, grid)
        d = decode(t["heat"], t["offset"], t["size"], grid, S1, core_bounds=(x0, y0, x1, y1))
        xs += list(d["x"]); ys += list(d["y"])
    m = match_trunks(np.column_stack([xs, ys]), np.zeros(len(xs)), scene.trees[:, :2],
                     np.zeros(len(scene.trees)), 0.1)
    assert m["recall"] >= 0.97 and m["precision"] == 1.0, m


def test_pixel_world_roundtrip():
    grid = window_grid((10, 20, 32, 42), S1.resolution)
    vu = world_to_pixel([11.3, 30.0], [41.9, 25.5], grid)
    x, y = pixel_to_world(vu[:, 0], vu[:, 1], grid)
    assert np.allclose(x, [11.3, 30.0]) and np.allclose(y, [41.9, 25.5])
