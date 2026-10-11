"""Stage 1 PyTorch parts. Skipped when PyTorch is not installed.

Run these first on a new machine: `pytest tests/sky2trail/test_stage1_torch.py -v`
(about a minute on CPU, seconds on a GPU).
"""

import json

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sky2trail.config import load_config

CFG = load_config()


def _tiny_data(tmp_path):
    from sky2trail.stage1.dataset import make_synthetic_dataset
    return make_synthetic_dataset(tmp_path / "data", CFG, n_train=2, n_val=1, size=40.0,
                                  log=lambda *a: None)


def test_model_shapes():
    if torch is None:
        return
    from sky2trail.stage1.features import channel_names
    from sky2trail.stage1.model import StemUNet
    c = len(channel_names(CFG.stage1))
    net = StemUNet(c, base=8)
    out = net(torch.zeros(2, c, 88, 88))
    for k in ("heat", "offset", "size"):
        assert out[k].shape == (2, 2, 88, 88), k
    # Starts with a low stem probability everywhere.
    assert float(torch.sigmoid(out["heat"]).mean()) < 0.2


def test_loss_decreases_on_one_batch(tmp_path):
    """Overfitting one batch is the basic check that targets, loss and gradients fit together."""
    if torch is None:
        return
    from sky2trail.stage1.dataset import StemSamples
    from sky2trail.stage1.model import StemUNet, stage1_loss
    from sky2trail.stage1.train import KEYS

    paths = _tiny_data(tmp_path)
    ds = StemSamples(paths["train"])
    items = [ds.get(i) for i in range(8)]
    batch = {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in KEYS}
    torch.manual_seed(0)
    net = StemUNet(ds.n_channels, base=8)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = stage1_loss(net(batch["image"]), batch)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.5 * first, (first, float(loss))


def test_train_checkpoint_and_predict(tmp_path):
    if torch is None:
        return
    from sky2trail.stage1.predict import Stage1Model, run_stage1
    from sky2trail.stage1.train import train
    from sky2trail.synthetic import make_hard_scene

    _tiny_data(tmp_path)
    res = train(str(tmp_path / "data"), str(tmp_path / "run"), epochs=2, batch_size=8, base=8,
                workers=0, device="cpu")
    run = tmp_path / "run"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    lines = (run / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and "f1_1m" in json.loads(lines[-1])
    # Resume continues after the last epoch without error.
    train(str(tmp_path / "data"), str(run), epochs=3, batch_size=8, base=8, workers=0,
          device="cpu", resume=True)
    model = Stage1Model(run / "best.pt", device="cpu")
    scene = make_hard_scene(size=40.0, pulses_per_m2=4.0, seed=99)
    out = run_stage1(scene.points, CFG, model)
    assert out.method == "stage1-unet"
    assert res["best_f1_1m"] >= 0.0
