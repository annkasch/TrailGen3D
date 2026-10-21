"""ForestGen3D training loop on CPU with a tiny model: train, resume, sample, evaluate."""

import json

import pytest

torch = pytest.importorskip("torch")

from sky2trail.fg3d import synthetic  # noqa: E402
from sky2trail.fg3d.pairs import list_pairs  # noqa: E402
from sky2trail.fg3d.sample import Fg3dModel, evaluate, sample_pairs  # noqa: E402
from sky2trail.fg3d.train import TrainCfg, train  # noqa: E402


def test_train_resume_sample_eval(tmp_path):
    data = tmp_path / "pairs"
    synthetic.write_pairs(data, 10, log=None)
    cfg = TrainCfg(epochs=2, batch_size=4, n_als=64, n_tls=128, workers=0, tiny=True, steps=50,
                   sample_every=2, sample_trees=2, ddim_steps=5)
    run = tmp_path / "run"
    res = train(str(data), str(run), cfg, device="cpu", log=lambda *_: None)
    assert (run / "last.pt").exists() and (run / "best.pt").exists()
    lines = (run / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "val_cd" in json.loads(lines[-1])
    meta = json.loads((run / "config.json").read_text())
    assert meta["model_cfg"]["npoints"][0] <= 128 and "torch" in meta

    cfg.epochs = 3
    train(str(data), str(run), cfg, device="cpu", resume=True, log=lambda *_: None)
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 3

    model = Fg3dModel(run / "best.pt", device="cpu")
    out = tmp_path / "samples"
    sample_pairs(model, list_pairs(data, "test"), out, n_samples=2, steps=5, log=lambda *_: None)
    s = evaluate(out, log=None)
    assert s["trees"] == 1 and s["cd"] > 0 and 0 <= s["epc"] <= 1
    assert res["best"] > 0
