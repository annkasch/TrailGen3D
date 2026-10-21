"""TLS .ptx reader and .npz conversion on a tiny synthetic file."""

import numpy as np

from sky2trail import tls


def _ptx(path, shift=(0.0, 0.0, 0.0)):
    m = np.eye(4)
    m[3, :3] = shift                       # row-vector form, as Leica writes it
    pts = [(1, 2, 3, 0.5, 10, 20, 30), (0, 0, 0, 0, 0, 0, 0), (1.004, 2, 3, 0.4, 1, 2, 3), (-4, 0, 1, 1, 9, 9, 9)]
    lines = ["2", "2", "0 0 0", "1 0 0", "0 1 0", "0 0 1"]
    lines += [" ".join(str(v) for v in row) for row in m]
    lines += [" ".join(str(v) for v in p) for p in pts]
    path.write_text("\n".join(lines) + "\n")


def test_read_ptx_drops_empty_cells_and_applies_transform(tmp_path):
    _ptx(tmp_path / "s.ptx", shift=(100.0, 200.0, 5.0))
    [s] = tls.read_ptx(tmp_path / "s.ptx", chunk_rows=3)
    assert len(s) == 3
    assert s.rgb[0].tolist() == [10, 20, 30]
    np.testing.assert_allclose(s.world()[0], [101, 202, 8])


def test_ptx_to_npz_voxel_and_range(tmp_path):
    _ptx(tmp_path / "s.ptx")
    s = tls.ptx_to_npz(tmp_path / "s.ptx", tmp_path / "s.npz", voxel=0.01, max_range=3.0)
    assert len(s) == 1                       # (-4, 0) out of range, the two near points share a voxel
    back = tls.load_npz(tmp_path / "s.npz")
    np.testing.assert_allclose(back.xyz, s.xyz)
    assert abs(back.intensity[0] - 0.5) < 0.01
