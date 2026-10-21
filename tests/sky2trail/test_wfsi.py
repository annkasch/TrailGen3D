"""WFSI loader: plot table and sample selection, offline (no portal calls)."""

from sky2trail import wfsi


def _plots(tmp_path):
    p = tmp_path / "plots.csv"
    rows = ["﻿BurnUnit,Forest_type,Macroplot,Easting,Northing,Tree_survey"]
    for unit in ("E", "F6.3"):
        for ftype, code in (("flatwoods", "F"), ("edge", "E"), ("wetlands", "W")):
            for i in range(3):
                rows.append(f"{unit},{ftype},{code}{unit[-1]}{i:02d},{430000 + i},{3546000 + i},"
                            f"{'Yes' if i == 2 else 'No'}")
    rows.append("E,edge,E999,NA,NA,Yes")
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return wfsi.read_plots(p)


def test_read_plots_skips_missing_coordinates(tmp_path):
    plots = _plots(tmp_path)
    assert len(plots) == 18
    assert all(p.name != "E999" for p in plots)
    assert plots[0].x == 430000.0


def test_pick_sample_spreads_and_prefers_tree_survey(tmp_path):
    sample = wfsi.pick_sample(_plots(tmp_path), 6)
    assert len(sample) == 6
    assert len({(p.burn_unit, p.forest_type) for p in sample}) == 6
    assert all(p.tree_survey for p in sample)
    assert len(wfsi.pick_sample(_plots(tmp_path), 100)) == 18


def test_scan_name():
    assert wfsi.scan_name("F041", "pre") == "FTSTF0410222.ptx"
    assert wfsi.scan_name("F041", "post") == "FTSTF0410322.ptx"
