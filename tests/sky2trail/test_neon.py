"""NEON loader: table parsing and stem positions, offline (no API calls)."""

import json
import math
import zipfile

from sky2trail import neon


def _zip(path, name, rows):
    header = list(rows[0])
    text = ",".join(header) + "\n" + "\n".join(",".join(str(r[h]) for h in header) for r in rows) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, text)


def test_build_stems_from_tables(tmp_path):
    site = tmp_path / "SJER"
    (site / "vst").mkdir(parents=True)
    _zip(site / "vst" / "a.zip", "NEON.D17.SJER.DP1.10098.001.vst_mappingandtagging.basic.csv", [
        {"individualID": "T1", "plotID": "SJER_001", "pointID": "41", "stemDistance": 10.0,
         "stemAzimuth": 90.0, "date": "2019-10-01"},
        {"individualID": "T2", "plotID": "SJER_001", "pointID": "41", "stemDistance": 5.0,
         "stemAzimuth": 0.0, "date": "2019-10-01"},
        {"individualID": "T3", "plotID": "SJER_001", "pointID": "", "stemDistance": "",
         "stemAzimuth": "", "date": "2019-10-01"},
    ])
    _zip(site / "vst" / "b.zip", "NEON.D17.SJER.DP1.10098.001.vst_apparentindividual.2019-10.basic.csv", [
        {"individualID": "T1", "stemDiameter": 25.0, "height": 12.0, "date": "2019-10-02",
         "plantStatus": "Live", "growthForm": "single bole tree"},
        {"individualID": "T2", "stemDiameter": "", "height": 3.0, "date": "2019-10-02",
         "plantStatus": "Live", "growthForm": "small tree"},
        {"individualID": "T3", "stemDiameter": 30.0, "height": 15.0, "date": "2019-10-02",
         "plantStatus": "Live", "growthForm": "single bole tree"},
    ])
    # Cached reference point, so no locations API call.
    (site / "reference_points.json").write_text(json.dumps(
        {"SJER_001.basePlot.vst.41": [1000.0, 2000.0, "11N", 0.1]}))

    stems = neon.build_stems(site, "SJER")
    by = {s.individual: s for s in stems}
    assert set(by) == {"T1", "T2"}  # T3 has no mapped position
    assert math.isclose(by["T1"].x, 1010.0) and math.isclose(by["T1"].y, 2000.0, abs_tol=1e-9)
    assert math.isclose(by["T2"].x, 1000.0, abs_tol=1e-9) and math.isclose(by["T2"].y, 2005.0)
    assert math.isclose(by["T1"].dbh, 0.25) and math.isnan(by["T2"].dbh)
    assert math.isclose(by["T1"].xy_uncertainty, 0.7)

    neon.save_stems(stems, site / "stems.csv")
    back = neon.load_stems(site / "stems.csv")
    assert [s.individual for s in back] == [s.individual for s in stems]
    s = neon.summary(site)
    assert s["plots"] == 1 and s["individuals"] == 2 and s["dbh_ge_10cm"] == 1


def test_tiles_and_flight_pairing():
    assert neon.tile_of(258277.1, 4110161.5) == (258000, 4110000)
    assert neon.nearest_flight("2019-10", ["2018-03", "2019-03", "2021-03"]) == "2019-03"
    assert neon.nearest_flight("2022-11", ["2021-03", "2023-04"]) == "2023-04"
    assert neon.nearest_flight("2022-11", []) is None
