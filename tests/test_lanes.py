import zipfile

import pandas as pd
import pytest
from conftest import dot_row, lion_row

from bike_network import lanes, lion, sources


def fac(rows):
    out = lanes.facilities(pd.DataFrame(rows))
    return out.set_index("segment_id")


def test_only_current_records_count():
    f = fac([dot_row(1, "Shared", None, status="Retired"), dot_row(2, "Protected", None)])
    assert list(f.index) == ["0000002"]


def test_classes_and_missing_direction():
    f = fac([dot_row(1, "Conventional Buffered", None)])
    assert (f.loc["0000001", "fwd"], f.loc["0000001", "rev"]) == ("painted", "none")


def test_best_facility_wins_across_records_and_second_columns():
    f = fac(
        [
            dot_row(1, "Shared", "Shared"),
            dot_row(1, "Conventional", None),
            dot_row(2, "Shared", None, ft2facilit="Protected"),
        ]
    )
    assert (f.loc["0000001", "fwd"], f.loc["0000001", "rev"]) == ("painted", "shared")
    assert f.loc["0000002", "fwd"] == "separated"


def test_unknown_facility_name_is_an_error():
    with pytest.raises(ValueError, match="Hoverlane"):
        fac([dot_row(1, "Hoverlane", None)])


def test_zip_slip_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="escapes"):
        sources._safe_extract(archive, tmp_path / "out")


def test_lion_fills_in_separated_paths_dot_lacks():
    seg = lion.prepare(
        pd.DataFrame(
            [
                lion_row(1, 1, 2, "P", BikeLane=" 1", BIKE_TRAFDIR="TW"),  # greenway, no DOT record
                lion_row(2, 2, 3, "W", BikeLane=" 1", BIKE_TRAFDIR="FT"),  # one-way path
                lion_row(3, 3, 4, "T", BikeLane=" 2", BIKE_TRAFDIR="TW"),  # paint: DOT decides
                lion_row(4, 4, 5, "P", BikeLane=" 1", BIKE_TRAFDIR="TW"),  # DOT has it: kept
            ]
        )
    )
    fac = lanes.facilities(pd.DataFrame([dot_row(4, "Shared", "Shared")]))
    f = lanes.with_lion_paths(fac, seg).set_index("segment_id")
    assert (f.loc["0000001", "fwd"], f.loc["0000001", "rev"]) == ("separated", "separated")
    assert (f.loc["0000002", "fwd"], f.loc["0000002", "rev"]) == ("separated", "none")
    assert "0000003" not in f.index
    assert f.loc["0000004", "fwd"] == "shared"
