import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from bike_network import pavement


def segments(rows):
    """``rows``: (traf_dir, rb_layer, line) per segment, in EPSG:2263 feet."""
    return gpd.GeoDataFrame(
        {
            "segment_id": [f"{i:07d}" for i in range(len(rows))],
            "traf_dir": [r[0] for r in rows],
            "rb_layer": [r[1] for r in rows],
            "vehicle_only": False,
        },
        geometry=[MultiLineString([r[2]]) for r in rows],
        crs=2263,
    )


def rated(rows):
    """``rows``: (rating, line) per rated block, in EPSG:2263 feet."""
    return gpd.GeoDataFrame(
        {"rating": [r[0] for r in rows]},
        geometry=[MultiLineString([r[1]]) for r in rows],
        crs=2263,
    )


def test_match_takes_the_rated_line_the_segment_lies_on():
    seg = segments(
        [
            ("T", "B", [(0, 0), (400, 0)]),  # on a rated block
            ("W", "B", [(0, 20), (400, 20)]),  # beside it, within reach
            ("T", "B", [(200, -300), (200, -10)]),  # crosses near the rated block's middle
            ("P", "N", [(0, 500), (400, 500)]),  # a path on a rated line: not a roadway
        ]
    )
    r = rated(
        [
            (6.5, [(0, 0), (400, 0)]),
            (9.0, [(200, -50), (200, -400)]),  # the crossing street's own block, further off
            (4.0, [(0, 500), (400, 500)]),
        ]
    )
    got = pavement.match(seg, r).tolist()
    assert got[0] == 6.5
    assert got[1] == 6.5
    assert got[2] == 9.0  # the cross street at right angles is never its rating
    assert np.isnan(got[3])


def test_only_a_divided_streets_roadbeds_look_further():
    # The rated centerline runs between two roadbeds 100 ft either side of it;
    # an alley at the same offset is not on that street.
    seg = segments(
        [
            ("W", "R", [(0, 100), (400, 100)]),
            ("A", "R", [(400, -100), (0, -100)]),
            ("T", "B", [(0, 100 + 1000), (400, 100 + 1000)]),
        ]
    )
    r = rated([(5.0, [(0, 0), (400, 0)]), (8.0, [(0, 1000), (400, 1000)])])
    got = pavement.match(seg, r).tolist()
    assert got[:2] == [5.0, 5.0]
    assert np.isnan(got[2])


def test_latest_keeps_each_blocks_newest_rating():
    line = MultiLineString([[(-73.95, 40.70), (-73.95, 40.701)]])
    raw = gpd.GeoDataFrame(
        {
            "oftcode": ["1", "1", "2", "3"],
            "direction": [None, None, None, None],
            "road_type": [None, None, None, None],
            "systemrating": ["4.5", "8.25", "7", "0"],
            "inspection": ["2019-05-01", "2025-06-01", "2024-01-01", "2026-01-01"],
        },
        geometry=[line] * 4,
        crs=4326,
    )
    got = pavement.latest(raw)
    assert sorted(got["rating"]) == [7.0, 8.25]  # block 1 repaved; block 3 not rated
    assert got.crs.to_epsg() == 2263
    assert isinstance(got.geometry.iloc[0], MultiLineString | LineString)
    assert got["inspected"].min() == pd.Timestamp("2024-01-01")


def test_match_with_nothing_rated():
    seg = segments([("T", "B", [(0, 0), (400, 0)])])
    empty = gpd.GeoDataFrame({"rating": []}, geometry=[], crs=2263)
    assert pavement.match(seg, empty).isna().all()
