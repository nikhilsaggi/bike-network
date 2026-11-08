import geopandas as gpd
import pandas as pd
import pytest
from conftest import lion_row
from shapely.geometry import MultiLineString

from bike_network import lion, traffic


def day(seg, direction, vol, n=96, yr=2020, d=1, req=1):
    return {
        "requestid": str(req),
        "segmentid": f"{seg:07d}",
        "direction": direction,
        "yr": str(yr),
        "m": "6",
        "d": str(d),
        "vol": str(vol),
        "n": str(n),
    }


def test_daily_volumes_keeps_full_days_and_scales_dst():
    rows = pd.DataFrame(
        [
            day(1, "NB", 1000, d=1),
            day(1, "NB", 920, n=92, d=2),  # a daylight-saving day: 1000 a full day
            day(1, "NB", 50, n=4, d=3),  # the stub of a count: dropped
            day(1, "SB", 300, d=1),
        ]
    )
    v = traffic.daily_volumes(rows).set_index("direction")
    assert v.loc["NB", "volume"] == pytest.approx(1000 * 0.5 + 920 * 96 / 92 * 0.5)
    assert v.loc["NB", "days"] == 2
    assert v.loc["SB", "volume"] == 300


def segs(lines):
    rows = [lion_row(i + 1, 2 * i + 1, 2 * i + 2) for i in range(len(lines))]
    seg = lion.prepare(pd.DataFrame(rows))
    return gpd.GeoDataFrame(seg, geometry=[MultiLineString([pts]) for pts in lines], crs=2263)


def test_orient_maps_compass_to_segment_direction():
    seg = segs([[(0, 0), (1000, 100)], [(0, 0), (50, 1000)]])  # eastbound, northbound
    counts = pd.DataFrame(
        {
            "segment_id": ["0000001", "0000001", "0000002", "0000002"],
            "direction": ["EB", "WB", "SB", "EB"],
            "volume": [1.0, 2.0, 3.0, 4.0],
            "days": [7, 7, 7, 7],
            "last_year": [2020] * 4,
        }
    )
    o = traffic.orient(counts, seg).set_index(["segment_id", "dir"])["volume"]
    assert o[("0000001", "fwd")] == 1.0
    assert o[("0000001", "rev")] == 2.0
    assert o[("0000002", "rev")] == 3.0  # southbound against a northbound segment
    assert len(o) == 3  # EB on a north-running segment says nothing


def test_car_edges_follow_traffic_direction():
    seg = lion.prepare(
        pd.DataFrame(
            [
                lion_row(1, 1, 2, "T"),
                lion_row(2, 2, 3, "W"),
                lion_row(3, 3, 4, "A"),
                lion_row(4, 4, 5, "P"),
                lion_row(5, 5, 6, "T", NonPed="V"),
            ]
        )
    )
    e = traffic.car_edges(seg)
    got = sorted(zip(e["segment_id"], e["dir"], strict=True))
    assert got == [("0000001", "fwd"), ("0000001", "rev"), ("0000002", "fwd"), ("0000003", "rev")]
    two_way = e[e["segment_id"] == "0000001"].iloc[0]
    assert (two_way["lanes_dir"], two_way["lanes_total"], two_way["one_way"]) == (1, 2, 0)
