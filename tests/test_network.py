import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row
from scipy.sparse import csgraph

from bike_network import config, lanes, lion
from bike_network.network import (
    Graph,
    build_edges,
    direction_volumes,
    mark_shared_streets,
    traffic_penalty,
)


def build(lion_rows, dot_rows=(), volumes=None):
    """``volumes`` maps (segment, "fwd"/"rev") to vehicles/day; unlisted streets are quiet."""
    seg = lion.prepare(pd.DataFrame(lion_rows))
    fac = lanes.facilities(
        pd.DataFrame(list(dot_rows), columns=["segmentid", "status", "ft_facilit", "tf_facilit"])
    )
    if volumes:
        est = pd.DataFrame(
            [(f"{s:07d}", d, v) for (s, d), v in volumes.items()],
            columns=["segment_id", "dir", "volume_est"],
        )
        seg = direction_volumes(seg, est)
    return build_edges(seg, fac)


REF = config.VOLUME_REFERENCE


def edge(edges, seg, direction):
    rows = edges[(edges["segment_id"] == f"{seg:07d}") & (edges["dir"] == direction)]
    assert len(rows) == 1, rows
    return rows.iloc[0]


QUIET = 1 + config.TRAFFIC_BASE  # a quiet 25 mph street with cars and no lane


def test_two_way_street_rides_both_ways():
    e = build([lion_row(1, 1, 2, "T")])
    for d in ("fwd", "rev"):
        r = edge(e, 1, d)
        assert r["mode"] == "ride"
        assert r["factor"] == pytest.approx(QUIET)
        assert r["cost"] == pytest.approx(1000 * QUIET)
    assert (edge(e, 1, "fwd")["u"], edge(e, 1, "fwd")["v"]) == (1, 2)
    assert (edge(e, 1, "rev")["u"], edge(e, 1, "rev")["v"]) == (2, 1)


@pytest.mark.parametrize(("traf", "rides", "walks"), [("W", "fwd", "rev"), ("A", "rev", "fwd")])
def test_one_way_street_walks_against_traffic(traf, rides, walks):
    e = build([lion_row(1, 1, 2, traf, Number_Travel_Lanes=" 1")])
    assert edge(e, 1, rides)["mode"] == "ride"
    assert edge(e, 1, walks)["mode"] == "walk"
    assert edge(e, 1, walks)["factor"] == config.WALK_FACTOR


def test_contraflow_lane_rides_against_traffic():
    e = build([lion_row(1, 1, 2, "W", Number_Travel_Lanes=" 1")], [dot_row(1, tf="Protected")])
    r = edge(e, 1, "rev")
    assert (r["mode"], r["facility"], r["factor"]) == ("ride", "separated", 1.0)
    assert edge(e, 1, "fwd")["facility"] == "none"


def test_lane_counts_only_in_its_own_direction():
    e = build(
        [lion_row(1, 1, 2, "T", Number_Travel_Lanes=" 4")],
        [dot_row(1, ft="Conventional")],
        {(1, "fwd"): 4 * REF, (1, "rev"): 4 * REF},
    )
    penalty = config.TRAFFIC_BASE + 2 * config.TRAFFIC_PER_DOUBLING  # two doublings
    assert edge(e, 1, "fwd")["factor"] == pytest.approx(1 + config.EXPOSURE["painted"] * penalty)
    assert edge(e, 1, "rev")["factor"] == pytest.approx(1 + penalty)


def test_pedestrian_path_walks_unless_designated():
    e = build(
        [lion_row(1, 1, 2, "P"), lion_row(2, 2, 3, "P")], [dot_row(2, "Protected", "Protected")]
    )
    assert set(e[e["segment_id"] == "0000001"]["mode"]) == {"walk"}
    path = e[e["segment_id"] == "0000002"]
    assert set(path["mode"]) == {"ride"}
    assert set(path["factor"]) == {1.0}  # no cars on a path


def test_vehicle_only_road_is_neither_ridden_nor_walked():
    e = build(
        [
            lion_row(1, 1, 2, "T", NonPed="V", RW_TYPE=" 2"),
            lion_row(2, 2, 3, "T", NonPed="V"),
        ],
        [dot_row(2, ft="Protected")],
    )
    assert "0000001" not in set(e["segment_id"])
    lane = e[e["segment_id"] == "0000002"]
    assert list(lane["dir"]) == ["fwd"]  # only the lane, and only its way
    assert lane.iloc[0]["mode"] == "ride"


def test_step_street_is_carried():
    e = build([lion_row(1, 1, 2, "P", RW_TYPE=" 7")])
    assert set(e["factor"]) == {config.STEP_FACTOR}


def test_riding_a_terrible_street_loses_to_walking(monkeypatch):
    monkeypatch.setattr(config, "WALK_FACTOR", 1.5)
    e = build(
        [lion_row(1, 1, 2, "T", TRUCK_ROUTE_TYPE="3"), lion_row(2, 2, 3, "T")],
        volumes={(1, "fwd"): 16 * REF, (1, "rev"): 16 * REF},
    )
    assert set(e[e["segment_id"] == "0000001"]["mode"]) == {"walk"}
    assert set(e[e["segment_id"] == "0000002"]["mode"]) == {"ride"}


def test_traffic_penalty_terms():
    seg = pd.DataFrame(
        {
            "traf_dir": ["T", "W", "T", "P", "T"],
            "speed_mph": [np.nan, 35.0, 25.0, 25.0, 25.0],
            "truck_route": ["", "", "3", "", ""],
        }
    )
    volume = pd.Series([4.0 * REF, REF, REF, 8.0 * REF, REF / 4])
    got = traffic_penalty(seg, volume).tolist()
    assert got[0] == pytest.approx(config.TRAFFIC_BASE + 2 * config.TRAFFIC_PER_DOUBLING)
    assert got[1] == pytest.approx(config.TRAFFIC_BASE + 2 * config.TRAFFIC_PER_5_MPH)
    assert got[2] == pytest.approx(config.TRAFFIC_BASE + config.TRUCK_PENALTY["3"])
    assert got[3] == 0.0  # no cars on a path, whatever the volume says
    assert got[4] == pytest.approx(config.TRAFFIC_BASE)  # quieter than quiet is no bonus


def test_contraflow_rider_shares_the_oncoming_volume():
    e = build(
        [lion_row(1, 1, 2, "W"), lion_row(2, 2, 3, "T")],
        [dot_row(1, tf="Conventional")],
        {(1, "fwd"): 8 * REF, (2, "fwd"): 2 * REF},
    )
    oncoming = config.TRAFFIC_BASE + 3 * config.TRAFFIC_PER_DOUBLING
    assert edge(e, 1, "rev")["volume"] == 8 * REF
    assert edge(e, 1, "rev")["factor"] == pytest.approx(1 + config.EXPOSURE["painted"] * oncoming)
    # A two-way street missing one direction's estimate borrows the other's.
    assert edge(e, 2, "rev")["volume"] == 2 * REF


def test_prepare_filters_and_dedupes():
    rows = [
        lion_row(1, 1, 2),
        lion_row(1, 1, 2, Street="ALT NAME"),  # same segment, another address row
        lion_row(2, 2, 3, FeatureTyp="1"),  # railroad
        lion_row(3, 3, 4, RB_Layer="G"),  # generic centreline of a divided street
        lion_row(4, 4, 5, Status="3"),  # paper street
        lion_row(5, 5, 6, RW_TYPE="14"),  # ferry route
        lion_row(6, 6, 7, TrafDir=" "),  # non-street feature
        lion_row(7, 7, 7),  # loop
        lion_row(8, 8, 9, RB_Layer="N", TrafDir="P"),  # greenway layer is kept
    ]
    seg = lion.prepare(pd.DataFrame(rows))
    assert sorted(seg["segment_id"]) == ["0000001", "0000008"]
    assert seg.loc[seg["segment_id"] == "0000001", "length_m"].item() == pytest.approx(1000)


def test_graph_keeps_cheapest_parallel_edge_and_maps_back():
    e = build(
        [lion_row(1, 1, 2, "T"), lion_row(2, 1, 2, "T")],
        [dot_row(2, "Protected", "Protected")],
    )
    g = Graph.from_edges(e)
    assert g.matrix.nnz == 2
    i, j = g.node_index(np.array([1, 2]))
    assert g.matrix[i, j] == pytest.approx(1000)
    assert e.iloc[g.edge_of[i, j] - 1]["segment_id"] == "0000002"


def test_route_detours_onto_a_protected_lane():
    # 1 -> 2 directly on a busy avenue, or 1 -> 3 -> 2 on a protected path 20% longer.
    e = build(
        [
            lion_row(1, 1, 2, "T", TRUCK_ROUTE_TYPE="3"),
            lion_row(2, 1, 3, "T", feet=600 / 0.3048),
            lion_row(3, 3, 2, "T", feet=600 / 0.3048),
        ],
        [dot_row(2, "Protected", "Protected"), dot_row(3, "Protected", "Protected")],
        {(1, "fwd"): 16 * REF, (1, "rev"): 16 * REF},
    )
    g = Graph.from_edges(e)
    src, dst = g.node_index(np.array([1, 2]))
    dist, pred = csgraph.dijkstra(g.matrix, indices=src, return_predecessors=True)
    assert dist[dst] == pytest.approx(1200)
    assert g.nodes[pred[dst]] == 3


def test_node_index_rejects_unknown_ids():
    g = Graph.from_edges(build([lion_row(1, 1, 2)]))
    with pytest.raises(KeyError):
        g.node_index(np.array([99]))


def test_shared_street_is_priced_quiet():
    seg = lion.prepare(
        pd.DataFrame(
            [
                lion_row(1, 1, 2, "A", TRUCK_ROUTE_TYPE="2"),  # shared street: Shared both ways
                lion_row(2, 2, 3, "T", TRUCK_ROUTE_TYPE="2"),  # sharrows on a two-way street
                lion_row(3, 3, 4, "W"),  # sharrows with traffic only
                lion_row(4, 10, 11, "A", street="BROADWAY"),  # shared street
                lion_row(5, 11, 12, "A", street="BROADWAY"),  # a block DOT coded otherwise
                lion_row(6, 12, 13, "A", street="BROADWAY"),  # shared street
                lion_row(7, 13, 14, "A", street="BROADWAY"),  # past its end: untouched
                lion_row(8, 20, 21, "A"),  # a signed route both ways is not a shared street
            ]
        )
    )
    fac = lanes.facilities(
        pd.DataFrame(
            [
                dot_row(1, "Shared", "Shared"),
                dot_row(2, "Shared", "Shared"),
                dot_row(3, "Shared"),
                dot_row(4, "Shared", "Shared"),
                dot_row(5, "Protected", "Shared"),
                dot_row(6, "Shared", "Shared"),
                dot_row(8, "Signed Route", "Signed Route"),
            ]
        )
    )
    busy = pd.DataFrame(
        [(f"{s:07d}", d, 16.0 * REF) for s in range(1, 9) for d in ("fwd", "rev")],
        columns=["segment_id", "dir", "volume_est"],
    )
    seg = mark_shared_streets(direction_volumes(seg, busy), fac)
    assert list(seg["shared_street"]) == [True, False, False, True, True, True, False, False]
    e = build_edges(seg, fac)
    assert edge(e, 1, "fwd")["factor"] == pytest.approx(QUIET)  # northbound on a southbound street
    assert edge(e, 1, "rev")["factor"] == pytest.approx(QUIET)
    assert edge(e, 2, "fwd")["factor"] > 2
