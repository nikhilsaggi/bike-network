import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row

from bike_network import assignment, config, effect, lanes, lion
from bike_network.network import Graph, build_edges, direction_volumes

REF = config.VOLUME_REFERENCE

# The square of test_assignment: 1 -> 2 -> 3 is a busy avenue (1 km a side),
# 1 -> 4 -> 3 a protected path 10% longer. One-way 3 -> 5 hangs off the
# corner, and 5 -> 6 is a quiet street with a painted lane.
SQUARE = [
    lion_row(1, 1, 2, "T"),
    lion_row(2, 2, 3, "T"),
    lion_row(3, 1, 4, "T", feet=1100 / 0.3048),
    lion_row(4, 4, 3, "T", feet=1100 / 0.3048),
    lion_row(5, 3, 5, "W"),
    lion_row(6, 5, 6, "T"),
]
LANES = [
    dot_row(3, "Protected", "Protected"),
    dot_row(4, "Protected", "Protected"),
    dot_row(6, "Conventional", "Conventional"),
]
BUSY = {(s, d): 16 * REF for s in (1, 2) for d in ("fwd", "rev")}
# The plan: the avenue northbound (corridor 1) and the painted street made protected (2).
BLOCKS = pd.DataFrame(
    {
        "segment_id": ["0000001", "0000002", "0000006"],
        "dir": ["fwd", "fwd", "fwd"],
        "u": [1, 2, 5],
        "v": [2, 3, 6],
        "corridor": [1, 1, 2],
        "new": [True, True, True],
        "kind": ["none", "none", "painted"],
    }
)
OD = pd.DataFrame({"o": [1, 5, 3], "d": [3, 6, 1], "trips": [10.0, 4.0, 7.0]})


def network():
    seg = lion.prepare(pd.DataFrame(SQUARE))
    fac = lanes.facilities(
        pd.DataFrame(LANES, columns=["segmentid", "status", "ft_facilit", "tf_facilit"])
    )
    est = pd.DataFrame(
        [(f"{s:07d}", d, v) for (s, d), v in BUSY.items()],
        columns=["segment_id", "dir", "volume_est"],
    )
    return seg, build_edges(direction_volumes(seg, est), fac)


def routed():
    seg, edges = network()
    plan, rows = effect.plan_edges(edges, BLOCKS)
    g_today, g_plan = Graph.from_edges(edges), Graph.from_edges(plan)
    today, after = assignment.assign(g_today, OD), assignment.assign(g_plan, OD)
    short = assignment.assign(Graph.from_edges(assignment.baseline_edges(edges)), OD)
    return seg, edges, plan, rows, g_today, g_plan, today, after, short


def rows_of(edges, *pairs):
    return [
        int(np.flatnonzero((edges["segment_id"] == f"{s:07d}") & (edges["dir"] == d))[0])
        for s, d in pairs
    ]


def test_plan_edges_protects_only_the_new_blocks():
    _, edges = network()
    plan, rows = effect.plan_edges(edges, BLOCKS)
    assert sorted(rows) == sorted(rows_of(edges, (1, "fwd"), (2, "fwd"), (6, "fwd")))
    assert (plan.loc[rows, "facility"] == "separated").all()
    assert plan.loc[rows, "cost"].tolist() == pytest.approx(plan.loc[rows, "length_m"].tolist())
    untouched = np.setdiff1d(np.arange(len(edges)), rows)
    pd.testing.assert_frame_equal(plan.iloc[untouched], edges.iloc[untouched])

    stale = BLOCKS.assign(u=[1, 2, 99])
    with pytest.raises(ValueError, match="not in the network"):
        effect.plan_edges(edges, stale)


def test_riders_move_to_the_new_lane_and_an_upgrade_keeps_its_route():
    _, edges, plan, rows, *_, today, after, short = routed()
    pairs = effect.compare(today.od, after.od).set_index(["o", "d"])
    assert pairs.loc[(1, 3), "length_m_today"] == pytest.approx(2200)  # the path
    assert pairs.loc[(1, 3), "length_m_plan"] == pytest.approx(2000)  # the avenue, built
    assert pairs.loc[(5, 6), "length_m_plan"] == pytest.approx(1000)  # same street, now protected
    assert pairs.loc[(5, 6), "cost_plan"] < pairs.loc[(5, 6), "cost_today"]
    assert pairs.loc[(3, 1), "cost_plan"] == pairs.loc[(3, 1), "cost_today"]

    s = effect.summarize(
        pairs.reset_index(),
        edges,
        plan,
        today.edge_trips,
        after.edge_trips,
        short.edge_trips,
        rows,
    )
    p = pairs
    saved = 10 * (p.loc[(1, 3), "cost_today"] - 2000) + 4 * (
        p.loc[(5, 6), "cost_today"] - p.loc[(5, 6), "cost_plan"]
    )
    assert s["saved_share"] == pytest.approx(saved / (p["trips"] * p["cost_today"]).sum(), abs=1e-4)
    assert s["helped_share"] == pytest.approx(14 / 21, abs=1e-4)
    assert s["same_route_share"] == pytest.approx(4 / 14, abs=1e-4)
    # today 1 -> 3 rides the 2.2 km path; the shortest route and the plan take the avenue
    assert s["detour"]["plan"] < s["detour"]["today"]
    assert s["trip_km"]["today"] - s["trip_km"]["plan"] == pytest.approx(2.0, abs=1)
    assert s["share"]["plan"]["painted"] == 0
    assert s["share"]["plan"]["separated"] > s["share"]["today"]["separated"]
    # the avenue's 2 km for 10 trips and the painted street's 1 km for 4, of 39.4 trip-km
    assert s["on_new"] == pytest.approx((10 * 2 + 4 * 1) / (10 * 2 + 4 * 1 + 7 * 2.2), abs=1e-4)


def test_route_paths_are_the_assigned_routes():
    _, edges, *_, g_plan, _, after, _ = routed()
    paths = effect.route_paths(g_plan, OD["o"].to_numpy(), OD["d"].to_numpy())
    routes = after.od.set_index(["o", "d"])
    for (o, d), path in zip(zip(OD["o"], OD["d"], strict=True), paths, strict=True):
        e = g_plan.edges.iloc[path]
        assert e["u"].iloc[0] == o
        assert e["v"].iloc[-1] == d
        assert (e["v"].to_numpy()[:-1] == e["u"].to_numpy()[1:]).all()
        assert e["length_m"].sum() == pytest.approx(routes.loc[(o, d), "length_m"])
    assert paths[0].tolist() == rows_of(edges, (1, "fwd"), (2, "fwd"))


def test_station_gains_count_each_trip_at_both_ends():
    *_, today, after, _ = routed()
    pairs = effect.compare(today.od, after.od)
    stations = pd.DataFrame(
        {
            "station": ["a", "b", "c", "d", "far"],
            "lat": [40.0, 40.2, 40.5, 40.6, 41.0],
            "lng": [-74.0] * 5,
            "ends": [30, 10, 21, 8, 5],
            "node": [1, 1, 3, 6, -1],
        }
    )
    got = effect.station_gains(pairs, stations).set_index("node")
    assert sorted(got.index) == [1, 3, 6]  # node 5 has trips but no station
    p = pairs.set_index(["o", "d"])
    at3 = p.loc[[(1, 3), (3, 1)]]
    assert got.loc[3, "trips"] == 17
    assert got.loc[3, "saved_share"] == pytest.approx(
        10
        * (at3["cost_today"] - at3["cost_plan"]).iloc[0]
        / (at3["trips"] * at3["cost_today"]).sum()
    )
    assert got.loc[3, "helped_share"] == pytest.approx(10 / 17)
    assert got.loc[1, "stations"] == 2
    assert got.loc[1, "lat"] == pytest.approx((40.0 * 30 + 40.2 * 10) / 40)


def test_each_corridor_gets_a_trip_that_rides_its_new_lane():
    _, edges, _, rows, g_today, g_plan, today, after, _ = routed()
    pairs = effect.compare(today.od, after.od)
    corridor = np.full(len(edges), -1)
    corridor[rows] = [0, 0, 1]
    new_m = np.zeros(len(edges))
    new_m[rows] = edges.loc[rows, "length_m"].to_numpy()
    ex = effect.examples(g_today, g_plan, pairs, corridor, new_m).set_index(["o", "d"])
    assert sorted(ex.index) == [(1, 3), (5, 6)]
    avenue = ex.loc[(1, 3)]
    assert list(avenue["corridors"]) == [0]
    assert list(avenue["today_edges"]) == rows_of(edges, (3, "fwd"), (4, "fwd"))
    assert list(avenue["plan_edges"]) == rows_of(edges, (1, "fwd"), (2, "fwd"))
    assert avenue["today_separated_m"] == pytest.approx(2200)
    assert avenue["plan_separated_m"] == pytest.approx(2000)
    assert avenue["new_m"] == pytest.approx(2000)
    upgrade = ex.loc[(5, 6)]
    assert upgrade["today_painted_m"] == pytest.approx(1000)
    assert upgrade["plan_separated_m"] == pytest.approx(1000)
    assert list(upgrade["today_edges"]) == list(upgrade["plan_edges"])


def test_place_names_take_the_two_shortest_streets():
    seg = pd.DataFrame(
        {
            "u": [1, 2, 1],
            "v": [2, 3, 4],
            "street": ["5 AVENUE", "WEST 49 STREET", "AVENUE OF THE AMERICAS"],
            "length_m": [80.0, 250.0, 80.0],
        }
    )
    names = effect.place_names(seg)
    assert names[2] == "5 Ave & W 49 St"
    assert names[1] == "5 Ave & Ave Of The Americas"
    # node 4 is the far end of 6th Ave's block: named for the cross street at its other end
    assert names[4] == "Ave Of The Americas near 5 Ave"
    assert names[3] == "W 49 St near 5 Ave"


def test_a_mid_block_node_takes_the_nearest_cross_street_along_its_own_street():
    m = [100.0, 30.0, 500.0, 50.0]
    seg = pd.DataFrame(
        {
            # Greenwich Ave runs A(1) - 2 - 3 - B(4); Charles St crosses at 1, Bank St at 4,
            # and a far-off street that is not Greenwich Ave touches node 3
            "u": [1, 2, 3, 1],
            "v": [2, 3, 4, 9],
            "street": [
                "GREENWICH AVENUE",
                "GREENWICH AVENUE",
                "GREENWICH AVENUE",
                "CHARLES STREET",
            ],
            "length_m": m,
        }
    )
    seg = pd.concat(
        [seg, pd.DataFrame({"u": [4], "v": [8], "street": ["BANK STREET"], "length_m": [60.0]})],
        ignore_index=True,
    )
    names = effect.place_names(seg)
    assert names[2] == "Greenwich Ave near Charles St"  # 100 m back, not 530 m on
    assert names[3] == "Greenwich Ave near Charles St"  # 130 m against 500 m
    lone = effect.place_names(
        pd.DataFrame({"u": [1], "v": [2], "street": ["ONE STREET"], "length_m": [9.0]})
    )
    assert lone[1] == "One St"


def test_distinct_sorts_by_cost_saved_and_lists_long_moved_unlike_trips():
    _, edges = network()
    ex = pd.DataFrame(
        {
            "o": [1, 3, 5, 2],
            "d": [3, 1, 6, 3],
            "trips": [10.0, 4.0, 40.0, 30.0],
            "cost_today": [3000.0, 3000.0, 1060.0, 1500.0],
            "cost_plan": [2000.0, 1500.0, 1000.0, 1000.0],
            "today_edges": [
                rows_of(edges, (3, "fwd"), (4, "fwd")),
                rows_of(edges, (4, "rev"), (3, "rev")),
                rows_of(edges, (6, "fwd")),
                rows_of(edges, (2, "fwd")),
            ],
            "plan_edges": [
                rows_of(edges, (1, "fwd"), (2, "fwd")),
                rows_of(edges, (2, "rev"), (1, "rev")),
                rows_of(edges, (6, "fwd")),
                rows_of(edges, (2, "fwd")),
            ],
        }
    )
    got = effect.distinct(ex, edges)
    # 15,000, 10,000, 6,000 and 2,400 cost-meters a year saved
    assert got["o"].tolist() == [2, 1, 3, 5]
    # 2 -> 3 and 5 -> 6 keep today's route; 3 -> 1 rides the streets of 1 -> 3 the other way
    assert got["listed"].tolist() == [False, True, False, False]
    loose = effect.distinct(ex, edges, overlap=1.0, min_m=0.0, min_moved=0.0)
    assert loose["listed"].all()
    # 1 -> 3 is 2 km today and with the plan
    assert not effect.distinct(ex, edges, min_m=2500.0)["listed"].any()
