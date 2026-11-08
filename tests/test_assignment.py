import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row
from pyproj import Transformer

from bike_network import assignment, config, demand, lanes, lion
from bike_network.network import Graph, build_edges, direction_volumes

REF = config.VOLUME_REFERENCE


def network(lion_rows, dot_rows=(), volumes=None):
    seg = lion.prepare(pd.DataFrame(lion_rows))
    fac = lanes.facilities(
        pd.DataFrame(list(dot_rows), columns=["segmentid", "status", "ft_facilit", "tf_facilit"])
    )
    est = None
    if volumes:
        est = pd.DataFrame(
            [(f"{s:07d}", d, v) for (s, d), v in volumes.items()],
            columns=["segment_id", "dir", "volume_est"],
        )
    return build_edges(direction_volumes(seg, est), fac)


def trips_on(result, graph, seg, direction):
    rows = (graph.edges["segment_id"] == f"{seg:07d}") & (graph.edges["dir"] == direction)
    return result.edge_trips[rows.to_numpy()].sum()


# A square: 1 -> 2 -> 3 is a busy avenue (1 km a side), 1 -> 4 -> 3 a protected
# path 10% longer. One-way 3 -> 5 hangs off the corner.
SQUARE = [
    lion_row(1, 1, 2, "T"),
    lion_row(2, 2, 3, "T"),
    lion_row(3, 1, 4, "T", feet=1100 / 0.3048),
    lion_row(4, 4, 3, "T", feet=1100 / 0.3048),
    lion_row(5, 3, 5, "W"),
]
LANES = [dot_row(3, "Protected", "Protected"), dot_row(4, "Protected", "Protected")]
BUSY = {(s, d): 16 * REF for s in (1, 2) for d in ("fwd", "rev")}


def test_trips_follow_the_cheapest_route_and_the_baseline_the_shortest():
    edges = network(SQUARE, LANES, BUSY)
    od = pd.DataFrame({"o": [1, 1, 3], "d": [3, 5, 1], "trips": [10.0, 4.0, 7.0]})
    g = Graph.from_edges(edges)
    safe = assignment.assign(g, od)
    assert trips_on(safe, g, 3, "fwd") == 14  # 1 -> 3 and 1 -> 5 take the path
    assert trips_on(safe, g, 4, "fwd") == 14
    assert trips_on(safe, g, 4, "rev") == 7  # 3 -> 1 comes back on it too
    assert trips_on(safe, g, 1, "fwd") == 0
    assert trips_on(safe, g, 5, "fwd") == 4
    routes = safe.od.set_index(["o", "d"])
    assert routes.loc[(1, 3), "length_m"] == pytest.approx(2200)
    assert routes.loc[(1, 3), "cost"] == pytest.approx(2200)  # separated: factor 1

    b = Graph.from_edges(assignment.baseline_edges(edges))
    short = assignment.assign(b, od)
    assert trips_on(short, b, 1, "fwd") == 14  # the avenue is shorter
    assert short.od.set_index(["o", "d"]).loc[(1, 3), "length_m"] == pytest.approx(2000)


def test_wrong_way_trips_walk_and_say_so():
    edges = network(SQUARE, LANES, BUSY)
    g = Graph.from_edges(edges)
    res = assignment.assign(g, pd.DataFrame({"o": [5], "d": [3], "trips": [2.0]}))
    route = res.od.iloc[0]
    assert route["walk_m"] == pytest.approx(route["length_m"]) == pytest.approx(1000)
    assert route["cost"] == pytest.approx(1000 * config.WALK_FACTOR)


def test_unreachable_pairs_are_flagged_not_counted():
    edges = network([*SQUARE, lion_row(9, 8, 9, "T")])
    g = Graph.from_edges(edges)
    res = assignment.assign(g, pd.DataFrame({"o": [1, 1], "d": [9, 2], "trips": [5.0, 1.0]}))
    routes = res.od.set_index("d")
    assert not routes.loc[9, "reachable"]
    assert routes.loc[2, "reachable"]
    assert res.edge_trips.sum() == 1  # only the reachable trip's one edge


def test_reduce_trips_drops_loops_and_lost_bikes():
    trips = pd.DataFrame(
        {
            "start_station_id": ["A", "A", "A", "B", "C"],
            "end_station_id": ["B", "B", "A", "A", None],
            "start_lat": [40.0, 40.0, 40.0, 41.0, 42.0],
            "start_lng": [-74.0] * 5,
            "end_lat": [41.0, 41.0, 40.0, 40.0, np.nan],
            "end_lng": [-74.0] * 4 + [np.nan],
        }
    )
    od, pos = demand.reduce_trips(trips)
    assert sorted(map(tuple, od.to_numpy().tolist())) == [("A", "B", 2), ("B", "A", 1)]
    pos = pos.set_index("station")
    assert pos.loc["A", "n"] == 3  # two starts, one end
    assert pos.loc["A", "lat"] / pos.loc["A", "n"] == pytest.approx(40.0)


def test_snap_attaches_to_the_nearest_allowed_node():
    nodes = pd.DataFrame(
        {"node": [1, 2, 3], "x": [987000.0, 987300.0, 990000.0], "y": [211000.0] * 3}
    )
    lng, lat = Transformer.from_crs(2263, 4326, always_xy=True).transform(987250.0, 211000.0)
    far_lng, far_lat = Transformer.from_crs(2263, 4326, always_xy=True).transform(
        999000.0, 211000.0
    )
    stations = pd.DataFrame({"station": ["s", "far"], "lat": [lat, far_lat], "lng": [lng, far_lng]})
    got = demand.snap(stations, nodes, np.array([1, 3])).set_index("station")
    assert got.loc["s", "node"] == 1  # node 2 is nearer but not allowed
    assert got.loc["far", "node"] == -1
