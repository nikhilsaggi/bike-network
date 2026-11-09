import itertools

import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row

from bike_network import assignment, config, corridors, lanes, lion
from bike_network.network import Graph, build_edges, direction_volumes

REF = config.VOLUME_REFERENCE
M = 1 / 0.3048  # feet per metre

# Fifth is a busy one-way avenue running south with no lane, then a lane past
# F4. Seventh, one short block west, is protected all the way. Short quiet
# cross streets join them at every node.
#
#   S0 - F0
#   |    |
#   S1 - F1        F: nodes 10..15, south of F4 a protected block to F5
#   |    |         S: nodes 20..24
#   ...  ...
#   S4 - F4
#        |
#        F5
AVE = 80  # metres per avenue block
CROSS = 100  # metres per cross street
F = [10, 11, 12, 13, 14, 15]
S = [20, 21, 22, 23, 24]
ROWS = [lion_row(1 + i, F[i], F[i + 1], "W", street="FIFTH", feet=AVE * M) for i in range(5)]
ROWS += [lion_row(10 + i, S[i], S[i + 1], "W", street="SEVENTH", feet=AVE * M) for i in range(4)]
ROWS += [lion_row(20 + i, S[i], F[i], "T", street=f"STREET {i}", feet=CROSS * M) for i in range(5)]
DOT = [dot_row(5, "Protected", None)] + [dot_row(10 + i, "Protected", None) for i in range(4)]
XY = pd.DataFrame(
    {
        "x": [0.0] * len(F) + [-CROSS * M] * len(S),
        "y": [(4 - i) * AVE * M for i in range(len(F))]
        + [(4 - i) * AVE * M for i in range(len(S))],
    },
    index=F + S,
)
TOY_PRICES = np.geomspace(1, 1e4, 37)  # this grid's savings are a few hundred cost-m per metre
OD = pd.DataFrame({"o": [10, 11, 15], "d": [15, 15, 10], "trips": [100.0, 50.0, 20.0]})


def network():
    seg = lion.prepare(pd.DataFrame(ROWS))
    fac = lanes.facilities(
        pd.DataFrame(DOT, columns=["segmentid", "status", "ft_facilit", "tf_facilit"])
    )
    busy = pd.DataFrame(
        [(f"{s:07d}", d, 8 * REF) for s in range(1, 5) for d in ("fwd", "rev")],
        columns=["segment_id", "dir", "volume_est"],
    )
    seg = direction_volumes(seg, busy)
    return seg, build_edges(seg, fac)


def setup():
    seg, edges = network()
    graph = Graph.from_edges(edges)
    table = corridors.street_runs(edges, seg, XY)
    runs = corridors.Runs.from_table(table, graph, edges)
    dist = corridors.Distances.compute(graph, OD.rename(columns={"trips": "weight"}))
    return seg, edges, table, runs, dist


def total_cost(edges):
    res = assignment.assign(Graph.from_edges(edges), OD)
    return (res.od["trips"] * res.od["cost"]).sum()


def run_of(table, street, first_node):
    hit = table[(table["street"] == street) & (table["pos"] == 0) & (table["u"] == first_node)]
    assert len(hit) == 1, (street, first_node)
    return int(hit["run"].iloc[0])


def test_runs_chain_each_street_in_each_direction():
    _, edges, table, _, _ = setup()
    south = table[table["run"] == run_of(table, "FIFTH", 10)]
    assert list(south["u"]) == F[:-1]
    assert list(south["v"]) == F[1:]
    north = table[table["run"] == run_of(table, "FIFTH", 15)]  # against traffic: a contraflow run
    assert list(north["u"]) == F[:0:-1]
    assert set(edges.loc[north["edge"], "mode"]) == {"walk"}
    assert table.groupby("street")["run"].nunique().to_dict() == {
        "FIFTH": 2,
        "SEVENTH": 2,
        **{f"STREET {i}": 2 for i in range(5)},  # one block each, both ways
    }
    assert not table.duplicated("edge").any()


def test_a_sharp_turn_ends_a_run():
    rows = [
        lion_row(1, 1, 2, "W", street="BENT"),
        lion_row(2, 2, 3, "W", street="BENT"),  # 90 degrees left
        lion_row(3, 3, 4, "W", street="BENT"),  # straight on from there
    ]
    seg = lion.prepare(pd.DataFrame(rows))
    edges = build_edges(
        seg,
        lanes.facilities(pd.DataFrame(columns=["segmentid", "status", "ft_facilit", "tf_facilit"])),
    )
    xy = pd.DataFrame(
        {"x": [0.0, 1000.0, 1000.0, 1000.0], "y": [0.0, 0.0, 1000.0, 2000.0]}, index=[1, 2, 3, 4]
    )
    t = corridors.street_runs(edges, seg, xy)
    with_traffic = t[t["u"].isin([1, 2, 3]) & (t["v"] == t["u"] + 1)]
    assert with_traffic.groupby("run")["u"].apply(list).sort_values().tolist() == [[1], [2, 3]]


def test_stretch_matches_rerouting_every_trip():
    _, edges, _, runs, dist = setup()
    before = total_cost(edges)
    checked = 0
    for run in range(len(runs.ptr) - 1):
        saved, _ = corridors.stretch_matrix(runs, run, dist)
        k = runs.ptr[run + 1] - runs.ptr[run]
        for s, t in itertools.combinations(range(k), 2):
            rows = corridors.stretch_edges(runs, run, s, t)
            after = total_cost(corridors.protect(edges, rows))
            assert before - after == pytest.approx(saved[s, t], abs=1e-3), (run, s, t)
            assert corridors.stretch_value(runs, run, s, t, dist)[0] == pytest.approx(saved[s, t])
            checked += 1
    assert checked > 30


def test_a_corridor_is_worth_more_than_its_blocks():
    # The 5th Ave case: no single block of Fifth beats the detour to Seventh,
    # but the four blocks to the lane at F4 do together.
    _, _, table, runs, dist = setup()
    run = run_of(table, "FIFTH", 10)
    saved, _, blocks = corridors.stretch_value(runs, run, 0, 4, dist)
    assert blocks == 0
    assert saved == pytest.approx(240 * 100 + 240 * 50, rel=0.02)
    best = corridors.best_stretches(runs, dist, [1.0]).set_index("run").loc[run]
    assert (best["s"], best["t"]) == (0, 4)  # the protected block past F4 is not part of the build
    assert best["new_m"] == pytest.approx(4 * AVE)
    assert best["alone"] == 0


def test_a_stretch_is_built_only_where_each_metre_pays_the_price():
    _, _, table, runs, dist = setup()
    run = run_of(table, "FIFTH", 10)
    ratio = (240 * 100 + 240 * 50) / (4 * AVE)  # cost-m saved per new metre, 112.5
    got = corridors.best_stretches(runs, dist, [ratio * 0.9, ratio * 1.1])
    got = got[got["run"] == run]
    assert list(got["price"]) == [pytest.approx(ratio * 0.9)]  # not worth it at the higher price


def test_round_price_is_the_highest_that_fills_the_room():
    grid = pd.DataFrame(
        {"price": [1.0, 1.0, 2.0, 2.0, 4.0], "new_m": [300.0, 300.0, 200.0, 100.0, 50.0]}
    )
    assert corridors.round_price(grid, 250) == 2.0
    assert corridors.round_price(grid, 600) == 1.0
    assert corridors.round_price(grid, 10_000) == 1.0  # nothing fills it: the lowest price


def test_build_out_protects_the_corridor_and_stops_when_nothing_helps():
    _, edges, table, _, _ = setup()
    picks, final = corridors.build_out(
        edges,
        table,
        OD.rename(columns={"trips": "weight"}),
        budget_m=10_000,
        log=lambda _: None,
        prices=TOY_PRICES,
    )
    first = picks[picks["run"] == run_of(table, "FIFTH", 10)].iloc[0]
    assert set(edges.loc[first["edges"], "segment_id"]) == {f"{i:07d}" for i in range(1, 5)}
    assert (final.loc[first["edges"], "cost"] == final.loc[first["edges"], "length_m"]).all()
    assert total_cost(final) < total_cost(edges)
    # Every trip ends up on protected blocks, so the last round finds nothing to build.
    assert picks["new_m"].sum() < 10_000


def test_budget_caps_the_build_out():
    _, edges, table, _, _ = setup()
    picks, _ = corridors.build_out(
        edges,
        table,
        OD.rename(columns={"trips": "weight"}),
        budget_m=1,
        log=lambda _: None,
        prices=TOY_PRICES,
    )
    assert len(picks) == 1  # the first pick spends the budget


def test_merge_joins_touching_picks_on_one_run():
    ptr = np.array([0, 6])
    new = np.array(
        [0.0, 80, 160, 160, 240, 320]
    )  # the block from node 2 to 3 was protected already
    picks = pd.DataFrame(
        {
            "run": [0, 0],
            "s": [3, 0],
            "t": [5, 2],
            "round": [0, 1],
            "new_m": [160.0, 160.0],
            "saved": [5.0, 7.0],
        }
    )
    m = corridors.merge_picks(picks, new, ptr)
    assert m[["s", "t", "picks", "round"]].to_dict("records") == [
        {"s": 0, "t": 5, "picks": 2, "round": 0}
    ]
    assert m["saved_in_plan"].iloc[0] == 12.0


@pytest.mark.parametrize(
    ("bearing", "boro", "expected"),
    [(209, 1, "S"), (29, 1, "N"), (119, 1, "E"), (180, 3, "S"), (270, 3, "W")],
)
def test_heading(bearing, boro, expected):
    b = np.radians(bearing)
    assert corridors.heading(np.sin(b), np.cos(b), boro) == expected


def test_a_protected_plaza_joins_the_run_but_is_never_built():
    rows = [
        lion_row(1, 1, 2, "W", street="BROAD"),
        lion_row(2, 2, 3, "P", street="BROAD"),  # a plaza, ridden on DOT's say-so
        lion_row(3, 3, 4, "W", street="BROAD"),
    ]
    seg = lion.prepare(pd.DataFrame(rows))
    fac = lanes.facilities(
        pd.DataFrame(
            [dot_row(2, "Ped Plaza", None)],
            columns=["segmentid", "status", "ft_facilit", "tf_facilit"],
        )
    )
    edges = build_edges(seg, fac)
    xy = pd.DataFrame({"x": [0.0] * 4, "y": [3000.0, 2000.0, 1000.0, 0.0]}, index=[1, 2, 3, 4])
    t = corridors.street_runs(edges, seg, xy)
    south = t[t["run"] == t.loc[t["u"] == 1, "run"].iloc[0]]
    assert list(south["u"]) == [1, 2, 3]
    runs = corridors.Runs.from_table(t, Graph.from_edges(edges), edges)
    run = int(south["run"].iloc[0])
    a = runs.ptr[run]
    km = 1000.0  # lion_row's default length
    assert runs.new[a : a + 4].tolist() == pytest.approx([0, km, km, 2 * km])


def divided_street():
    # Two one-way roadbeds of one divided avenue: 1 -> 2 -> 3 north, 6 -> 5 -> 4 south.
    rows = [
        lion_row(1, 1, 2, "W", street="PARK", RB_Layer="R"),
        lion_row(2, 2, 3, "W", street="PARK", RB_Layer="R"),
        lion_row(3, 4, 5, "A", street="PARK", RB_Layer="R"),
        lion_row(4, 5, 6, "A", street="PARK", RB_Layer="R"),
    ]
    seg = lion.prepare(pd.DataFrame(rows))
    edges = build_edges(
        seg,
        lanes.facilities(pd.DataFrame(columns=["segmentid", "status", "ft_facilit", "tf_facilit"])),
    )
    xy = pd.DataFrame(
        {"x": [0.0, 0.0, 0.0, 60.0, 60.0, 60.0], "y": [0.0, 3000.0, 6000.0, 0.0, 3000.0, 6000.0]},
        index=[1, 2, 3, 4, 5, 6],
    )
    return seg, edges, xy


def test_a_divided_street_takes_each_direction_on_its_own_roadbed():
    seg, edges, xy = divided_street()
    t = corridors.street_runs(edges, seg, xy)
    paths = sorted((*g["u"], g["v"].iloc[-1]) for _, g in t.groupby("run"))
    assert paths == [(1, 2, 3), (6, 5, 4)]  # no run against either roadbed's traffic


def test_describe_says_what_each_corridor_changes():
    seg, edges, table, runs, _ = setup()
    picks, _ = corridors.build_out(
        edges,
        table,
        OD.rename(columns={"trips": "weight"}),
        budget_m=10_000,
        log=lambda _: None,
        prices=TOY_PRICES,
    )
    plan = corridors.merge_picks(picks, runs.new, runs.ptr)
    plan[["saved_alone", "trips_alone", "blocks_alone"]] = 0.0
    out, blocks = corridors.describe(plan, runs, edges, seg, XY)
    south = out[(out["street"] == "FIFTH") & (out["heading"] == "S")].iloc[0]
    assert (south["kind"], south["none_m"], south["contraflow_m"]) == (
        "none",
        pytest.approx(4 * AVE),
        0,
    )
    assert south["opposite"] == "none"  # northbound Fifth is walked today
    north = out[(out["street"] == "FIFTH") & (out["heading"] == "N")].iloc[0]
    assert north["kind"] == "contraflow"
    assert north["opposite"] == "none"
    mine = blocks[blocks["corridor"] == south["corridor"]]
    assert set(mine.loc[mine["new"], "kind"]) == {"none"}
