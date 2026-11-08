import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row

from bike_network import assignment, candidates, config, lanes, lion
from bike_network.network import Graph, build_edges, direction_volumes

REF = config.VOLUME_REFERENCE

# A 3x3 grid of 300 m blocks: busy two-way avenues, a one-way street with no
# lane, a protected path, and a footpath across the middle.
#   7 - 8 - 9
#   |   |   |
#   4 - 5 - 6
#   |   |   |
#   1 - 2 - 3
BLOCK = 300 / 0.3048
GRID = [
    lion_row(1, 1, 2, "T", feet=BLOCK),
    lion_row(2, 2, 3, "T", feet=BLOCK),
    lion_row(3, 4, 5, "W", feet=BLOCK),  # one-way east
    lion_row(4, 5, 6, "W", feet=BLOCK),
    lion_row(5, 7, 8, "T", feet=BLOCK),
    lion_row(6, 8, 9, "T", feet=BLOCK),
    lion_row(7, 1, 4, "T", feet=BLOCK),
    lion_row(8, 4, 7, "T", feet=BLOCK),
    lion_row(9, 2, 5, "P", feet=BLOCK),  # footpath
    lion_row(10, 5, 8, "P", feet=BLOCK),
    lion_row(11, 3, 6, "T", feet=BLOCK),
    lion_row(12, 6, 9, "T", feet=BLOCK),
]
DOT = [dot_row(11, "Protected", "Protected"), dot_row(12, "Protected", "Protected")]
BUSY = {(s, d): 8 * REF for s in (1, 2, 5, 6, 7, 8) for d in ("fwd", "rev")}
OD = pd.DataFrame(
    {
        "o": [1, 1, 3, 7, 9, 6, 4, 2],
        "d": [9, 6, 7, 3, 1, 4, 6, 8],
        "trips": [100.0, 40.0, 70.0, 25.0, 60.0, 30.0, 50.0, 80.0],
    }
)


def network():
    seg = lion.prepare(pd.DataFrame(GRID))
    fac = lanes.facilities(
        pd.DataFrame(DOT, columns=["segmentid", "status", "ft_facilit", "tf_facilit"])
    )
    est = pd.DataFrame(
        [(f"{s:07d}", d, v) for (s, d), v in BUSY.items()],
        columns=["segment_id", "dir", "volume_est"],
    )
    seg = direction_volumes(seg, est)
    return seg, build_edges(seg, fac)


def total_cost(edges):
    res = assignment.assign(Graph.from_edges(edges), OD)
    return (res.od["trips"] * res.od["cost"]).sum()


def test_kinds():
    seg, edges = network()
    c = candidates.candidate_edges(edges, seg).set_index(["segment_id", "dir"])["kind"]
    assert c[("0000001", "fwd")] == "lane"
    assert c[("0000003", "rev")] == "contraflow"  # west on an eastbound street
    assert c[("0000009", "fwd")] == "walkway"
    assert ("0000011", "fwd") not in c.index  # already protected


def test_benefit_matches_rerouting_every_trip():
    seg, edges = network()
    cands = candidates.candidate_edges(edges, seg)
    pairs = OD.rename(columns={"trips": "weight"})
    scored = candidates.score(Graph.from_edges(edges), cands, pairs)
    before = total_cost(edges)
    checked = 0
    for row in scored.itertuples():
        upgraded = edges.copy()
        hit = (upgraded["segment_id"] == row.segment_id) & (upgraded["dir"] == row.dir)
        upgraded.loc[hit, ["cost", "mode"]] = [row.new_cost, "ride"]
        assert before - total_cost(upgraded) == pytest.approx(row.saved, abs=1e-6), row
        checked += 1
    assert checked == len(cands) > 10
    assert scored["saved"].max() > 0


def test_sample_weights_add_up_to_the_trips():
    pairs = candidates.sample_pairs(OD, n=500)
    assert pairs["weight"].sum() == pytest.approx(OD["trips"].sum())
    assert set(zip(pairs["o"], pairs["d"], strict=True)) <= set(zip(OD["o"], OD["d"], strict=True))
    assert np.isclose(pairs["weight"].iloc[0], OD["trips"].sum() / 500)
