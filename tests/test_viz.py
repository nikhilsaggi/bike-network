import json
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row
from shapely.geometry import MultiLineString

from bike_network import config, effect, lanes, lion, viz
from bike_network.network import build_edges, direction_volumes

NODE = shutil.which("node")

# Every case the cost model distinguishes, so cost.js is held to each.
LION = [
    lion_row(1, 1, 2, "T"),
    lion_row(2, 2, 3, "W", Number_Travel_Lanes=" 1"),
    lion_row(3, 3, 4, "A", Number_Travel_Lanes=" 3", POSTED_SPEED="35"),
    lion_row(4, 4, 5, "W", Number_Travel_Lanes=" 1"),  # contraflow lane below
    lion_row(5, 5, 6, "T", Number_Travel_Lanes=" 6", TRUCK_ROUTE_TYPE="3"),
    lion_row(6, 6, 7, "P"),
    lion_row(7, 7, 8, "P"),  # designated path below
    lion_row(8, 8, 9, "P", RW_TYPE=" 7"),  # step street
    lion_row(9, 9, 10, "T", NonPed="V", RW_TYPE=" 2"),  # highway
    lion_row(10, 10, 11, "T", NonPed="V"),  # vehicle-only with a lane below
    lion_row(11, 11, 12, "T", Number_Travel_Lanes="  ", TRUCK_ROUTE_TYPE="1"),
    lion_row(12, 12, 13, "T", Number_Travel_Lanes=" 4", TRUCK_ROUTE_TYPE="2"),
]
DOT = [
    dot_row(4, None, "Protected"),
    dot_row(5, "Conventional", "Shared"),
    dot_row(7, "Protected", "Protected"),
    dot_row(10, "Protected", None),
    dot_row(12, "Curbside Buffered", None),
]


# Estimated vehicles/day per car direction; segments 4 and 11 have none and
# fall back (4 is one-way, so its contraflow lane borrows the car direction).
VOLUMES = {
    (1, "fwd"): 900,
    (1, "rev"): 2500,
    (2, "fwd"): 4000,
    (3, "rev"): 30000,
    (4, "fwd"): 6000,
    (5, "fwd"): 20000,
    (5, "rev"): 18000,
    (12, "fwd"): 7000,
}


def segments():
    seg = lion.prepare(pd.DataFrame(LION))
    fac = lanes.facilities(pd.DataFrame(DOT))
    est = pd.DataFrame(
        [(f"{s:07d}", d, v) for (s, d), v in VOLUMES.items()],
        columns=["segment_id", "dir", "volume_est"],
    )
    return direction_volumes(seg, est), fac


def run_cost_js(attrs, knobs):
    script = """
const api = require(process.argv[1]);
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = [];
for (let i = 0; i < input.p.length; i++)
  for (const dir of [0, 1]) {
    const r = api.edge(input.p[i], input.speed[i], input.vf[i], input.vr[i], dir, input.knobs);
    out.push(r === null ? null : [r.ride, r.factor]);
  }
console.log(JSON.stringify(out));
"""
    payload = {
        "p": attrs["p"].tolist(),
        "speed": attrs["speed"].tolist(),
        "vf": attrs["vf"].tolist(),
        "vr": attrs["vr"].tolist(),
        "knobs": knobs,
    }
    res = subprocess.run(  # noqa: S603 -- node on a file in this repo
        [NODE, "-e", script, str(config.ROOT / "viz" / "cost.js")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("walk_factor", [config.WALK_FACTOR, 1.3])
def test_cost_js_matches_python(monkeypatch, walk_factor):
    monkeypatch.setattr(config, "WALK_FACTOR", walk_factor)
    seg, fac = segments()
    edges = build_edges(seg, fac).set_index(["segment_id", "dir"])
    js = run_cost_js(viz.pack_attributes(seg, fac), viz.knobs())
    for i, sid in enumerate(seg["segment_id"]):
        for d, name in enumerate(("fwd", "rev")):
            got = js[2 * i + d]
            if (sid, name) not in edges.index:
                assert got is None, (sid, name)
                continue
            want = edges.loc[(sid, name)]
            assert got is not None, (sid, name)
            assert got[0] == (want["mode"] == "ride"), (sid, name)
            assert got[1] == pytest.approx(want["factor"]), (sid, name)


def test_export_round_trips_geometry():
    seg, fac = segments()
    seg = seg.head(3).copy()
    lines = [
        [(980000, 200000), (980500, 200300), (981000, 200000)],
        [(981000, 200000), (982000, 201000)],
        [(1000000, 190000), (999000, 191500)],
    ]
    gseg = gpd.GeoDataFrame(seg, geometry=[MultiLineString([pts]) for pts in lines], crs=2263)
    edges = build_edges(seg, fac)
    payload = viz.export(gseg, fac, edges)

    decoded, ci, prev = [], 0, np.zeros(2)
    for nv in payload["nv"]:
        pts = np.cumsum(np.array(payload["c"][ci : ci + 2 * nv]).reshape(nv, 2), axis=0) + prev
        prev = pts[0]
        ci += 2 * nv
        decoded.append((pts + payload["origin"]) / payload["quant"])
    want = gseg.to_crs(4326).geometry.map(lambda g: np.array(g.geoms[0].coords))
    # Rows come back in Morton order; match each decoded line by its first vertex.
    for line in want:
        match = min(decoded, key=lambda d: np.abs(d[0] - line[0]).sum())
        assert match.shape == line.shape
        assert np.abs(match - line).max() < 1e-5


def test_trips_land_on_their_own_direction():
    seg = pd.DataFrame({"segment_id": ["0000001", "0000002"]})
    flows = pd.DataFrame(
        {
            "segment_id": ["0000001", "0000001", "0000002"],
            "dir": ["fwd", "rev", "rev"],
            "trips": [10.4, 3.0, 7.0],
        }
    )
    fwd, rev = viz._per_direction(seg, flows, "trips")
    assert fwd.tolist() == [10, 0]
    assert rev.tolist() == [3, 7]


def test_plan_blocks_point_at_their_own_segment_and_direction():
    seg, fac = segments()
    seg = seg.head(3).copy()
    lines = [
        [(980000, 200000), (981000, 200000)],
        [(981000, 200000), (982000, 201000)],
        [(990000, 190000), (991000, 191000)],
    ]
    gseg = gpd.GeoDataFrame(seg, geometry=[MultiLineString([pts]) for pts in lines], crs=2263)
    edges = build_edges(seg, fac)
    ids = list(seg["segment_id"])
    plan = {
        "corridors": pd.DataFrame(
            {
                "corridor": [1],
                "street": ["STREET"],
                "boro": [1],
                "heading": ["S"],
                "from_street": ["A"],
                "to_street": ["B"],
                "new_m": [1000.0],
                "kind": ["contraflow"],
                "contraflow_m": [1000.0],
                "painted_m": [0.0],
                "shared_m": [0.0],
                "none_m": [0.0],
                "opposite": ["separated"],
                "gaps": [1],
                "round": [1],
                "trips_alone": [365.0],
                "saved_alone": [5000.0],
                "blocks_alone": [1000.0],
                "saved_in_plan": [5000.0],
            }
        ),
        "blocks": pd.DataFrame(
            {
                "segment_id": [ids[2], ids[0]],
                "dir": ["rev", "fwd"],
                "corridor": [1, 1],
                "new": [True, False],
                "kind": ["contraflow", ""],
            }
        ),
        "summary": {"budget_km": 80},
    }
    payload = viz.export(gseg, fac, edges, plan=plan)
    p = payload["plan"]
    assert p["budget_km"] == 80
    assert p["corridors"][0]["trips_day"] == 1
    street_of = dict(zip(seg["segment_id"], seg["street"], strict=True))
    got = {(payload["names"][payload["name"][s]], d, new, k) for s, d, _, new, k in p["blocks"]}
    assert got == {(street_of[ids[2]], 1, 1, 0), (street_of[ids[0]], 0, 0, -1)}
    assert p["kinds"][0] == "contraflow"
    assert p["corridors"][0]["kinds"]["contraflow"] == 1000
    assert p["corridors"][0]["opposite"] == "separated"


def plan_payload_inputs():
    """A three-segment network with a one-corridor plan, as export() takes it."""
    seg, fac = segments()
    seg = seg.head(3).copy()
    lines = [
        [(980000, 200000), (981000, 200000)],
        [(981000, 200000), (982000, 201000)],
        [(982000, 201000), (983000, 201000)],
    ]
    gseg = gpd.GeoDataFrame(seg, geometry=[MultiLineString([pts]) for pts in lines], crs=2263)
    edges = build_edges(seg, fac)
    ids = list(seg["segment_id"])
    corridor = {
        "corridor": [1],
        "street": ["STREET 2"],
        "boro": [1],
        "heading": ["E"],
        "from_street": ["A"],
        "to_street": ["B"],
        "new_m": [1000.0],
        "kind": ["none"],
        **{
            f"{k}_m": [1000.0 if k == "none" else 0.0]
            for k in ("contraflow", "painted", "shared", "none")
        },
        "opposite": ["none"],
        "gaps": [1],
        "round": [1],
        "trips_alone": [365.0],
        "saved_alone": [5000.0],
        "blocks_alone": [1000.0],
        "saved_in_plan": [5000.0],
    }
    blocks = pd.DataFrame(
        {"segment_id": [ids[1]], "dir": ["fwd"], "corridor": [1], "new": [True], "kind": ["none"]}
    )
    plan = {"corridors": pd.DataFrame(corridor), "blocks": blocks, "summary": {"budget_km": 80}}
    return gseg, fac, edges, ids, plan


def test_effect_trips_and_example_routes_land_on_their_own_direction():
    gseg, fac, edges, ids, plan = plan_payload_inputs()
    row = {
        (s, d): i for i, (s, d) in enumerate(zip(edges["segment_id"], edges["dir"], strict=True))
    }
    today = edges[["segment_id", "dir", "street", "facility", "mode", "length_m"]].assign(
        boro=1, trips=0.0, trips_shortest=0.0
    )
    after = today.copy()
    today.loc[row[(ids[0], "fwd")], "trips"] = 365 * 40  # riders leave segment 1 fwd ...
    after.loc[row[(ids[1], "fwd")], "trips"] = 365 * 40  # ... for the new lane on segment 2
    mix = {f"{w}_{k}_m": 0.0 for w in ("today", "plan") for k in effect.MIX}
    examples = pd.DataFrame(
        [
            {
                "o": 1,
                "d": 3,
                "trips": 3650.0,
                "corridors": [0],
                "cost_today": 2000.0,
                "cost_plan": 1500.0,
                **mix,
                "new_m": 1000.0,
                "today_edges": [row[(ids[0], "fwd")], row[(ids[2], "rev")]],
                "plan_edges": [row[(ids[1], "fwd")]],
            }
        ]
    )
    stations = pd.DataFrame(
        {
            "node": [1, 3, 4],
            "lat": [40.7, 40.71, 40.72],
            "lng": [-74.0, -73.99, -73.98],
            "stations": [1, 2, 1],
            "trips": [3650.0, 7300.0, 100.0],
            "saved_share": [0.1, 0.2, 0.0],
            "helped_share": [0.5, 0.6, 0.0],
        }
    )
    eff = {
        "summary": {"saved_share": 0.1},
        "flows": after,
        "examples": examples,
        "stations": stations,
    }
    payload = viz.export(gseg, fac, edges, flows=today, plan=plan, effect=eff)
    seg_of = {payload["names"][payload["name"][s]]: s for s in range(payload["n"])}
    s1, s2, s3 = (seg_of[f"STREET {i}"] for i in (1, 2, 3))
    assert payload["pf"][s2] == 365 * 40
    assert payload["pf"][s1] == 0
    assert sum(payload["pr"]) == 0
    e = payload["plan"]["effect"]
    assert e["saved_share"] == 0.1
    x = e["examples"][0]
    assert x["tp"] == [2 * s1, 2 * s3 + 1]
    assert x["pp"] == [2 * s2]
    assert x["trips_day"] == 10
    assert x["from"] == "St 1"  # node 1 meets one street
    assert e["shifts"]["join"][0]["street"] == "STREET 2"
    assert e["shifts"]["join"][0]["km_day"] == 40
    assert e["shifts"]["leave"][0]["street"] == "STREET 1"
    # a station with under a trip a day is left out
    assert e["stations"]["n"] == [1, 2]
    assert e["stations"]["name"][1] == "St 2 & St 3"
