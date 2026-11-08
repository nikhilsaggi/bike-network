import json
import shutil
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from conftest import dot_row, lion_row
from shapely.geometry import MultiLineString

from bike_network import config, lanes, lion, viz
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
