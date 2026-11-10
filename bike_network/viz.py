"""Export the network for the edge-cost viewer in ``viz/``.

The viewer recomputes every edge's cost in the browser (``viz/cost.js``) so
the knobs in ``config.py`` can be moved live; it therefore ships the *inputs*
of the cost model per segment, not the costs. ``tests/test_viz.py`` holds the
two implementations to the same answer.

Geometry is WGS84 quantized to 1e-5 degrees (about a meter), each segment's
first vertex stored as a delta from the previous segment's and the rest as
deltas along the line. Segments are sorted along a Morton curve so those
deltas stay small. Every LION geometry starts at its FROM node, so the stored
vertex order is the ``fwd`` direction.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from . import config
from .corridors import KINDS

if TYPE_CHECKING:
    import geopandas as gpd

QUANT = 1e5
SIMPLIFY_FT = 3.0
TRAF_CODE = {"T": 1, "W": 2, "A": 3}  # anything else: no cars
FAC_CODE = {"none": 0, "shared": 1, "painted": 2, "separated": 3}

# Bit layout of the per-segment ``p`` value; viz/cost.js reads the same.
BIT_STEP = 1 << 4
BIT_VEHICLE_ONLY = 1 << 5
BIT_HIGHWAY = 1 << 6
BIT_SHARED_STREET = 1 << 11  # display only: its volume and truck term are already quiet


def knobs() -> dict[str, Any]:
    """Return the cost model's tunable numbers, as the viewer's starting values."""
    return {
        "exposure_painted": config.EXPOSURE["painted"],
        "exposure_shared": config.EXPOSURE["shared"],
        "traffic_base": config.TRAFFIC_BASE,
        "per_doubling": config.TRAFFIC_PER_DOUBLING,
        "volume_ref": config.VOLUME_REFERENCE,
        "per_5_mph": config.TRAFFIC_PER_5_MPH,
        "truck_limited": config.TRUCK_PENALTY["1"],
        "truck_local": config.TRUCK_PENALTY["2"],
        "truck_through": config.TRUCK_PENALTY["3"],
        "walk_factor": config.WALK_FACTOR,
        "step_factor": config.STEP_FACTOR,
        "default_speed": config.DEFAULT_SPEED_MPH,
    }


def pack_attributes(seg: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    """Per segment: packed flags, lanes, speed, length and each direction's volume.

    ``seg`` must carry ``vol_fwd``/``vol_rev`` (:func:`network.direction_volumes`).
    """
    seg = seg.merge(fac, on="segment_id", how="left")
    traf = seg["traf_dir"].map(TRAF_CODE).fillna(0).to_numpy(dtype=np.int64)
    truck = pd.to_numeric(seg["truck_route"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    fwd = seg["fwd"].fillna("none").map(FAC_CODE).to_numpy(dtype=np.int64)
    rev = seg["rev"].fillna("none").map(FAC_CODE).to_numpy(dtype=np.int64)
    rw = seg["rw_type"].to_numpy()
    shared = seg["shared_street"].to_numpy() if "shared_street" in seg else np.zeros(len(seg), bool)
    packed = (
        traf
        | (truck << 2)
        | np.where(rw == config.RW_STEP_STREET, BIT_STEP, 0)
        | np.where(seg["vehicle_only"].to_numpy(), BIT_VEHICLE_ONLY, 0)
        | np.where(rw == config.RW_HIGHWAY, BIT_HIGHWAY, 0)
        | (fwd << 7)
        | (rev << 9)
        | np.where(shared, BIT_SHARED_STREET, 0)
    )
    return pd.DataFrame(
        {
            "p": packed,
            "lanes": seg["travel_lanes"].fillna(0).astype(int),
            "speed": seg["speed_mph"].fillna(0).astype(int),
            "len": seg["length_m"].round(1),
            "vf": seg["vol_fwd"].round().astype(int),
            "vr": seg["vol_rev"].round().astype(int),
        },
        index=seg.index,
    )


def _morton(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64) & np.uint64(0xFFFF)
        out = np.zeros_like(v)
        for bit in range(16):
            out |= ((v >> np.uint64(bit)) & np.uint64(1)) << np.uint64(2 * bit)
        return out

    return spread(x) | (spread(y) << np.uint64(1))


BORO = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
TOP_FACILITIES = 15


def _per_direction(
    seg: pd.DataFrame, flows: pd.DataFrame, col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Trips a year in each direction of every segment row, 0 where none."""
    f = flows.drop_duplicates(["segment_id", "dir"]).set_index(["segment_id", "dir"])[col]
    out = []
    for d in ("fwd", "rev"):
        idx = pd.MultiIndex.from_arrays([seg["segment_id"], [d] * len(seg)])
        out.append(np.rint(f.reindex(idx).fillna(0).to_numpy()).astype(np.int64))
    return out[0], out[1]


def flow_summary(seg: gpd.GeoDataFrame, flows: pd.DataFrame) -> dict[str, Any]:
    """Trip-km shares by facility, both routings, and the busiest bike facilities.

    ``flows`` is the ``assign`` command's output. A facility is a street name in
    a borough, summed over its directed edges with any bike facility; its
    ``bbox`` frames the blocks that carry it, for the viewer to fly to.
    """
    kind = (
        np.where(flows["mode"] == "walk", "walked", flows["facility"])
        if "mode" in flows
        else flows["facility"]
    )
    share = {}
    for col in ("trips", "trips_shortest"):
        km = pd.Series(flows[col].to_numpy() * flows["length_m"].to_numpy()).groupby(kind).sum()
        share[col] = (km / km.sum()).round(4).to_dict()
    on_lanes = flows[(flows["facility"] != "none") & (flows["trips"] > 0)]
    on_lanes = on_lanes.assign(trip_km=on_lanes["trips"] * on_lanes["length_m"] / 1000)
    top = (
        on_lanes.groupby(["street", "boro"])
        .agg(trip_km=("trip_km", "sum"), peak=("trips", "max"))
        .sort_values("trip_km", ascending=False)
        .head(TOP_FACILITIES)
        .reset_index()
    )
    geom = seg.drop_duplicates("segment_id").set_index("segment_id").geometry.to_crs(4326)
    rows = []
    for r in top.itertuples():
        ids = on_lanes.loc[
            (on_lanes["street"] == r.street) & (on_lanes["boro"] == r.boro), "segment_id"
        ]
        w, s, e, n = geom.reindex(ids.unique()).total_bounds
        rows.append(
            {
                "street": r.street,
                "boro": BORO.get(int(r.boro), ""),
                "trip_km": round(r.trip_km),
                "peak_day": round(r.peak / 365),
                "bbox": [round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
            }
        )
    return {
        "year": config.DEMAND_YEAR,
        "trips": round(flows.attrs.get("trips_routed", 0)),
        "share": {"safest": share["trips"], "shortest": share["trips_shortest"]},
        "top": rows,
    }


def plan_layer(seg: gpd.GeoDataFrame, rank: pd.Series, plan: dict[str, Any]) -> dict[str, Any]:
    """Shape the ``corridors`` command's build-out for the viewer's New corridors view.

    ``plan`` holds ``corridors`` and ``blocks`` (its two parquet files) and
    ``summary`` (its JSON). ``rank`` maps segment id to the payload's
    segment order. Each block is ``[segment, direction, corridor index,
    new, kind]``: ``new`` is 0 for a block the corridor bridges that was
    protected already, and ``kind`` indexes ``corridors.KINDS`` (-1 when not
    new).
    """
    cors, blocks = plan["corridors"], plan["blocks"]
    blocks = blocks[blocks["segment_id"].isin(rank.index)]
    geom = seg.drop_duplicates("segment_id").set_index("segment_id").geometry.to_crs(4326)
    rows = []
    for r in cors.itertuples():
        ids = blocks.loc[blocks["corridor"] == r.corridor, "segment_id"].unique()
        w, s, e, n = geom.reindex(ids).total_bounds
        rows.append(
            {
                "street": r.street,
                "boro": BORO.get(int(r.boro), ""),
                "heading": r.heading,
                "from": r.from_street,
                "to": r.to_street,
                "new_m": round(r.new_m),
                "kind": r.kind,
                "kinds": {k: round(getattr(r, f"{k}_m")) for k in KINDS},
                "opposite": r.opposite,
                "gaps": int(r.gaps),
                "round": int(r.round),
                "trips_day": round(r.trips_alone / 365),
                "saved": round(r.saved_alone / 1000),
                "blocks": round(r.blocks_alone / 1000),
                "in_plan": round(r.saved_in_plan / 1000),
                "bbox": [round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
            }
        )
    code = {k: i for i, k in enumerate(KINDS)}
    cells = [
        [int(rank[sid]), 0 if d == "fwd" else 1, int(c) - 1, int(new), code.get(kind, -1)]
        for sid, d, c, new, kind in zip(
            blocks["segment_id"],
            blocks["dir"],
            blocks["corridor"],
            blocks["new"],
            blocks["kind"],
            strict=True,
        )
    ]
    return {**plan["summary"], "kinds": list(KINDS), "corridors": rows, "blocks": cells}


def export(
    seg: gpd.GeoDataFrame,
    fac: pd.DataFrame,
    edges: pd.DataFrame,
    counted: pd.DataFrame | None = None,
    flows: pd.DataFrame | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the viewer's JSON payload.

    ``counted`` (``segment_id, dir, volume_counted``) adds DOT's measured
    volumes where they exist, for display. ``flows`` (the ``assign``
    command's ``flows.parquet``) adds routed trips a year per direction under
    both routings and a summary for the Trips view. ``plan`` adds the
    ``corridors`` command's build-out (see :func:`plan_layer`).
    """
    seg = seg[seg["segment_id"].isin(edges["segment_id"])].reset_index(drop=True)
    attrs = pack_attributes(seg, fac)
    geom = shapely.get_geometry(seg.geometry.simplify(SIMPLIFY_FT).to_crs(4326).values, 0)
    coords = shapely.get_coordinates(geom)
    counts = shapely.get_num_coordinates(geom)
    q = np.rint(coords * QUANT).astype(np.int64)
    origin = q.min(axis=0)
    q -= origin
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(_morton(q[starts, 0] >> 4, q[starts, 1] >> 4), kind="stable")

    flat: list[int] = []
    prev = np.zeros(2, dtype=np.int64)
    for i in order:
        pts = q[starts[i] : starts[i] + counts[i]]
        first = pts[0] - prev
        rest = np.diff(pts, axis=0)
        flat.extend(first.tolist())
        flat.extend(rest.ravel().tolist())
        prev = pts[0]

    names, name_idx = np.unique(seg["street"].to_numpy(), return_inverse=True)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    measured: list[list[int]] = []
    if counted is not None:
        pos = pd.Series(rank, index=seg["segment_id"])
        pos = pos[~pos.index.duplicated()]
        c = counted.dropna(subset=["volume_counted"])
        c = c[c["segment_id"].isin(pos.index)]
        measured = [
            [int(pos[s]), 0 if d == "fwd" else 1, round(v)]
            for s, d, v in zip(c["segment_id"], c["dir"], c["volume_counted"], strict=True)
        ]
    trips: dict[str, Any] = {}
    if flows is not None:
        tf, tr = _per_direction(seg, flows, "trips")
        sf, sr = _per_direction(seg, flows, "trips_shortest")
        trips = {
            "tf": tf[order].tolist(),
            "tr": tr[order].tolist(),
            "sf": sf[order].tolist(),
            "sr": sr[order].tolist(),
            "flow": flow_summary(seg, flows),
        }
    extra: dict[str, Any] = {}
    if plan is not None:
        pos = pd.Series(rank, index=seg["segment_id"])
        extra["plan"] = plan_layer(seg, pos[~pos.index.duplicated()], plan)
    return {
        **trips,
        **extra,
        "lion_release": config.LION_RELEASE,
        "origin": [int(origin[0]), int(origin[1])],
        "quant": QUANT,
        "knobs": knobs(),
        "names": names.tolist(),
        "n": len(order),
        "nv": counts[order].tolist(),
        "c": flat,
        "name": name_idx[order].tolist(),
        "p": attrs["p"].to_numpy()[order].tolist(),
        "lanes": attrs["lanes"].to_numpy()[order].tolist(),
        "speed": attrs["speed"].to_numpy()[order].tolist(),
        "len": attrs["len"].to_numpy()[order].tolist(),
        "vf": attrs["vf"].to_numpy()[order].tolist(),
        "vr": attrs["vr"].to_numpy()[order].tolist(),
        "counted": measured,
    }


def page() -> str:
    """Return the viewer page as a whole document.

    ``viz/index.html`` is written as a page body (the artifact host adds the
    document around it); a standalone copy gets that skeleton here.
    """
    body = (config.ROOT / "viz" / "index.html").read_text()
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '<meta name="description" content="Every NYC street costed by cycling stress, a year of '
        'Citi Bike trips routed over it, and where new protected lanes would help most.">'
        f"</head><body>\n{body}\n</body></html>\n"
    )


def write(payload: dict[str, Any]) -> None:
    """Write the payload next to a copy of the viewer page in ``data/build/viz``."""
    out = config.BUILD_DIR / "viz"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out / "cost.js").write_text((config.ROOT / "viz" / "cost.js").read_text())
    (out / "index.html").write_text(page())
