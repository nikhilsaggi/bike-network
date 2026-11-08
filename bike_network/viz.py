"""Export the network for the edge-cost viewer in ``viz/``.

The viewer recomputes every edge's cost in the browser (``viz/cost.js``) so
the knobs in ``config.py`` can be moved live; it therefore ships the *inputs*
of the cost model per segment, not the costs. ``tests/test_viz.py`` holds the
two implementations to the same answer.

Geometry is WGS84 quantised to 1e-5 degrees (about a metre), each segment's
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
    packed = (
        traf
        | (truck << 2)
        | np.where(rw == config.RW_STEP_STREET, BIT_STEP, 0)
        | np.where(seg["vehicle_only"].to_numpy(), BIT_VEHICLE_ONLY, 0)
        | np.where(rw == config.RW_HIGHWAY, BIT_HIGHWAY, 0)
        | (fwd << 7)
        | (rev << 9)
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


def export(
    seg: gpd.GeoDataFrame,
    fac: pd.DataFrame,
    edges: pd.DataFrame,
    counted: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the viewer's JSON payload.

    ``counted`` (``segment_id, dir, volume_counted``) adds DOT's measured
    volumes where they exist, for display.
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
    return {
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


def write(payload: dict[str, Any]) -> None:
    """Write the payload next to a copy of the viewer page in ``data/build/viz``."""
    out = config.BUILD_DIR / "viz"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out / "cost.js").write_text((config.ROOT / "viz" / "cost.js").read_text())
    # viz/index.html is written as a page body (the artifact host adds the
    # document around it); a local copy gets that skeleton here.
    page = (config.ROOT / "viz" / "index.html").read_text()
    (out / "index.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        f"</head><body>\n{page}\n</body></html>\n"
    )
