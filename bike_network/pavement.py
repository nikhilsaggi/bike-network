"""Pavement condition per street segment, from DOT's street pavement ratings.

DOT inspects every street block and rates its pavement 1-10 (Open Data
``6yyb-pb25``). A block is identified by its on, from and to street codes,
not LION's segment id, but its line is LION's own: a matched segment's
midpoint lies on the rated line (median offset well under a foot). So each
LION car segment takes the rating of the nearest rated line that runs the
same way, within ``MATCH_FT``. A divided street's roadbeds (``rb_layer`` R)
are rated on the street's centerline, which sits between them, so they look
further, to ``ROADBED_MATCH_FT``. Alleys and driveways are not rated, and
looking further for them would only find the street beside them.

Only the latest rating of each block counts: pavement is resurfaced, and a
2019 rating says little about a street repaved since. A rating of 0 is DOT's
way of saying it did not rate the block (construction, a duplicate record).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from . import config
from .network import CAR_DIRS
from .sources import USER_AGENT

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

PAGE = 100_000
MATCH_FT = 40.0
ROADBED_MATCH_FT = 150.0
# |cos| of the angle between a segment and the rated line; 0.9 is about 25 degrees.
MIN_ALIGNMENT = 0.9
TANGENT_FT = 10.0  # half the run over which a line's direction is measured
# One block, one rating: a multi-pass street is rated once per roadway.
BLOCK = ["oftcode", "direction", "road_type"]


def _fetch_page(offset: int) -> list[dict[str, Any]]:
    params = {
        "$select": "the_geom,oftcode,direction,road_type,systemrating,inspection",
        "$where": f"systemrating > 0 AND inspection >= '{config.PAVEMENT_SINCE}'",
        "$order": ":id",
        "$limit": str(PAGE),
        "$offset": str(offset),
    }
    url = f"{config.PAVEMENT_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 -- fixed https host
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        return json.loads(resp.read())["features"]


def ensure_ratings() -> Path:
    """Return the path to every rating since ``PAVEMENT_SINCE``, downloading if absent."""
    path = config.RAW_DIR / "pavement" / f"ratings_since_{config.PAVEMENT_SINCE[:4]}.geojson"
    if path.exists():
        return path
    print("  Downloading DOT street pavement ratings...")
    features: list[dict[str, Any]] = []
    while True:
        page = _fetch_page(len(features))
        features.extend(page)
        if len(page) < PAGE:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    tmp.replace(path)
    return path


def latest(ratings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Each block's latest rating, as ``rating, inspected, geometry`` in LION's CRS."""
    r = ratings.assign(
        rating=pd.to_numeric(ratings["systemrating"]),
        inspected=pd.to_datetime(ratings["inspection"]),
    )
    for col in BLOCK:
        r[col] = r[col].fillna("")
    r = r[(r["rating"] > 0) & r.geometry.notna()]
    r = r.sort_values("inspected").drop_duplicates(BLOCK, keep="last")
    if r.crs is None:
        r = r.set_crs(4326)
    return r[["rating", "inspected", "geometry"]].to_crs(2263).reset_index(drop=True)


def _tangent(lines: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit direction of each line where it passes nearest each point."""
    along = shapely.line_locate_point(lines, points)
    length = shapely.length(lines)
    a = shapely.line_interpolate_point(lines, np.clip(along - TANGENT_FT, 0, length))
    b = shapely.line_interpolate_point(lines, np.clip(along + TANGENT_FT, 0, length))
    dx = shapely.get_x(b) - shapely.get_x(a)
    dy = shapely.get_y(b) - shapely.get_y(a)
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 0, norm, np.nan)
    return dx / norm, dy / norm


def _single_line(geoms: np.ndarray) -> np.ndarray:
    """Return one LineString per geometry: its parts merged where they join, else the first."""
    merged = shapely.line_merge(geoms)
    return np.where(shapely.get_type_id(merged) == 1, merged, shapely.get_geometry(geoms, 0))


def _nearest_aligned(
    lines: np.ndarray, rated: gpd.GeoDataFrame, tree: shapely.STRtree, within: float
) -> pd.DataFrame:
    """For each line, the nearest rated line within ``within`` feet that runs the same way."""
    mid = shapely.line_interpolate_point(lines, 0.5, normalized=True)
    si, ri = tree.query(mid, predicate="dwithin", distance=within)
    near = _single_line(rated.geometry.to_numpy()[ri])
    sx, sy = _tangent(_single_line(lines[si]), mid[si])
    rx, ry = _tangent(near, mid[si])
    pairs = pd.DataFrame(
        {
            "si": si,
            "ri": ri,
            "dist": shapely.distance(mid[si], near),
            "cos": np.abs(sx * rx + sy * ry),
        }
    )
    pairs = pairs[pairs["cos"] >= MIN_ALIGNMENT]
    return pairs.sort_values(["dist", "ri"]).drop_duplicates("si")


def match(seg: gpd.GeoDataFrame, rated: gpd.GeoDataFrame) -> pd.Series:
    """Pavement rating for each row of ``seg`` (NaN where none applies).

    Only segments cars use are matched: a rating describes a roadway, and a
    greenway or path beside one is not on it.
    """
    out = pd.Series(np.nan, index=seg.index, name="pavement")
    car = (seg["traf_dir"].isin(CAR_DIRS) & ~seg["vehicle_only"]).to_numpy()
    if not car.any() or rated.empty:
        return out
    tree = shapely.STRtree(rated.geometry.to_numpy())
    lines = seg.geometry.to_numpy()
    ratings = rated["rating"].to_numpy()
    rows = np.flatnonzero(car)
    hit = _nearest_aligned(lines[rows], rated, tree, MATCH_FT)
    out.iloc[rows[hit["si"].to_numpy()]] = ratings[hit["ri"].to_numpy()]
    roadbed = car & out.isna().to_numpy() & (seg["rb_layer"] == "R").to_numpy()
    rows = np.flatnonzero(roadbed)
    if len(rows):
        hit = _nearest_aligned(lines[rows], rated, tree, ROADBED_MATCH_FT)
        out.iloc[rows[hit["si"].to_numpy()]] = ratings[hit["ri"].to_numpy()]
    return out


def ratings_for(seg: gpd.GeoDataFrame) -> pd.Series:
    """Download (once), reduce and match DOT's ratings to ``seg``."""
    import geopandas as gpd  # noqa: PLC0415 -- geometry is only needed here

    return match(seg, latest(gpd.read_file(ensure_ratings())))
