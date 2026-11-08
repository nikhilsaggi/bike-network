"""Fit the cost model's main knobs to DOT's automated bike counters.

DOT runs permanent bike counters on the East River bridges, a few avenue
lanes and some greenways (Open Data ``smn3-rzf9`` for where, ``uczf-rk3c``
for the 15-minute counts). They count every bike; the routed trips are Citi
Bike's alone, a share of all riding that is roughly steady from place to
place. So a knob setting is scored by how much the ratio of routed trips to
counted bikes varies across counters -- the spread (standard deviation) of
``log(routed / counted)`` -- not by matching totals. Lower is better, and a
setting has to beat the shortest legal route's spread to be worth keeping.

Two things besides route choice move that ratio, and the score removes both:

- Citi Bike's share of all riding is higher where it is denser. The score
  regresses the log ratio on the log of Citi Bike trip ends within
  ``STATION_RADIUS_M`` of the counter and keeps the residual spread.
- A counter only Citi Bike routes rarely pass (the outer-borough greenways)
  is ratio noise. The scored counters are fixed once, as those the shortest
  routing sends ``MIN_ROUTED_PER_DAY`` trips through, so every knob setting
  is judged on the same set.

Each counter is tied to the LION street it counts, named below rather than
found by proximity alone: DOT gives Columbus Ave at 86th St the same
coordinates as Amsterdam Ave at 86th St. A bridge counter sees every
crossing, so it is compared with the busiest block of the bridge's bike
facility; in the network, riders join a bridge path at access points along
its approach that in reality lie before the counter.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from . import config
from .sources import _download

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

COUNTERS_URL = "https://data.cityofnewyork.us/resource/smn3-rzf9.json?$limit=500"
COUNTS_URL = "https://data.cityofnewyork.us/resource/uczf-rk3c.json"
INTERVALS_PER_DAY = 96  # 15-minute counts

# (DOT counter id, the LION street it counts, a cross street that locates the
# block, or None to take the street's facility block nearest the counter).
# One counter per site, bikes only where DOT splits bikes from pedestrians.
COUNTERS = [
    ("100009427", "WILLIAMSBURG BRIDGE BIKE PATH", None),
    ("100009428", "ED KOCH QUEENSBORO BR BIKE PATH", None),
    ("100062893", "MANHATTAN BRIDGE BIKE PATH", None),
    ("300020904", "BROOKLYN BRIDGE", None),
    ("100010019", "KENT AVENUE", "NORTH 8 STREET"),
    ("100010018", "PULASKI BRIDGE BIKE PATH", None),
    ("300029647", "WILLIS AVENUE BRIDGE", None),
    ("100057316", "8 AVENUE", "WEST 50 STREET"),
    ("100057319", "AMSTERDAM AVENUE", "WEST 86 STREET"),
    ("100057320", "COLUMBUS AVENUE", "WEST 86 STREET"),
    ("100009425", "PROSPECT PARK WEST", None),
    ("300043078", "HIGH BRIDGE GREENWAY", None),
    ("300041240", "OCEAN PARKWAY BIKE PATH", None),
    ("300043076", "SHORE PARKWAY GREENWAY", None),
    ("300043074", "CONCRETE PLANT PARK GREENWAY", None),
    ("300024007", "111 STREET", "50 AVENUE"),
    ("300042336", "PELHAM PARKWAY GREENWAY", None),
]
# Counters on these streets count a crossing: compared with the busiest block.
CROSSINGS = frozenset(
    {
        "WILLIAMSBURG BRIDGE BIKE PATH",
        "ED KOCH QUEENSBORO BR BIKE PATH",
        "MANHATTAN BRIDGE BIKE PATH",
        "BROOKLYN BRIDGE",
        "PULASKI BRIDGE BIKE PATH",
        "WILLIS AVENUE BRIDGE",
    }
)
STATION_RADIUS_M = 1000.0
MIN_ROUTED_PER_DAY = 100.0


def ensure_counts(year: int) -> tuple[Path, Path]:
    """Return paths to the counter list and one year's per-counter totals."""
    folder = config.RAW_DIR / "bike_counters"
    where = f"date between '{year}-01-01T00:00:00' and '{year}-12-31T23:59:59'"
    params = {
        "$select": "id,sum(counts) as n,count(*) as k",
        "$where": where,
        "$group": "id",
        "$limit": "500",
    }
    paths = (folder / "counters.json", folder / f"counts_{year}.json")
    urls = (COUNTERS_URL, f"{COUNTS_URL}?{urllib.parse.urlencode(params)}")
    for path, url in zip(paths, urls, strict=True):
        if not path.exists():
            _download(url, path)
    return paths


def counter_table(year: int) -> pd.DataFrame:
    """Return the chosen counters: name, position, and mean bikes per counted day."""
    where, counts = ensure_counts(year)
    sites = pd.DataFrame(json.loads(where.read_text())).drop_duplicates("id").set_index("id")
    totals = pd.DataFrame(json.loads(counts.read_text())).set_index("id")
    rows = []
    for cid, street, cross in COUNTERS:
        n = float(totals.loc[cid, "n"]) if cid in totals.index else np.nan
        k = float(totals.loc[cid, "k"]) if cid in totals.index else np.nan
        rows.append(
            {
                "id": cid,
                "name": sites.loc[cid, "name"],
                "lat": float(sites.loc[cid, "latitude"]),
                "lng": float(sites.loc[cid, "longitude"]),
                "street": street,
                "cross": cross,
                "per_day": n / (k / INTERVALS_PER_DAY) if k else np.nan,
                "coverage": k / (365 * INTERVALS_PER_DAY),
            }
        )
    return pd.DataFrame(rows)


def locate(counters: pd.DataFrame, seg: gpd.GeoDataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Tie each counter to the LION segment it counts (``segment_id``, ``snap_m``)."""
    import geopandas as gpd  # noqa: PLC0415 -- runtime use only here
    from shapely.geometry import Point  # noqa: PLC0415

    with_lane = set(edges.loc[edges["facility"] != "none", "segment_id"])
    points = gpd.GeoSeries(
        [Point(xy) for xy in zip(counters["lng"], counters["lat"], strict=True)], crs=4326
    ).to_crs(seg.crs)
    found = []
    for (_, c), pt in zip(counters.iterrows(), points, strict=True):
        cand = seg[(seg["street"] == c["street"]) & seg["segment_id"].isin(with_lane)]
        if pd.notna(c["cross"]):
            cross = seg[seg["street"] == c["cross"]]
            nodes = set(cross["u"]) | set(cross["v"])
            cand = cand[cand["u"].isin(nodes) | cand["v"].isin(nodes)]
        if cand.empty:
            msg = f"no bike-facility block of {c['street']} for counter {c['id']} ({c['name']})"
            raise ValueError(msg)
        dist = cand.geometry.distance(pt)
        best = dist.idxmin()
        found.append((cand.loc[best, "segment_id"], dist[best] * config.FEET_TO_M))
    out = counters.copy()
    out["segment_id"] = [f[0] for f in found]
    out["snap_m"] = [f[1] for f in found]
    return out


def trip_ends_near(counters: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """Citi Bike trip ends a year at stations within ``STATION_RADIUS_M`` of each counter."""
    lat0 = np.radians(40.7)

    def xy(lat: pd.Series, lng: pd.Series) -> np.ndarray:
        return np.column_stack([np.radians(lng) * np.cos(lat0), np.radians(lat)]) * 6_371_000

    tree = cKDTree(xy(stations["lat"], stations["lng"]))
    hits = tree.query_ball_point(xy(counters["lat"], counters["lng"]), STATION_RADIUS_M)
    ends = stations["ends"].to_numpy()
    return np.array([ends[h].sum() for h in hits], dtype=float)


def crossing_segments(
    counters: pd.DataFrame, seg: pd.DataFrame, edges: pd.DataFrame
) -> list[list[str]]:
    """Return the segments each counter is compared over: its block, or a bridge's facility."""
    with_lane = set(edges.loc[edges["facility"] != "none", "segment_id"])
    out = []
    for c in counters.itertuples():
        if c.street in CROSSINGS:
            ids = seg.loc[
                (seg["street"] == c.street) & seg["segment_id"].isin(with_lane), "segment_id"
            ]
            out.append(sorted(set(ids)))
        else:
            out.append([c.segment_id])
    return out


def routed_at(segments: list[list[str]], edges: pd.DataFrame, edge_trips: np.ndarray) -> np.ndarray:
    """Routed trips a day at each counter: both directions, busiest of its segments."""
    per_seg = pd.Series(edge_trips / 365).groupby(edges["segment_id"].to_numpy()).sum()
    return np.array([per_seg.reindex(ids).fillna(0).max() for ids in segments])


def score(routed: np.ndarray, counted: np.ndarray, ends: np.ndarray) -> dict[str, float]:
    """Spread of log(routed / counted) once local Citi Bike density is regressed out.

    ``raw`` is the spread before that adjustment; ``slope`` how strongly
    Citi Bike's share rises with its local trip ends.
    """
    ratio = np.log(np.maximum(routed, 1.0) / counted)
    design = np.column_stack([np.ones_like(ends), np.log(ends)])
    coef = np.linalg.lstsq(design, ratio, rcond=None)[0]
    resid = ratio - design @ coef
    return {
        "spread": float(np.sqrt((resid**2).sum() / (len(ratio) - design.shape[1]))),
        "raw": float(np.std(ratio, ddof=1)),
        "slope": float(coef[1]),
        "spearman": float(spearmanr(routed, counted).statistic),
    }
