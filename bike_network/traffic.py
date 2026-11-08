"""Traffic volume per direction of travel: measured where DOT counted, estimated elsewhere.

DOT's Automated Traffic Volume Counts (Open Data 7ym2-wayt) are road-tube
counts, usually a week or two long, in 15-minute intervals on one LION
segment and one compass direction. They are downloaded already summed to
daily totals per count, segment, direction and day.

A count's compass direction (NB/SB/EB/WB) becomes ``fwd`` or ``rev`` by
comparing it with the segment's own bearing from FROM to TO node. Counts on
a segment running nearly perpendicular to the counted direction are dropped,
as are the few two-way (EW/NS) totals.

The estimate is a gradient-boosted model of log daily volume from what LION
records on every street. :func:`evaluate` compares it, on streets held out
of the fit, with the two things it would replace: lane count alone, and the
rule-based traffic penalty the network uses today.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from . import config
from .network import CAR_DIRS, traffic_penalty
from .sources import USER_AGENT

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

COUNTS_URL = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
COUNTS_SINCE = 2010
PAGE = 50_000
COMPASS = {"NB": (0.0, 1.0), "SB": (0.0, -1.0), "EB": (1.0, 0.0), "WB": (-1.0, 0.0)}
# |cos| between the counted compass direction and the segment's bearing below
# this, and the count cannot say which way along the segment it was taken.
MIN_ALIGNMENT = 0.3
# A full day is 96 fifteen-minute intervals; daylight-saving days have 92 or 100.
FULL_DAY = (92, 100)
SNOW_CODE = {"C": 1, "S": 2, "V": 3, "H": 4}


def _fetch_page(offset: int) -> list[dict[str, Any]]:
    params = {
        "$select": "requestid,segmentid,direction,yr,m,d,sum(vol) as vol,count(*) as n",
        "$where": f"yr >= {COUNTS_SINCE} AND direction in ('NB','SB','EB','WB')",
        "$group": "requestid,segmentid,direction,yr,m,d",
        "$order": "requestid,segmentid,direction,yr,m,d",
        "$limit": str(PAGE),
        "$offset": str(offset),
    }
    url = f"{COUNTS_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 -- fixed https host
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        return json.loads(resp.read())


def ensure_counts() -> Path:
    """Return the path to daily count totals, downloading them if absent."""
    path = config.RAW_DIR / "traffic_counts" / f"daily_since_{COUNTS_SINCE}.json"
    if path.exists():
        return path
    print("  Downloading DOT traffic counts (daily totals)...")
    rows: list[dict[str, Any]] = []
    while True:
        page = _fetch_page(len(rows))
        rows.extend(page)
        if len(page) < PAGE:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)
    return path


def daily_volumes(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean full-day volume per (segment, compass direction)."""
    df = rows.assign(
        segment_id=rows["segmentid"].astype(str).str.strip().str.zfill(7),
        vol=pd.to_numeric(rows["vol"]),
        n=pd.to_numeric(rows["n"]),
        yr=pd.to_numeric(rows["yr"]),
    )
    df = df[df["n"].between(*FULL_DAY)]
    df = df.assign(day_vol=df["vol"] * 96 / df["n"])
    return (
        df.groupby(["segment_id", "direction"])
        .agg(volume=("day_vol", "mean"), days=("day_vol", "size"), last_year=("yr", "max"))
        .reset_index()
    )


def orient(counts: pd.DataFrame, seg: gpd.GeoDataFrame) -> pd.DataFrame:
    """Turn compass directions into ``fwd``/``rev`` against each segment's geometry."""
    geom = seg.set_index("segment_id").geometry
    geom = geom[~geom.index.duplicated()]
    lines = geom.reindex(counts["segment_id"]).to_numpy()
    known = ~pd.isna(lines)
    counts, lines = counts[known].copy(), lines[known]
    parts = shapely.get_geometry(lines, 0)
    start = shapely.get_point(parts, 0)
    end = shapely.get_point(parts, -1)
    dx = shapely.get_x(end) - shapely.get_x(start)
    dy = shapely.get_y(end) - shapely.get_y(start)
    norm = np.hypot(dx, dy)
    cx = counts["direction"].map(lambda c: COMPASS[c][0]).to_numpy()
    cy = counts["direction"].map(lambda c: COMPASS[c][1]).to_numpy()
    cos = (dx * cx + dy * cy) / np.where(norm > 0, norm, np.nan)
    counts["dir"] = np.where(cos > 0, "fwd", "rev")
    counts = counts[np.abs(cos) >= MIN_ALIGNMENT]
    # Two compass directions can fold onto one edge (a diagonal street counted
    # NB and EB); keep the better-measured.
    return (
        counts.sort_values("days", ascending=False)
        .drop_duplicates(["segment_id", "dir"])
        .drop(columns="direction")
    )


def car_edges(seg: pd.DataFrame) -> pd.DataFrame:
    """One row per direction cars may travel, with the predictors the model uses."""
    parts = []
    for direction, with_traffic in (("fwd", ("T", "W")), ("rev", ("T", "A"))):
        s = seg[seg["traf_dir"].isin(with_traffic) & ~seg["vehicle_only"]]
        parts.append(s.assign(dir=direction))
    df = pd.concat(parts, ignore_index=True)
    two_way = df["traf_dir"] == "T"
    lanes = df["travel_lanes"].where(df["travel_lanes"] > 0)
    feats = pd.DataFrame(
        {
            "segment_id": df["segment_id"],
            "dir": df["dir"],
            "street": df["street"],
            "boro": df["boro"].fillna(0).astype(int),
            "lanes_dir": np.ceil(lanes / np.where(two_way, 2, 1)),
            "lanes_total": lanes,
            "one_way": (~two_way).astype(int),
            "speed": df["speed_mph"],
            "truck": pd.to_numeric(df["truck_route"], errors="coerce").fillna(0).astype(int),
            "width_ft": df["width_ft"].where(df["width_ft"] > 0),
            "park_lanes": df["park_lanes"],
            "snow": df["snow_priority"].map(SNOW_CODE).fillna(0).astype(int),
            "carto": df["carto_level"].map({10: 3, 20: 2, 30: 1}).fillna(0).astype(int),
            "bridge": (df["rw_type"] == 3).astype(int),
            "rule_penalty": traffic_penalty(df).to_numpy(),
        }
    )
    return feats[df["traf_dir"].isin(CAR_DIRS).to_numpy()]


FEATURES = [
    "boro",
    "lanes_dir",
    "lanes_total",
    "one_way",
    "speed",
    "truck",
    "width_ft",
    "park_lanes",
    "snow",
    "carto",
    "bridge",
]
CATEGORICAL = ["boro", "truck", "snow"]


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=30,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        random_state=0,
    )


@dataclass
class Score:
    """Hold-out fit of one way of predicting log daily volume."""

    name: str
    r2: float
    spearman: float
    median_factor: float  # typical error, as a multiple of the true volume


def _score(name: str, y: np.ndarray, pred: np.ndarray) -> Score:
    resid = y - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    return Score(name, r2, spearmanr(y, pred).statistic, float(np.exp(np.median(np.abs(resid)))))


def evaluate(train: pd.DataFrame, folds: int = 5) -> list[Score]:
    """Score each predictor by cross-validation that never splits a street across folds.

    Adjacent blocks of one street carry nearly the same traffic, so holding
    out single segments would let the model look up its neighbour's answer.
    """
    y = np.log(train["volume"].to_numpy())
    groups = (train["boro"].astype(str) + "|" + train["street"]).to_numpy()
    lanes_only = (
        train["lanes_dir"].fillna(config.DEFAULT_LANES_PER_DIR).clip(1, 4).to_numpy()[:, None]
    )
    lanes_onehot = (lanes_only == np.arange(1, 5)).astype(float)
    rule = train["rule_penalty"].to_numpy()[:, None]
    x = train[FEATURES].to_numpy(dtype=float)
    preds = {name: np.zeros_like(y) for name in ("lanes", "rule", "model")}
    for fit, test in GroupKFold(n_splits=folds).split(x, y, groups):
        preds["lanes"][test] = (
            LinearRegression().fit(lanes_onehot[fit], y[fit]).predict(lanes_onehot[test])
        )
        preds["rule"][test] = LinearRegression().fit(rule[fit], y[fit]).predict(rule[test])
        preds["model"][test] = _model().fit(x[fit], y[fit]).predict(x[test])
    return [
        _score("Lanes per direction alone", y, preds["lanes"]),
        _score("Today's rule-based penalty", y, preds["rule"]),
        _score("Volume model (all features)", y, preds["model"]),
    ]


def estimate(edges: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Fit on every counted edge and predict daily volume for every car edge."""
    model = _model().fit(train[FEATURES].to_numpy(dtype=float), np.log(train["volume"].to_numpy()))
    out = edges[["segment_id", "dir"]].copy()
    out["volume_est"] = np.exp(model.predict(edges[FEATURES].to_numpy(dtype=float)))
    counted = train.set_index(["segment_id", "dir"])["volume"]
    out["volume_counted"] = counted.reindex(
        pd.MultiIndex.from_frame(out[["segment_id", "dir"]])
    ).to_numpy()
    return out


def training_set(seg: gpd.GeoDataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all car edges with features, the counted subset with ``volume``)."""
    edges = car_edges(seg)
    counts = orient(daily_volumes(rows), seg)
    train = edges.merge(counts, on=["segment_id", "dir"], how="inner")
    return edges, train[train["volume"] > 0]
