"""Citi Bike trips as station-to-station demand, snapped onto the network.

Citi Bike publishes every trip, monthly, at https://s3.amazonaws.com/tripdata
(the ``JC-`` files are Jersey City's system and are not used). A trip names
its start and end station and their coordinates, never the route, so this
reduces each month to trip counts per (start station, end station) and a
mean position per station. Each month's reduction is cached, so the 8 GB of
zips are read once.

Dropped: trips with no end station (lost or stolen bikes) and trips that
return to the station they left, which are loops rather than journeys and
have no route to assign.
"""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config
from .sources import _download

if TYPE_CHECKING:
    from pathlib import Path

TRIPDATA_URL = "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip"
COLUMNS = ["start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng"]
CHUNK = 1_000_000
# A station further than this from any network node is not on the network
# (none should be; the guard is for a bad coordinate).
MAX_SNAP_M = 150.0


def months(year: int) -> list[str]:
    """Return the twelve ``YYYYMM`` names of a calendar year."""
    return [f"{year}{m:02d}" for m in range(1, 13)]


def ensure_month(month: str) -> Path:
    """Return the path to one month's trip zip, downloading it if absent."""
    path = config.RAW_DIR / "citibike" / f"{month}-citibike-tripdata.zip"
    if not path.exists():
        print(f"  Downloading Citi Bike {month}...")
        _download(TRIPDATA_URL.format(month=month), path, timeout=1800)
    return path


def reduce_trips(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (OD counts, per-station coordinate sums) for one batch of trips."""
    t = trips.dropna(subset=["start_station_id", "end_station_id"])
    t = t.assign(
        start_station_id=t["start_station_id"].astype(str).str.strip(),
        end_station_id=t["end_station_id"].astype(str).str.strip(),
    )
    t = t[t["start_station_id"] != t["end_station_id"]]
    od = t.groupby(["start_station_id", "end_station_id"]).size().rename("trips").reset_index()
    ends = pd.concat(
        [
            t[["start_station_id", "start_lat", "start_lng"]].set_axis(
                ["station", "lat", "lng"], axis=1
            ),
            t[["end_station_id", "end_lat", "end_lng"]].set_axis(["station", "lat", "lng"], axis=1),
        ]
    ).dropna()
    pos = ends.groupby("station").agg(lat=("lat", "sum"), lng=("lng", "sum"), n=("lat", "size"))
    return od, pos.reset_index()


def reduce_month(month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one month's (OD counts, station coordinate sums), cached."""
    cache = config.BUILD_DIR / "demand" / month
    od_path, pos_path = cache / "od.parquet", cache / "stations.parquet"
    if od_path.exists() and pos_path.exists():
        return pd.read_parquet(od_path), pd.read_parquet(pos_path)
    ods, poss = [], []
    with zipfile.ZipFile(ensure_month(month)) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            with zf.open(name) as f:
                reader = pd.read_csv(
                    f,
                    usecols=COLUMNS,
                    dtype={"start_station_id": str, "end_station_id": str},
                    chunksize=CHUNK,
                )
                for chunk in reader:
                    od, pos = reduce_trips(chunk)
                    ods.append(od)
                    poss.append(pos)
    od = (
        pd.concat(ods)
        .groupby(["start_station_id", "end_station_id"], as_index=False)["trips"]
        .sum()
    )
    pos = pd.concat(poss).groupby("station", as_index=False)[["lat", "lng", "n"]].sum()
    cache.mkdir(parents=True, exist_ok=True)
    od.to_parquet(od_path)
    pos.to_parquet(pos_path)
    return od, pos


def year_demand(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a year's (OD trips, stations with mean position and trip ends)."""
    ods, poss = [], []
    for month in months(year):
        od, pos = reduce_month(month)
        print(f"  {month}: {od['trips'].sum():,} trips between stations")
        ods.append(od)
        poss.append(pos)
    od = (
        pd.concat(ods)
        .groupby(["start_station_id", "end_station_id"], as_index=False)["trips"]
        .sum()
    )
    pos = pd.concat(poss).groupby("station", as_index=False)[["lat", "lng", "n"]].sum()
    stations = pd.DataFrame(
        {
            "station": pos["station"],
            "lat": pos["lat"] / pos["n"],
            "lng": pos["lng"] / pos["n"],
            "ends": pos["n"],
        }
    )
    return od, stations


def snap(stations: pd.DataFrame, nodes: pd.DataFrame, allowed: np.ndarray) -> pd.DataFrame:
    """Attach each station to the nearest node in ``allowed`` (LION node ids).

    ``nodes`` is :func:`lion.read_nodes` output (EPSG:2263 feet). Stations
    further than ``MAX_SNAP_M`` from every allowed node get ``node = -1``.
    """
    from pyproj import Transformer  # noqa: PLC0415 -- only snapping projects

    to_ft = Transformer.from_crs(4326, 2263, always_xy=True)
    x, y = to_ft.transform(stations["lng"].to_numpy(), stations["lat"].to_numpy())
    pool = nodes[nodes["node"].isin(allowed)]
    dist_ft, idx = cKDTree(pool[["x", "y"]].to_numpy()).query(np.column_stack([x, y]))
    out = stations.copy()
    out["snap_m"] = dist_ft * config.FEET_TO_M
    out["node"] = np.where(out["snap_m"] <= MAX_SNAP_M, pool["node"].to_numpy()[idx], -1)
    return out
