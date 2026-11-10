"""Read LION into one row per physical street segment.

LION is NYC DCP's street centerline. Its codes, from the release metadata:

- ``TrafDir``: ``W`` one-way from the FROM node to the TO node, ``A`` one-way
  TO -> FROM, ``T`` two-way, ``P`` pedestrian path, blank for non-streets.
- ``BIKE_TRAFDIR``: ``FT`` / ``TF`` / ``TW``, the same three directions for
  bicycles, set only on DOT's bike network. It is how a contraflow lane on a
  one-way street shows up.
- ``NonPed``: ``V`` is vehicle-only (highways and the like).
- ``Number_Travel_Lanes`` counts the lanes of one carriageway, both directions.
- ``Snow_Priority`` ``C`` marks DSNY's critical routes (arterials, bus and
  truck routes); ``Carto_Display_Level`` 10/20/30 flags the roads DCP draws at
  city, borough and neighborhood scale. Both are used as traffic predictors.

A segment can appear on several rows that differ only in address ranges or
alternate names, sometimes drawn the other way (``66 STREET`` from one node,
``BQE EB ENTRANCE QUEENS BLVD WB`` back from the other, its one-way flag
flipped to match). Rows are deduplicated on the segment and its pair of
nodes in either order, keeping the first, so a segment id names one row and
each direction of it one edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyogrio

from . import config

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

COLUMNS = [
    "SegmentID",
    "Street",
    "FeatureTyp",
    "SegmentTyp",
    "RB_Layer",
    "NonPed",
    "TrafDir",
    "Status",
    "RW_TYPE",
    "NodeIDFrom",
    "NodeIDTo",
    "BikeLane",
    "BIKE_TRAFDIR",
    "POSTED_SPEED",
    "Number_Travel_Lanes",
    "Number_Park_Lanes",
    "TRUCK_ROUTE_TYPE",
    "StreetWidth_Min",
    "Snow_Priority",
    "Carto_Display_Level",
    "LBoro",
    "SHAPE_Length",
]


def _code(s: pd.Series) -> pd.Series:
    """LION pads its coded strings with spaces; normalize to stripped strings."""
    return s.fillna("").astype(str).str.strip()


def _number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_code(s).replace("", np.nan), errors="coerce")


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter and normalize raw LION rows (LION's own column names).

    Returns one row per kept segment with snake_case columns. Works on a plain
    DataFrame or a GeoDataFrame; a geometry column is carried through.
    """
    df = raw.copy()
    for col in ("FeatureTyp", "RB_Layer", "NonPed", "TrafDir", "Status", "BIKE_TRAFDIR"):
        df[col] = _code(df[col])
    rw = _number(df["RW_TYPE"])

    keep = (
        df["FeatureTyp"].isin(config.KEEP_FEATURE_TYPES)
        & df["RB_Layer"].isin(config.KEEP_RB_LAYERS)
        & (df["Status"] == config.CONSTRUCTED)
        & ~rw.isin(config.DROP_RW_TYPES)
        & ((df["TrafDir"] != "") | (df["BIKE_TRAFDIR"] != ""))
    )
    df = df[keep]
    rw = rw[keep]

    out = pd.DataFrame(
        {
            "segment_id": _code(df["SegmentID"]).str.zfill(7),
            "street": _code(df["Street"]),
            "u": _number(df["NodeIDFrom"]).astype("int64"),
            "v": _number(df["NodeIDTo"]).astype("int64"),
            "traf_dir": df["TrafDir"],
            "rb_layer": df["RB_Layer"],
            "bike_dir": df["BIKE_TRAFDIR"],
            "bike_lane": _code(df["BikeLane"]),
            "vehicle_only": df["NonPed"] == "V",
            "rw_type": rw.fillna(0).astype("int64"),
            "speed_mph": _number(df["POSTED_SPEED"]),
            "travel_lanes": _number(df["Number_Travel_Lanes"]),
            "park_lanes": _number(df["Number_Park_Lanes"]),
            "truck_route": _code(df["TRUCK_ROUTE_TYPE"]),
            "width_ft": pd.to_numeric(df["StreetWidth_Min"], errors="coerce"),
            "snow_priority": _code(df["Snow_Priority"]),
            "carto_level": _number(df["Carto_Display_Level"]),
            "boro": _number(df["LBoro"]).astype("Int64"),
            "length_m": df["SHAPE_Length"].astype(float) * config.FEET_TO_M,
        },
        index=df.index,
    )
    if "geometry" in df:
        out["geometry"] = df["geometry"]
    out = out[out["u"] != out["v"]]  # a loop carries no one anywhere
    ends = pd.DataFrame(
        {"lo": out[["u", "v"]].min(axis=1), "hi": out[["u", "v"]].max(axis=1)}, index=out.index
    )
    alias = pd.concat([out["segment_id"], ends], axis=1).duplicated()
    return out[~alias].reset_index(drop=True)


def read(gdb: Path, *, geometry: bool = True) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read and prepare LION's ``lion`` layer from the file geodatabase."""
    raw = pyogrio.read_dataframe(gdb, layer="lion", columns=COLUMNS, read_geometry=geometry)
    out = prepare(raw)
    if geometry:
        import geopandas as gpd  # noqa: PLC0415 -- only needed with geometry

        out = gpd.GeoDataFrame(out, geometry="geometry", crs=raw.crs)
    return out


def read_nodes(gdb: Path) -> pd.DataFrame:
    """Return LION node coordinates as ``node, x, y`` in LION's CRS (EPSG:2263)."""
    nodes = pyogrio.read_dataframe(gdb, layer="node")
    id_col = next(c for c in nodes.columns if c.lower() == "nodeid")
    return pd.DataFrame(
        {
            "node": pd.to_numeric(nodes[id_col]).astype("int64"),
            "x": nodes.geometry.x,
            "y": nodes.geometry.y,
        }
    ).drop_duplicates("node")
