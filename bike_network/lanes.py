"""NYC DOT's bike routes, reduced to the facility on each side of a segment.

DOT keeps a lane's history: an upgrade retires the old record and adds a new
one, so only ``status == "Current"`` records describe the network today.
``ft_facilit`` is the facility for travel from LION's FROM node to its TO
node and ``tf_facilit`` the other way (DOT's ``bikedir`` R/L/2 agrees with
LION's ``BIKE_TRAFDIR`` FT/TF/TW on every current record that joins). The
``*2`` columns name a second facility in the same direction; the better of
the two wins, as does the better of two records on one segment.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from . import config

if TYPE_CHECKING:
    from pathlib import Path


def _classify(names: pd.Series) -> pd.Series:
    cls = names.map(config.FACILITY_CLASS)
    unknown = names.notna() & (names != "") & cls.isna()
    if unknown.any():
        # A new DOT facility name is a modelling decision, not a default.
        msg = f"unmapped DOT facility names: {sorted(names[unknown].unique())}"
        raise ValueError(msg)
    return cls.fillna("none")


def _best(a: pd.Series, b: pd.Series) -> pd.Series:
    rank = config.FACILITY_RANK
    return a.where(a.map(rank) >= b.map(rank), b)


def facilities(records: pd.DataFrame) -> pd.DataFrame:
    """Return ``segment_id -> fwd, rev`` facility classes from DOT records."""
    cur = records[records["status"] == "Current"].copy()
    for col in ("ft_facilit", "tf_facilit", "ft2facilit", "tf2facilit"):
        if col not in cur:
            cur[col] = None
    out = pd.DataFrame(
        {
            "segment_id": cur["segmentid"].astype(str).str.strip().str.zfill(7),
            "fwd": _best(_classify(cur["ft_facilit"]), _classify(cur["ft2facilit"])),
            "rev": _best(_classify(cur["tf_facilit"]), _classify(cur["tf2facilit"])),
        }
    )
    rank = config.FACILITY_RANK
    inverse = {r: name for name, r in rank.items()}
    agg = out.groupby("segment_id").agg(
        fwd=("fwd", lambda s: max(s.map(rank))),
        rev=("rev", lambda s: max(s.map(rank))),
    )
    return agg.apply(lambda col: col.map(inverse)).reset_index()


def load(path: Path) -> pd.DataFrame:
    """Read the downloaded DOT records and reduce them to facilities."""
    return facilities(pd.DataFrame(json.loads(path.read_text())))
