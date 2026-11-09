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


SHARED = "Shared"


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
            # DOT's literal "Shared" (not the class): how a shared street shows up.
            "fwd_shared": (cur["ft_facilit"] == SHARED) | (cur["ft2facilit"] == SHARED),
            "rev_shared": (cur["tf_facilit"] == SHARED) | (cur["tf2facilit"] == SHARED),
        }
    )
    rank = config.FACILITY_RANK
    inverse = {r: name for name, r in rank.items()}
    agg = out.groupby("segment_id").agg(
        fwd=("fwd", lambda s: max(s.map(rank))),
        rev=("rev", lambda s: max(s.map(rank))),
        fwd_shared=("fwd_shared", "any"),
        rev_shared=("rev_shared", "any"),
    )
    agg["fwd"] = agg["fwd"].map(inverse)
    agg["rev"] = agg["rev"].map(inverse)
    return agg.reset_index()


def load(path: Path) -> pd.DataFrame:
    """Read the downloaded DOT records and reduce them to facilities."""
    return facilities(pd.DataFrame(json.loads(path.read_text())))


def recorded_segments(path: Path) -> set[str]:
    """Every segment DOT has a bike route record for, current or retired."""
    records = pd.DataFrame(json.loads(path.read_text()))
    return set(records["segmentid"].astype(str).str.strip().str.zfill(7))


# LION's BikeLane classes that are a separated path whatever else they combine
# with: 1 Class I, 4 link, and the Class I combinations 5, 8, 9 and 10.
LION_SEPARATED = frozenset({"1", "4", "5", "8", "9", "10"})


def with_lion_paths(
    fac: pd.DataFrame, seg: pd.DataFrame, recorded: set[str] | None = None
) -> pd.DataFrame:
    """Add separated paths LION marks as bike network but DOT has never recorded.

    LION and DOT's bike routes have drifted apart in segment ids (DOT's current
    records name some ids LION 26C does not have, and the reverse), which left
    stretches of greenway and bridge path -- the Williamsburg Bridge's
    Manhattan end, Randalls Island, the East River Esplanade -- walk-only.
    Only separated paths fall back to LION: they are physical and rarely
    removed, while painted and signed on-street lanes do get removed and DOT is
    the authority on those.

    ``recorded`` holds every segment DOT has a record for, current or retired.
    A retired record is DOT saying a block is no longer a facility, and LION
    can lag it: the Pulaski Bridge's pedestrian path was a shared path until
    DOT opened the bike path beside it in 2016, and LION still marks both.
    Reviving it split the bridge's riders across two paths.
    """
    known = set(fac["segment_id"]) | (recorded or set())
    s = seg[
        seg["bike_lane"].isin(LION_SEPARATED)
        & seg["bike_dir"].isin(["FT", "TF", "TW"])
        & ~seg["segment_id"].isin(known)
    ].drop_duplicates("segment_id")
    extra = pd.DataFrame(
        {
            "segment_id": s["segment_id"],
            "fwd": s["bike_dir"].map({"FT": "separated", "TW": "separated"}).fillna("none"),
            "rev": s["bike_dir"].map({"TF": "separated", "TW": "separated"}).fillna("none"),
            "fwd_shared": False,
            "rev_shared": False,
        }
    )
    return pd.concat([fac, extra], ignore_index=True)
