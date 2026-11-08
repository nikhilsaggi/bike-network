"""Paths, sources and every tunable number of the cost model.

The cost model is ``cost = length_m * factor`` on every directed edge, with
``factor = 1 + exposure(facility) * traffic_penalty(edge)`` for riding and a
flat ``WALK_FACTOR`` for walking the bike. Every factor is positive, so
Dijkstra applies and no loop can lower a route's cost.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BUILD_DIR = DATA_DIR / "build"

# --- Sources -----------------------------------------------------------------

# NYC DCP's LION street centreline. Releases are named YYx (26a, 26b, ...) and
# published at a stable path; the DCP download page renders its links in
# script, so the URL is built here rather than scraped.
LION_RELEASE = "26c"
LION_URL = (
    "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/lion/"
    "nyclion_{release}.zip"
)

# NYC DOT's bike routes (Open Data mzxg-pwib). Attributes only: the geometry is
# LION's, joined on segmentid. A response that fills the limit is treated as
# truncated rather than trusted.
BIKE_ROUTES_LIMIT = 200_000
BIKE_ROUTES_URL = (
    "https://data.cityofnewyork.us/resource/mzxg-pwib.json"
    "?$select=segmentid,status,ft_facilit,tf_facilit,ft2facilit,tf2facilit,"
    "facilitycl,onoffst,bikedir,instdate,ret_date"
    f"&$limit={BIKE_ROUTES_LIMIT}&$order=:id"
)

# Citi Bike demand: every station-to-station trip of one calendar year. 2025
# is the latest year whose months all look complete (April and October 2026
# are a fraction of their 2025 size).
DEMAND_YEAR = 2025

# --- Which LION records are network ------------------------------------------

# FeatureTyp: 0 street, 6 private street, A alley, W addressable path, C street
# dedicated for public use. Dropped: railroad, shoreline, boundaries, paper
# streets, ferry routes.
KEEP_FEATURE_TYPES = frozenset({"0", "6", "A", "W", "C"})
# RB_Layer: R roadbed-only and B both are the physical carriageways; N holds
# pedestrian paths and greenways (2,197 current bike-lane records sit there).
# G is the generic centreline of a divided street, which would duplicate its
# roadbeds.
KEEP_RB_LAYERS = frozenset({"B", "R", "N"})
CONSTRUCTED = "2"  # Status
# RW_TYPE codes.
RW_HIGHWAY = 2
RW_STEP_STREET = 7
RW_NON_PHYSICAL = 12
RW_FERRY = 14
DROP_RW_TYPES = frozenset({RW_NON_PHYSICAL, RW_FERRY})

FEET_TO_M = 0.3048

# --- Cost model ----------------------------------------------------------------

# How much of a street's traffic penalty a rider still feels, by the facility
# in their direction of travel. Separated means no car shares the space:
# protected lanes, greenways, boardwalks, DOT-designated bike sidewalks,
# off-street links and plazas. Paint is a line drivers mostly keep clear of.
EXPOSURE = {
    "separated": 0.0,
    "painted": 0.3,
    "shared": 1.0,
    "none": 1.0,
}

# DOT facility names (ft_facilit / tf_facilit and their *2 columns).
FACILITY_CLASS = {
    "Protected": "separated",
    "Boardwalk": "separated",
    "Sidewalk": "separated",
    "Unpaved": "separated",
    "Ped Plaza": "separated",
    "Link": "separated",  # every current Link record is off-street
    "Greenway": "separated",
    "Conventional": "painted",
    "Conventional Buffered": "painted",
    "Curbside": "painted",
    "Curbside Buffered": "painted",
    "Shared": "shared",
    "Signed Route": "shared",
    "Wide Parking Lane": "shared",
}
FACILITY_RANK = {"separated": 3, "painted": 2, "shared": 1, "none": 0}

# Traffic penalty for a street open to cars. Volume is vehicles per day in the
# rider's direction (oncoming, in a contraflow lane), estimated for every street
# by bike_network.traffic from DOT's counts. Each doubling above a quiet street
# adds the same amount, so 3,200 -> 6,400 weighs as much as 6,400 -> 12,800.
TRAFFIC_BASE = 0.2  # any street with cars on it
TRAFFIC_PER_DOUBLING = 1.0  # calibrated: see docs/methodology.md
VOLUME_REFERENCE = 1600  # vehicles/day one way: the median one-lane street's estimate
TRAFFIC_PER_5_MPH = 0.15  # each 5 mph of posted speed above the default
DEFAULT_SPEED_MPH = 25  # NYC's citywide default limit
# TRUCK_ROUTE_TYPE: 1 limited local, 2 local, 3 through.
TRUCK_PENALTY = {"1": 0.2, "2": 0.3, "3": 0.5}

# Walking the bike is legal on any sidewalk in either direction, so every
# segment a pedestrian may use stays traversable. Walking is about four times
# slower than riding; the counters say riders treat it as worse still (the
# hassle of dismounting), so it is calibrated to 8. A step street means
# carrying the bike, set above walking by judgment.
WALK_FACTOR = 8.0  # calibrated: see docs/methodology.md
STEP_FACTOR = 12.0
