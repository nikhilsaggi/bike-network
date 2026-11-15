"""Paths, sources and every tunable number of the cost model.

The cost model is ``cost = length_m * factor`` on every directed edge, with
``factor = 1 + exposure(facility) * felt_penalty(edge) + pavement_term(edge)``
for riding, where the felt penalty is the traffic penalty adjusted for room
where the rider shares the lane with cars, and a flat ``WALK_FACTOR`` for walking the
bike. Every factor is positive, so
Dijkstra applies and no loop can lower a route's cost.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BUILD_DIR = DATA_DIR / "build"

# --- Sources -----------------------------------------------------------------

# NYC DCP's LION street centerline. Releases are named YYx (26a, 26b, ...) and
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

# NYC DOT's street pavement ratings (Open Data 6yyb-pb25): 1-10 per block,
# keyed by on/from/to street codes rather than LION's segment id, so they are
# matched to LION by geometry. DOT moved to its Pavement Condition Index
# inspections in 2018; older ratings are not on the same scale.
PAVEMENT_URL = "https://data.cityofnewyork.us/resource/6yyb-pb25.geojson"
PAVEMENT_SINCE = "2018-01-01"

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
# G is the generic centerline of a divided street, which would duplicate its
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

# Room to be passed. The same traffic is easier to share with more roadway in
# the rider's direction: curb to curb, less parking, all of it on a one-way
# street and half on a two-way one. Where a rider shares the lane with cars
# (no facility, or sharrows), the traffic penalty is scaled by
# (ROOM_REFERENCE / room) ** ROOM_EXPONENT, with room held to [ROOM_MIN,
# ROOM_MAX], and room takes at most ROOM_MAX_DISCOUNT off it: a wide arterial
# is roomier, but its traffic is still an arterial's. Past two lanes, more
# width brings more lanes of traffic, not more room for the rider, and below
# ROOM_MIN the width is likely a data error. A painted lane's stripe is the
# rider's room, and the curb-to-curb width counts the lane itself, so paint
# keeps its exposure whatever the width.
ROOM_REFERENCE_FT = 12.0  # the median one-lane-per-direction street (one-way 14, two-way 9)
ROOM_MIN_FT = 7.0
ROOM_MAX_FT = 24.0  # two travel lanes
ROOM_EXPONENT = 1.0  # see docs/methodology.md
ROOM_MAX_DISCOUNT = 1.25  # judgment: a quarter of a busy arterial's penalty
PARKING_LANE_FT = 8.0
PARK_LANES_UNKNOWN = 2  # LION leaves a quarter of streets blank; 2 is by far the most common

# Rough pavement. DOT rates 8-10 good, 4-7 fair and 1-3 poor. Each point below
# good adds the same to the riding factor, whatever the traffic. Streets with
# no rating (about one block in eight) take none.
PAVEMENT_GOOD = 8.0
PAVEMENT_PER_POINT = 0.05  # judgment: a poor street (3) costs 1.25x a good one

# Walking the bike is legal on any sidewalk in either direction, so every
# segment a pedestrian may use stays traversable. Walking is about four times
# slower than riding; the counters say riders treat it as worse still (the
# hassle of dismounting), so it is calibrated to 8. A step street means
# carrying the bike, set above walking by judgment.
WALK_FACTOR = 8.0  # calibrated: see docs/methodology.md
STEP_FACTOR = 12.0

# --- Corridors (bike_network.corridors) ------------------------------------------

# A sharper bend ends a street's run. Broadway's own bends (Madison Square,
# Columbus Circle) reach 59 degrees between LION nodes.
CORRIDOR_MAX_TURN_DEG = 60
# Every project costs at least about a block. Without a floor, a 9 m sliver
# at an intersection outranks every corridor on saving per new meter.
CORRIDOR_MIN_PROJECT_M = 100
# The build-out's default budget: about 50 miles, NYC's Streets Plan target
# for protected bike lanes a year.
CORRIDOR_BUDGET_KM = 80
CORRIDOR_ROUND_KM = 10  # new lane per round before every street is scored again
CORRIDOR_CANDIDATES = 2000  # best stretches checked for shared riders each round
# A stretch sharing more of its riders than this with one already taken this
# round waits for the next round's rescore, which sees the first one built.
CORRIDOR_MAX_SHARED = 0.1
MANHATTAN_GRID_DEG = 29  # Manhattan's "north" is 29 degrees east of true north

# --- The plan's effect on riders (bike_network.effect) ----------------------------

# A trip whose route with the plan is within this of today's length, walking
# included, is taken to keep its route: its own street got the lane.
SAME_ROUTE_M = 1.0
# Example trips are drawn from this many station pairs that gain the most
# (trips times cost saved); enough to give nearly every corridor one.
EXAMPLE_CANDIDATES = 20000
# Two example trips are alike when they share more than this share of the
# shorter one's streets; the list shows the one that saves more.
EXAMPLE_OVERLAP = 0.3
# The list leaves out trips too short to show much (either route under this)
# and trips whose route barely moves: less than this share of the route with
# the plan on streets today's route does not ride.
EXAMPLE_MIN_M = 1200.0
EXAMPLE_MIN_MOVED = 0.3
