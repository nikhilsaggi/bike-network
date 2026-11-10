# bike-network

A directed NYC street network where every edge costs more the less safe it is
to ride: more for traffic, less for a bike lane in the direction of travel,
least for a protected one. It is the base for routing experiments: which lanes
carry the most riding, and where a new lane would help most.

All inputs are public:

| Source | Used for |
|---|---|
| [LION](https://www.nyc.gov/content/planning/pages/resources/datasets/lion) (NYC DCP), release `26c` | streets, paths, traffic direction, travel lanes, posted speed, truck routes |
| [NYC Bike Routes](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib) (NYC DOT) | bike facility in each direction, joined on LION `SegmentID` |

## Run

```
uv venv && uv pip install -e . --group dev   # or: pip install -e . pytest 'ruff==0.16.3'
python -m bike_network build
pytest -q
ruff format . && ruff check .
```

`build` downloads the inputs into `data/raw/` once, then writes
`data/build/segments.parquet` (one row per LION segment, with geometry in
EPSG:2263) and `data/build/edges.parquet` (one row per directed edge).

`python -m bike_network viz` then exports the network to `data/build/viz/`
for the edge-cost viewer (`viz/index.html`): every street drawn once per
direction, colored by cost, with a slider for each knob below. Serve that
folder (`python -m http.server -d data/build/viz`) and open it. The viewer
recomputes costs in the browser with `viz/cost.js`, which
`tests/test_viz.py` holds to the Python model. Once `assign` has run, its Citi Bike
trips mode shows routed trips a day per direction under the safest and the
shortest routing, their difference, where the trip-km go by facility, and
the busiest bike facilities.

The published map is at **https://nikhilsaggi.github.io/bike-network/**.
`python -m bike_network publish` puts the exported folder there: it commits
`data/build/viz/` as the whole of the `gh-pages` branch (one commit, no
history, since it is build output) and force-pushes it, and GitHub Pages
serves that branch. It refuses an export older than the checkout's viewer or
than any build file it was made from, so run `viz` first. `--no-push` builds
the branch without pushing it. A fork needs Pages turned on once, serving
`gh-pages` from `/`.

## Routing Citi Bike trips

`python -m bike_network assign` downloads Citi Bike's
[trip data](https://s3.amazonaws.com/tripdata/index.html) for
`DEMAND_YEAR` (2025; about 8 GB of monthly zips, each reduced once to
station-to-station counts under `data/build/demand/`), snaps every station
to the nearest node of the main network, and routes each trip on its
cheapest route. It routes them again on the shortest legal route (riding
at 1x wherever riding is allowed) as the baseline the safety cost has to
beat, and writes `flows.parquet` (trips a year on each directed edge, both
ways), `routes.parquet` (each station pair's route) and `stations.parquet`.

Trips that start and end at the same station are loops with no route and
are dropped; so are trips with no end station.

## Calibration and new-lane candidates

`python -m bike_network calibrate` re-routes the year's trips under a grid of
the three main knobs (traffic weight, painted-lane share, walking factor) and
scores each against DOT's automated bike counters (about an hour; pass
`--per-doubling`, `--painted` or `--walk-factor` lists to run a smaller grid,
whose results merge into `calibration.parquet`). How the score works, what
it decided and what it could not are in
**[docs/methodology.md](docs/methodology.md)**.

`python -m bike_network candidates` scores every block that could take a
protected facility (a lane where there is none, a contraflow lane against
one-way traffic on an undivided street, a riding path on a walkway) by the stress a year of Citi
Bike trips would save if it had one, including trips that avoid the block
today. It scores two independent samples of 40,000 station pairs and reports
how far their rankings agree, then writes `candidates.parquet`.

One block at a time misses corridors: a block of mixed traffic between two
others is worth little on its own, because the detour to a parallel lane
still wins. `python -m bike_network corridors` scores whole stretches
instead: consecutive blocks of one street in one direction, which a trip can
join at any intersection and leave at any later one. It then spends a budget
of new protected lane greedily (`--budget-km`, default 80, about NYC's yearly
Streets Plan target). Each round builds the stretches that save the most
stress per new meter, re-routes, and scores every street again, about 90
seconds a round. It writes `corridors.parquet` (one row per corridor, in
build order), `corridor_edges.parquet` (its blocks) and `corridors.json` (the
plan's totals), and `viz` then adds a New corridors view to the map. The
method and what it found:
[docs/methodology.md](docs/methodology.md#7-where-new-lanes-would-help).

The plan is picked on a sample, so `corridors` ends by running
`python -m bike_network effect` (about a minute, and runnable alone): every
one of the year's trips routed again with the plan built, set beside
today's routes. It writes `plan_flows.parquet` (trips a year on each
directed edge, as `flows.parquet`), `plan_routes.parquet` (each station
pair's route), `plan_stations.parquet` (each station's trips and the share
of their stress saved), `plan_examples.parquet` (one real trip per corridor,
both routes) and `plan_effect.json` (the totals). The map's New corridors
mode then has two more views: Where riders go (trips a day with the plan
minus today, the facility mix of the trip-km, the streets riders move to
and leave, and the example trips drawn both ways) and Who gains (each
station by the share of its trips' stress saved). The trips are held fixed:
the same riders between the same stations, so the views show where they
would ride, not how many more would.

## Cost model

Each directed edge costs `length_m * factor`. A bike either rides the edge or
is walked on the sidewalk, whichever is cheaper:

- **Riding** is legal with the flow of traffic, and in any direction DOT has a
  bike facility for (contraflow lanes, two-way greenways).
  `factor = 1 + exposure * traffic_penalty`.
- **Walking** the bike is legal on any sidewalk in either direction:
  `factor = 8` (12 on a step street). Vehicle-only roads have no walk edge.

| Facility in the direction of travel | exposure |
|---|---|
| Protected, greenway, boardwalk, off-street link | 0 |
| Painted (conventional, buffered, curbside) | 0.3 |
| Shared, signed route, wide parking lane, none | 1 |

`traffic_penalty` is 0 where no car may go. Otherwise it is 0.2, plus 1.0
for each doubling of traffic above a quiet street's 1,600 vehicles a day,
plus 0.15 per 5 mph over 25, plus 0.2 / 0.3 / 0.5 for a limited-local /
local / through truck route. Traffic is the volume in the rider's direction
(the oncoming stream, in a contraflow lane). All of these numbers live in
`bike_network/config.py`.

The traffic weight and the walking factor are calibrated against DOT's bike
counters; the rest are judgment. **[docs/methodology.md](docs/methodology.md)**
sets out how each weight was chosen and the evidence for it.

## Traffic volume

DOT has counted traffic on under 1% of car directions, so `build` estimates
the rest: a gradient-boosted model of daily volume per direction, trained on
DOT's [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt)
since 2010 and fed what LION records on every street (width, travel lanes,
one-way, truck route, snow-plough priority, borough, display level). Each
count's compass direction is matched to the segment's own bearing.
`python -m bike_network traffic` scores it on streets held out of the fit:

| Predictor of log daily volume | R² | Rank correlation | Typical error |
|---|---|---|---|
| Lanes per direction alone | 0.32 | 0.55 | 1.62x |
| The lane-count rule it replaced | 0.39 | 0.67 | 1.52x |
| Volume model | 0.47 | 0.73 | 1.47x |

The counted streets skew busy and the median count dates from 2016, so the
estimate ranks streets better than it measures any one of them.

## Known gaps

- DOT's bike routes and LION have drifted apart in segment ids. Separated
  paths DOT has never recorded take their directions from LION (586 blocks);
  see [docs/methodology.md](docs/methodology.md).
- About 30 DOT lane records are attached to the generic centerline of a
  divided street (LION layer `G`) rather than its carriageways, so they do not
  reach the network. Another ~270 current records name a segment LION does
  not have.
- Staten Island and Governors Island are their own components: ferries are
  not ridden, so they are left out.
- LION lags street redesigns. Lane counts, width and truck routes can
  describe a street as it was before DOT rebuilt it, and the traffic model
  inherits that: a road diet keeps its old estimate. The one case handled is
  DOT's shared streets (Broadway in Midtown and the Flatiron, Berry Street):
  a `Shared` facility offered against one-way car traffic marks one, and it
  is priced as a quiet street.
- No turn or intersection costs yet.
