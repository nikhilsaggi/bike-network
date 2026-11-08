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
direction, coloured by cost, with a slider for each knob below. Serve that
folder (`python -m http.server -d data/build/viz`) and open it. The viewer
recomputes costs in the browser with `viz/cost.js`, which
`tests/test_viz.py` holds to the Python model.

## Cost model

Each directed edge costs `length_m * factor`. A bike either rides the edge or
is walked on the sidewalk, whichever is cheaper:

- **Riding** is legal with the flow of traffic, and in any direction DOT has a
  bike facility for (contraflow lanes, two-way greenways).
  `factor = 1 + exposure * traffic_penalty`.
- **Walking** the bike is legal on any sidewalk in either direction:
  `factor = 4` (8 on a step street). Vehicle-only roads have no walk edge.

| Facility in the direction of travel | exposure |
|---|---|
| Protected, greenway, boardwalk, off-street link | 0 |
| Painted (conventional, buffered, curbside) | 0.3 |
| Shared, signed route, wide parking lane, none | 1 |

`traffic_penalty` is 0 where no car may go, otherwise 0.2, plus 0.4 per
travel lane per direction beyond the first (up to 3), plus 0.15 per 5 mph
over 25, plus 0.2 / 0.3 / 0.5 for a limited-local / local / through truck
route. All of these numbers live in `bike_network/config.py`.

## Known gaps

- About 30 DOT lane records are attached to the generic centreline of a
  divided street (LION layer `G`) rather than its carriageways, so they do not
  reach the network. Another ~270 current records name a segment LION does
  not have.
- Staten Island and Governors Island are their own components: ferries are
  not ridden, so they are left out.
- No turn or intersection costs yet.
