# Development

## Setup

```
uv venv && uv pip install -e . --group dev   # or: pip install -e . pytest 'ruff==0.16.3'
pytest -q
ruff format . && ruff check .
```

## The pipeline

Each step reads the ones before it. Inputs download into `data/raw/` once;
everything else goes in `data/build/`. Tuning knobs live in
`bike_network/config.py`.

| Command | What it does | Writes |
|---|---|---|
| `build` | Network, traffic model and costs | `segments.parquet` (one row per LION segment, geometry in EPSG:2263), `edges.parquet` (one row per directed edge) |
| `traffic` | Scores the volume model on streets held out of the fit | |
| `assign` | Routes the `DEMAND_YEAR` Citi Bike trips on the safest and the shortest legal route. The first run downloads about 8 GB of monthly zips, each reduced once to station-to-station counts under `demand/` | `flows.parquet`, `routes.parquet`, `stations.parquet` |
| `calibrate` | Re-routes the trips under a grid of traffic weight, painted-lane share and walking factor, and scores each against DOT's bike counters (about an hour; `--per-doubling`, `--painted`, `--walk-factor` run a smaller grid, merged into the same file) | `calibration.parquet` |
| `candidates` | Scores single blocks for a new protected facility, on two samples of 40,000 station pairs | `candidates.parquet` |
| `corridors` | Builds the greedy plan of new protected lane (`--budget-km`, default 80; `--seed`), about 90 seconds a round, then runs `effect` | `corridors.parquet`, `corridor_edges.parquet`, `corridors.json` |
| `effect` | Routes every trip again with the plan built (about a minute) | `plan_flows.parquet`, `plan_routes.parquet`, `plan_stations.parquet`, `plan_examples.parquet`, `plan_effect.json` |
| `viz` | Exports the map | `viz/` |
| `publish` | Pushes the export to GitHub Pages | |

Run each as `python -m bike_network <command>`.

## The map

`viz` exports the network to `data/build/viz/` for the viewer in
`viz/index.html`. Serve it with `python -m http.server -d data/build/viz`.
The viewer recomputes costs in the browser with `viz/cost.js`, and
`tests/test_viz.py` holds that to the Python model, so a change to the cost
model needs both.

## Publishing

`publish` commits `data/build/viz/` as the whole of the `gh-pages` branch
(one commit, no history) and force-pushes it. It refuses an export older
than the checkout's viewer or than any build file it was made from, so run
`viz` first. `--no-push` builds the branch without pushing. A fork needs
Pages turned on once, serving `gh-pages` from `/`.

A push to `main` that changes only the viewer (`viz/index.html`,
`viz/cost.js`) needs none of that: the Pages workflow
(`.github/workflows/pages.yml`) puts the new page over the `data.json`
already published.

**When the viewer needs new data**, bump `FORMAT` in `bike_network/viz.py`
and `DATA_FORMAT` in `viz/index.html` together. If the page's format differs
from the published data's, the workflow fails without publishing, and the
site waits for a local `viz` and `publish`.
