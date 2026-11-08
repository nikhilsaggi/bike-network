"""Command line: ``python -m bike_network build``."""

from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from . import assignment, calibration, config, demand, lanes, lion, sources, traffic, viz
from .network import CostParams, Graph, build_edges, direction_volumes, mark_shared_streets


def _summary(seg: pd.DataFrame, fac: pd.DataFrame, edges: pd.DataFrame, graph: Graph) -> None:
    joined = fac["segment_id"].isin(seg["segment_id"])
    print(
        f"  DOT segments with a current facility: {len(fac):,}, "
        f"{joined.sum():,} in the network ({(~joined).sum():,} not found)"
    )
    km = edges.groupby(["mode", "facility"])["length_m"].sum().div(1000).round(0)
    print("  Directed km by mode and facility:")
    for (mode, facility), value in km.items():
        print(f"    {mode:4s} {facility:9s} {value:9,.0f}")
    labels = graph.strong_components()
    sizes = np.bincount(labels)
    print(
        f"  Nodes {len(graph.nodes):,}, edges {graph.matrix.nnz:,}; "
        f"largest strongly connected component {sizes.max() / len(labels):.1%} of nodes"
    )
    ride = edges[edges["mode"] == "ride"]
    bands = pd.cut(ride["factor"], [0, 1.0001, 1.25, 1.5, 2, 3, np.inf])
    share = ride.groupby(bands, observed=False)["length_m"].sum() / ride["length_m"].sum()
    print("  Ridden km by cost factor:")
    for band, value in share.items():
        print(f"    {band!s:16s} {value:6.1%}")


def build() -> None:
    """Fetch inputs, build the costed network, and write it to ``data/build``."""
    print("Sources")
    gdb = sources.ensure_lion()
    routes = sources.ensure_bike_routes()
    print("Reading LION...")
    seg = lion.read(gdb)
    print(f"  {len(seg):,} segments")
    fac = lanes.load(routes)
    print("Traffic")
    est, counted = traffic.volumes(seg)
    print(
        f"  Volume model trained on {len(counted):,} counted directions; "
        f"estimates for {len(est):,} car directions"
    )
    seg = mark_shared_streets(direction_volumes(seg, est), fac)
    print(f"  {int(seg['shared_street'].sum())} segments priced as DOT shared streets")
    edges = build_edges(seg, fac)
    graph = Graph.from_edges(edges)
    _summary(seg, fac, edges, graph)

    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(config.BUILD_DIR / "segments.parquet")
    edges.to_parquet(config.BUILD_DIR / "edges.parquet")
    est.to_parquet(config.BUILD_DIR / "traffic.parquet")
    print(f"Wrote segments, edges and traffic parquet files to {config.BUILD_DIR}")


def build_viz() -> None:
    """Export the built network for the edge-cost viewer."""
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet", columns=["segment_id"])
    fac = lanes.load(sources.ensure_bike_routes())
    counted = pd.read_parquet(config.BUILD_DIR / "traffic.parquet")
    flows_path = config.BUILD_DIR / "flows.parquet"
    flows = None
    if flows_path.exists():
        flows = pd.read_parquet(flows_path)
        routes = pd.read_parquet(config.BUILD_DIR / "routes.parquet", columns=["trips"])
        flows.attrs["trips_routed"] = float(routes["trips"].sum())
    payload = viz.export(seg, fac, edges, counted, flows)
    viz.write(payload)
    print(f"Wrote {config.BUILD_DIR / 'viz'}: {payload['n']:,} segments")


def evaluate_traffic() -> None:
    """Score the traffic volume model against the predictors it replaced."""
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    rows = pd.read_json(traffic.ensure_counts(), dtype=False)
    edges, train = traffic.training_set(seg, rows)
    years = train["last_year"]
    print(
        f"  {len(train):,} counted directions on {train['segment_id'].nunique():,} segments "
        f"({len(train) / len(edges):.1%} of {len(edges):,} car directions), "
        f"last counted {years.min()}-{years.max()}, median {int(years.median())}"
    )
    print("  Hold-out fit to log daily volume (5 folds, grouped by street):")
    print(f"    {'':30s} {'R2':>6s} {'rank r':>7s} {'typical error':>14s}")
    for s in traffic.evaluate(train):
        print(f"    {s.name:30s} {s.r2:6.2f} {s.spearman:7.2f} {s.median_factor:13.2f}x")


def _trip_km_by_facility(edges: pd.DataFrame, trips: np.ndarray) -> pd.Series:
    kind = np.where(edges["mode"] == "walk", "walked", edges["facility"])
    km = pd.Series(trips * edges["length_m"].to_numpy() / 1000).groupby(kind).sum()
    return km / km.sum()


WORKERS = max(1, min(8, os.cpu_count() or 1))


def _network_od(graph: Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Citi Bike trips as ``o, d, trips`` on network nodes, and the snapped stations."""
    labels = graph.strong_components()
    main_component = graph.nodes[labels == np.bincount(labels).argmax()]
    print(f"Citi Bike {config.DEMAND_YEAR}")
    od, stations = demand.year_demand(config.DEMAND_YEAR)
    nodes = lion.read_nodes(sources.ensure_lion())
    stations = demand.snap(stations, nodes, main_component)
    node_of = stations.set_index("station")["node"]
    od = od.assign(o=od["start_station_id"].map(node_of), d=od["end_station_id"].map(node_of))
    usable = (od["o"] >= 0) & (od["d"] >= 0)
    print(
        f"  {len(stations):,} stations, {(stations['node'] < 0).sum()} off the network; "
        f"{od['trips'].sum():,} trips between stations, {od.loc[~usable, 'trips'].sum():,} with an end off it"
    )
    od = od[usable]
    od = od.assign(o=od["o"].astype("int64"), d=od["d"].astype("int64"))
    return od[["o", "d", "trips"]], stations


def run_assignment() -> None:
    """Route a year of Citi Bike trips and write per-edge trip counts."""
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet")
    seg = pd.read_parquet(config.BUILD_DIR / "segments.parquet", columns=["segment_id", "boro"])
    graph = Graph.from_edges(edges)
    base = Graph.from_edges(assignment.baseline_edges(edges))
    od, stations = _network_od(graph)

    print("Routing (safety cost, then shortest legal distance)...")
    safe = assignment.assign(graph, od, workers=WORKERS)
    short = assignment.assign(base, od, workers=WORKERS)
    r, s = safe.od, short.od
    same_node = od.loc[od["o"] == od["d"], "trips"].sum()  # neighbouring stations
    print(
        f"  {r['trips'].sum():,.0f} trips routed ({same_node:,.0f} start and end at one node); "
        f"{(~r['reachable']).sum()} pairs unreachable"
    )
    km_safe = (r["trips"] * r["length_m"]).sum() / 1000
    km_short = (s["trips"] * s["length_m"]).sum() / 1000
    print(
        f"  Trip-km: {km_safe:,.0f} on the safest routes, {km_short:,.0f} on the shortest "
        f"(safest routes are {km_safe / km_short - 1:.1%} longer)"
    )
    share = pd.DataFrame(
        {
            "safest": _trip_km_by_facility(graph.edges, safe.edge_trips),
            "shortest": _trip_km_by_facility(base.edges, short.edge_trips),
        }
    ).fillna(0)
    print("  Share of trip-km by facility:")
    for kind, row in share.iterrows():
        print(f"    {kind:10s} {row['safest']:6.1%} {row['shortest']:9.1%}")

    flows = edges[["segment_id", "dir", "street", "facility", "mode", "length_m"]].copy()
    flows["trips"] = safe.edge_trips
    flows["trips_shortest"] = short.edge_trips
    flows = flows.merge(seg.drop_duplicates("segment_id"), on="segment_id", how="left")
    lanes_only = flows[flows["facility"] != "none"]
    top = (
        lanes_only.assign(trip_km=lanes_only["trips"] * lanes_only["length_m"] / 1000)
        .groupby(["street", "boro"])
        .agg(trip_km=("trip_km", "sum"), peak_per_day=("trips", lambda x: x.max() / 365))
        .sort_values("trip_km", ascending=False)
        .head(15)
    )
    print("  Busiest bike facilities (trip-km a year, busiest block's trips a day):")
    for (street, boro), row in top.iterrows():
        print(
            f"    {street[:34]:34s} {int(boro)} {row['trip_km']:12,.0f} {row['peak_per_day']:8,.0f}"
        )

    flows.to_parquet(config.BUILD_DIR / "flows.parquet")
    r.to_parquet(config.BUILD_DIR / "routes.parquet")
    stations.to_parquet(config.BUILD_DIR / "stations.parquet")
    print(f"Wrote flows, routes and stations parquet files to {config.BUILD_DIR}")


# The knobs calibration searches; everything else stays at config.py's values.
GRID = {
    "per_doubling": [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    "painted": [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    "walk_factor": [2.5, 4.0, 6.0],
}


def run_calibration() -> None:
    """Score a grid of knob settings against DOT's bike counters."""
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    fac = lanes.load(sources.ensure_bike_routes())
    edges = build_edges(seg, fac)
    od, stations = _network_od(Graph.from_edges(edges))
    stations = stations[stations["node"] >= 0]

    counters = calibration.locate(calibration.counter_table(config.DEMAND_YEAR), seg, edges)
    counters["ends"] = calibration.trip_ends_near(counters, stations)
    where = calibration.crossing_segments(counters, seg, edges)

    base_edges = assignment.baseline_edges(edges)
    base = assignment.assign(Graph.from_edges(base_edges), od, workers=WORKERS)
    base_routed = calibration.routed_at(where, base_edges, base.edge_trips)
    scored = (base_routed >= calibration.MIN_ROUTED_PER_DAY) & (counters["ends"] > 0).to_numpy()
    counted, ends = counters["per_day"].to_numpy()[scored], counters["ends"].to_numpy()[scored]
    print(f"Scoring on {scored.sum()} counters: " + ", ".join(counters.loc[scored, "name"]))
    baseline = calibration.score(base_routed[scored], counted, ends)
    print(f"  Shortest legal route: spread {baseline['spread']:.3f} (raw {baseline['raw']:.3f})")

    rows = []
    combos = list(itertools.product(*GRID.values()))
    for i, (per_doubling, painted, walk) in enumerate(combos, 1):
        start = time.time()
        params = CostParams.from_config(
            per_doubling=per_doubling,
            exposure={**config.EXPOSURE, "painted": painted},
            walk_factor=walk,
        )
        e = build_edges(seg, fac, params)
        res = assignment.assign(Graph.from_edges(e), od, workers=WORKERS)
        routed = calibration.routed_at(where, e, res.edge_trips)
        s = calibration.score(routed[scored], counted, ends)
        trip_km = (res.od["trips"] * res.od["length_m"]).sum() / 1000
        rows.append(
            {
                "per_doubling": per_doubling,
                "painted": painted,
                "walk_factor": walk,
                **s,
                "trip_km": trip_km,
                **{f"routed_{cid}": v for cid, v in zip(counters["id"], routed, strict=True)},
            }
        )
        print(
            f"  [{i:3d}/{len(combos)}] doubling {per_doubling:4.2f} painted {painted:4.2f} "
            f"walk {walk:3.1f}: spread {s['spread']:.3f} ({time.time() - start:.0f}s)",
            flush=True,
        )
    results = pd.DataFrame(rows).sort_values("spread")
    results.attrs["baseline_spread"] = baseline["spread"]
    results.to_parquet(config.BUILD_DIR / "calibration.parquet")
    counters.assign(scored=scored, routed_shortest=base_routed).to_parquet(
        config.BUILD_DIR / "counters.parquet"
    )
    print("Best settings:")
    for r in results.head(10).itertuples():
        print(
            f"  doubling {r.per_doubling:4.2f} painted {r.painted:4.2f} walk {r.walk_factor:3.1f}: "
            f"spread {r.spread:.3f}  raw {r.raw:.3f}  rank r {r.spearman:.2f}"
        )
    print(f"Wrote {config.BUILD_DIR / 'calibration.parquet'}")


def main() -> None:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="bike_network")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build the costed street network")
    sub.add_parser("viz", help="export the built network for the viewer in viz/")
    sub.add_parser("traffic", help="score the traffic volume model on held-out streets")
    sub.add_parser("assign", help="route a year of Citi Bike trips over the network")
    sub.add_parser("calibrate", help="score a grid of cost knobs against DOT's bike counters")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "viz":
        build_viz()
    elif args.command == "traffic":
        evaluate_traffic()
    elif args.command == "assign":
        run_assignment()
    elif args.command == "calibrate":
        run_calibration()
