"""Command line: ``python -m bike_network build``."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from . import (
    assignment,
    calibration,
    candidates,
    config,
    corridors,
    demand,
    effect,
    lanes,
    lion,
    pages,
    sources,
    traffic,
    viz,
)
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
    n_dot = len(fac)
    fac = lanes.with_lion_paths(fac, seg, lanes.recorded_segments(routes))
    print(f"  {len(fac) - n_dot} separated paths taken from LION where DOT has never recorded one")
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
    fac.to_parquet(config.BUILD_DIR / "facilities.parquet")
    print(f"Wrote segments, edges and traffic parquet files to {config.BUILD_DIR}")


def build_viz() -> None:
    """Export the built network for the edge-cost viewer."""
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet", columns=["segment_id", "dir"])
    fac = pd.read_parquet(config.BUILD_DIR / "facilities.parquet")
    counted = pd.read_parquet(config.BUILD_DIR / "traffic.parquet")
    flows_path = config.BUILD_DIR / "flows.parquet"
    flows = None
    if flows_path.exists():
        flows = pd.read_parquet(flows_path)
        routes = pd.read_parquet(config.BUILD_DIR / "routes.parquet", columns=["trips"])
        flows.attrs["trips_routed"] = float(routes["trips"].sum())
    plan = None
    if (config.BUILD_DIR / "corridors.json").exists():
        plan = {
            "corridors": pd.read_parquet(config.BUILD_DIR / "corridors.parquet"),
            "blocks": pd.read_parquet(config.BUILD_DIR / "corridor_edges.parquet"),
            "summary": json.loads((config.BUILD_DIR / "corridors.json").read_text()),
        }
    effect = None
    if plan is not None and (config.BUILD_DIR / "plan_effect.json").exists():
        effect = {
            "summary": json.loads((config.BUILD_DIR / "plan_effect.json").read_text()),
            "flows": pd.read_parquet(config.BUILD_DIR / "plan_flows.parquet"),
            "examples": pd.read_parquet(config.BUILD_DIR / "plan_examples.parquet"),
            "stations": pd.read_parquet(config.BUILD_DIR / "plan_stations.parquet"),
        }
    payload = viz.export(seg, fac, edges, counted, flows, plan, effect)
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
    km = assignment.trip_km_by_facility(edges, trips)
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
    same_node = od.loc[od["o"] == od["d"], "trips"].sum()  # neighboring stations
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


def run_calibration(grid: dict[str, list[float]] | None = None) -> None:
    """Score a grid of knob settings against DOT's bike counters.

    Results merge into ``calibration.parquet`` by setting, so a follow-up grid
    (``--walk 8,12``) extends the earlier one instead of replacing it.
    """
    grid = {**GRID, **(grid or {})}
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    fac = pd.read_parquet(config.BUILD_DIR / "facilities.parquet")
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
    combos = list(itertools.product(*grid.values()))
    for i, (per_doubling, painted, walk) in enumerate(combos, 1):
        start = time.time()
        params = CostParams.from_config(
            per_doubling=per_doubling,
            exposure={**config.EXPOSURE, "painted": painted},
            walk_factor=walk,
            # Carrying a bike up steps never costs less than walking it.
            step_factor=max(config.STEP_FACTOR, walk),
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
    knobs = ["per_doubling", "painted", "walk_factor"]
    results = pd.DataFrame(rows)
    path = config.BUILD_DIR / "calibration.parquet"
    if path.exists():
        results = pd.concat([pd.read_parquet(path), results]).drop_duplicates(knobs, keep="last")
    results = results.sort_values("spread").reset_index(drop=True)
    results["baseline_spread"] = baseline["spread"]
    results.to_parquet(path)
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


def run_candidates() -> None:
    """Score every possible single-block upgrade by the cost it would save Citi Bike trips."""
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet")
    seg = pd.read_parquet(
        config.BUILD_DIR / "segments.parquet",
        columns=["segment_id", "traf_dir", "rb_layer", "boro"],
    )
    graph = Graph.from_edges(edges)
    od, _ = _network_od(graph)
    cands = candidates.candidate_edges(edges, seg)
    print(
        f"{len(cands):,} candidate directions: "
        + ", ".join(f"{k} {n:,}" for k, n in cands["kind"].value_counts().items())
    )
    runs = []
    for seed in (0, 1):
        start = time.time()
        pairs = candidates.sample_pairs(od, seed=seed)
        runs.append(candidates.score(graph, cands, pairs))
        print(f"  Sample {seed}: {len(pairs):,} station pairs scored in {time.time() - start:.0f}s")
    top = [set(r.nlargest(50, "saved").index) for r in runs]
    print(f"  The two samples agree on {len(top[0] & top[1])} of their top 50 blocks")

    out = runs[0].assign(
        saved=(runs[0]["saved"] + runs[1]["saved"]) / 2,
        trips=(runs[0]["trips"] + runs[1]["trips"]) / 2,
    )
    out = out.merge(
        seg.drop_duplicates("segment_id")[["segment_id", "boro"]], on="segment_id", how="left"
    )
    out["saved_per_m"] = out["saved"] / out["length_m"]
    keep = [
        "segment_id",
        "dir",
        "street",
        "boro",
        "kind",
        "facility",
        "length_m",
        "factor",
        "saved",
        "trips",
        "saved_per_m",
    ]
    out[keep].to_parquet(config.BUILD_DIR / "candidates.parquet")

    boro = {1: "MN", 2: "BX", 3: "BK", 4: "QN", 5: "SI"}
    print("Best single blocks (cost-km a year saved, trips a day that would use it):")
    for r in out.nlargest(15, "saved").itertuples():
        print(
            f"  {r.kind:10s} {r.street[:30]:30s} {boro.get(r.boro, '')} {r.dir} {r.length_m:5.0f} m  {r.saved / 1000:9,.0f}  {r.trips / 365:7,.0f}"
        )
    print("Best streets, summing their blocks:")
    streets = out.groupby(["street", "boro", "kind"]).agg(
        saved=("saved", "sum"), km=("length_m", lambda s: s.sum() / 1000)
    )
    for (street, b, kind), r in streets.nlargest(20, "saved").iterrows():
        print(
            f"  {kind:10s} {street[:30]:30s} {boro.get(b, '')} {r['km']:5.1f} km  {r['saved'] / 1000:10,.0f}"
        )
    print(f"Wrote {config.BUILD_DIR / 'candidates.parquet'}")


def run_corridors(budget_km: float, seed: int = 0) -> None:
    """Build out protected corridors greedily, best cost saved per new meter first."""
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet")
    seg = pd.read_parquet(
        config.BUILD_DIR / "segments.parquet",
        columns=["segment_id", "street", "u", "v", "traf_dir", "rb_layer", "boro"],
    )
    graph = Graph.from_edges(edges)
    od, _ = _network_od(graph)
    xy = lion.read_nodes(sources.ensure_lion()).set_index("node")
    table = corridors.street_runs(edges, seg, xy)
    print(f"{table['run'].nunique():,} street runs over {len(table):,} directed blocks")
    pairs = candidates.sample_pairs(od, seed=seed)
    print(
        f"Build-out: {budget_km:g} km of new protected lane, {len(pairs):,} sampled station pairs"
    )
    picks, final = corridors.build_out(edges, table, pairs, budget_km * 1000)
    if picks.empty:
        print("Nothing to build.")
        return

    runs = corridors.Runs.from_table(table, graph, edges)
    plan = corridors.merge_picks(picks, runs.new, runs.ptr)
    print("Scoring each corridor alone against today's network")
    today = corridors.Distances.compute(graph, pairs)
    plan[["saved_alone", "trips_alone", "blocks_alone"]] = [
        corridors.stretch_value(runs, c.run, c.s, c.t, today) for c in plan.itertuples()
    ]
    weight, before = today.pairs["weight"].to_numpy(), today.cost
    del today
    after = corridors.Distances.compute(Graph.from_edges(final), pairs, reverse=False).cost
    gain = before - after

    out, blocks = corridors.describe(plan, runs, edges, seg, xy)
    out.to_parquet(config.BUILD_DIR / "corridors.parquet")
    blocks.to_parquet(config.BUILD_DIR / "corridor_edges.parquet")

    total = (weight * before).sum()
    helped = weight[gain > corridors.EPS].sum()
    summary = {
        "budget_km": budget_km,
        "new_km": round(out["new_m"].sum() / 1000, 2),
        "rounds": int(picks["round"].max() + 1),
        "saved": round((weight * gain).sum() / 1000),
        "saved_share": round((weight * gain).sum() / total, 4),
        "helped_day": round(helped / 365),
        "helped_share": round(helped / weight.sum(), 4),
        "year": config.DEMAND_YEAR,
    }
    (config.BUILD_DIR / "corridors.json").write_text(json.dumps(summary, indent=1))
    print(
        f"Plan: {out['new_m'].sum() / 1000:.1f} km of new protected lane in {len(out)} corridors, "
        f"built over {picks['round'].max() + 1} rounds"
    )
    print(
        f"  Saves {(weight * gain).sum() / 1000:,.0f} cost-km a year, "
        f"{(weight * gain).sum() / total:.1%} of the cost of every trip today; "
        f"{helped / 365:,.0f} trips a day ({helped / weight.sum():.1%}) ride at least one new block"
    )
    print(
        "Corridors in build order. Alone: protected on its own against today's network.\n"
        "Blocks: its blocks scored one at a time, summed. Cost-km a year; trips a day."
    )
    boro_code = {1: "MN", 2: "BX", 3: "BK", 4: "QN", 5: "SI"}
    print(
        f"  {'#':>3} {'street':24s} {'':2s}  {'from -> to':40s} {'new km':>6s} {'trips':>6s} {'alone':>8s} {'blocks':>8s}"
    )
    for r in out.itertuples():
        span = f"{r.kind[:6]} {r.from_street} -> {r.to_street}" + (
            f" ({r.gaps} gaps)" if r.gaps > 1 else ""
        )
        print(
            f"  {r.corridor:3d} {corridors.short_name(r.street)[:24]:24s} {boro_code.get(r.boro, '')} {r.heading}  "
            f"{span[:40]:40s} {r.new_m / 1000:6.2f} {r.trips_alone / 365:6,.0f} "
            f"{r.saved_alone / 1000:8,.0f} {r.blocks_alone / 1000:8,.0f}"
        )
    print(
        f"Wrote {config.BUILD_DIR / 'corridors.parquet'}, corridor_edges.parquet and corridors.json"
    )
    run_effect()


def run_effect() -> None:
    """Route every trip again with the corridor plan built, and compare with today's routes."""
    b = config.BUILD_DIR
    edges = pd.read_parquet(b / "edges.parquet")
    flows = pd.read_parquet(b / "flows.parquet")
    if (
        len(flows) != len(edges)
        or not (flows["segment_id"].to_numpy() == edges["segment_id"].to_numpy()).all()
    ):
        msg = "flows.parquet is not from this edges.parquet: run `assign` first."
        raise SystemExit(msg)
    routes = pd.read_parquet(b / "routes.parquet")
    blocks = pd.read_parquet(b / "corridor_edges.parquet")
    stations = pd.read_parquet(b / "stations.parquet")

    plan, new_rows = effect.plan_edges(edges, blocks)
    print(
        f"Routing {routes['trips'].sum():,.0f} trips with the plan's "
        f"{edges['length_m'].to_numpy()[new_rows].sum() / 1000:.1f} km of new lane built..."
    )
    graph_plan = Graph.from_edges(plan)
    res = assignment.assign(graph_plan, routes[["o", "d", "trips"]], workers=WORKERS)
    pairs = effect.compare(routes, res.od)
    summary = effect.summarize(
        pairs,
        edges,
        plan,
        flows["trips"].to_numpy(),
        res.edge_trips,
        flows["trips_shortest"].to_numpy(),
        new_rows,
    )
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    print(
        f"  Saves {pct(summary['saved_share'])} of the cost of every trip; "
        f"{summary['helped_day']:,} trips a day ({pct(summary['helped_share'])}) gain, "
        f"{pct(summary['same_route_share'])} of them on the route they ride today "
        f"({pct(summary['same_route_saved'])} of the saving)"
    )
    d, c, k = summary["detour"], summary["cost_per_km"], summary["walked_km"]
    print(
        f"  Trip-km over the shortest routes: {pct(d['today'])} today, {pct(d['plan'])} with the plan; "
        f"cost per km {c['today']:.3f} -> {c['plan']:.3f}; walked {k['today']:,} -> {k['plan']:,} km"
    )
    print("  Share of trip-km by facility, today and with the plan:")
    for kind in effect.MIX:
        print(
            f"    {kind:10s} {summary['share']['today'][kind]:6.1%} {summary['share']['plan'][kind]:6.1%}"
        )

    corridor = np.full(len(edges), -1)
    pos = pd.MultiIndex.from_frame(edges[effect.KEY]).get_indexer(
        pd.MultiIndex.from_frame(blocks[effect.KEY])
    )
    corridor[pos] = blocks["corridor"].to_numpy() - 1
    new_m = np.zeros(len(edges))
    new_m[new_rows] = edges["length_m"].to_numpy()[new_rows]
    start = time.time()
    ex = effect.distinct(
        effect.examples(Graph.from_edges(edges), graph_plan, pairs, corridor, new_m), edges
    )
    covered = {c for cs in ex["corridors"] for c in cs} if len(ex) else set()
    print(
        f"  {len(ex)} example trips for {len(covered)} of {blocks['corridor'].nunique()} corridors, "
        f"{int(ex['listed'].sum()) if len(ex) else 0} unlike one another ({time.time() - start:.0f}s)"
    )

    out = plan[["segment_id", "dir", "u", "v", "street", "facility", "mode", "length_m"]].copy()
    out["trips"] = res.edge_trips
    out["new"] = False
    out.loc[new_rows, "new"] = True
    out.to_parquet(b / "plan_flows.parquet")
    res.od.to_parquet(b / "plan_routes.parquet")
    ex.to_parquet(b / "plan_examples.parquet")
    effect.station_gains(pairs, stations).to_parquet(b / "plan_stations.parquet")
    (b / "plan_effect.json").write_text(json.dumps(summary, indent=1))
    print(
        f"Wrote plan_flows, plan_routes, plan_examples and plan_stations parquet and plan_effect.json to {b}"
    )


def run_publish(*, push: bool, remote: str) -> None:
    """Commit the exported viewer as the gh-pages branch and push it."""
    site = config.BUILD_DIR / "viz"
    problems = pages.stale(site, config.BUILD_DIR)
    if problems:
        raise SystemExit(
            "Not publishing: " + "; ".join(problems) + ". Run `python -m bike_network viz` first."
        )
    head = pages.git(config.ROOT, "rev-parse", "--short", "HEAD")
    dirty = pages.git(config.ROOT, "status", "--porcelain", "--untracked-files=no")
    message = f"Publish the map built from {head}" + (" with uncommitted changes" if dirty else "")
    commit = pages.commit_site(site, config.ROOT, message)
    print(f"{pages.BRANCH} is now {commit[:7]}: {message}")
    if not push:
        return
    pages.git(config.ROOT, "push", "--force", remote, f"{commit}:refs/heads/{pages.BRANCH}")
    url = pages.pages_url(pages.git(config.ROOT, "remote", "get-url", remote))
    print(
        f"Pushed to {remote}"
        + (f"; Pages serves it at {url} within a minute or two" if url else "")
    )


def main() -> None:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="bike_network")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build the costed street network")
    sub.add_parser("viz", help="export the built network for the viewer in viz/")
    sub.add_parser("traffic", help="score the traffic volume model on held-out streets")
    sub.add_parser("assign", help="route a year of Citi Bike trips over the network")
    cal = sub.add_parser("calibrate", help="score a grid of cost knobs against DOT's bike counters")
    for knob, values in GRID.items():
        cal.add_argument(
            f"--{knob.replace('_', '-')}",
            dest=knob,
            type=lambda s: [float(x) for x in s.split(",")],
            help=f"comma-separated values (default {','.join(map(str, values))})",
        )
    sub.add_parser("candidates", help="rank single-block upgrades by the cost they would save")
    cor = sub.add_parser("corridors", help="build out protected corridors greedily under a budget")
    cor.add_argument(
        "--budget-km",
        type=float,
        default=config.CORRIDOR_BUDGET_KM,
        help=f"km of new protected lane (default {config.CORRIDOR_BUDGET_KM})",
    )
    cor.add_argument("--seed", type=int, default=0, help="seed for the station-pair sample")
    sub.add_parser("effect", help="route every trip again with the corridor plan built")
    pub = sub.add_parser("publish", help="push the exported viewer to the gh-pages branch")
    pub.add_argument("--no-push", action="store_true", help="commit the branch but do not push it")
    pub.add_argument("--remote", default="origin", help="remote to push to (default origin)")
    args = parser.parse_args()
    commands = {
        "build": build,
        "viz": build_viz,
        "traffic": evaluate_traffic,
        "assign": run_assignment,
        "calibrate": lambda: run_calibration(
            {k: getattr(args, k) for k in GRID if getattr(args, k)}
        ),
        "candidates": run_candidates,
        "corridors": lambda: run_corridors(args.budget_km, args.seed),
        "effect": run_effect,
        "publish": lambda: run_publish(push=not args.no_push, remote=args.remote),
    }
    commands[args.command]()
