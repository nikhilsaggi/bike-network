"""Command line: ``python -m bike_network build``."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, lanes, lion, sources, traffic, viz
from .network import Graph, build_edges


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
    edges = build_edges(seg, fac)
    graph = Graph.from_edges(edges)
    _summary(seg, fac, edges, graph)

    config.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(config.BUILD_DIR / "segments.parquet")
    edges.to_parquet(config.BUILD_DIR / "edges.parquet")
    print(f"Wrote {config.BUILD_DIR / 'segments.parquet'} and edges.parquet")


def build_viz() -> None:
    """Export the built network for the edge-cost viewer."""
    import geopandas as gpd  # noqa: PLC0415 -- only this command needs geometry

    seg = gpd.read_parquet(config.BUILD_DIR / "segments.parquet")
    edges = pd.read_parquet(config.BUILD_DIR / "edges.parquet", columns=["segment_id"])
    fac = lanes.load(sources.ensure_bike_routes())
    payload = viz.export(seg, fac, edges)
    viz.write(payload)
    print(f"Wrote {config.BUILD_DIR / 'viz'}: {payload['n']:,} segments")


def build_traffic() -> None:
    """Fit and evaluate the traffic volume model, and write per-edge estimates."""
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
    est = traffic.estimate(edges, train)
    est.to_parquet(config.BUILD_DIR / "traffic.parquet")
    print(f"Wrote {config.BUILD_DIR / 'traffic.parquet'}")


def main() -> None:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="bike_network")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build the costed street network")
    sub.add_parser("viz", help="export the built network for the viewer in viz/")
    sub.add_parser("traffic", help="fit and evaluate the traffic volume model")
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "viz":
        build_viz()
    elif args.command == "traffic":
        build_traffic()
