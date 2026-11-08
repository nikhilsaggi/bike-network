"""Command line: ``python -m bike_network build``."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, lanes, lion, sources
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


def main() -> None:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="bike_network")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build the costed street network")
    args = parser.parse_args()
    if args.command == "build":
        build()
