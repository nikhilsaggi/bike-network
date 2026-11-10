"""Route every Citi Bike trip over the network and count trips on each edge.

All-or-nothing assignment: each station-to-station trip takes the one
cheapest route under the edge costs. Bikes do not congest one another, so a
route's cost does not depend on how many others take it and one pass is
exact. ``scipy.sparse.csgraph.dijkstra`` runs once per origin node and gives
the cheapest route from it to every node at once; a numba loop then walks
each destination's route back through the predecessor tree, adding its trips
to every edge on the way.
"""

from __future__ import annotations

import itertools
import multiprocessing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csgraph

if TYPE_CHECKING:
    from .network import Graph

BATCH = 64  # origins per dijkstra call: 64 x 105k nodes x 12 bytes is ~80 MB


@njit(cache=True)
def _walk_routes(
    origin: int,
    dests: np.ndarray,
    trips: np.ndarray,
    pred: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_row: np.ndarray,
    length: np.ndarray,
    walked: np.ndarray,
    edge_trips: np.ndarray,
    out_len: np.ndarray,
    out_walk: np.ndarray,
    out_ok: np.ndarray,
) -> None:  # pragma: no cover -- compiled; covered through assign()
    for k in range(dests.shape[0]):
        node = dests[k]
        total = 0.0
        walk = 0.0
        ok = True
        while node != origin:
            p = pred[node]
            if p < 0:
                ok = False
                break
            row = -1
            for j in range(indptr[p], indptr[p + 1]):
                if indices[j] == node:
                    row = edge_row[j]
                    break
            edge_trips[row] += trips[k]
            total += length[row]
            if walked[row]:
                walk += length[row]
            node = p
        out_len[k] = total
        out_walk[k] = walk
        out_ok[k] = ok


@dataclass
class Assignment:
    """Trips per edge row of ``graph.edges``, and each OD pair's route."""

    edge_trips: np.ndarray
    od: pd.DataFrame  # o, d (LION nodes), trips, cost, length_m, walk_m, reachable


# The routing state a worker reads: set directly in-process, or handed once to
# each spawned worker by _init_worker. Spawn rather than fork: by the time
# assign() runs, numba and pyarrow have threads, and forking a threaded
# process can deadlock the child.
_WORK: dict[str, Any] = {}


def _init_worker(state: dict[str, Any]) -> None:
    _WORK.update(state)


def _route_origins(
    first: int, last: int
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route origins ``first:last`` (positions in the sorted origin list)."""
    w = _WORK
    origins, bounds = w["origins"], w["bounds"]
    lo, hi = bounds[first], bounds[last]
    edge_trips = np.zeros(w["n_edges"], dtype=np.float64)
    cost = np.full(hi - lo, np.inf)
    out_len = np.zeros(hi - lo)
    out_walk = np.zeros(hi - lo)
    out_ok = np.zeros(hi - lo, dtype=np.bool_)
    for b in range(first, last, w["batch"]):
        chunk = origins[b : min(b + w["batch"], last)]
        dist, pred = csgraph.dijkstra(
            w["matrix"], directed=True, indices=chunk, return_predecessors=True
        )
        for r, origin in enumerate(chunk):
            a, z = bounds[b + r] - lo, bounds[b + r + 1] - lo
            dests = w["di"][lo + a : lo + z].astype(np.int64)
            cost[a:z] = dist[r, dests]
            _walk_routes(
                int(origin),
                dests,
                w["trips"][lo + a : lo + z],
                pred[r].astype(np.int64),
                w["indptr"],
                w["indices"],
                w["edge_row"],
                w["length"],
                w["walked"],
                edge_trips,
                out_len[a:z],
                out_walk[a:z],
                out_ok[a:z],
            )
    return lo, hi, edge_trips, cost, out_len, out_walk, out_ok


def assign(graph: Graph, od: pd.DataFrame, batch: int = BATCH, workers: int = 1) -> Assignment:
    """Assign ``od`` (``o``, ``d`` LION node ids, ``trips``) to the cheapest routes.

    ``workers > 1`` splits the origins across forked processes; the answer is
    the same, since every origin is routed independently.
    """
    od = od.groupby(["o", "d"], as_index=False)["trips"].sum()
    od = od[od["o"] != od["d"]].reset_index(drop=True)
    oi = graph.node_index(od["o"].to_numpy())
    di = graph.node_index(od["d"].to_numpy())
    order = np.argsort(oi, kind="stable")
    oi, di = oi[order], di[order]
    trips = od["trips"].to_numpy(dtype=np.float64)[order]
    origins, starts = np.unique(oi, return_index=True)
    edge_of = graph.edge_of
    state = {
        "matrix": graph.matrix,
        "indptr": edge_of.indptr.astype(np.int64),
        "indices": edge_of.indices.astype(np.int64),
        "edge_row": (edge_of.data - 1).astype(np.int64),
        "length": graph.edges["length_m"].to_numpy(dtype=np.float64),
        "walked": (graph.edges["mode"] == "walk").to_numpy(),
        "n_edges": len(graph.edges),
        "origins": origins,
        "bounds": np.append(starts, len(oi)),
        "di": di,
        "trips": trips,
        "batch": batch,
    }
    cuts = np.linspace(0, len(origins), max(workers, 1) + 1).astype(int)
    spans = [(int(a), int(b)) for a, b in itertools.pairwise(cuts) if b > a]
    if workers > 1 and len(spans) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(len(spans), initializer=_init_worker, initargs=(state,)) as pool:
            parts = pool.starmap(_route_origins, spans)
    else:
        _init_worker(state)
        parts = [_route_origins(a, b) for a, b in spans]
        _WORK.clear()

    edge_trips = np.zeros(len(graph.edges), dtype=np.float64)
    cost = np.full(len(od), np.inf)
    out_len = np.zeros(len(od))
    out_walk = np.zeros(len(od))
    out_ok = np.zeros(len(od), dtype=np.bool_)
    for lo, hi, e, c, ln, wk, ok in parts:
        edge_trips += e
        cost[lo:hi], out_len[lo:hi], out_walk[lo:hi], out_ok[lo:hi] = c, ln, wk, ok
    routes = pd.DataFrame(
        {
            "o": graph.nodes[oi],
            "d": graph.nodes[di],
            "trips": trips,
            "cost": cost,
            "length_m": out_len,
            "walk_m": out_walk,
            "reachable": out_ok,
        }
    )
    return Assignment(edge_trips=edge_trips, od=routes)


def trip_km_by_facility(edges: pd.DataFrame, trips: np.ndarray) -> pd.Series:
    """Trip-km on each kind of facility, with walked edges as their own kind."""
    kind = np.where(edges["mode"] == "walk", "walked", edges["facility"])
    return pd.Series(trips * edges["length_m"].to_numpy() / 1000).groupby(kind).sum()


def baseline_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Return the same edges costed by distance alone: riding at 1x wherever it is legal.

    Walking keeps its factor, so the baseline still routes the bike legally
    rather than the wrong way down a one-way street.
    """
    out = edges.copy()
    out["cost"] = np.where(out["mode"] == "ride", out["length_m"], out["cost"])
    return out
