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

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def assign(graph: Graph, od: pd.DataFrame, batch: int = BATCH) -> Assignment:
    """Assign ``od`` (``o``, ``d`` LION node ids, ``trips``) to the cheapest routes."""
    od = od.groupby(["o", "d"], as_index=False)["trips"].sum()
    od = od[od["o"] != od["d"]].reset_index(drop=True)
    oi = graph.node_index(od["o"].to_numpy())
    di = graph.node_index(od["d"].to_numpy())
    order = np.argsort(oi, kind="stable")
    oi, di = oi[order], di[order]
    trips = od["trips"].to_numpy(dtype=np.float64)[order]

    edge_of = graph.edge_of
    indptr = edge_of.indptr.astype(np.int64)
    indices = edge_of.indices.astype(np.int64)
    edge_row = (edge_of.data - 1).astype(np.int64)
    length = graph.edges["length_m"].to_numpy(dtype=np.float64)
    walked = (graph.edges["mode"] == "walk").to_numpy()
    edge_trips = np.zeros(len(graph.edges), dtype=np.float64)
    cost = np.full(len(od), np.inf)
    out_len = np.zeros(len(od))
    out_walk = np.zeros(len(od))
    out_ok = np.zeros(len(od), dtype=np.bool_)

    origins, starts = np.unique(oi, return_index=True)
    bounds = np.append(starts, len(oi))
    for b in range(0, len(origins), batch):
        chunk = origins[b : b + batch]
        dist, pred = csgraph.dijkstra(
            graph.matrix, directed=True, indices=chunk, return_predecessors=True
        )
        for r, origin in enumerate(chunk):
            lo, hi = bounds[b + r], bounds[b + r + 1]
            dests = di[lo:hi].astype(np.int64)
            cost[lo:hi] = dist[r, dests]
            _walk_routes(
                int(origin),
                dests,
                trips[lo:hi],
                pred[r].astype(np.int64),
                indptr,
                indices,
                edge_row,
                length,
                walked,
                edge_trips,
                out_len[lo:hi],
                out_walk[lo:hi],
                out_ok[lo:hi],
            )
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


def baseline_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Return the same edges costed by distance alone: riding at 1x wherever it is legal.

    Walking keeps its factor, so the baseline still routes the bike legally
    rather than the wrong way down a one-way street.
    """
    out = edges.copy()
    out["cost"] = np.where(out["mode"] == "ride", out["length_m"], out["cost"])
    return out
