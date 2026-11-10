"""Where a new bike facility would help most: the value of upgrading one directed edge.

Giving edge ``u -> v`` a protected facility lowers its cost to ``c'`` (its
length: a factor of 1). A trip from ``o`` to ``d`` then saves
``max(0, C_od - (D_o[u] + c' + R_d[v]))``, where ``C_od`` is its cost today,
``D_o`` the cheapest costs from ``o`` and ``R_d`` the cheapest costs to ``d``.
That is exact for one upgraded edge under all-or-nothing assignment, and it
counts the streets riders avoid today, which carry no flow and would be
missed by ranking edges on flow times saving.

Summing it over every station pair for every candidate is about 10^11
operations, so the pairs are sampled in proportion to their trips and each
sampled pair stands for ``total trips / sample size`` trips: an unbiased
estimate whose noise falls with the sample. The unit of benefit is cost-
meters a year -- meters weighted by how stressful they are -- saved across
all trips.

Three kinds of candidate, all on segments that exist:

- ``lane``: a direction riders may use today without a protected facility;
- ``contraflow``: the direction against car traffic on a one-way street,
  walked today (not on one roadbed of a divided street, whose other roadbed
  carries that direction);
- ``walkway``: a path walked today because riding is not allowed on it.

Highways and other vehicle-only roads have no edges and are never candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from scipy.sparse import csgraph

from . import config
from .network import other_roadbed

if TYPE_CHECKING:
    import pandas as pd

    from .network import Graph

SAMPLE_PAIRS = 40_000
CAR_DIRS = ("T", "W", "A")


def candidate_edges(edges: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Return the directed edges an upgrade could improve, with their upgraded cost."""
    traf = seg.drop_duplicates("segment_id").set_index("segment_id")["traf_dir"]
    e = edges.assign(traf_dir=edges["segment_id"].map(traf).to_numpy())
    walked = e["mode"] == "walk"
    on_street = e["traf_dir"].isin(CAR_DIRS)
    kind = np.select(
        [
            ~walked & (e["facility"] != "separated"),
            walked & on_street & ~other_roadbed(edges, seg),
            walked & ~on_street & (e["factor"] < config.STEP_FACTOR),
        ],
        ["lane", "contraflow", "walkway"],
        default="",
    )
    out = e.assign(kind=kind)[kind != ""].copy()
    out["new_cost"] = out["length_m"].clip(lower=0.01)
    return out


def sample_pairs(od: pd.DataFrame, n: int = SAMPLE_PAIRS, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` station pairs in proportion to trips; ``weight`` is trips each stands for."""
    rng = np.random.default_rng(seed)
    p = od["trips"].to_numpy(dtype=float)
    pick = rng.choice(len(od), size=n, replace=True, p=p / p.sum())
    out = od.iloc[pick][["o", "d"]].reset_index(drop=True)
    out["weight"] = p.sum() / n
    return out


def _distance_rows(matrix, sources: np.ndarray) -> np.ndarray:  # noqa: ANN001 -- scipy sparse
    """Cheapest costs from each source to every node, as float32 to halve memory."""
    rows = np.empty((len(sources), matrix.shape[0]), dtype=np.float32)
    for b in range(0, len(sources), 64):
        rows[b : b + 64] = csgraph.dijkstra(matrix, directed=True, indices=sources[b : b + 64])
    return rows


@njit(parallel=True, cache=True)
def _benefits(
    cand_u: np.ndarray,
    cand_v: np.ndarray,
    cand_cost: np.ndarray,
    pair_o: np.ndarray,
    pair_d: np.ndarray,
    pair_cost: np.ndarray,
    pair_w: np.ndarray,
    dist_from: np.ndarray,
    dist_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover -- compiled; covered via score()
    n = cand_u.shape[0]
    saved = np.zeros(n)
    users = np.zeros(n)
    for i in prange(n):
        u = cand_u[i]
        v = cand_v[i]
        c = cand_cost[i]
        s = 0.0
        k = 0.0
        for j in range(pair_o.shape[0]):
            via = dist_from[pair_o[j], u] + c + dist_to[pair_d[j], v]
            gain = pair_cost[j] - via
            if gain > 1e-6:
                s += pair_w[j] * gain
                k += pair_w[j]
        saved[i] = s
        users[i] = k
    return saved, users


def score(graph: Graph, cands: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Add ``saved`` (cost-meters a year) and ``trips`` (a year, that would use it) per candidate."""
    oi = graph.node_index(pairs["o"].to_numpy())
    di = graph.node_index(pairs["d"].to_numpy())
    origins, o_row = np.unique(oi, return_inverse=True)
    dests, d_row = np.unique(di, return_inverse=True)
    dist_from = _distance_rows(graph.matrix, origins)
    dist_to = _distance_rows(graph.matrix.T.tocsr(), dests)
    pair_cost = dist_from[o_row, di].astype(np.float64)
    keep = np.isfinite(pair_cost)
    saved, users = _benefits(
        graph.node_index(cands["u"].to_numpy()),
        graph.node_index(cands["v"].to_numpy()),
        cands["new_cost"].to_numpy(dtype=np.float64),
        o_row[keep].astype(np.int64),
        d_row[keep].astype(np.int64),
        pair_cost[keep],
        pairs["weight"].to_numpy(dtype=np.float64)[keep],
        dist_from,
        dist_to,
    )
    return cands.assign(saved=saved, trips=users)
