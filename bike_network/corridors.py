"""Where a run of new protected blocks along one street would help most, and in what order.

Scoring blocks one at a time misses corridors. One mixed-traffic block of an
avenue is worth little on its own: a rider still faces the blocks either
side of it, so the detour to a parallel lane stays cheaper. Protect the
whole stretch and the detour stops paying. A **stretch** is consecutive
blocks of one street in one direction (a :func:`street_runs` run). Protect
it and a trip from ``o`` to ``d`` can join at any node ``i`` of it and
leave at any later node ``j``:

    via = D_o[n_i] + (P_j - P_i) + R_d[n_j]

where ``P`` is the stretch's running length (a protected block costs its
length), and the trip saves ``max(0, C_od - min_{i<j} via)``. One sweep
along the street finds the minimum, keeping the best ``D_o[n_i] - P_i`` so
far. It reuses the single-block score's distances, ``D_o`` from each sampled
origin and ``R_d`` to each sampled destination.

The score is exact when the trip's best new route uses the stretch in one
piece. Every block costs at least its length, and a protected one exactly
that, so leaving a straight stretch and rejoining it never pays. On a
curving street the score is a lower bound.

A longer stretch never saves less, so how far one runs needs a price per
new metre: extend while each one saves at least that. Each round of the
build-out (:func:`build_out`) sets the price that its candidates would
spend the remaining budget at, sizes every street's stretch to it, and
builds the best by saving per new metre. Blocks already protected cost
nothing, so a stretch may join gaps across them. After each round the
picks join the network and every street is scored again. That accounts for
what scoring one corridor at a time cannot: two parallel avenues compete
for the same riders, and one corridor can make another worth more.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit, prange

from . import candidates, config
from .network import CAR_DIRS, MIN_COST, Graph

if TYPE_CHECKING:
    from collections.abc import Callable

# Prices a round can set, in cost-metres a year saved per new metre: a
# geometric grid, 12 steps to a factor of ten.
PRICES = np.geomspace(1e4, 1e7, 37)

# A saving below this many cost-metres is float32 rounding, not a new route:
# distances keep about seven significant figures, so the sum of three of
# them can undercut a 20 km route's true cost by a few thousandths of a metre.
EPS = 0.5


# --- Runs ---------------------------------------------------------------------


def _bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Degrees clockwise from north."""
    return np.degrees(np.arctan2(dx, dy)) % 360


def street_runs(edges: pd.DataFrame, seg: pd.DataFrame, xy: pd.DataFrame) -> pd.DataFrame:
    """Chain each street's directed edges into runs, one row per edge in order.

    Consecutive edges share a node and a street name (and borough), and each
    pairing takes the straightest continuation at that node. Turns sharper
    than ``config.CORRIDOR_MAX_TURN_DEG`` and U-turns end a run. Streets
    open to cars are chained in both directions, so the wrong way down a
    one-way street is a run too (a contraflow corridor). Blocks of the same
    name that are protected already, such as a rideable plaza, join the run
    but are never built. ``xy`` maps LION node id to ``x, y`` feet.
    Columns: ``run, pos, edge`` (the row label in ``edges``), ``u, v,
    street, boro``.
    """
    traf = seg.drop_duplicates("segment_id").set_index("segment_id")
    e = edges.assign(
        edge=edges.index,
        traf_dir=edges["segment_id"].map(traf["traf_dir"]).to_numpy(),
        boro=edges["segment_id"].map(traf["boro"]).to_numpy(),
    )
    protected = e["cost"] <= np.maximum(e["length_m"], MIN_COST) * (1 + 1e-9)
    e = e[(e["traf_dir"].isin(CAR_DIRS) | protected) & e["street"].notna()]
    e = e[e["street"].str.strip() != ""]
    # The graph keeps the cheapest edge per ordered node pair; chain that one.
    e = e.sort_values("cost", kind="stable").drop_duplicates(["u", "v"])
    e = e[e["u"].isin(xy.index) & e["v"].isin(xy.index) & (e["u"] != e["v"])]
    e = e.assign(
        bearing=_bearing(
            xy["x"].reindex(e["v"]).to_numpy() - xy["x"].reindex(e["u"]).to_numpy(),
            xy["y"].reindex(e["v"]).to_numpy() - xy["y"].reindex(e["u"]).to_numpy(),
        )
    )
    cols = ["edge", "u", "v", "street", "boro", "bearing"]
    joins = e[cols].merge(
        e[cols],
        left_on=["v", "street", "boro"],
        right_on=["u", "street", "boro"],
        suffixes=("_a", "_b"),
    )
    joins = joins[joins["v_b"] != joins["u_a"]]
    turn = (joins["bearing_b"] - joins["bearing_a"]).abs() % 360
    joins = joins.assign(turn=np.minimum(turn, 360 - turn))
    joins = joins[joins["turn"] <= config.CORRIDOR_MAX_TURN_DEG].sort_values("turn", kind="stable")

    nxt: dict[int, int] = {}
    has_prev: set[int] = set()
    for a, b in zip(joins["edge_a"].to_numpy(), joins["edge_b"].to_numpy(), strict=True):
        if a not in nxt and b not in has_prev:
            nxt[a] = b
            has_prev.add(b)

    rows: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    order = list(e["edge"].to_numpy())
    heads = [x for x in order if x not in has_prev]
    # Heads first; whatever is left lies on a closed loop, cut where it is entered.
    run = 0
    for start in heads + order:
        if start in seen:
            continue
        x, pos = start, 0
        while x is not None and x not in seen:
            seen.add(x)
            rows.append((run, pos, x))
            pos += 1
            x = nxt.get(x)
        run += 1
    out = pd.DataFrame(rows, columns=["run", "pos", "edge"])
    return out.merge(e[["edge", "u", "v", "street", "boro"]], on="edge", how="left")


# --- Scoring ------------------------------------------------------------------


@dataclass
class Runs:
    """Runs as flat arrays for the compiled scorers.

    Run ``r`` has nodes ``nodes[ptr[r]:ptr[r+1]]`` (one more than its
    edges). ``length`` is the running protected cost along the run, and
    ``new`` the running length of blocks not yet protected, so a stretch's
    new metres are a difference of two entries.
    """

    table: pd.DataFrame
    ptr: np.ndarray
    nodes: np.ndarray
    length: np.ndarray
    new: np.ndarray

    @classmethod
    def from_table(cls, table: pd.DataFrame, graph: Graph, edges: pd.DataFrame) -> Runs:
        """Index ``table`` (from :func:`street_runs`) against ``graph``; ``edges`` gives costs."""
        table = table.sort_values(["run", "pos"]).reset_index(drop=True)
        ed = edges.loc[table["edge"]]
        protected_cost = np.maximum(ed["length_m"].to_numpy(), MIN_COST)
        built = ed["cost"].to_numpy() <= protected_cost * (1 + 1e-9)
        sizes = table.groupby("run", sort=True).size().to_numpy()
        ptr = np.zeros(len(sizes) + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(sizes + 1)
        nodes = np.empty(ptr[-1], dtype=np.int64)
        length = np.zeros(ptr[-1])
        new = np.zeros(ptr[-1])
        u = graph.node_index(table["u"].to_numpy())
        v = graph.node_index(table["v"].to_numpy())
        # edge i of run r sits between node slots ptr[r]+pos and ptr[r]+pos+1
        slot = ptr[table["run"].to_numpy()] + table["pos"].to_numpy()
        nodes[slot] = u
        nodes[slot + 1] = v
        length[slot + 1] = protected_cost
        new[slot + 1] = np.where(built, 0.0, ed["length_m"].to_numpy())
        # Running totals within each run: a global cumsum less its value where the run starts.
        first_slot = np.repeat(ptr[:-1], sizes + 1)
        length = np.cumsum(length)
        length -= length[first_slot]
        new = np.cumsum(new)
        new -= new[first_slot]
        return cls(table=table, ptr=ptr, nodes=nodes, length=length, new=new)


@njit(cache=True)
def _accumulate(
    nd: np.ndarray,
    plen: np.ndarray,
    pair_o: np.ndarray,
    pair_d: np.ndarray,
    pair_cost: np.ndarray,
    pair_w: np.ndarray,
    dist_from: np.ndarray,
    dist_to: np.ndarray,
    saved: np.ndarray,
    trips: np.ndarray,
) -> None:  # pragma: no cover -- compiled; covered via stretch_matrix()
    """Add every pair's saving to ``saved[s, t]`` for each stretch from node ``s`` to ``t``."""
    k = nd.shape[0]
    enter = np.empty(k)
    leave = np.empty(k)
    tail = np.empty(k)
    for p in range(pair_o.shape[0]):
        o = pair_o[p]
        d = pair_d[p]
        for t in range(k):
            enter[t] = dist_from[o, nd[t]] - plen[t]
            leave[t] = dist_to[d, nd[t]] + plen[t]
        # tail[s]: the best via over any stretch that starts at s or later
        tail[k - 1] = np.inf
        best_leave = np.inf
        for s in range(k - 2, -1, -1):
            best_leave = min(best_leave, leave[s + 1])
            tail[s] = min(tail[s + 1], enter[s] + best_leave)
        bar = pair_cost[p] - EPS
        if tail[0] >= bar:
            continue  # not even the whole run helps, so no part of it does
        for s in range(k - 1):
            if tail[s] >= bar:
                break
            best_enter = enter[s]
            cur = np.inf
            for t in range(s + 1, k):
                cur = min(cur, best_enter + leave[t])
                if cur < bar:
                    saved[s, t] += pair_w[p] * (pair_cost[p] - cur)
                    trips[s, t] += pair_w[p]
                best_enter = min(best_enter, enter[t])


@njit(cache=True)
def _best_net(
    saved: np.ndarray, new: np.ndarray, min_project: float, price: float
) -> tuple[int, int]:  # pragma: no cover -- compiled; covered via best_stretches()
    """Return the stretch saving the most net of ``price`` per new metre, as (start, end); -1 if none pays."""
    best, bs, bt = 0.0, -1, -1
    k = new.shape[0]
    for s in range(k - 1):
        for t in range(s + 1, k):
            metres = new[t] - new[s]
            if saved[s, t] > 0.0 and metres > 0.0:
                x = saved[s, t] - price * max(metres, min_project)
                if x > best:
                    best, bs, bt = x, s, t
    return bs, bt


@njit(parallel=True, cache=True)
def _best_stretches(
    ptr: np.ndarray,
    nodes: np.ndarray,
    length: np.ndarray,
    new: np.ndarray,
    order: np.ndarray,
    pair_o: np.ndarray,
    pair_d: np.ndarray,
    pair_cost: np.ndarray,
    pair_w: np.ndarray,
    dist_from: np.ndarray,
    dist_to: np.ndarray,
    min_project: float,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """Per run and price, the stretch saving the most net of the price: start, end, saved, trips, blocks alone."""
    n = ptr.shape[0] - 1
    m = prices.shape[0]
    out_s = np.full((n, m), -1, dtype=np.int64)
    out_t = np.full((n, m), -1, dtype=np.int64)
    saved = np.zeros((n, m))
    trips = np.zeros((n, m))
    alone = np.zeros((n, m))
    for q in prange(n):
        r = order[q]
        a = ptr[r]
        k = ptr[r + 1] - a
        if new[a + k - 1] <= 0.0:
            continue  # all protected already
        f = np.zeros((k, k))
        u = np.zeros((k, k))
        _accumulate(
            nodes[a : a + k],
            length[a : a + k],
            pair_o,
            pair_d,
            pair_cost,
            pair_w,
            dist_from,
            dist_to,
            f,
            u,
        )
        for j in range(m):
            bs, bt = _best_net(f, new[a : a + k], min_project, prices[j])
            if bs < 0:
                break  # a higher price cannot pay either
            # Protected blocks at either end change nothing: drop them.
            while new[a + bs + 1] == new[a + bs]:
                bs += 1
            while new[a + bt] == new[a + bt - 1]:
                bt -= 1
            out_s[r, j], out_t[r, j] = bs, bt
            saved[r, j], trips[r, j] = f[bs, bt], u[bs, bt]
            for i in range(bs, bt):
                alone[r, j] += f[i, i + 1]
    return out_s, out_t, saved, trips, alone


@njit(cache=True)
def _helped(
    nd: np.ndarray,
    plen: np.ndarray,
    pair_o: np.ndarray,
    pair_d: np.ndarray,
    pair_cost: np.ndarray,
    dist_from: np.ndarray,
    dist_to: np.ndarray,
) -> np.ndarray:  # pragma: no cover -- compiled; covered via build_out()
    """Which pairs the whole of a stretch would help."""
    out = np.zeros(pair_o.shape[0], dtype=np.bool_)
    for p in range(pair_o.shape[0]):
        best_enter = dist_from[pair_o[p], nd[0]] - plen[0]
        cur = np.inf
        for t in range(1, nd.shape[0]):
            cur = min(cur, best_enter + dist_to[pair_d[p], nd[t]] + plen[t])
            best_enter = min(best_enter, dist_from[pair_o[p], nd[t]] - plen[t])
        out[p] = cur < pair_cost[p] - EPS
    return out


@dataclass
class Distances:
    """Cheapest costs from the sampled origins and to the sampled destinations."""

    pairs: pd.DataFrame  # o_row, d_row, weight, plus o and d node ids
    dist_from: np.ndarray
    dist_to: np.ndarray
    cost: np.ndarray  # each pair's cost today

    @classmethod
    def compute(cls, graph: Graph, pairs: pd.DataFrame, *, reverse: bool = True) -> Distances:
        """Run Dijkstra from every sampled origin, and back from every sampled destination.

        ``reverse=False`` skips the second half when only the pairs' costs are wanted.
        """
        pairs = pairs.groupby(["o", "d"], as_index=False)["weight"].sum()
        oi = graph.node_index(pairs["o"].to_numpy())
        di = graph.node_index(pairs["d"].to_numpy())
        origins, o_row = np.unique(oi, return_inverse=True)
        dests, d_row = np.unique(di, return_inverse=True)
        dist_from = candidates._distance_rows(graph.matrix, origins)  # noqa: SLF001
        dist_to = (
            candidates._distance_rows(graph.matrix.T.tocsr(), dests)  # noqa: SLF001
            if reverse
            else np.empty((0, 0), dtype=np.float32)
        )
        cost = dist_from[o_row, di].astype(np.float64)
        keep = np.isfinite(cost)
        pairs = pairs.assign(o_row=o_row, d_row=d_row)[keep].reset_index(drop=True)
        return cls(pairs=pairs, dist_from=dist_from, dist_to=dist_to, cost=cost[keep])

    def args(self) -> tuple[np.ndarray, ...]:
        """Return the pair arrays the compiled scorers take."""
        p = self.pairs
        return (
            p["o_row"].to_numpy(np.int64),
            p["d_row"].to_numpy(np.int64),
            self.cost,
            p["weight"].to_numpy(np.float64),
        )


def stretch_matrix(runs: Runs, run: int, dist: Distances) -> tuple[np.ndarray, np.ndarray]:
    """Return ``saved[s, t]`` and ``trips[s, t]`` for every stretch of one run (nodes ``s`` to ``t``)."""
    a, b = runs.ptr[run], runs.ptr[run + 1]
    k = b - a
    saved, trips = np.zeros((k, k)), np.zeros((k, k))
    o, d, cost, w = dist.args()
    _accumulate(
        runs.nodes[a:b], runs.length[a:b], o, d, cost, w, dist.dist_from, dist.dist_to, saved, trips
    )
    return saved, trips


def best_stretches(runs: Runs, dist: Distances, prices: np.ndarray) -> pd.DataFrame:
    """Each run's best stretch at each price, ascending: the most saved net of price x new metres.

    A stretch extends while each new metre saves at least the price, so a
    lower price gives a longer stretch. ``saved`` and ``trips`` are a year;
    ``ratio`` is saved per new metre (at least ``CORRIDOR_MIN_PROJECT_M``).
    """
    prices = np.sort(np.asarray(prices, dtype=np.float64))
    order = np.random.default_rng(0).permutation(len(runs.ptr) - 1)  # spread long runs over threads
    s, t, saved, trips, alone = _best_stretches(
        runs.ptr,
        runs.nodes,
        runs.length,
        runs.new,
        order,
        *dist.args(),
        dist.dist_from,
        dist.dist_to,
        float(config.CORRIDOR_MIN_PROJECT_M),
        prices,
    )
    run, j = np.nonzero(s >= 0)
    out = pd.DataFrame(
        {
            "run": run,
            "price": prices[j],
            "s": s[run, j],
            "t": t[run, j],
            "saved": saved[run, j],
            "trips": trips[run, j],
            "alone": alone[run, j],
        }
    )
    a = runs.ptr[out["run"].to_numpy()]
    out["new_m"] = runs.new[a + out["t"].to_numpy()] - runs.new[a + out["s"].to_numpy()]
    out["ratio"] = out["saved"] / np.maximum(out["new_m"], config.CORRIDOR_MIN_PROJECT_M)
    return out


def round_price(stretches: pd.DataFrame, room_m: float) -> float:
    """Return the highest price whose stretches, one per run, fill ``room_m`` of new lane.

    Scoring each run alone counts two parallel avenues twice, so the price
    comes out a little high; the next round sets it again on what is left.
    """
    filled = stretches.groupby("price")["new_m"].sum().sort_index()
    enough = filled[filled >= room_m]
    return float(enough.index.max()) if len(enough) else float(filled.index.min())


def helped(runs: Runs, run: int, s: int, t: int, dist: Distances) -> np.ndarray:
    """Boolean per pair: does protecting nodes ``s`` to ``t`` of ``run`` help it."""
    a = runs.ptr[run]
    o, d, cost, _ = dist.args()
    return _helped(
        runs.nodes[a + s : a + t + 1],
        runs.length[a + s : a + t + 1],
        o,
        d,
        cost,
        dist.dist_from,
        dist.dist_to,
    )


# --- Build-out ----------------------------------------------------------------


def stretch_edges(runs: Runs, run: int, s: int, t: int) -> np.ndarray:
    """Row labels in ``edges`` of the blocks between nodes ``s`` and ``t`` of ``run``."""
    first = runs.ptr[run] - run  # each earlier run has one more node than edges
    return runs.table["edge"].to_numpy()[first + s : first + t]


def stretch_value(
    runs: Runs, run: int, s: int, t: int, dist: Distances
) -> tuple[float, float, float]:
    """Stress saved and trips (a year) by protecting nodes ``s`` to ``t``, and its blocks' savings one at a time."""
    a = runs.ptr[run]
    k = t - s + 1
    saved, trips = np.zeros((k, k)), np.zeros((k, k))
    o, d, cost, w = dist.args()
    _accumulate(
        runs.nodes[a + s : a + t + 1],
        runs.length[a + s : a + t + 1],
        o,
        d,
        cost,
        w,
        dist.dist_from,
        dist.dist_to,
        saved,
        trips,
    )
    return saved[0, k - 1], trips[0, k - 1], float(np.diagonal(saved, 1).sum())


def protect(edges: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Give ``rows`` of ``edges`` a protected lane: ridden, at a cost of their length."""
    edges = edges.copy()
    edges.loc[rows, "facility"] = "separated"
    edges.loc[rows, "mode"] = "ride"
    edges.loc[rows, "factor"] = 1.0
    edges.loc[rows, "cost"] = np.maximum(edges.loc[rows, "length_m"], MIN_COST)
    return edges


def _pick(cands: pd.DataFrame, runs: Runs, dist: Distances, room_m: float) -> list[int]:
    """Choose this round's picks: best ratio first, skipping any that shares riders with one taken.

    Two stretches that help disjoint sets of trips are each worth what they
    were scored at, so both can be built in one round. One that shares more
    than ``CORRIDOR_MAX_SHARED`` of its riders with a stretch already taken
    waits for the next round's rescore.
    """
    w = dist.pairs["weight"].to_numpy()
    taken = np.zeros(len(w), dtype=bool)
    chosen: list[int] = []
    spent = 0.0
    for i, c in enumerate(cands.head(config.CORRIDOR_CANDIDATES).itertuples()):
        mask = helped(runs, c.run, c.s, c.t, dist)
        share = w[mask & taken].sum() / max(w[mask].sum(), 1e-9)
        if chosen and share > config.CORRIDOR_MAX_SHARED:
            continue
        chosen.append(i)
        taken |= mask
        spent += c.new_m
        if spent >= room_m:
            break
    return chosen


def build_out(
    edges: pd.DataFrame,
    table: pd.DataFrame,
    pairs: pd.DataFrame,
    budget_m: float,
    round_m: float | None = None,
    log: Callable[[str], None] = print,
    prices: np.ndarray = PRICES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily protect stretches until ``budget_m`` of new lane is spent.

    Each round scores every run against the network as built so far and
    sets a price per new metre (:func:`round_price`). Each run's stretch
    extends while every new metre saves at least that price. The round
    takes the best of those by saving per new metre, skipping competitors,
    up to ``round_m`` metres, protects them, and goes again. Returns (picks, the final edges). A pick's
    ``saved`` and ``trips`` (a year) are its marginal value when it was
    picked, given every earlier pick.
    """
    round_m = round_m or config.CORRIDOR_ROUND_KM * 1000
    picks = []
    spent, rnd = 0.0, 0
    while spent < budget_m:
        start = time.time()
        graph = Graph.from_edges(edges)
        runs = Runs.from_table(table, graph, edges)
        dist = Distances.compute(graph, pairs)
        grid = best_stretches(runs, dist, prices)
        if grid.empty:
            break
        price = round_price(grid, budget_m - spent)
        cands = grid[grid["price"] == price].sort_values("ratio", ascending=False)
        chosen = cands.iloc[_pick(cands, runs, dist, min(round_m, budget_m - spent))]
        for c in chosen.itertuples():
            rows = stretch_edges(runs, c.run, c.s, c.t)
            picks.append(
                {
                    "round": rnd,
                    "run": c.run,
                    "s": c.s,
                    "t": c.t,
                    "edges": rows,
                    "new_m": c.new_m,
                    "saved": c.saved,
                    "trips": c.trips,
                    "alone": c.alone,
                    "ratio": c.ratio,
                    "price": price,
                }
            )
            edges = protect(edges, rows)
        spent += chosen["new_m"].sum()
        log(
            f"  Round {rnd + 1}: {len(chosen)} stretches, {chosen['new_m'].sum() / 1000:.1f} km "
            f"({spent / 1000:.1f} km in all), price {price:,.0f} cost-m a year per new m, "
            f"{time.time() - start:.0f}s"
        )
        rnd += 1
        del dist
    return pd.DataFrame(picks), edges


# --- Reporting ----------------------------------------------------------------

_ABBREV = {
    "EAST": "E",
    "WEST": "W",
    "NORTH": "N",
    "SOUTH": "S",
    "STREET": "St",
    "AVENUE": "Ave",
    "BOULEVARD": "Blvd",
    "PLACE": "Pl",
    "ROAD": "Rd",
    "PARKWAY": "Pkwy",
    "DRIVE": "Dr",
    "BRIDGE": "Br",
    "EXPRESSWAY": "Expy",
    "SQUARE": "Sq",
    "TERRACE": "Ter",
    "COURT": "Ct",
    "LANE": "Ln",
    "HIGHWAY": "Hwy",
}


def short_name(street: str) -> str:
    """``EAST 49 STREET`` -> ``E 49 St``."""
    return " ".join(_ABBREV.get(w, w.title()) for w in str(street).split())


def cross_streets(seg: pd.DataFrame) -> pd.Series:
    """Every street name meeting each LION node, as a set."""
    ends = pd.concat(
        [
            seg[["u", "street"]].rename(columns={"u": "n"}),
            seg[["v", "street"]].rename(columns={"v": "n"}),
        ]
    )
    return ends.dropna().groupby("n")["street"].agg(set)


def node_label(nodes: list[int], at: int, street: str, names: pd.Series) -> str:
    """Return the cross street at position ``at`` of a run's nodes, else the nearest one along it."""
    for step in range(len(nodes)):
        for i in (at - step, at + step):
            if 0 <= i < len(nodes):
                others = names.get(nodes[i], set()) - {street}
                if others:
                    name = short_name(min(others, key=lambda x: (len(x), x)))
                    return name if step == 0 else f"near {name}"
    return "?"


def heading(dx: float, dy: float, boro: int) -> str:
    """Compass heading of a stretch; Manhattan's is against its street grid, 29 degrees east of north."""
    b = math.degrees(math.atan2(dx, dy)) - (config.MANHATTAN_GRID_DEG if boro == 1 else 0.0)
    return ["N", "E", "S", "W"][round((b % 360) / 90) % 4]


def merge_picks(picks: pd.DataFrame, original_new: np.ndarray, ptr: np.ndarray) -> pd.DataFrame:
    """Join picks on the same run that overlap or touch, across already protected blocks.

    ``original_new`` is :attr:`Runs.new` for the network before any pick, so
    a gap of blocks that were protected already does not split a corridor.
    """
    rows = []
    for run, group in picks.sort_values("s").groupby("run"):
        a = ptr[run]
        spans: list[list] = []
        for p in group.itertuples():
            if spans and original_new[a + p.s] <= original_new[a + spans[-1][1]]:
                last = spans[-1]
                last[1] = max(last[1], p.t)
                last[2].append(p.Index)
            else:
                spans.append([p.s, p.t, [p.Index]])
        for s, t, members in spans:
            m = picks.loc[members]
            rows.append(
                {
                    "run": run,
                    "s": s,
                    "t": t,
                    "round": m["round"].min(),
                    "first_pick": min(members),
                    "picks": len(members),
                    "new_m": m["new_m"].sum(),
                    "saved_in_plan": m["saved"].sum(),
                }
            )
    return pd.DataFrame(rows).sort_values("first_pick").reset_index(drop=True)
