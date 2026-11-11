"""What the corridor plan changes for riders: every trip routed again with it built.

``corridors`` picks the lanes on a sample of station pairs. This routes all of
the year's trips on the network with the plan's new blocks protected and sets
the result beside ``assign``'s routes today, trip by trip. The trips are held
fixed: the same riders between the same stations. So it shows where they
would ride and how much less stress it would cost them, not how many more
people would ride.

A map of where trips move shows only the riders who change route. Many of the
trips that gain keep theirs, because the lane went on the street they already
ride; :func:`summarize` counts them, and the facility mix of the trip-km shows
what they gain.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.sparse import csgraph

from . import config
from .assignment import trip_km_by_facility
from .corridors import EPS, cross_streets, protect, short_name

if TYPE_CHECKING:
    from .network import Graph

KEY = ["segment_id", "dir", "u", "v"]
PLACE_SEARCH_M = 400  # how far along its street a mid-block node looks for a cross street
# How a route's meters are split, in the viewer's order.
MIX = ("separated", "painted", "shared", "none", "walked")


def plan_edges(edges: pd.DataFrame, blocks: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``edges`` with the plan's new blocks protected, and those blocks' rows.

    ``blocks`` is the ``corridors`` command's ``corridor_edges.parquet``; a
    block it marks ``new`` must be in ``edges``. ``edges`` must have a
    default index, as read from ``edges.parquet``.
    """
    new = blocks.loc[blocks["new"], KEY].drop_duplicates()
    hit = pd.MultiIndex.from_frame(edges[KEY]).isin(pd.MultiIndex.from_frame(new))
    rows = np.flatnonzero(hit)
    if len(rows) != len(new):
        msg = (
            f"{len(new) - len(rows)} of the plan's new blocks are not in the network; "
            "run `corridors` again on this build"
        )
        raise ValueError(msg)
    return protect(edges, rows), rows


def compare(today: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Join each station pair's route today and with the plan, keeping pairs reachable in both.

    Both are :func:`assignment.assign` route tables. Columns: ``o, d, trips``
    and ``cost``, ``length_m``, ``walk_m`` suffixed ``_today`` and ``_plan``.
    """
    cols = ["o", "d", "cost", "length_m", "walk_m", "reachable"]
    pairs = today[[*cols, "trips"]].merge(plan[cols], on=["o", "d"], suffixes=("_today", "_plan"))
    ok = pairs["reachable_today"] & pairs["reachable_plan"]
    return pairs[ok].drop(columns=["reachable_today", "reachable_plan"]).reset_index(drop=True)


def gains(pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each pair's stress saved per trip, whether it gains, and whether it keeps its route."""
    gain = (pairs["cost_today"] - pairs["cost_plan"]).to_numpy()
    helped = gain > EPS
    same = (
        helped
        & (
            np.abs(pairs["length_m_plan"] - pairs["length_m_today"]) < config.SAME_ROUTE_M
        ).to_numpy()
        & (np.abs(pairs["walk_m_plan"] - pairs["walk_m_today"]) < config.SAME_ROUTE_M).to_numpy()
    )
    return gain, helped, same


def summarize(
    pairs: pd.DataFrame,
    edges: pd.DataFrame,
    plan: pd.DataFrame,
    trips_today: np.ndarray,
    trips_plan: np.ndarray,
    trips_shortest: np.ndarray,
    new_rows: np.ndarray,
) -> dict[str, Any]:
    """Sum up the plan's effect on the year's trips, as plain numbers for the viewer and the log.

    ``trips_*`` are trips a year on each row of ``edges`` (``plan`` for
    ``trips_plan``): routed today, with the plan, and on the shortest legal
    route. Shares are of trips or of trip-km; ``detour`` is how much longer
    the trip-km are than on the shortest routes.
    """
    w = pairs["trips"].to_numpy()
    gain, helped, same = gains(pairs)
    cost_today = (w * pairs["cost_today"]).sum()
    length = edges["length_m"].to_numpy()
    km = {
        "today": (trips_today * length).sum() / 1000,
        "plan": (trips_plan * length).sum() / 1000,
        "shortest": (trips_shortest * length).sum() / 1000,
    }
    share = {}
    for when, e, t in (("today", edges, trips_today), ("plan", plan, trips_plan)):
        by = trip_km_by_facility(e, t)
        share[when] = {k: round(float(by.get(k, 0.0) / by.sum()), 4) for k in MIX}
    walked = {when: float((w * pairs[f"walk_m_{when}"]).sum() / 1000) for when in ("today", "plan")}
    return {
        "year": config.DEMAND_YEAR,
        "trips": round(float(w.sum())),
        "saved": round(float((w * gain).sum() / 1000)),
        "saved_share": round(float((w * gain).sum() / cost_today), 4),
        "helped_day": round(float(w[helped].sum() / 365)),
        "helped_share": round(float(w[helped].sum() / w.sum()), 4),
        # of the trips that gain: those whose route stays put, and their part of the saving
        "same_route_share": round(float(w[same].sum() / max(w[helped].sum(), 1)), 4),
        "same_route_saved": round(float((w * gain)[same].sum() / max((w * gain).sum(), 1)), 4),
        "trip_km": {k: round(v) for k, v in km.items()},
        "detour": {
            "today": round(km["today"] / km["shortest"] - 1, 4),
            "plan": round(km["plan"] / km["shortest"] - 1, 4),
        },
        "cost_per_km": {
            when: round(
                float((w * pairs[f"cost_{when}"]).sum() / (w * pairs[f"length_m_{when}"]).sum()),
                3,
            )
            for when in ("today", "plan")
        },
        "walked_km": {k: round(v) for k, v in walked.items()},
        "share": share,
        "on_new": round(
            float((trips_plan[new_rows] * length[new_rows]).sum() / 1000 / km["plan"]), 4
        ),
    }


def station_gains(pairs: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Per network node with a station: its trips and the share of their stress the plan saves.

    A trip counts at both ends. ``stations`` is ``assign``'s
    ``stations.parquet``; stations snapped to one node are pooled, placed at
    their trip-weighted mean position. Columns: ``node, lat, lng,
    stations, trips`` (ends a year), ``saved_share``, ``helped_share``.
    """
    w = pairs["trips"].to_numpy()
    gain, helped, _ = gains(pairs)
    part = pd.DataFrame(
        {
            "trips": w,
            "saved": w * gain,
            "cost": w * pairs["cost_today"].to_numpy(),
            "helped": np.where(helped, w, 0.0),
        }
    )
    ends = pd.concat(
        [part.assign(node=pairs["o"].to_numpy()), part.assign(node=pairs["d"].to_numpy())]
    )
    by = ends.groupby("node").sum()
    st = stations[stations["node"] >= 0]
    pos = (
        st.assign(wlat=st["lat"] * st["ends"], wlng=st["lng"] * st["ends"])
        .groupby("node")
        .agg(
            wlat=("wlat", "sum"),
            wlng=("wlng", "sum"),
            ends=("ends", "sum"),
            stations=("station", "size"),
        )
    )
    out = pos.join(by, how="inner")
    return pd.DataFrame(
        {
            "node": out.index.to_numpy(),
            "lat": (out["wlat"] / out["ends"]).to_numpy(),
            "lng": (out["wlng"] / out["ends"]).to_numpy(),
            "stations": out["stations"].to_numpy(),
            "trips": out["trips"].to_numpy(),
            "saved_share": (out["saved"] / out["cost"]).to_numpy(),
            "helped_share": (out["helped"] / out["trips"]).to_numpy(),
        }
    )


def route_paths(graph: Graph, o: np.ndarray, d: np.ndarray, batch: int = 64) -> list[np.ndarray]:
    """Edge rows along the cheapest route of each pair (LION node ids), origin first.

    The same routes :func:`assignment.assign` counts: the same matrix and
    Dijkstra. An unreachable pair gets an empty array.
    """
    oi, di = graph.node_index(np.asarray(o)), graph.node_index(np.asarray(d))
    ptr, col, row = graph.edge_of.indptr, graph.edge_of.indices, graph.edge_of.data - 1
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(oi)
    order = np.argsort(oi, kind="stable")
    origins, starts = np.unique(oi[order], return_index=True)
    bounds = np.append(starts, len(oi))
    for b in range(0, len(origins), batch):
        chunk = origins[b : b + batch]
        _, pred = csgraph.dijkstra(
            graph.matrix, directed=True, indices=chunk, return_predecessors=True
        )
        for r, origin in enumerate(chunk):
            for j in order[bounds[b + r] : bounds[b + r + 1]]:
                node, path = di[j], []
                while node != origin:
                    p = pred[r, node]
                    if p < 0:
                        path = []
                        break
                    lo, hi = ptr[p], ptr[p + 1]
                    path.append(row[lo + np.flatnonzero(col[lo:hi] == node)[0]])
                    node = p
                out[j] = np.array(path[::-1], dtype=np.int64)
    return out


def route_mix(edges: pd.DataFrame, rows: np.ndarray) -> dict[str, float]:
    """Meters of a route by what it rides on (``MIX``); walked meters whatever the facility."""
    e = edges.iloc[rows]
    kind = np.where(e["mode"] == "walk", "walked", e["facility"])
    m = pd.Series(e["length_m"].to_numpy()).groupby(kind).sum()
    return {k: float(m.get(k, 0.0)) for k in MIX}


def examples(
    graph_today: Graph,
    graph_plan: Graph,
    pairs: pd.DataFrame,
    corridor: np.ndarray,
    new_m: np.ndarray,
    candidates: int | None = None,
) -> pd.DataFrame:
    """One real trip per corridor that shows what it changes, today and with the plan.

    ``corridor`` gives each edge row's corridor number (-1 for none) and
    ``new_m`` its meters of new lane (0 unless the plan builds it). From the
    ``candidates`` pairs that gain most (trips times stress saved), each
    corridor takes the one scoring highest on that gain times the share of
    the corridor's new lane its route with the plan rides. One pair can be
    the example for several corridors; it is listed once, with all of them in
    ``corridors``, in order of that score. Sorted by gain.
    """
    candidates = candidates or config.EXAMPLE_CANDIDATES
    gain, _, _ = gains(pairs)
    weighted = pairs["trips"].to_numpy() * gain
    top = np.argsort(-weighted, kind="stable")[: min(candidates, int((weighted > EPS).sum()))]
    paths = route_paths(graph_plan, pairs["o"].to_numpy()[top], pairs["d"].to_numpy()[top])
    total = pd.Series(new_m).groupby(corridor).sum()
    best: dict[int, tuple[float, int]] = {}
    for j, path in zip(top, paths, strict=True):
        c, m = corridor[path], new_m[path]
        on = (c >= 0) & (m > 0)
        if not on.any():
            continue
        for ci, meters in pd.Series(m[on]).groupby(c[on]).sum().items():
            score = weighted[j] * min(1.0, meters / total[ci])
            if score > best.get(int(ci), (0.0, -1))[0]:
                best[int(ci)] = (score, int(j))
    if not best:
        return pd.DataFrame()
    by_pair: dict[int, list[tuple[float, int]]] = {}
    for ci, (score, j) in best.items():
        by_pair.setdefault(j, []).append((score, ci))
    chosen = sorted(by_pair, key=lambda j: -weighted[j])
    plan_path = dict(zip(top, paths, strict=True))
    today_paths = route_paths(
        graph_today, pairs["o"].to_numpy()[chosen], pairs["d"].to_numpy()[chosen]
    )
    rows = []
    for j, today in zip(chosen, today_paths, strict=True):
        p = pairs.iloc[j]
        mix_today = route_mix(graph_today.edges, today)
        mix_plan = route_mix(graph_plan.edges, plan_path[j])
        rows.append(
            {
                "o": int(p["o"]),
                "d": int(p["d"]),
                "trips": float(p["trips"]),
                "corridors": [ci for _, ci in sorted(by_pair[j], reverse=True)],
                "cost_today": float(p["cost_today"]),
                "cost_plan": float(p["cost_plan"]),
                **{f"today_{k}_m": v for k, v in mix_today.items()},
                **{f"plan_{k}_m": v for k, v in mix_plan.items()},
                "new_m": float(new_m[plan_path[j]].sum()),
                "today_edges": today.tolist(),
                "plan_edges": plan_path[j].tolist(),
            }
        )
    return pd.DataFrame(rows)


def distinct(
    examples: pd.DataFrame,
    edges: pd.DataFrame,
    overlap: float | None = None,
    min_m: float | None = None,
    min_moved: float | None = None,
) -> pd.DataFrame:
    """Sort example trips by the stress they save a year, and mark the ones to list.

    The stress saved is trips times the drop in cost, so a long, busy trip
    outranks a short one that rides nothing but new lane. A trip is listed
    when both its routes are at least ``min_m`` long, at least ``min_moved``
    of its route with the plan is on streets today's route does not ride
    (a trip whose route stays put shows little on the map), and it is unlike
    every trip listed above it. Neighboring corridors often pick much the
    same trip, sometimes the other way: two trips are alike when the streets
    they share, in either direction and on either route, make up more than
    ``overlap`` of the shorter one's. Every trip stays, so each corridor
    keeps its own.
    """
    overlap = config.EXAMPLE_OVERLAP if overlap is None else overlap
    min_m = config.EXAMPLE_MIN_M if min_m is None else min_m
    min_moved = config.EXAMPLE_MIN_MOVED if min_moved is None else min_moved
    if examples.empty:
        return examples.assign(listed=pd.Series(dtype=bool))
    saved = 1 - examples["cost_plan"] / examples["cost_today"]
    saved_year = examples["trips"] * (examples["cost_today"] - examples["cost_plan"])
    out = (
        examples.assign(saved_share=saved, saved_year=saved_year)
        .sort_values(["saved_year", "trips"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    sid = edges["segment_id"].to_numpy()
    meters = edges["length_m"].to_numpy()
    length = edges.drop_duplicates("segment_id").set_index("segment_id")["length_m"]
    listed: list[int] = []
    streets: dict[int, set] = {}
    total: dict[int, float] = {}
    for i, (today, plan) in enumerate(zip(out["today_edges"], out["plan_edges"], strict=True)):
        t, p = np.asarray(today, dtype=int), np.asarray(plan, dtype=int)
        plan_m = meters[p].sum()
        moved = meters[p][~np.isin(sid[p], sid[t])].sum()
        if min(meters[t].sum(), plan_m) < min_m or moved < min_moved * plan_m:
            continue
        streets[i] = set(sid[np.concatenate([t, p])])
        total[i] = float(length.reindex(list(streets[i])).sum())
        alike = any(
            float(length.reindex(list(streets[i] & streets[j])).sum())
            > overlap * min(total[i], total[j])
            for j in listed
        )
        if not alike:
            listed.append(i)
    out["listed"] = False
    out.loc[listed, "listed"] = True
    return out


def place_names(seg: pd.DataFrame) -> pd.Series:
    """Name every LION node by the two shortest street names meeting there: ``5 Ave & W 49 St``.

    A node partway along a block, where only one street meets, is named for
    the nearest cross street along that street, by length:
    ``Greenwich Ave near Charles St``. With none within ``PLACE_SEARCH_M``
    it keeps the street's name alone.
    """
    names = cross_streets(seg)
    shortest = lambda streets: sorted(streets, key=lambda s: (len(s), s))  # noqa: E731
    along: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for u, v, street, m in zip(seg["u"], seg["v"], seg["street"], seg["length_m"], strict=True):
        along.setdefault((u, street), []).append((v, m))
        along.setdefault((v, street), []).append((u, m))

    def near(node: int, street: str) -> str | None:
        best, todo, seen = {node: 0.0}, [(0.0, node)], set()
        while todo:
            d, n = heapq.heappop(todo)
            if n in seen or d > PLACE_SEARCH_M:
                continue
            seen.add(n)
            others = names.get(n, set()) - {street}
            if others:
                return shortest(others)[0]
            for nxt, m in along.get((n, street), []):
                if d + m < best.get(nxt, np.inf):
                    best[nxt] = d + m
                    heapq.heappush(todo, (d + m, nxt))
        return None

    def name(node: int, streets: set[str]) -> str:
        picked = shortest(streets)[:2]
        if len(picked) == 2:
            return " & ".join(short_name(s) for s in sorted(picked))
        cross = near(node, picked[0])
        return (
            f"{short_name(picked[0])} near {short_name(cross)}" if cross else short_name(picked[0])
        )

    return pd.Series({n: name(n, s) for n, s in names.items()}, dtype=object)
