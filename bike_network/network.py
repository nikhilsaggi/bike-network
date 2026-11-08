"""Turn LION segments and DOT facilities into a directed, costed edge table.

Every segment yields up to two directed edges: ``fwd`` from LION's FROM node to
its TO node and ``rev`` back. On each, a bike either rides (where riding that
way is legal) or is walked on the sidewalk (anywhere a pedestrian may go),
whichever costs less. Riding is legal with the flow of traffic, and in any
direction DOT has a bike facility for -- which is how contraflow lanes and
greenways get their directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from . import config

# Floor on an edge's cost, so a zero-length segment is still an edge in a
# sparse matrix (an explicit 0 can be dropped as "no edge").
MIN_COST = 0.01

CAR_DIRS = frozenset({"T", "W", "A"})


def direction_volumes(seg: pd.DataFrame, volumes: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``vol_fwd`` / ``vol_rev``: vehicles/day a rider shares each direction with.

    ``volumes`` holds ``segment_id, dir, volume_est`` for each direction cars
    travel (:func:`traffic.estimate`). Riding with traffic, that is the traffic
    in the rider's direction. Riding a contraflow lane on a one-way street,
    it is the street's one stream of oncoming traffic. A direction with no
    estimate takes the other direction's, then ``VOLUME_REFERENCE``.
    """
    seg = seg.copy()
    if volumes is None:
        seg["vol_fwd"] = seg["vol_rev"] = float(config.VOLUME_REFERENCE)
        return seg
    est = volumes.set_index(["segment_id", "dir"])["volume_est"]
    est = est[~est.index.duplicated()]
    own = {
        d: pd.Series(
            est.reindex(pd.MultiIndex.from_arrays([seg["segment_id"], [d] * len(seg)])).to_numpy(),
            index=seg.index,
        )
        for d in ("fwd", "rev")
    }
    seg["vol_fwd"] = own["fwd"].fillna(own["rev"]).fillna(config.VOLUME_REFERENCE)
    seg["vol_rev"] = own["rev"].fillna(own["fwd"]).fillna(config.VOLUME_REFERENCE)
    return seg


def traffic_penalty(seg: pd.DataFrame, volume: pd.Series) -> pd.Series:
    """Traffic penalty for one direction of each segment; 0 where no car is allowed."""
    doublings = np.log2(volume / config.VOLUME_REFERENCE).clip(lower=0)
    speed = seg["speed_mph"].fillna(config.DEFAULT_SPEED_MPH)
    over = ((speed - config.DEFAULT_SPEED_MPH) / 5).clip(lower=0)
    truck = seg["truck_route"].map(config.TRUCK_PENALTY).fillna(0.0)
    penalty = (
        config.TRAFFIC_BASE
        + config.TRAFFIC_PER_DOUBLING * doublings
        + config.TRAFFIC_PER_5_MPH * over
        + truck
    )
    return penalty.where(seg["traf_dir"].isin(CAR_DIRS), 0.0)


def build_edges(seg: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    """Return one row per traversable directed edge.

    ``seg`` is :func:`lion.prepare` output, with ``vol_fwd``/``vol_rev`` from
    :func:`direction_volumes` (every street counts as quiet without them), and
    ``fac`` is :func:`lanes.facilities` output. Columns: ``segment_id, dir,
    u, v, length_m, facility, volume, mode, factor, cost`` plus the inputs
    that set them.
    """
    if "vol_fwd" not in seg:
        seg = direction_volumes(seg, None)
    seg = seg.merge(fac, on="segment_id", how="left")
    seg["fwd"] = seg["fwd"].fillna("none")
    seg["rev"] = seg["rev"].fillna("none")
    step = seg["rw_type"] == config.RW_STEP_STREET
    no_ride = step | seg["vehicle_only"] | (seg["rw_type"] == config.RW_HIGHWAY)
    can_walk = ~seg["vehicle_only"] & (seg["rw_type"] != config.RW_HIGHWAY)
    walk_factor = np.where(step, config.STEP_FACTOR, config.WALK_FACTOR)

    parts = []
    for direction, u, v, with_traffic in (
        ("fwd", "u", "v", ("T", "W")),
        ("rev", "v", "u", ("T", "A")),
    ):
        facility = seg[direction]
        volume = seg[f"vol_{direction}"]
        penalty = traffic_penalty(seg, volume)
        has_lane = facility != "none"
        cars_allow = seg["traf_dir"].isin(with_traffic) & ~no_ride
        can_ride = cars_allow | has_lane
        exposure = facility.map(config.EXPOSURE)
        ride_factor = 1.0 + exposure * penalty
        ride = can_ride & (~can_walk | (ride_factor <= walk_factor))
        part = pd.DataFrame(
            {
                "segment_id": seg["segment_id"],
                "dir": direction,
                "u": seg[u],
                "v": seg[v],
                "street": seg["street"],
                "length_m": seg["length_m"],
                "facility": facility,
                "volume": volume.where(seg["traf_dir"].isin(CAR_DIRS)),
                "traffic_penalty": penalty,
                "mode": np.where(ride, "ride", "walk"),
                "factor": np.where(ride, ride_factor, walk_factor),
            }
        )
        parts.append(part[can_ride | can_walk])
    edges = pd.concat(parts, ignore_index=True)
    edges["cost"] = (edges["length_m"] * edges["factor"]).clip(lower=MIN_COST)
    return edges


@dataclass
class Graph:
    """A routable view of the edge table.

    ``matrix[i, j]`` is the cheapest edge from node ``nodes[i]`` to
    ``nodes[j]``; ``edge_of`` holds that edge's row in ``edges`` at the same
    sparse position, so a route found on the matrix maps back to segments.
    """

    edges: pd.DataFrame
    nodes: np.ndarray
    matrix: sparse.csr_array
    edge_of: sparse.csr_array

    @classmethod
    def from_edges(cls, edges: pd.DataFrame, weight: str = "cost") -> Graph:
        """Index nodes and keep the cheapest edge per ordered node pair."""
        nodes = np.unique(np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()]))
        ui = np.searchsorted(nodes, edges["u"].to_numpy())
        vi = np.searchsorted(nodes, edges["v"].to_numpy())
        w = edges[weight].to_numpy(dtype=float)
        order = np.lexsort((w, vi, ui))
        ui, vi, w = ui[order], vi[order], w[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = (ui[1:] != ui[:-1]) | (vi[1:] != vi[:-1])
        n = len(nodes)
        shape = (n, n)
        matrix = sparse.csr_array((w[first], (ui[first], vi[first])), shape=shape)
        # +1 so edge row 0 is not an implicit zero.
        edge_of = sparse.csr_array((order[first] + 1, (ui[first], vi[first])), shape=shape)
        return cls(edges=edges, nodes=nodes, matrix=matrix, edge_of=edge_of)

    def node_index(self, node_ids: np.ndarray) -> np.ndarray:
        """Matrix indices of LION node ids (all must be in the graph)."""
        idx = np.searchsorted(self.nodes, node_ids)
        if (idx >= len(self.nodes)).any() or (
            self.nodes[idx.clip(max=len(self.nodes) - 1)] != node_ids
        ).any():
            msg = "node id not in graph"
            raise KeyError(msg)
        return idx

    def strong_components(self) -> np.ndarray:
        """Label each node by strongly connected component."""
        _, labels = csgraph.connected_components(self.matrix, directed=True, connection="strong")
        return labels
