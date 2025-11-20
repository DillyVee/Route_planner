"""
Standalone directed graph implementation without GUI dependencies.

This module provides the DirectedGraph class and haversine distance function
without requiring PyQt6 or other GUI libraries. This allows the DRPP pipeline
to work in headless environments.
"""

import heapq
from math import asin, cos, radians, sin, sqrt
from typing import List, Tuple, Optional

# Try to import V4 path reconstruction, fall back to simple method
try:
    from drpp_core.path_reconstruction import reconstruct_path
    V4_AVAILABLE = True
except ImportError:
    V4_AVAILABLE = False


def haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculate distance between two lat/lon points in meters.

    Args:
        a: First point as (latitude, longitude)
        b: Second point as (latitude, longitude)

    Returns:
        Distance in meters
    """
    lat1, lon1 = a
    lat2, lon2 = b
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(aa))
    return 6371000 * c


class DirectedGraph:
    """
    Directed graph with Dijkstra shortest path.

    This is a lightweight implementation without GUI dependencies,
    extracted from Route_Planner.py for use in headless environments.
    """

    def __init__(self):
        """Initialize empty directed graph."""
        self.node_to_id = {}  # Dict[Coordinate, int]
        self.id_to_node = []  # List[Coordinate]
        self.adj = []  # List[List[Tuple[int, float]]]

    def _ensure(self, node: Tuple[float, float]) -> int:
        """
        Ensure node exists in graph, add if missing.

        Args:
            node: Coordinate tuple (lat, lon)

        Returns:
            Node ID (integer index)
        """
        if node in self.node_to_id:
            return self.node_to_id[node]
        idx = len(self.id_to_node)
        self.node_to_id[node] = idx
        self.id_to_node.append(node)
        self.adj.append([])
        return idx

    def add_edge(self, a: Tuple[float, float], b: Tuple[float, float], w: float) -> None:
        """
        Add directed edge from a to b with weight w.

        Args:
            a: Start node (lat, lon)
            b: End node (lat, lon)
            w: Edge weight (distance in meters)
        """
        ia = self._ensure(a)
        ib = self._ensure(b)
        # Avoid duplicate edges
        if not any(v == ib for v, _ in self.adj[ia]):
            self.adj[ia].append((ib, w))

    def dijkstra(self, source_id: int, max_distance: Optional[float] = None) -> Tuple[List[float], List[int]]:
        """
        Compute shortest paths from source to all nodes using Dijkstra's algorithm.

        Args:
            source_id: Starting node ID
            max_distance: Optional distance threshold for early termination (meters)
                         If specified, stops exploring nodes beyond this distance

        Returns:
            Tuple of (distances, predecessors) where:
                - distances[node_id] = shortest distance from source
                - predecessors[node_id] = previous node in shortest path

        Raises:
            ValueError: If source_id is out of bounds
        """
        n = len(self.id_to_node)

        # Validate source_id bounds
        if source_id < 0 or source_id >= n:
            raise ValueError(f"source_id {source_id} out of bounds [0, {n})")

        dist = [float("inf")] * n
        prev = [-1] * n
        dist[source_id] = 0.0
        h = [(0.0, source_id)]

        while h:
            d, u = heapq.heappop(h)

            # Early termination: stop if we've exceeded max distance
            if max_distance and d > max_distance:
                break

            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(h, (nd, v))

        return dist, prev

    def shortest_path(self, start_node: Tuple[float, float], end_node: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], float]:
        """
        Find shortest path between two nodes.

        Args:
            start_node: Starting coordinate (lat, lon)
            end_node: Ending coordinate (lat, lon)

        Returns:
            Tuple of (path_coordinates, distance) where:
                - path_coordinates is list of (lat, lon) points, or None if no path exists
                - distance is total path distance in meters, or inf if no path exists
        """
        if start_node not in self.node_to_id or end_node not in self.node_to_id:
            return None, float("inf")

        s = self.node_to_id[start_node]
        t = self.node_to_id[end_node]

        dist, prev = self.dijkstra(s)

        if dist[t] == float("inf"):
            return None, float("inf")

        # Use robust path reconstruction with cycle detection
        if V4_AVAILABLE:
            path_ids = reconstruct_path(prev, s, t)
            if not path_ids:
                return None, float("inf")
        else:
            # Fallback to simple reconstruction with iteration limit
            cur = t
            rev = []
            max_iterations = len(self.id_to_node)
            for _ in range(max_iterations):
                rev.append(cur)
                if cur == -1:
                    break
                cur = prev[cur]
            rev.reverse()
            path_ids = rev

        coords = [self.id_to_node[i] for i in path_ids]
        return coords, dist[t]
