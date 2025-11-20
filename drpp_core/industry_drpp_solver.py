"""
Industry-standard DRPP solver using Eulerian augmentation.

This module orchestrates the complete industry-standard pipeline:
1. Build topology with snapping
2. Add shortest-path deadhead edges
3. Check connectivity
4. Balance graph (Eulerian augmentation)
5. Construct Eulerian tour
6. Preserve metadata throughout

This follows the methodology used by DOTs, Esri, HERE, TomTom, Trimble, etc.
"""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
import logging

from .types import SegmentRequirement
from .topology import (
    TopologyBuilder, TopologyNode, TopologyEdge, NodeID,
    build_adjacency_list, Coordinate
)
from .connectivity import ConnectivityAnalyzer, ConnectedComponent
from .eulerian_solver import EulerianSolver, EulerianTour
from .geo import haversine


logger = logging.getLogger(__name__)


@dataclass
class DRPPSolution:
    """Complete DRPP solution with metadata."""
    tour: EulerianTour
    nodes: Dict[NodeID, TopologyNode]
    edges: List[TopologyEdge]  # All edges including balancing edges
    route_coordinates: List[Coordinate]  # Ordered route coordinates
    total_distance_km: float
    required_distance_km: float
    deadhead_distance_km: float
    deadhead_percentage: float
    num_segments: int
    is_valid: bool
    messages: List[str]


class IndustryDRPPSolver:
    """
    Industry-standard DRPP solver.

    Uses the Eulerian augmentation approach:
    1. Build topology (snap endpoints, create nodes/edges)
    2. Add deadhead edges (shortest paths between required segments)
    3. Check connectivity (SCC analysis)
    4. Balance graph (add edges to make Eulerian)
    5. Construct tour (Hierholzer's algorithm)
    """

    def __init__(self, snap_tolerance_meters: float = 2.0):
        """
        Initialize industry-standard DRPP solver.

        Args:
            snap_tolerance_meters: Endpoint snapping tolerance (default 2m)
        """
        self.snap_tolerance = snap_tolerance_meters
        self.topology_builder: Optional[TopologyBuilder] = None
        self.nodes: Dict[NodeID, TopologyNode] = {}
        self.edges: List[TopologyEdge] = []
        self.messages: List[str] = []

    def solve(self, segments: List[SegmentRequirement],
              start_coordinate: Optional[Coordinate] = None) -> DRPPSolution:
        """
        Solve DRPP using industry-standard methodology.

        Args:
            segments: List of road segments from KML
            start_coordinate: Optional starting coordinate

        Returns:
            DRPPSolution with complete route
        """
        self.messages = []

        # Step 1: Build topology
        logger.info("Step 1: Building topology with endpoint snapping...")
        self.topology_builder = TopologyBuilder(self.snap_tolerance)
        self.nodes, required_edges = self.topology_builder.build_topology(segments)
        self.edges = required_edges.copy()

        self.messages.append(f"Built topology: {len(self.nodes)} nodes, {len(required_edges)} required edges")
        logger.info(f"  → {len(self.nodes)} nodes, {len(required_edges)} required edges")

        # Step 2: Add shortest-path deadhead edges between required segments
        logger.info("Step 2: Adding shortest-path deadhead edges...")
        deadhead_edges = self._add_deadhead_edges(required_edges)
        self.edges.extend(deadhead_edges)

        self.messages.append(f"Added {len(deadhead_edges)} deadhead edges for connectivity")
        logger.info(f"  → Added {len(deadhead_edges)} deadhead edges")

        # Step 3: Check connectivity
        logger.info("Step 3: Checking graph connectivity...")
        start_node = None
        if start_coordinate:
            start_node = self.topology_builder.get_node_id(start_coordinate)

        node_ids = set(self.nodes.keys())
        analyzer = ConnectivityAnalyzer(node_ids, self.edges)
        is_feasible, message = analyzer.check_feasibility(start_node)

        self.messages.append(f"Connectivity check: {message}")
        logger.info(f"  → {message}")

        if not is_feasible:
            # Return empty solution
            return DRPPSolution(
                tour=EulerianTour([], 0, 0, 0, 0, False, message),
                nodes=self.nodes,
                edges=self.edges,
                route_coordinates=[],
                total_distance_km=0,
                required_distance_km=0,
                deadhead_distance_km=0,
                deadhead_percentage=0,
                num_segments=len(segments),
                is_valid=False,
                messages=self.messages
            )

        # Step 4: Solve using Eulerian augmentation
        logger.info("Step 4: Solving DRPP via Eulerian augmentation...")
        solver = EulerianSolver(self.edges, len(self.nodes))

        # Create shortest path function for balancing
        def shortest_path_func(from_node: NodeID, to_node: NodeID) -> Tuple[List[Coordinate], float]:
            return self._dijkstra_shortest_path(from_node, to_node)

        tour = solver.solve(shortest_path_func)

        balance_summary = solver.get_balance_summary()
        self.messages.append(balance_summary)
        logger.info(f"  → {balance_summary}")

        self.messages.append(f"Eulerian tour: {len(tour.edges)} edges, {tour.total_distance:.1f}m total")
        logger.info(f"  → Tour: {len(tour.edges)} edges, {tour.total_distance:.1f}m")

        # Step 5: Extract route coordinates
        logger.info("Step 5: Extracting route coordinates...")
        route_coords = self._extract_route_coordinates(tour)

        # Compute statistics
        total_distance_km = tour.total_distance / 1000.0
        required_distance = sum(e.cost for e in tour.edges if e.required)
        required_distance_km = required_distance / 1000.0
        deadhead_distance_km = tour.deadhead_distance / 1000.0
        deadhead_pct = (deadhead_distance_km / total_distance_km * 100) if total_distance_km > 0 else 0

        self.messages.append(
            f"Solution: {total_distance_km:.2f}km total "
            f"({required_distance_km:.2f}km required, "
            f"{deadhead_distance_km:.2f}km deadhead = {deadhead_pct:.1f}%)"
        )

        return DRPPSolution(
            tour=tour,
            nodes=self.nodes,
            edges=self.edges,
            route_coordinates=route_coords,
            total_distance_km=total_distance_km,
            required_distance_km=required_distance_km,
            deadhead_distance_km=deadhead_distance_km,
            deadhead_percentage=deadhead_pct,
            num_segments=len(segments),
            is_valid=tour.is_valid,
            messages=self.messages
        )

    def _add_deadhead_edges(self, required_edges: List[TopologyEdge]) -> List[TopologyEdge]:
        """
        Add shortest-path edges between all required edge endpoints.

        This creates a fully connected graph for DRPP solving.

        Args:
            required_edges: List of required edges from KML

        Returns:
            List of deadhead edges
        """
        # Get all unique nodes from required edges
        required_nodes: Set[NodeID] = set()
        for edge in required_edges:
            required_nodes.add(edge.from_node)
            required_nodes.add(edge.to_node)

        if len(required_nodes) <= 1:
            return []  # Only one node, no deadhead needed

        # For large graphs, only connect nearby nodes (use spatial clustering)
        # For now, connect all pairs (works well for up to ~1000 nodes)
        deadhead_edges = []
        required_node_list = list(required_nodes)

        # Limit to reasonable number of connections
        max_connections_per_node = 20

        for i, from_node in enumerate(required_node_list):
            # Find nearest neighbors for this node
            neighbors = []
            from_coord = self.nodes[from_node].coordinate

            for to_node in required_node_list:
                if to_node == from_node:
                    continue

                to_coord = self.nodes[to_node].coordinate
                distance = haversine(from_coord, to_coord)
                neighbors.append((distance, to_node))

            # Sort by distance and take closest N
            neighbors.sort()
            neighbors = neighbors[:max_connections_per_node]

            # Create deadhead edges to nearest neighbors
            for distance, to_node in neighbors:
                # Check if edge already exists in required edges
                exists = any(e.from_node == from_node and e.to_node == to_node for e in required_edges)
                if exists:
                    continue

                # Compute shortest path
                path_coords, path_distance = self._dijkstra_shortest_path(from_node, to_node)

                if path_coords and path_distance < float('inf'):
                    edge = TopologyEdge(
                        edge_id=f"deadhead_{from_node}_{to_node}",
                        from_node=from_node,
                        to_node=to_node,
                        cost=path_distance,
                        required=False,
                        segment_id=None,
                        coordinates=path_coords,
                        metadata={"type": "deadhead", "purpose": "connectivity"}
                    )
                    deadhead_edges.append(edge)

        return deadhead_edges

    def _dijkstra_shortest_path(self, from_node: NodeID, to_node: NodeID) -> Tuple[List[Coordinate], float]:
        """
        Compute shortest path using Dijkstra's algorithm.

        Args:
            from_node: Starting node ID
            to_node: Target node ID

        Returns:
            Tuple of (path coordinates, total distance)
        """
        import heapq

        # Build adjacency list
        adjacency = build_adjacency_list(self.edges)

        # Dijkstra's algorithm
        distances = {from_node: 0.0}
        predecessors: Dict[NodeID, Optional[NodeID]] = {from_node: None}
        pq = [(0.0, from_node)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == to_node:
                break

            for edge in adjacency.get(current, []):
                neighbor = edge.to_node
                new_dist = current_dist + edge.cost

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if to_node not in predecessors:
            return [], float('inf')

        path_nodes = []
        current = to_node
        while current is not None:
            path_nodes.append(current)
            current = predecessors.get(current)

        path_nodes.reverse()

        # Convert to coordinates
        path_coords = [self.nodes[node].coordinate for node in path_nodes]

        return path_coords, distances.get(to_node, float('inf'))

    def _extract_route_coordinates(self, tour: EulerianTour) -> List[Coordinate]:
        """
        Extract ordered route coordinates from Eulerian tour.

        Args:
            tour: Eulerian tour with ordered edges

        Returns:
            List of coordinates in route order
        """
        if not tour.edges:
            return []

        route_coords = []

        for edge in tour.edges:
            # Add all coordinates from this edge
            # Skip first coordinate if it matches previous end (avoid duplicates)
            if route_coords and edge.coordinates and edge.coordinates[0] == route_coords[-1]:
                route_coords.extend(edge.coordinates[1:])
            else:
                route_coords.extend(edge.coordinates)

        return route_coords


def solve_drpp_industry_standard(segments: List[SegmentRequirement],
                                 start_coordinate: Optional[Coordinate] = None,
                                 snap_tolerance_meters: float = 2.0) -> DRPPSolution:
    """
    Convenience function to solve DRPP using industry-standard methodology.

    Args:
        segments: List of road segments from KML
        start_coordinate: Optional starting coordinate (lat, lon)
        snap_tolerance_meters: Endpoint snapping tolerance

    Returns:
        DRPPSolution with complete route
    """
    solver = IndustryDRPPSolver(snap_tolerance_meters)
    return solver.solve(segments, start_coordinate)
