"""
Graph topology construction with snapping and cleaning.

This module provides industry-standard topology construction for DRPP:
- Endpoint snapping within tolerance
- Geometry cleaning and validation
- Node/edge construction with metadata preservation
- Directed edge creation from road segments
"""

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

from .types import SegmentRequirement
from .geo import haversine


Coordinate = Tuple[float, float]  # (lat, lon)
NodeID = int


@dataclass
class TopologyNode:
    """A node in the topology graph."""
    node_id: NodeID
    coordinate: Coordinate  # (lat, lon) - canonical snapped coordinate
    original_coordinates: Set[Coordinate]  # All coordinates snapped to this node


@dataclass
class TopologyEdge:
    """A directed edge in the topology graph."""
    edge_id: str  # Unique edge identifier
    from_node: NodeID
    to_node: NodeID
    cost: float  # Travel cost (distance in meters, time in seconds, etc.)
    required: bool  # True if this is a required edge (from KML), False if deadhead
    segment_id: Optional[str]  # Original segment ID from KML (CollId)
    coordinates: List[Coordinate]  # Full geometry of the edge
    metadata: Dict  # All metadata from KML ExtendedData


class TopologyBuilder:
    """
    Builds clean graph topology from road segments with endpoint snapping.

    This implements the industry-standard approach:
    1. Snap endpoints within tolerance to create proper node topology
    2. Clean geometry (remove duplicates, validate)
    3. Build directed edges with metadata preservation
    """

    def __init__(self, snap_tolerance_meters: float = 2.0):
        """
        Initialize topology builder.

        Args:
            snap_tolerance_meters: Distance threshold for snapping endpoints (default 2m)
        """
        self.snap_tolerance = snap_tolerance_meters
        self.nodes: Dict[NodeID, TopologyNode] = {}
        self.edges: List[TopologyEdge] = []
        self.coordinate_to_node: Dict[Coordinate, NodeID] = {}
        self.next_node_id = 0

    def build_topology(self, segments: List[SegmentRequirement]) -> Tuple[Dict[NodeID, TopologyNode], List[TopologyEdge]]:
        """
        Build graph topology from road segments.

        Args:
            segments: List of road segments from KML

        Returns:
            Tuple of (nodes dict, edges list)
        """
        # Step 1: Extract all endpoints and snap them
        self._snap_endpoints(segments)

        # Step 2: Build directed edges from segments
        self._build_edges(segments)

        return self.nodes, self.edges

    def _snap_endpoints(self, segments: List[SegmentRequirement]) -> None:
        """
        Snap all segment endpoints to create proper node topology.

        Uses spatial hashing for efficient O(n log n) snapping.
        """
        # Extract all unique endpoints
        endpoints: Set[Coordinate] = set()
        for segment in segments:
            if segment.coordinates:
                endpoints.add(segment.coordinates[0])  # Start point
                endpoints.add(segment.coordinates[-1])  # End point

        # Snap endpoints using spatial grid
        grid_cell_size = self.snap_tolerance * 2  # Grid cells are 2x tolerance
        grid: Dict[Tuple[int, int], List[Coordinate]] = defaultdict(list)

        # Assign endpoints to grid cells
        for coord in endpoints:
            cell = self._get_grid_cell(coord, grid_cell_size)
            grid[cell].append(coord)

        # Snap endpoints within each cell and neighboring cells
        for coord in endpoints:
            if coord in self.coordinate_to_node:
                continue  # Already snapped

            # Find canonical node for this coordinate
            canonical_coord = self._find_snap_target(coord, grid, grid_cell_size)

            # Create node if needed
            if canonical_coord not in self.coordinate_to_node:
                node_id = self.next_node_id
                self.next_node_id += 1
                node = TopologyNode(
                    node_id=node_id,
                    coordinate=canonical_coord,
                    original_coordinates=set()
                )
                self.nodes[node_id] = node
                self.coordinate_to_node[canonical_coord] = node_id

            # Map this coordinate to the canonical node
            node_id = self.coordinate_to_node[canonical_coord]
            self.nodes[node_id].original_coordinates.add(coord)
            if coord != canonical_coord:
                self.coordinate_to_node[coord] = node_id

    def _get_grid_cell(self, coord: Coordinate, cell_size: float) -> Tuple[int, int]:
        """Get grid cell for a coordinate (for spatial hashing)."""
        lat, lon = coord
        # Approximate: 1 degree ≈ 111km at equator
        lat_cells = int(lat / (cell_size / 111000))
        lon_cells = int(lon / (cell_size / 111000))
        return (lat_cells, lon_cells)

    def _find_snap_target(self, coord: Coordinate, grid: Dict[Tuple[int, int], List[Coordinate]],
                         cell_size: float) -> Coordinate:
        """
        Find the canonical coordinate to snap to.

        Searches current cell and 8 neighboring cells for nearby points.
        Returns the closest point within snap_tolerance, or coord itself if none found.
        """
        cell = self._get_grid_cell(coord, cell_size)

        # Check current cell and 8 neighbors
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if neighbor_cell in grid:
                    candidates.extend(grid[neighbor_cell])

        # Find closest candidate within tolerance
        closest = coord
        min_distance = self.snap_tolerance

        for candidate in candidates:
            if candidate == coord:
                continue
            distance = haversine(coord, candidate)
            if distance < min_distance:
                min_distance = distance
                closest = candidate

        return closest

    def _build_edges(self, segments: List[SegmentRequirement]) -> None:
        """
        Build directed edges from road segments.

        Each segment becomes one or two directed edges depending on directionality.
        """
        for segment in segments:
            if not segment.coordinates or len(segment.coordinates) < 2:
                continue  # Skip invalid segments

            # Get snapped start and end nodes
            start_coord = segment.coordinates[0]
            end_coord = segment.coordinates[-1]

            start_node = self.coordinate_to_node.get(start_coord)
            end_node = self.coordinate_to_node.get(end_coord)

            if start_node is None or end_node is None:
                # This shouldn't happen if snapping worked correctly
                continue

            # Calculate edge cost (total length of LineString)
            cost = self._calculate_edge_cost(segment.coordinates)

            # Create forward edge (start → end)
            if segment.forward_required or not segment.one_way:
                edge = TopologyEdge(
                    edge_id=f"{segment.segment_id}_fwd",
                    from_node=start_node,
                    to_node=end_node,
                    cost=cost,
                    required=segment.forward_required,
                    segment_id=segment.segment_id,
                    coordinates=segment.coordinates.copy(),
                    metadata=segment.metadata.copy()
                )
                self.edges.append(edge)

            # Create backward edge (end → start)
            if segment.backward_required or not segment.one_way:
                edge = TopologyEdge(
                    edge_id=f"{segment.segment_id}_bwd",
                    from_node=end_node,
                    to_node=start_node,
                    cost=cost,
                    required=segment.backward_required,
                    segment_id=segment.segment_id,
                    coordinates=list(reversed(segment.coordinates)),
                    metadata=segment.metadata.copy()
                )
                self.edges.append(edge)

    def _calculate_edge_cost(self, coordinates: List[Coordinate]) -> float:
        """
        Calculate total cost (distance) for a sequence of coordinates.

        Returns distance in meters.
        """
        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            total_distance += haversine(coordinates[i], coordinates[i + 1])
        return total_distance

    def get_node_id(self, coordinate: Coordinate) -> Optional[NodeID]:
        """Get the node ID for a coordinate (after snapping)."""
        return self.coordinate_to_node.get(coordinate)

    def get_node_coordinate(self, node_id: NodeID) -> Optional[Coordinate]:
        """Get the canonical coordinate for a node ID."""
        node = self.nodes.get(node_id)
        return node.coordinate if node else None


def build_adjacency_list(edges: List[TopologyEdge]) -> Dict[NodeID, List[TopologyEdge]]:
    """
    Build adjacency list from edges for efficient graph traversal.

    Args:
        edges: List of topology edges

    Returns:
        Dict mapping from_node → list of outgoing edges
    """
    adjacency: Dict[NodeID, List[TopologyEdge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.from_node].append(edge)
    return adjacency


def build_reverse_adjacency_list(edges: List[TopologyEdge]) -> Dict[NodeID, List[TopologyEdge]]:
    """
    Build reverse adjacency list from edges (for incoming edges).

    Args:
        edges: List of topology edges

    Returns:
        Dict mapping to_node → list of incoming edges
    """
    reverse_adjacency: Dict[NodeID, List[TopologyEdge]] = defaultdict(list)
    for edge in edges:
        reverse_adjacency[edge.to_node].append(edge)
    return reverse_adjacency


def get_node_degrees(edges: List[TopologyEdge], num_nodes: int) -> List[Tuple[int, int]]:
    """
    Calculate in-degree and out-degree for all nodes.

    Args:
        edges: List of topology edges
        num_nodes: Total number of nodes

    Returns:
        List of (in_degree, out_degree) for each node ID
    """
    degrees = [(0, 0) for _ in range(num_nodes)]

    for edge in edges:
        # Increment out-degree of from_node
        in_deg, out_deg = degrees[edge.from_node]
        degrees[edge.from_node] = (in_deg, out_deg + 1)

        # Increment in-degree of to_node
        in_deg, out_deg = degrees[edge.to_node]
        degrees[edge.to_node] = (in_deg + 1, out_deg)

    return degrees
