"""
Type definitions for DRPP solver.

This module provides type aliases and dataclasses for type safety and clarity.
All coordinate operations should use these types for consistency.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

# Type Aliases for clarity (Python 3.9 compatible)
Coordinate = Tuple[float, float]  # (latitude, longitude) in decimal degrees
NodeID = int  # Integer node identifier
SegmentIndex = int  # Index into required_edges list
Distance = float  # Distance in meters
ClusterID = int  # Cluster identifier


class PathResult(NamedTuple):
    """Result from routing a single cluster.

    Attributes:
        path: List of coordinates representing the route
        distance: Total route distance in meters
        cluster_id: Identifier of the cluster that was routed
        segments_covered: Number of segments successfully covered
        segments_unreachable: Number of segments that couldn't be reached
        computation_time: Time taken to compute this route in seconds
    """

    path: List[Coordinate]
    distance: Distance
    cluster_id: ClusterID
    segments_covered: int
    segments_unreachable: int
    computation_time: float


class ClusterResult(NamedTuple):
    """Result from clustering segments.

    Attributes:
        clusters: Mapping from cluster ID to list of segment indices
        noise_count: Number of segments classified as noise
        method_used: Name of clustering method that was used
    """

    clusters: Dict[ClusterID, List[SegmentIndex]]
    noise_count: int
    method_used: str


@dataclass
class SegmentRequirement:
    """Defines traversal requirements for a segment.

    Attributes:
        segment_id: Unique segment identifier from KML
        forward_required: Must traverse start->end
        backward_required: Must traverse end->start
        one_way: Only one direction is allowed
        coordinates: List of (lat, lon) points
        metadata: Additional KML data (speed, name, etc.)
    """

    segment_id: str
    forward_required: bool
    backward_required: bool
    one_way: bool = False
    coordinates: Optional[List[Tuple[float, float]]] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Initialize default mutable fields."""
        if self.coordinates is None:
            self.coordinates = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_two_way_required(self) -> bool:
        """Both directions must be traversed."""
        return self.forward_required and self.backward_required

    @property
    def required_traversals(self) -> int:
        """Number of required traversals (0, 1, or 2)."""
        return int(self.forward_required) + int(self.backward_required)


@dataclass
class SegmentMetadata:
    """Metadata for a required segment.

    Attributes:
        index: Unique segment index
        start_node: Starting node (as coordinate or ID)
        end_node: Ending node (as coordinate or ID)
        coordinates: Full path coordinates for the segment
        length: Segment length in meters
        speed_limit: Optional speed limit in km/h
        highway_type: Optional OSM highway type
    """

    index: SegmentIndex
    start_node: Union[Coordinate, NodeID]
    end_node: Union[Coordinate, NodeID]
    coordinates: List[Coordinate]
    length: Distance
    speed_limit: Optional[float] = None
    highway_type: Optional[str] = None


@dataclass
class UnreachableSegment:
    """Information about a segment that couldn't be reached.

    Attributes:
        segment_index: Index of the unreachable segment
        reason: Human-readable reason code
        attempted_from: Node ID from which routing was attempted
        distance_to_nearest: Distance to nearest reachable point (if known)
    """

    segment_index: SegmentIndex
    reason: str
    attempted_from: Optional[NodeID] = None
    distance_to_nearest: Optional[Distance] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        msg = f"Segment {self.segment_index}: {self.reason}"
        if self.attempted_from is not None:
            msg += f" (from node {self.attempted_from})"
        if self.distance_to_nearest is not None:
            msg += f" [nearest: {self.distance_to_nearest:.1f}m]"
        return msg


class UnreachableReason(Enum):
    """Standard reason codes for unreachable segments."""

    NO_PATH_IN_MATRIX = "no_path_in_precomputed_matrix"
    NO_PATH_FROM_CURRENT = "no_path_from_current_position"
    DISCONNECTED_COMPONENT = "disconnected_graph_component"
    DIJKSTRA_FAILED = "dijkstra_computation_failed"
    INVALID_NODE_ID = "invalid_or_missing_node_id"


@dataclass
class GraphInterface:
    """Interface requirements for graph objects.

    Any graph object passed to DRPP functions must provide these attributes/methods.

    Attributes:
        node_to_id: Mapping from coordinates to integer node IDs
        id_to_node: Mapping from integer node IDs to coordinates
        dijkstra: Function that computes shortest paths from a source node
    """

    node_to_id: Dict[Coordinate, NodeID]
    id_to_node: Dict[NodeID, Coordinate]

    def dijkstra(self, source_id: NodeID) -> Tuple[List[Distance], List[Optional[NodeID]]]:
        """Compute shortest paths from source using Dijkstra's algorithm.

        Args:
            source_id: Starting node ID

        Returns:
            Tuple of (distances, predecessors) where:
            - distances[i] is the shortest distance to node i
            - predecessors[i] is the previous node on the path to i
              (None if unreachable, source_id if it's the source)
        """
        raise NotImplementedError("Graph must implement dijkstra method")
