"""
Greedy routing algorithm for DRPP with robust error handling.

Implements nearest-neighbor greedy approach with fallback strategies
for unreachable segments.
"""

from typing import List, Tuple, Set, Optional, Any, Dict
from dataclasses import dataclass

from .types import (
    Coordinate, NodeID, SegmentIndex, Distance,
    UnreachableSegment, UnreachableReason, PathResult
)
from .distance_matrix import DistanceMatrix
from .path_reconstruction import reconstruct_path
from .logging_config import get_logger, LogTimer, log_exception

logger = get_logger(__name__)


@dataclass
class NodeNormalizer:
    """Utility for converting between node representations.

    Ensures consistent use of integer node IDs internally while supporting
    both coordinate and ID representations in the API.

    Attributes:
        node_to_id: Mapping from coordinates to node IDs
        id_to_node: Mapping from node IDs to coordinates
    """
    node_to_id: Dict[Coordinate, NodeID]
    id_to_node: Dict[NodeID, Coordinate]

    def to_id(self, node: Coordinate | NodeID) -> Optional[NodeID]:
        """Convert node (coordinate or ID) to ID.

        Args:
            node: Either a coordinate tuple or a node ID

        Returns:
            Node ID, or None if not found

        Example:
            >>> normalizer = NodeNormalizer(node_to_id, id_to_node)
            >>> node_id = normalizer.to_id((40.7, -74.0))
        """
        if isinstance(node, int):
            return node
        return self.node_to_id.get(node)

    def to_coords(self, node: Coordinate | NodeID) -> Optional[Coordinate]:
        """Convert node (ID or coordinate) to coordinate.

        Args:
            node: Either a node ID or a coordinate tuple

        Returns:
            Coordinate tuple, or None if not found

        Example:
            >>> coords = normalizer.to_coords(42)
        """
        if isinstance(node, tuple):
            return node
        # Handle both dict and list for id_to_node
        if isinstance(self.id_to_node, dict):
            return self.id_to_node.get(node)
        elif isinstance(self.id_to_node, list):
            try:
                return self.id_to_node[node] if node < len(self.id_to_node) else None
            except (IndexError, TypeError):
                return None
        return None


def _calculate_segment_length(coordinates: List[Coordinate]) -> Distance:
    """Calculate total length of a segment from coordinates.

    Args:
        coordinates: List of (lat, lon) points

    Returns:
        Total length in meters
    """
    if len(coordinates) < 2:
        return 0.0

    from .clustering import haversine
    return sum(
        haversine(coordinates[i], coordinates[i + 1])
        for i in range(len(coordinates) - 1)
    )


def _try_dijkstra_fallback(
    graph: Any,
    required_edges: List[Tuple],
    remaining: Set[SegmentIndex],
    current_id: NodeID,
    normalizer: NodeNormalizer
) -> Optional[Tuple[SegmentIndex, Distance, List[NodeID]]]:
    """Attempt to find nearest segment using live Dijkstra computation.

    This is a fallback when precomputed matrix doesn't have the path.

    Args:
        graph: Graph object with dijkstra() method
        required_edges: List of all required edges
        remaining: Set of remaining segment indices
        current_id: Current position (node ID)
        normalizer: Node normalizer for ID conversion

    Returns:
        Tuple of (segment_index, distance, path_ids) if found, None otherwise
    """
    logger.debug(f"Attempting Dijkstra fallback from node {current_id}")

    try:
        distances, predecessors = graph.dijkstra(current_id)
    except Exception as e:
        logger.error(f"Dijkstra failed in fallback: {e}", exc_info=True)
        return None

    best_seg_idx: Optional[SegmentIndex] = None
    best_dist = float('inf')
    best_path_ids: Optional[List[NodeID]] = None

    for seg_idx in remaining:
        segment_start = required_edges[seg_idx][0]
        segment_start_id = normalizer.to_id(segment_start)

        if segment_start_id is None:
            logger.warning(f"Segment {seg_idx} start node has no ID")
            continue

        if distances[segment_start_id] < best_dist:
            # Try to reconstruct path
            path_ids = reconstruct_path(predecessors, current_id, segment_start_id)
            if path_ids:
                best_dist = distances[segment_start_id]
                best_seg_idx = seg_idx
                best_path_ids = path_ids

    if best_seg_idx is not None:
        logger.info(
            f"Dijkstra fallback succeeded: found segment {best_seg_idx} "
            f"at distance {best_dist:.1f}m"
        )
        return (best_seg_idx, best_dist, best_path_ids)

    logger.warning("Dijkstra fallback found no reachable segments")
    return None


def _greedy_route_ondemand(
    graph: Any,
    required_edges: List[Tuple],
    segment_indices: List[SegmentIndex],
    start_node: Coordinate | NodeID
) -> PathResult:
    """Route using on-demand Dijkstra (no precomputation).

    This is MUCH faster for large clusters because it only computes
    distances as needed, rather than all O(n²) pairs upfront.

    Each iteration computes ONE single-source Dijkstra from current position,
    which is O(n log n), vs O(n²) for all-pairs.

    For 11,060 nodes:
    - All-pairs: ~122M distance computations (very slow)
    - On-demand: ~11,060 Dijkstra calls (much faster)

    Args:
        graph: Graph object with dijkstra() method
        required_edges: List of all required edges
        segment_indices: Indices of segments to route through
        start_node: Starting position

    Returns:
        PathResult with route and diagnostics
    """
    import time
    start_time = time.perf_counter()

    # Setup normalizer
    normalizer = NodeNormalizer(graph.node_to_id, graph.id_to_node)
    current_id = normalizer.to_id(start_node)

    if current_id is None:
        raise ValueError(f"Start node {start_node} has no valid ID")

    # Initialize
    path_ids: List[NodeID] = []
    remaining = set(segment_indices)
    total_distance = 0.0
    unreachable: List[UnreachableSegment] = []

    logger.info(f"On-demand greedy routing: {len(segment_indices)} segments")

    iteration = 0
    with LogTimer(logger, f"On-demand greedy routing ({len(segment_indices)} segments)", level=10):
        while remaining:
            iteration += 1

            # Compute single-source Dijkstra from current position
            try:
                distances, predecessors = graph.dijkstra(current_id)
            except Exception as e:
                logger.error(f"Dijkstra failed at iteration {iteration}: {e}", exc_info=True)
                # Mark all remaining as unreachable
                for seg_idx in remaining:
                    unreachable.append(UnreachableSegment(
                        segment_index=seg_idx,
                        reason=UnreachableReason.NO_PATH_FROM_CURRENT.value,
                        attempted_from=current_id
                    ))
                break

            # Find nearest reachable segment
            best_seg_idx: Optional[SegmentIndex] = None
            best_dist = float('inf')
            best_path_ids: Optional[List[NodeID]] = None
            best_start_id: Optional[NodeID] = None

            for seg_idx in remaining:
                segment_start = required_edges[seg_idx][0]
                segment_start_id = normalizer.to_id(segment_start)

                if segment_start_id is None:
                    logger.warning(f"Segment {seg_idx} has invalid start node")
                    continue

                # Check if reachable and get distance
                if segment_start_id in distances:
                    dist = distances[segment_start_id]
                    if dist < best_dist:
                        # Reconstruct path
                        path = reconstruct_path(predecessors, current_id, segment_start_id)
                        if path:
                            best_dist = dist
                            best_seg_idx = seg_idx
                            best_path_ids = path
                            best_start_id = segment_start_id

            # No segment found - all remaining are unreachable
            if best_seg_idx is None:
                logger.error(
                    f"All {len(remaining)} remaining segments unreachable "
                    f"from node {current_id} at iteration {iteration}"
                )
                for seg_idx in remaining:
                    unreachable.append(UnreachableSegment(
                        segment_index=seg_idx,
                        reason=UnreachableReason.NO_PATH_FROM_CURRENT.value,
                        attempted_from=current_id
                    ))
                break

            # Route to best segment
            segment_start = required_edges[best_seg_idx][0]
            segment_end = required_edges[best_seg_idx][1]
            segment_coords = required_edges[best_seg_idx][2]
            segment_end_id = normalizer.to_id(segment_end)

            if segment_end_id is None:
                logger.error(f"Segment {best_seg_idx} has invalid end node")
                unreachable.append(UnreachableSegment(
                    segment_index=best_seg_idx,
                    reason=UnreachableReason.INVALID_NODE_ID.value
                ))
                remaining.remove(best_seg_idx)
                continue

            # Add approach path
            if best_path_ids:
                path_ids.extend(best_path_ids)
                total_distance += best_dist

            # Traverse segment
            segment_length = _calculate_segment_length(segment_coords)
            total_distance += segment_length

            # Update position
            current_id = segment_end_id
            remaining.remove(best_seg_idx)

            # Progress logging
            if iteration % 100 == 0:
                elapsed = time.perf_counter() - start_time
                rate = iteration / elapsed if elapsed > 0 else 0
                logger.debug(
                    f"Iteration {iteration}: covered {len(segment_indices) - len(remaining)}/{len(segment_indices)} "
                    f"segments [{rate:.1f} segments/sec]"
                )

    # Convert path to coordinates
    path_coords = []
    for nid in path_ids:
        coords = normalizer.to_coords(nid)
        if coords:
            path_coords.append(coords)

    elapsed = time.perf_counter() - start_time
    segments_covered = len(segment_indices) - len(unreachable)

    logger.info(
        f"On-demand routing complete: {segments_covered}/{len(segment_indices)} segments, "
        f"{total_distance / 1000:.1f}km, {elapsed:.2f}s "
        f"[{segments_covered / elapsed if elapsed > 0 else 0:.1f} segments/sec]"
    )

    if unreachable:
        logger.warning(f"{len(unreachable)} segments unreachable:")
        for ur in unreachable[:5]:
            logger.warning(f"  - {ur}")
        if len(unreachable) > 5:
            logger.warning(f"  ... and {len(unreachable) - 5} more")

    return PathResult(
        path=path_coords,
        distance=total_distance,
        cluster_id=-1,  # Set by caller
        segments_covered=segments_covered,
        segments_unreachable=len(unreachable),
        computation_time=elapsed
    )


def greedy_route_cluster(
    graph: Optional[Any],
    required_edges: List[Tuple],
    segment_indices: List[SegmentIndex],
    start_node: Coordinate | NodeID,
    distance_matrix: Optional[DistanceMatrix] = None,
    normalizer: Optional[NodeNormalizer] = None,
    enable_fallback: bool = True,
    use_ondemand: bool = False
) -> PathResult:
    """Route through segments using greedy nearest-neighbor approach.

    Args:
        graph: Graph object (required if enable_fallback=True or use_ondemand=True)
        required_edges: List of all required edges. Each edge is a tuple of
            (start_node, end_node, coordinates, ...)
        segment_indices: Indices of segments to route through
        start_node: Starting position (coordinate or node ID)
        distance_matrix: Precomputed distance matrix (optional)
        normalizer: Node normalizer (required if distance_matrix provided)
        enable_fallback: If True, use Dijkstra fallback for unreachable segments
        use_ondemand: If True, use on-demand Dijkstra instead of precomputing matrix
                     (MUCH faster for large clusters)

    Returns:
        PathResult with route, distance, and diagnostics

    Raises:
        ValueError: If invalid arguments provided

    Example:
        >>> result = greedy_route_cluster(
        ...     graph=graph,
        ...     required_edges=edges,
        ...     segment_indices=[0, 1, 2, 5],
        ...     start_node=(40.7, -74.0),
        ...     use_ondemand=True  # Recommended for large datasets
        ... )
        >>> print(f"Route: {result.distance:.1f}m, covered {result.segments_covered} segments")
    """
    import time
    start_time = time.perf_counter()

    if not segment_indices:
        logger.warning("No segments to route")
        return PathResult(
            path=[],
            distance=0.0,
            cluster_id=-1,
            segments_covered=0,
            segments_unreachable=0,
            computation_time=0.0
        )

    # Use on-demand routing for large clusters (avoids expensive all-pairs precomputation)
    if use_ondemand:
        logger.info(f"Using on-demand Dijkstra routing for {len(segment_indices)} segments")
        return _greedy_route_ondemand(
            graph, required_edges, segment_indices, start_node
        )

    # Compute matrix if not provided
    if distance_matrix is None or normalizer is None:
        if graph is None:
            raise ValueError("graph required when distance_matrix not provided")

        # Auto-switch to on-demand mode for large endpoint sets
        node_ids = set()
        for seg_idx in segment_indices:
            start_node_coords = required_edges[seg_idx][0]
            end_node_coords = required_edges[seg_idx][1]
            start_id = graph.node_to_id.get(start_node_coords)
            end_id = graph.node_to_id.get(end_node_coords)
            if start_id is not None:
                node_ids.add(start_id)
            if end_id is not None:
                node_ids.add(end_id)

        # If too many endpoints, switch to on-demand mode automatically
        if len(node_ids) > 1000:
            logger.warning(
                f"Cluster has {len(node_ids)} segment endpoints. "
                f"Switching to on-demand mode (precomputing {len(node_ids)}² distances would be very slow)"
            )
            return _greedy_route_ondemand(
                graph, required_edges, segment_indices, start_node
            )

        logger.info(
            f"Computing distance matrix for {len(segment_indices)} segments "
            f"({len(node_ids)} endpoints)"
        )
        from .distance_matrix import compute_distance_matrix

        # Extract unique nodes
        id_to_coords = {}

        for seg_idx in segment_indices:
            start_node_coords = required_edges[seg_idx][0]
            end_node_coords = required_edges[seg_idx][1]

            start_id = graph.node_to_id.get(start_node_coords)
            end_id = graph.node_to_id.get(end_node_coords)

            if start_id is not None:
                id_to_coords[start_id] = start_node_coords
            if end_id is not None:
                id_to_coords[end_id] = end_node_coords

        # Include start node
        if isinstance(start_node, tuple):
            start_id = graph.node_to_id.get(start_node)
        else:
            start_id = start_node
            start_node = graph.id_to_node.get(start_id)

        if start_id is not None:
            node_ids.add(start_id)
            id_to_coords[start_id] = start_node

        distance_matrix = compute_distance_matrix(graph, node_ids, id_to_coords)
        normalizer = NodeNormalizer(graph.node_to_id, graph.id_to_node)

    # Initialize
    path_ids: List[NodeID] = []
    remaining = set(segment_indices)
    current_id = normalizer.to_id(start_node)
    total_distance = 0.0
    unreachable: List[UnreachableSegment] = []

    if current_id is None:
        raise ValueError(f"Start node {start_node} has no valid ID")

    logger.debug(
        f"Starting greedy routing: {len(segment_indices)} segments "
        f"from node {current_id}"
    )

    # Main greedy loop
    iteration = 0
    with LogTimer(logger, f"Greedy routing ({len(segment_indices)} segments)", level=10):
        while remaining:
            iteration += 1
            best_seg_idx: Optional[SegmentIndex] = None
            best_approach_dist = float('inf')
            best_approach_path_ids: Optional[List[NodeID]] = None

            # Find nearest reachable segment
            for seg_idx in remaining:
                segment_start = required_edges[seg_idx][0]
                segment_start_id = normalizer.to_id(segment_start)

                if segment_start_id is None:
                    logger.warning(f"Segment {seg_idx} has invalid start node")
                    continue

                # Check precomputed matrix
                if distance_matrix.has_path(current_id, segment_start_id):
                    approach_dist = distance_matrix.get_distance(current_id, segment_start_id)
                    approach_path_ids = distance_matrix.get_path_ids(current_id, segment_start_id)

                    if approach_dist < best_approach_dist:
                        best_approach_dist = approach_dist
                        best_seg_idx = seg_idx
                        best_approach_path_ids = approach_path_ids

            # No segment found in matrix - try fallback
            if best_seg_idx is None:
                if enable_fallback and graph is not None:
                    logger.warning(
                        f"No precomputed path at iteration {iteration}, "
                        f"trying Dijkstra fallback"
                    )
                    fallback_result = _try_dijkstra_fallback(
                        graph, required_edges, remaining, current_id, normalizer
                    )
                    if fallback_result:
                        best_seg_idx, best_approach_dist, best_approach_path_ids = fallback_result
                    else:
                        # Fallback also failed
                        logger.error(
                            f"All {len(remaining)} remaining segments unreachable "
                            f"from node {current_id}"
                        )
                        for seg_idx in remaining:
                            unreachable.append(UnreachableSegment(
                                segment_index=seg_idx,
                                reason=UnreachableReason.NO_PATH_FROM_CURRENT.value,
                                attempted_from=current_id
                            ))
                        break
                else:
                    # Fallback disabled
                    logger.error(
                        f"No path in matrix for {len(remaining)} segments "
                        f"(fallback disabled)"
                    )
                    for seg_idx in remaining:
                        unreachable.append(UnreachableSegment(
                            segment_index=seg_idx,
                            reason=UnreachableReason.NO_PATH_IN_MATRIX.value,
                            attempted_from=current_id
                        ))
                    break

            # Route to best segment
            if best_seg_idx is not None:
                segment_start = required_edges[best_seg_idx][0]
                segment_end = required_edges[best_seg_idx][1]
                segment_coords = required_edges[best_seg_idx][2]

                segment_end_id = normalizer.to_id(segment_end)
                if segment_end_id is None:
                    logger.error(f"Segment {best_seg_idx} has invalid end node")
                    unreachable.append(UnreachableSegment(
                        segment_index=best_seg_idx,
                        reason=UnreachableReason.INVALID_NODE_ID.value
                    ))
                    remaining.remove(best_seg_idx)
                    continue

                # Add approach path
                if best_approach_path_ids:
                    path_ids.extend(best_approach_path_ids)
                    total_distance += best_approach_dist

                # Traverse segment
                segment_length = _calculate_segment_length(segment_coords)
                total_distance += segment_length

                # Update position
                current_id = segment_end_id
                remaining.remove(best_seg_idx)

                if iteration % 100 == 0:
                    logger.debug(
                        f"Iteration {iteration}: covered {len(segment_indices) - len(remaining)} "
                        f"segments, {len(remaining)} remaining"
                    )

    # Convert path to coordinates
    path_coords = [
        normalizer.id_to_node[nid]
        for nid in path_ids
        if nid in normalizer.id_to_node
    ]

    elapsed = time.perf_counter() - start_time
    segments_covered = len(segment_indices) - len(unreachable)

    logger.info(
        f"Greedy routing complete: {segments_covered}/{len(segment_indices)} segments, "
        f"{total_distance / 1000:.1f}km, {elapsed:.2f}s"
    )

    if unreachable:
        logger.warning(f"{len(unreachable)} segments unreachable:")
        for ur in unreachable[:5]:  # Log first 5
            logger.warning(f"  - {ur}")
        if len(unreachable) > 5:
            logger.warning(f"  ... and {len(unreachable) - 5} more")

    return PathResult(
        path=path_coords,
        distance=total_distance,
        cluster_id=-1,  # Set by caller
        segments_covered=segments_covered,
        segments_unreachable=len(unreachable),
        computation_time=elapsed
    )
