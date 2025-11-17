"""
Robust path reconstruction for shortest path algorithms.

Handles various sentinel values and edge cases that can occur with
different Dijkstra implementations.
"""

from typing import List, Optional, Set
from .types import NodeID
from .logging_config import get_logger

logger = get_logger(__name__)


def reconstruct_path(
    predecessors: List[Optional[NodeID]],
    source_id: NodeID,
    target_id: NodeID,
    max_iterations: Optional[int] = None
) -> List[NodeID]:
    """Reconstruct shortest path from Dijkstra predecessor array.

    This function handles all common sentinel values and edge cases:
    - predecessors[node] = -1 (C-style sentinel)
    - predecessors[node] = None (Python-style sentinel)
    - predecessors[source] = source (self-loop sentinel)
    - Cycle detection
    - Maximum iteration limits

    Args:
        predecessors: Predecessor array from Dijkstra's algorithm.
            predecessors[i] is the previous node on the path to node i.
        source_id: Starting node ID
        target_id: Destination node ID
        max_iterations: Maximum number of iterations before giving up.
            If None, uses len(predecessors) as the limit.

    Returns:
        List of node IDs representing the path from source to target.
        Returns empty list if:
        - No path exists
        - Invalid sentinel values encountered
        - Cycle detected
        - Maximum iterations exceeded

    Example:
        >>> predecessors = [None, 0, 0, 1, 2]
        >>> path = reconstruct_path(predecessors, source_id=0, target_id=4)
        >>> print(path)
        [0, 2, 4]

    Raises:
        ValueError: If source_id or target_id are out of bounds
    """
    # Validation
    if source_id < 0 or source_id >= len(predecessors):
        raise ValueError(f"source_id {source_id} out of bounds [0, {len(predecessors)})")
    if target_id < 0 or target_id >= len(predecessors):
        raise ValueError(f"target_id {target_id} out of bounds [0, {len(predecessors)})")

    # Special case: source equals target
    if source_id == target_id:
        return [source_id]

    # Set iteration limit
    if max_iterations is None:
        max_iterations = len(predecessors)

    path: List[NodeID] = []
    current = target_id
    visited: Set[NodeID] = set()

    for iteration in range(max_iterations):
        path.append(current)

        # Successfully reached source
        if current == source_id:
            path.reverse()
            logger.debug(f"Path reconstructed: {len(path)} nodes from {source_id} to {target_id}")
            return path

        # Cycle detection
        if current in visited:
            logger.warning(
                f"Cycle detected during path reconstruction from {source_id} to {target_id} "
                f"at node {current} (iteration {iteration})"
            )
            return []

        visited.add(current)

        # Get predecessor
        predecessor = predecessors[current]

        # Check for sentinel values indicating no path
        if predecessor is None:
            logger.debug(f"No path exists: predecessor[{current}] is None")
            return []

        if predecessor == -1:
            logger.debug(f"No path exists: predecessor[{current}] is -1 sentinel")
            return []

        # Check for invalid self-loop (not at source)
        if predecessor == current:
            logger.warning(
                f"Invalid self-loop at node {current} (not source) during path reconstruction"
            )
            return []

        # Validate predecessor is in bounds
        if predecessor < 0 or predecessor >= len(predecessors):
            logger.error(
                f"Invalid predecessor: predecessors[{current}] = {predecessor} "
                f"is out of bounds [0, {len(predecessors)})"
            )
            return []

        current = predecessor

    # Maximum iterations exceeded
    logger.warning(
        f"Path reconstruction exceeded maximum iterations ({max_iterations}) "
        f"from {source_id} to {target_id}"
    )
    return []


def validate_path(
    path: List[NodeID],
    source_id: NodeID,
    target_id: NodeID
) -> bool:
    """Validate that a path is correct.

    Args:
        path: List of node IDs representing a path
        source_id: Expected starting node
        target_id: Expected ending node

    Returns:
        True if path is valid, False otherwise

    Example:
        >>> path = [0, 2, 4]
        >>> validate_path(path, source_id=0, target_id=4)
        True
        >>> validate_path(path, source_id=0, target_id=5)
        False
    """
    if not path:
        logger.debug("Path validation failed: empty path")
        return False

    if path[0] != source_id:
        logger.debug(f"Path validation failed: starts at {path[0]}, expected {source_id}")
        return False

    if path[-1] != target_id:
        logger.debug(f"Path validation failed: ends at {path[-1]}, expected {target_id}")
        return False

    # Check for duplicate nodes (indicates cycle)
    if len(path) != len(set(path)):
        logger.debug("Path validation failed: contains duplicate nodes")
        return False

    logger.debug(f"Path validated: {len(path)} nodes from {source_id} to {target_id}")
    return True


def reconstruct_path_safe(
    predecessors: List[Optional[NodeID]],
    source_id: NodeID,
    target_id: NodeID
) -> Optional[List[NodeID]]:
    """Safe wrapper around reconstruct_path that catches exceptions.

    Args:
        predecessors: Predecessor array from Dijkstra
        source_id: Starting node ID
        target_id: Destination node ID

    Returns:
        List of node IDs if successful, None if any error occurs

    Example:
        >>> path = reconstruct_path_safe(predecessors, source_id=0, target_id=4)
        >>> if path is None:
        ...     print("Failed to reconstruct path")
    """
    try:
        path = reconstruct_path(predecessors, source_id, target_id)
        if path and validate_path(path, source_id, target_id):
            return path
        return None
    except Exception as e:
        logger.error(f"Exception during path reconstruction: {e}", exc_info=True)
        return None
