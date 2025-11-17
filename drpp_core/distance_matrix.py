"""
Memory-efficient distance matrix computation and storage.

Supports both dict-based and numpy-based storage for optimal performance
based on matrix size.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import sys

from .types import NodeID, Distance, Coordinate
from .path_reconstruction import reconstruct_path
from .logging_config import get_logger, LogTimer

logger = get_logger(__name__)

# Try to import numpy for large matrices
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.info("NumPy not available - using dict-based matrices only")


@dataclass
class MatrixStats:
    """Statistics about a distance matrix.

    Attributes:
        num_nodes: Number of nodes in the matrix
        num_paths: Number of valid paths stored
        memory_bytes: Approximate memory usage in bytes
        storage_type: Type of storage used ('dict' or 'numpy')
    """

    num_nodes: int
    num_paths: int
    memory_bytes: int
    storage_type: str


class DistanceMatrix:
    """Memory-efficient storage for precomputed shortest paths.

    Stores only node IDs and distances, not full coordinate paths.
    Automatically selects optimal storage format based on matrix size.

    For small matrices (< 1000 nodes): Uses dict-based storage (fast, flexible)
    For large matrices (>= 1000 nodes): Uses numpy arrays (compact, cache-friendly)

    Attributes:
        distances: Mapping or array of shortest distances
        path_node_ids: Mapping or array of paths (as node ID sequences)
        id_to_coords: Mapping from node IDs to coordinates
        storage_type: 'dict' or 'numpy'
    """

    def __init__(self, use_numpy: bool = False, num_nodes: Optional[int] = None):
        """Initialize distance matrix.

        Args:
            use_numpy: Force numpy storage (requires numpy to be installed)
            num_nodes: Expected number of nodes (for numpy pre-allocation)

        Raises:
            RuntimeError: If use_numpy=True but numpy is not available
        """
        if use_numpy and not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy requested but not available")

        self.use_numpy = use_numpy
        self.num_nodes = num_nodes

        # Dict-based storage
        self.distances: Dict[Tuple[NodeID, NodeID], Distance] = {}
        self.path_node_ids: Dict[Tuple[NodeID, NodeID], List[NodeID]] = {}
        self.id_to_coords: Dict[NodeID, Coordinate] = {}

        # NumPy-based storage (initialized on first use)
        self.distance_matrix: Optional[Any] = None  # np.ndarray if numpy available
        self.path_matrix: Optional[Any] = None  # List[List[List[NodeID]]]

        self.storage_type = "numpy" if use_numpy else "dict"

        logger.debug(f"DistanceMatrix initialized: storage={self.storage_type}, nodes={num_nodes}")

    def set(
        self, source_id: NodeID, target_id: NodeID, distance: Distance, path_ids: List[NodeID]
    ) -> None:
        """Store shortest path information.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID
            distance: Shortest distance between nodes (in meters)
            path_ids: Sequence of node IDs representing the path

        Example:
            >>> matrix = DistanceMatrix()
            >>> matrix.set(0, 5, 1234.5, [0, 2, 4, 5])
        """
        if self.use_numpy:
            # NumPy storage - lazy initialization
            if self.distance_matrix is None and self.num_nodes is not None:
                self._initialize_numpy_storage()

            if self.distance_matrix is not None:
                self.distance_matrix[source_id, target_id] = distance
                # Store path in dict (paths are variable length, hard to store in numpy)
                self.path_node_ids[(source_id, target_id)] = path_ids
        else:
            # Dict storage
            self.distances[(source_id, target_id)] = distance
            self.path_node_ids[(source_id, target_id)] = path_ids

    def get_distance(self, source_id: NodeID, target_id: NodeID) -> Distance:
        """Get precomputed shortest distance.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID

        Returns:
            Shortest distance in meters, or float('inf') if no path exists

        Example:
            >>> dist = matrix.get_distance(0, 5)
            >>> print(f"Distance: {dist:.1f}m")
        """
        if self.use_numpy and self.distance_matrix is not None:
            val = self.distance_matrix[source_id, target_id]
            return float(val) if val != np.inf else float("inf")
        else:
            return self.distances.get((source_id, target_id), float("inf"))

    def get_path_ids(self, source_id: NodeID, target_id: NodeID) -> List[NodeID]:
        """Get shortest path as sequence of node IDs.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID

        Returns:
            List of node IDs representing the path, or empty list if no path

        Example:
            >>> path = matrix.get_path_ids(0, 5)
            >>> print(f"Path: {path}")
        """
        return self.path_node_ids.get((source_id, target_id), [])

    def get_path_coords(self, source_id: NodeID, target_id: NodeID) -> List[Coordinate]:
        """Get shortest path as sequence of coordinates.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID

        Returns:
            List of (lat, lon) coordinates representing the path

        Example:
            >>> coords = matrix.get_path_coords(0, 5)
            >>> for lat, lon in coords:
            ...     print(f"({lat}, {lon})")
        """
        path_ids = self.get_path_ids(source_id, target_id)
        return [self.id_to_coords[nid] for nid in path_ids if nid in self.id_to_coords]

    def has_path(self, source_id: NodeID, target_id: NodeID) -> bool:
        """Check if a path exists between two nodes.

        Args:
            source_id: Starting node ID
            target_id: Ending node ID

        Returns:
            True if a path exists, False otherwise

        Example:
            >>> if matrix.has_path(0, 5):
            ...     print("Path exists!")
        """
        if self.use_numpy and self.distance_matrix is not None:
            return bool(self.distance_matrix[source_id, target_id] != np.inf)
        else:
            return (source_id, target_id) in self.distances

    def _initialize_numpy_storage(self) -> None:
        """Initialize NumPy arrays for storage."""
        if not NUMPY_AVAILABLE or self.num_nodes is None:
            return

        logger.debug(f"Initializing NumPy matrix: {self.num_nodes}x{self.num_nodes}")
        self.distance_matrix = np.full((self.num_nodes, self.num_nodes), np.inf, dtype=np.float32)

    def get_stats(self) -> MatrixStats:
        """Get statistics about memory usage and content.

        Returns:
            MatrixStats object with detailed information

        Example:
            >>> stats = matrix.get_stats()
            >>> print(f"Memory usage: {stats.memory_bytes / 1024 / 1024:.1f} MB")
        """
        if self.use_numpy and self.distance_matrix is not None:
            # NumPy storage
            num_paths = len(self.path_node_ids)
            matrix_bytes = self.distance_matrix.nbytes
            paths_bytes = sum(sys.getsizeof(path) for path in self.path_node_ids.values())
            coords_bytes = sys.getsizeof(self.id_to_coords)
            total_bytes = matrix_bytes + paths_bytes + coords_bytes

            return MatrixStats(
                num_nodes=self.num_nodes or 0,
                num_paths=num_paths,
                memory_bytes=total_bytes,
                storage_type="numpy",
            )
        else:
            # Dict storage
            num_paths = len(self.path_node_ids)
            distances_bytes = sys.getsizeof(self.distances)
            paths_bytes = sys.getsizeof(self.path_node_ids)
            coords_bytes = sys.getsizeof(self.id_to_coords)
            total_bytes = distances_bytes + paths_bytes + coords_bytes

            return MatrixStats(
                num_nodes=len(self.id_to_coords),
                num_paths=num_paths,
                memory_bytes=total_bytes,
                storage_type="dict",
            )


def compute_distance_matrix(
    graph: Any,  # GraphInterface
    node_ids: Set[NodeID],
    id_to_coords: Dict[NodeID, Coordinate],
    use_numpy: bool = False,
) -> DistanceMatrix:
    """Compute all-pairs shortest paths for a set of nodes.

    Args:
        graph: Graph object with dijkstra() method
        node_ids: Set of node IDs to compute paths between
        id_to_coords: Mapping from node IDs to coordinates
        use_numpy: Whether to use NumPy storage for large matrices

    Returns:
        DistanceMatrix containing all shortest paths

    Example:
        >>> matrix = compute_distance_matrix(graph, {0, 1, 2, 5}, id_to_coords)
        >>> distance = matrix.get_distance(0, 5)

    Raises:
        ValueError: If graph doesn't implement dijkstra() method
    """
    if not hasattr(graph, "dijkstra"):
        raise ValueError("Graph must implement dijkstra() method")

    num_nodes = len(node_ids)
    logger.info(f"Computing distance matrix for {num_nodes} nodes")

    # Check if node IDs are contiguous (required for NumPy storage)
    max_node_id = max(node_ids) if node_ids else 0
    min_node_id = min(node_ids) if node_ids else 0
    is_contiguous = (max_node_id - min_node_id + 1) == num_nodes

    # Auto-select numpy for large matrices (only if node IDs are contiguous)
    if not use_numpy and NUMPY_AVAILABLE and num_nodes >= 1000:
        if is_contiguous and max_node_id < 100000:  # Reasonable upper bound
            use_numpy = True
            logger.info(f"Auto-enabling NumPy storage for large matrix ({num_nodes} nodes)")
        else:
            logger.warning(
                f"Node IDs are non-contiguous (range: {min_node_id}-{max_node_id}, "
                f"count: {num_nodes}). Using dict storage instead of NumPy."
            )

    with LogTimer(logger, "Distance matrix computation"):
        matrix = DistanceMatrix(use_numpy=use_numpy, num_nodes=num_nodes)
        matrix.id_to_coords = id_to_coords.copy()

        invalid_paths = 0
        total_pairs = 0

        # Compute shortest paths from each node
        for source_id in node_ids:
            try:
                distances, predecessors = graph.dijkstra(source_id)
            except Exception as e:
                logger.error(f"Dijkstra failed from node {source_id}: {e}", exc_info=True)
                continue

            for target_id in node_ids:
                if source_id == target_id:
                    continue

                total_pairs += 1

                # Check if target is reachable
                if distances[target_id] == float("inf"):
                    logger.debug(f"No path from {source_id} to {target_id}")
                    continue

                # Reconstruct path
                path_ids = reconstruct_path(predecessors, source_id, target_id)

                if not path_ids:
                    invalid_paths += 1
                    logger.debug(f"Failed to reconstruct path {source_id} -> {target_id}")
                    continue

                # Store in matrix
                matrix.set(source_id, target_id, distances[target_id], path_ids)

    # Log statistics
    stats = matrix.get_stats()
    logger.info(
        f"Distance matrix complete: {stats.num_paths}/{total_pairs} paths, "
        f"{stats.memory_bytes / 1024 / 1024:.1f} MB, storage={stats.storage_type}"
    )

    if invalid_paths > 0:
        logger.warning(f"Failed to reconstruct {invalid_paths} paths")

    return matrix
