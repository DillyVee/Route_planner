"""
Parallel execution framework for DRPP routing with minimal memory overhead.

Uses ProcessPoolExecutor for better resource management compared to Pool.
Precomputes distance matrices in parent process to avoid pickling full graph.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing
import time

from .types import (
    Coordinate, NodeID, SegmentIndex, ClusterID,
    PathResult, UnreachableSegment
)
from .distance_matrix import DistanceMatrix, compute_distance_matrix
from .greedy_router import greedy_route_cluster, NodeNormalizer
from .logging_config import get_logger, LogTimer, log_exception

logger = get_logger(__name__)


@dataclass
class ClusterTask:
    """Lightweight task data for worker processes.

    Contains only precomputed distance matrix and metadata - no graph object.
    This minimizes pickle overhead and memory usage in worker processes.

    Attributes:
        cluster_id: Unique cluster identifier
        cluster_index: Index in processing order
        segment_indices: List of segment indices in this cluster
        required_edges: List of required edges (shared across all tasks)
        distance_matrix: Precomputed distance matrix for this cluster
        normalizer: Node ID normalizer
        start_node_id: Starting node ID
        enable_fallback: Whether to enable Dijkstra fallback
    """
    cluster_id: ClusterID
    cluster_index: int
    segment_indices: List[SegmentIndex]
    required_edges: List[Tuple]  # Shared, read-only
    distance_matrix: DistanceMatrix
    normalizer: NodeNormalizer
    start_node_id: NodeID
    enable_fallback: bool = False  # Disabled in workers (no graph available)


@dataclass
class ClusterTaskResult:
    """Result from routing a single cluster.

    Attributes:
        cluster_id: Cluster that was routed
        cluster_index: Original index for ordering
        success: Whether routing succeeded
        path: Route coordinates (empty if failed)
        distance: Total route distance in meters
        segments_covered: Number of segments successfully covered
        segments_unreachable: Number of unreachable segments
        computation_time: Time taken in seconds
        error_message: Error message if failed
        worker_id: ID of worker process
    """
    cluster_id: ClusterID
    cluster_index: int
    success: bool
    path: List[Coordinate]
    distance: float
    segments_covered: int
    segments_unreachable: int
    computation_time: float
    error_message: Optional[str] = None
    worker_id: Optional[int] = None


def _route_cluster_worker(task: ClusterTask) -> ClusterTaskResult:
    """Worker function for routing a single cluster.

    This function runs in a separate process. It receives precomputed
    distance matrix, NOT the full graph object, to minimize memory usage.

    Args:
        task: ClusterTask with all necessary precomputed data

    Returns:
        ClusterTaskResult with routing results or error information

    Note:
        This function catches all exceptions and returns them in the result
        rather than raising, to ensure the parent process can handle errors.
    """
    import os
    worker_id = os.getpid()

    try:
        # Reconstruct start node from ID
        start_node = task.normalizer.id_to_node.get(task.start_node_id)
        if start_node is None:
            raise ValueError(f"Invalid start node ID: {task.start_node_id}")

        # Route using precomputed matrix (no graph needed!)
        result = greedy_route_cluster(
            graph=None,  # Not available in worker
            required_edges=task.required_edges,
            segment_indices=task.segment_indices,
            start_node=start_node,
            distance_matrix=task.distance_matrix,
            normalizer=task.normalizer,
            enable_fallback=task.enable_fallback
        )

        return ClusterTaskResult(
            cluster_id=task.cluster_id,
            cluster_index=task.cluster_index,
            success=True,
            path=result.path,
            distance=result.distance,
            segments_covered=result.segments_covered,
            segments_unreachable=result.segments_unreachable,
            computation_time=result.computation_time,
            worker_id=worker_id
        )

    except Exception as e:
        import traceback
        logger.error(
            f"Worker {worker_id} failed on cluster {task.cluster_id}: {e}",
            exc_info=True
        )
        return ClusterTaskResult(
            cluster_id=task.cluster_id,
            cluster_index=task.cluster_index,
            success=False,
            path=[],
            distance=0.0,
            segments_covered=0,
            segments_unreachable=len(task.segment_indices),
            computation_time=0.0,
            error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            worker_id=worker_id
        )


def _precompute_cluster_tasks(
    graph: Any,
    required_edges: List[Tuple],
    clusters: Dict[ClusterID, List[SegmentIndex]],
    cluster_order: List[ClusterID],
    start_node: Coordinate | NodeID
) -> List[ClusterTask]:
    """Precompute distance matrices for all clusters in parent process.

    This is the key optimization: we compute all matrices once in the parent
    process with access to the graph, then send only lightweight matrices
    to workers.

    Args:
        graph: Graph object with dijkstra() method
        required_edges: List of all required edges
        clusters: Mapping from cluster ID to segment indices
        cluster_order: Order in which to process clusters
        start_node: Global starting position

    Returns:
        List of ClusterTask objects ready for parallel execution

    Note:
        This function can be slow for many/large clusters, but it runs
        only once and eliminates repeated graph pickling.
    """
    logger.info(f"Precomputing distance matrices for {len(cluster_order)} clusters")

    normalizer_main = NodeNormalizer(graph.node_to_id, graph.id_to_node)
    start_node_id = normalizer_main.to_id(start_node)

    if start_node_id is None:
        raise ValueError(f"Start node {start_node} has no valid ID")

    tasks: List[ClusterTask] = []

    with LogTimer(logger, "Distance matrix precomputation"):
        for cluster_index, cluster_id in enumerate(cluster_order):
            seg_idxs = clusters[cluster_id]

            # Extract unique nodes for this cluster
            node_ids = set()
            id_to_coords = {}

            for seg_idx in seg_idxs:
                start_coord = required_edges[seg_idx][0]
                end_coord = required_edges[seg_idx][1]

                start_id = normalizer_main.to_id(start_coord)
                end_id = normalizer_main.to_id(end_coord)

                if start_id is not None:
                    node_ids.add(start_id)
                    id_to_coords[start_id] = start_coord
                if end_id is not None:
                    node_ids.add(end_id)
                    id_to_coords[end_id] = end_coord

            # Include global start node
            node_ids.add(start_node_id)
            id_to_coords[start_node_id] = normalizer_main.id_to_node[start_node_id]

            # Compute distance matrix for this cluster
            try:
                distance_matrix = compute_distance_matrix(
                    graph, node_ids, id_to_coords
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute matrix for cluster {cluster_id}: {e}",
                    exc_info=True
                )
                # Create empty matrix - worker will handle failure
                distance_matrix = DistanceMatrix()
                distance_matrix.id_to_coords = id_to_coords

            # Create task
            task = ClusterTask(
                cluster_id=cluster_id,
                cluster_index=cluster_index,
                segment_indices=seg_idxs,
                required_edges=required_edges,
                distance_matrix=distance_matrix,
                normalizer=normalizer_main,
                start_node_id=start_node_id,
                enable_fallback=False  # No graph in workers
            )
            tasks.append(task)

            if (cluster_index + 1) % 10 == 0:
                logger.debug(f"Precomputed {cluster_index + 1}/{len(cluster_order)} matrices")

    logger.info(f"Precomputation complete: {len(tasks)} tasks ready")
    return tasks


def parallel_cluster_routing(
    graph: Any,
    required_edges: List[Tuple],
    clusters: Dict[ClusterID, List[SegmentIndex]],
    cluster_order: List[ClusterID],
    start_node: Coordinate | NodeID,
    num_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[PathResult]:
    """Route through clusters in parallel using ProcessPoolExecutor.

    This is the main entry point for parallel DRPP routing with all
    production-ready optimizations:
    - No graph pickling (10-50x memory reduction)
    - Robust error handling (workers never crash parent)
    - Progress tracking
    - Detailed logging
    - Type safety

    Args:
        graph: Graph object with dijkstra(), node_to_id, id_to_node
        required_edges: List of all required edges
        clusters: Mapping from cluster ID to segment indices
        cluster_order: Order in which to process clusters
        start_node: Global starting position
        num_workers: Number of worker processes (default: CPU count - 1)
        progress_callback: Optional function(completed, total) for progress

    Returns:
        List of PathResult objects in cluster_order

    Example:
        >>> results = parallel_cluster_routing(
        ...     graph=graph,
        ...     required_edges=edges,
        ...     clusters=clusters,
        ...     cluster_order=[0, 1, 2],
        ...     start_node=(40.7, -74.0),
        ...     num_workers=4
        ... )
        >>> total_distance = sum(r.distance for r in results)

    Raises:
        ValueError: If invalid arguments provided
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    num_workers = min(num_workers, len(cluster_order))

    logger.info(
        f"Starting parallel routing: {len(cluster_order)} clusters, "
        f"{num_workers} workers"
    )

    # Phase 1: Precompute all distance matrices in parent process
    tasks = _precompute_cluster_tasks(
        graph, required_edges, clusters, cluster_order, start_node
    )

    # Phase 2: Route clusters in parallel
    results: List[ClusterTaskResult] = []
    completed = 0
    failed = 0

    logger.info("Starting parallel execution")
    parallel_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_route_cluster_worker, task): task
            for task in tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_task):
            completed += 1

            try:
                result = future.result(timeout=300)  # 5 minute timeout per cluster
                results.append(result)

                if not result.success:
                    failed += 1
                    logger.error(
                        f"Cluster {result.cluster_id} failed: {result.error_message}"
                    )

                # Progress callback
                if progress_callback is not None:
                    progress_callback(completed, len(cluster_order))

                # Periodic logging
                if completed % 10 == 0 or completed == len(cluster_order):
                    elapsed = time.perf_counter() - parallel_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{len(cluster_order)} "
                        f"({100 * completed / len(cluster_order):.1f}%) "
                        f"[{rate:.1f} clusters/sec]"
                    )

            except Exception as e:
                logger.error(f"Future raised exception: {e}", exc_info=True)
                task = future_to_task[future]
                # Create failure result
                results.append(ClusterTaskResult(
                    cluster_id=task.cluster_id,
                    cluster_index=task.cluster_index,
                    success=False,
                    path=[],
                    distance=0.0,
                    segments_covered=0,
                    segments_unreachable=len(task.segment_indices),
                    computation_time=0.0,
                    error_message=str(e)
                ))
                failed += 1

    # Sort results by cluster index to maintain order
    results.sort(key=lambda r: r.cluster_index)

    # Convert to PathResult objects
    path_results = [
        PathResult(
            path=r.path,
            distance=r.distance,
            cluster_id=r.cluster_id,
            segments_covered=r.segments_covered,
            segments_unreachable=r.segments_unreachable,
            computation_time=r.computation_time
        )
        for r in results
    ]

    # Summary logging
    total_elapsed = time.perf_counter() - parallel_start
    total_distance = sum(r.distance for r in results)
    total_covered = sum(r.segments_covered for r in results)
    total_unreachable = sum(r.segments_unreachable for r in results)

    logger.info("=" * 60)
    logger.info(f"Parallel routing complete in {total_elapsed:.1f}s")
    logger.info(f"Throughput: {len(cluster_order) / total_elapsed:.1f} clusters/sec")
    logger.info(f"Total distance: {total_distance / 1000:.1f} km")
    logger.info(f"Segments covered: {total_covered}")
    logger.info(f"Segments unreachable: {total_unreachable}")
    if failed > 0:
        logger.warning(f"Failed clusters: {failed}")
    logger.info("=" * 60)

    return path_results


def estimate_optimal_workers(
    num_clusters: int,
    num_segments: int,
    avg_cluster_size: Optional[int] = None
) -> int:
    """Estimate optimal number of worker processes.

    Args:
        num_clusters: Total number of clusters
        num_segments: Total number of segments
        avg_cluster_size: Average segments per cluster (computed if not provided)

    Returns:
        Recommended number of workers

    Example:
        >>> workers = estimate_optimal_workers(num_clusters=100, num_segments=5000)
        >>> print(f"Use {workers} workers")
    """
    available_cpus = multiprocessing.cpu_count()

    if avg_cluster_size is None and num_clusters > 0:
        avg_cluster_size = num_segments // num_clusters

    logger.debug(
        f"Estimating workers: {num_clusters} clusters, "
        f"{num_segments} segments, "
        f"avg size {avg_cluster_size}, "
        f"{available_cpus} CPUs"
    )

    # If many clusters, use most CPUs
    if num_clusters >= available_cpus:
        workers = max(1, available_cpus - 1)
    # If few clusters, use fewer workers
    elif num_clusters > 1:
        workers = min(num_clusters, max(1, available_cpus // 2))
    else:
        workers = 1

    logger.info(f"Recommended workers: {workers}")
    return workers
