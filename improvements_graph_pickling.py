"""
IMPROVEMENT 1: Avoid Pickling Full Graph in Workers
====================================================

Problem: multiprocessing.Pool pickles the entire graph object for each worker,
causing memory duplication on Windows and serialization overhead on Unix.

Solution: Precompute distance matrices in parent process, pass only:
  - Distance matrix (dict of floats)
  - Path node IDs (list of integers)
  - Node coordinate mappings (dict)
"""

from typing import Dict, List, Tuple, Set
import multiprocessing as mp
from multiprocessing import Pool


# ============================================================================
# SERIALIZABLE DATA STRUCTURES
# ============================================================================

class SerializableClusterData:
    """
    Lightweight, picklable cluster data (no graph object).

    Memory footprint:
      - distance_only: Dict[(int, int), float] ≈ 24 bytes/entry
      - path_node_ids: Dict[(int, int), List[int]] ≈ 40 + 8N bytes/entry
      - id_to_coords: Dict[int, (float, float)] ≈ 40 bytes/entry
    """
    __slots__ = ['distance_matrix', 'path_node_ids', 'id_to_coords',
                 'required_edges_slim', 'seg_idxs', 'start_node_id']

    def __init__(self):
        # Distance only (no paths stored here for minimal memory)
        self.distance_matrix: Dict[Tuple[int, int], float] = {}

        # Path as node IDs (integers, not coordinates)
        self.path_node_ids: Dict[Tuple[int, int], List[int]] = {}

        # Node ID -> coordinates mapping (shared across all paths)
        self.id_to_coords: Dict[int, Tuple[float, float]] = {}

        # Slim required edges: (start_id, end_id, coords, orig_idx)
        self.required_edges_slim: List[Tuple[int, int, List, int]] = []

        self.seg_idxs: List[int] = []
        self.start_node_id: int = None


# ============================================================================
# PREPROCESSING IN PARENT PROCESS
# ============================================================================

def precompute_all_cluster_matrices(graph, required_edges, clusters,
                                    cluster_order, start_node):
    """
    Precompute distance matrices for ALL clusters in parent process.

    This runs sequentially but avoids pickling the graph N times.
    For large graphs, this is much faster than pickling.

    Args:
        graph: DirectedGraph instance (only used in parent)
        required_edges: List of (start, end, coords, idx)
        clusters: Dict[cluster_id] -> [seg_idx, ...]
        cluster_order: Ordered list of cluster IDs
        start_node: Starting location (coordinates)

    Returns:
        List[SerializableClusterData] in cluster_order
    """
    print(f"\n[PARENT] Preprocessing {len(cluster_order)} cluster matrices...")

    # Convert start_node to ID
    start_node_id = graph.node_to_id.get(start_node)

    cluster_data_list = []

    for cluster_idx, cid in enumerate(cluster_order):
        seg_idxs = clusters[cid]

        # Create serializable data structure
        cluster_data = SerializableClusterData()
        cluster_data.seg_idxs = seg_idxs
        cluster_data.start_node_id = start_node_id

        # Extract unique endpoints and convert to IDs
        endpoint_ids = set()
        node_coords_map = {}  # node_id -> coords

        for seg_idx in seg_idxs:
            start_node_coords = required_edges[seg_idx][0]
            end_node_coords = required_edges[seg_idx][1]

            start_id = graph.node_to_id.get(start_node_coords)
            end_id = graph.node_to_id.get(end_node_coords)

            if start_id is not None:
                endpoint_ids.add(start_id)
                node_coords_map[start_id] = start_node_coords
            if end_id is not None:
                endpoint_ids.add(end_id)
                node_coords_map[end_id] = end_node_coords

        if start_node_id is not None:
            endpoint_ids.add(start_node_id)
            node_coords_map[start_node_id] = start_node

        # Run Dijkstra from each endpoint
        for source_id in endpoint_ids:
            dist_array, prev_array = graph.dijkstra(source_id)

            for target_id in endpoint_ids:
                if source_id == target_id:
                    continue

                if dist_array[target_id] == float('inf'):
                    continue

                # Store distance
                cluster_data.distance_matrix[(source_id, target_id)] = dist_array[target_id]

                # Store path as node IDs (much smaller than coordinates)
                path_ids = reconstruct_path_ids_robust(prev_array, source_id, target_id)
                cluster_data.path_node_ids[(source_id, target_id)] = path_ids

        # Store node ID -> coordinate mapping
        cluster_data.id_to_coords = node_coords_map

        # Create slim required edges (with node IDs instead of coordinates as keys)
        for seg_idx in seg_idxs:
            start_coords = required_edges[seg_idx][0]
            end_coords = required_edges[seg_idx][1]
            segment_coords = required_edges[seg_idx][2]

            start_id = graph.node_to_id.get(start_coords)
            end_id = graph.node_to_id.get(end_coords)

            cluster_data.required_edges_slim.append(
                (start_id, end_id, segment_coords, seg_idx)
            )

        cluster_data_list.append(cluster_data)

        if (cluster_idx + 1) % 10 == 0:
            print(f"  Preprocessed {cluster_idx + 1}/{len(cluster_order)} clusters")

    print(f"[PARENT] ✓ Preprocessing complete")
    return cluster_data_list


def reconstruct_path_ids_robust(prev_array, source_id, target_id):
    """
    Robust path reconstruction that handles different sentinel values.

    Returns list of node IDs from source to target.
    """
    path_ids = []
    cur = target_id
    max_iterations = len(prev_array)  # Prevent infinite loops

    for _ in range(max_iterations):
        path_ids.append(cur)

        if cur == source_id:
            break

        prev = prev_array[cur]

        # Handle various sentinel values
        if prev is None or prev == -1 or prev == cur:
            # Path not found or invalid
            return []

        cur = prev

    path_ids.reverse()
    return path_ids


# ============================================================================
# WORKER FUNCTION (NO GRAPH NEEDED)
# ============================================================================

def _route_cluster_worker_no_graph(cluster_data: SerializableClusterData):
    """
    Worker function that operates on precomputed data only.

    NO GRAPH PICKLING - all data is precomputed and serialized.

    Args:
        cluster_data: SerializableClusterData with precomputed matrices

    Returns:
        Result dict with path (as coordinates) and distance
    """
    import time

    try:
        start_time = time.time()

        # Run greedy routing using precomputed data
        path_coords, total_dist, unreachable = greedy_route_from_matrix(cluster_data)

        elapsed = time.time() - start_time

        return {
            'success': True,
            'path': path_coords,
            'distance': total_dist,
            'unreachable': unreachable,
            'num_segments': len(cluster_data.seg_idxs),
            'time': elapsed
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def greedy_route_from_matrix(cluster_data: SerializableClusterData):
    """
    Greedy routing using only precomputed matrix data.

    No graph access - all lookups use node IDs.
    """
    path_coords = []
    remaining = set(cluster_data.seg_idxs)
    current_id = cluster_data.start_node_id
    total_dist = 0.0
    unreachable = []

    while remaining:
        best_seg_idx = None
        best_dist = float('inf')
        best_path_ids = None

        # Find nearest segment using precomputed distances
        for seg_idx_pos, (start_id, end_id, seg_coords, orig_idx) in enumerate(
            cluster_data.required_edges_slim
        ):
            if orig_idx not in remaining:
                continue

            # Lookup precomputed distance
            matrix_key = (current_id, start_id)

            if matrix_key in cluster_data.distance_matrix:
                dist = cluster_data.distance_matrix[matrix_key]

                if dist < best_dist:
                    best_dist = dist
                    best_seg_idx = orig_idx
                    best_path_ids = cluster_data.path_node_ids.get(matrix_key, [])

        if best_seg_idx is None:
            unreachable.extend(list(remaining))
            break

        # Find the segment data
        best_seg_data = None
        for start_id, end_id, seg_coords, orig_idx in cluster_data.required_edges_slim:
            if orig_idx == best_seg_idx:
                best_seg_data = (start_id, end_id, seg_coords)
                break

        if best_seg_data is None:
            break

        start_id, end_id, seg_coords = best_seg_data

        # Add approach path (convert IDs to coordinates)
        if best_path_ids:
            approach_coords = [
                cluster_data.id_to_coords[node_id]
                for node_id in best_path_ids
                if node_id in cluster_data.id_to_coords
            ]
            path_coords = append_path(path_coords, approach_coords)

        total_dist += best_dist

        # Traverse segment
        path_coords = append_path(path_coords, seg_coords)
        segment_length = path_length(seg_coords)
        total_dist += segment_length

        # Update position
        current_id = end_id
        remaining.remove(best_seg_idx)

    return path_coords, total_dist, unreachable


def append_path(total_path, new_coords):
    """Append path coordinates, avoiding duplicates"""
    if not new_coords:
        return total_path
    if total_path and len(new_coords) > 0 and total_path[-1] == new_coords[0]:
        new_coords = new_coords[1:]
    total_path.extend(new_coords)
    return total_path


def path_length(coords):
    """Calculate total path length"""
    if len(coords) < 2:
        return 0.0
    from math import radians, cos, sin, asin, sqrt

    def haversine(a, b):
        lat1, lon1 = a
        lat2, lon2 = b
        lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        aa = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2*asin(sqrt(aa))
        return 6371000 * c

    return sum(haversine(coords[i], coords[i+1]) for i in range(len(coords)-1))


# ============================================================================
# MAIN PARALLEL ROUTING (UPDATED)
# ============================================================================

def parallel_cluster_routing_no_graph_pickling(graph, required_edges, clusters,
                                               cluster_order, start_node,
                                               num_workers=None):
    """
    Parallel routing WITHOUT pickling the graph to workers.

    Architecture:
      1. Parent: Precompute all cluster distance matrices sequentially
      2. Parent: Serialize matrices as lightweight data structures
      3. Workers: Receive only matrices + metadata (no graph)
      4. Workers: Run greedy routing using precomputed data

    Memory savings: 10-100x less data transfer on Windows
    Time savings: Avoids repeated graph pickling
    """
    import multiprocessing

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    num_workers = min(num_workers, len(cluster_order))

    print(f"\n[NO-GRAPH-PICKLING] Routing {len(cluster_order)} clusters")
    print(f"[NO-GRAPH-PICKLING] Phase 1: Preprocessing in parent process...")

    # PHASE 1: Precompute all matrices in parent (sequential, but no pickling)
    cluster_data_list = precompute_all_cluster_matrices(
        graph, required_edges, clusters, cluster_order, start_node
    )

    # PHASE 2: Parallel routing with lightweight data
    print(f"\n[NO-GRAPH-PICKLING] Phase 2: Routing {len(cluster_order)} clusters with {num_workers} workers...")

    results = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(_route_cluster_worker_no_graph, cluster_data_list)

    # Format results
    ordered_results = [
        (r['path'], r['distance'], cluster_order[i])
        for i, r in enumerate(results)
        if r['success']
    ]

    total_distance = sum(r['distance'] for r in results if r['success'])

    print(f"\n[NO-GRAPH-PICKLING] ✓ Complete")
    print(f"[NO-GRAPH-PICKLING] Total distance: {total_distance/1000:.1f} km")

    return ordered_results


# ============================================================================
# MEMORY COMPARISON
# ============================================================================

def estimate_memory_savings(num_endpoints, graph_size_mb=50):
    """
    Estimate memory savings from avoiding graph pickling.

    Example:
        - Graph size: 50 MB (typical urban area)
        - 100 endpoints per cluster
        - Distance matrix: 100² × 24 bytes = 240 KB
        - Path IDs: 100² × 8 × 10 (avg path length) = 800 KB
        - Total serializable data: ~1 MB

        Savings: 50 MB → 1 MB = 50x reduction
    """
    matrix_size_kb = (num_endpoints ** 2) * 24 / 1024
    path_ids_size_kb = (num_endpoints ** 2) * 8 * 10 / 1024  # Assume avg path length 10
    total_kb = matrix_size_kb + path_ids_size_kb

    savings_ratio = (graph_size_mb * 1024) / total_kb if total_kb > 0 else 1

    return {
        'graph_size_mb': graph_size_mb,
        'matrix_size_kb': matrix_size_kb,
        'path_ids_size_kb': path_ids_size_kb,
        'total_serializable_kb': total_kb,
        'savings_ratio': savings_ratio
    }


if __name__ == '__main__':
    # Example memory savings
    savings = estimate_memory_savings(num_endpoints=200, graph_size_mb=50)
    print("="*60)
    print("MEMORY SAVINGS ANALYSIS")
    print("="*60)
    print(f"Graph size: {savings['graph_size_mb']} MB")
    print(f"Distance matrix: {savings['matrix_size_kb']:.1f} KB")
    print(f"Path node IDs: {savings['path_ids_size_kb']:.1f} KB")
    print(f"Total serializable: {savings['total_serializable_kb']:.1f} KB")
    print(f"Savings ratio: {savings['savings_ratio']:.1f}x")
    print("="*60)
