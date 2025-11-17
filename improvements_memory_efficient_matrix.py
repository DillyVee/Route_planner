"""
IMPROVEMENT 2: Memory-Efficient Distance Matrix Storage
========================================================

Problem: Current storage: Dict[(node, node)] = (distance, path_coords)
  - 200 endpoints → 40,000 entries
  - Each entry stores full path coordinates (list of tuples)
  - Memory usage: ~4-40 MB per cluster

Solutions:
  1. Store only distances, reconstruct paths on-demand
  2. Store path as node IDs (integers), convert to coords when needed
  3. Use sparse matrix representation for large clusters
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


# ============================================================================
# OPTION 1: DISTANCE-ONLY MATRIX (Minimal Memory)
# ============================================================================

class DistanceOnlyMatrix:
    """
    Store ONLY distances, reconstruct paths on-demand.

    Memory: Dict[(int, int), float] ≈ 24 bytes/entry
    Trade-off: Must recompute paths when needed (via Dijkstra or stored prev_array)

    Best for: Very large clusters (500+ endpoints) where memory is critical
    """

    def __init__(self):
        self.distances: Dict[Tuple[int, int], float] = {}
        # Store prev_arrays from Dijkstra for path reconstruction
        self.prev_arrays: Dict[int, np.ndarray] = {}
        self.id_to_coords: Dict[int, Tuple[float, float]] = {}

    def precompute_from_graph(self, graph, endpoint_ids: set):
        """
        Precompute distances and store prev_arrays for reconstruction.

        Memory: O(K × V) where K = endpoints, V = graph nodes
        """
        for source_id in endpoint_ids:
            dist_array, prev_array = graph.dijkstra(source_id)

            # Store prev_array for this source (enables path reconstruction)
            self.prev_arrays[source_id] = prev_array

            # Store only distances to target endpoints
            for target_id in endpoint_ids:
                if source_id != target_id and dist_array[target_id] != float('inf'):
                    self.distances[(source_id, target_id)] = dist_array[target_id]

    def get_distance(self, source_id: int, target_id: int) -> float:
        """Get precomputed distance"""
        return self.distances.get((source_id, target_id), float('inf'))

    def reconstruct_path(self, source_id: int, target_id: int) -> List[int]:
        """
        Reconstruct path on-demand using stored prev_array.

        Time: O(path_length) - fast, just backtracking
        """
        if source_id not in self.prev_arrays:
            return []

        prev_array = self.prev_arrays[source_id]
        path_ids = []
        cur = target_id
        max_iterations = len(prev_array)

        for _ in range(max_iterations):
            path_ids.append(cur)

            if cur == source_id:
                break

            prev = prev_array[cur]

            if prev is None or prev == -1 or prev == cur:
                return []

            cur = prev

        path_ids.reverse()
        return path_ids

    def get_path_coords(self, source_id: int, target_id: int) -> List[Tuple[float, float]]:
        """Get path as coordinates (reconstructed on-demand)"""
        path_ids = self.reconstruct_path(source_id, target_id)
        return [self.id_to_coords[nid] for nid in path_ids if nid in self.id_to_coords]

    def memory_footprint(self) -> dict:
        """Estimate memory usage"""
        distance_bytes = len(self.distances) * 24  # Dict entry overhead + 2 ints + 1 float
        prev_arrays_bytes = sum(arr.nbytes for arr in self.prev_arrays.values())
        coords_bytes = len(self.id_to_coords) * 40  # Dict entry + 2 floats

        return {
            'distances_kb': distance_bytes / 1024,
            'prev_arrays_kb': prev_arrays_bytes / 1024,
            'coords_kb': coords_bytes / 1024,
            'total_kb': (distance_bytes + prev_arrays_bytes + coords_bytes) / 1024
        }


# ============================================================================
# OPTION 2: NODE-ID PATH STORAGE (Balanced)
# ============================================================================

class NodeIDPathMatrix:
    """
    Store distances + paths as node IDs (not coordinates).

    Memory: Distance: 24 bytes/entry + Path IDs: ~8N bytes/entry
    Trade-off: Small memory increase, but no recomputation needed

    Best for: Medium clusters (50-500 endpoints) - good balance
    """

    def __init__(self):
        self.distances: Dict[Tuple[int, int], float] = {}
        self.path_node_ids: Dict[Tuple[int, int], List[int]] = {}
        self.id_to_coords: Dict[int, Tuple[float, float]] = {}

    def precompute_from_graph(self, graph, endpoint_ids: set):
        """Precompute distances and paths (as node IDs)"""
        for source_id in endpoint_ids:
            dist_array, prev_array = graph.dijkstra(source_id)

            for target_id in endpoint_ids:
                if source_id == target_id:
                    continue

                if dist_array[target_id] == float('inf'):
                    continue

                # Store distance
                self.distances[(source_id, target_id)] = dist_array[target_id]

                # Store path as node IDs
                path_ids = reconstruct_path_ids_robust(prev_array, source_id, target_id)
                self.path_node_ids[(source_id, target_id)] = path_ids

    def get_distance(self, source_id: int, target_id: int) -> float:
        return self.distances.get((source_id, target_id), float('inf'))

    def get_path_ids(self, source_id: int, target_id: int) -> List[int]:
        return self.path_node_ids.get((source_id, target_id), [])

    def get_path_coords(self, source_id: int, target_id: int) -> List[Tuple[float, float]]:
        """Convert path IDs to coordinates"""
        path_ids = self.get_path_ids(source_id, target_id)
        return [self.id_to_coords[nid] for nid in path_ids if nid in self.id_to_coords]

    def memory_footprint(self) -> dict:
        """Estimate memory usage"""
        distance_bytes = len(self.distances) * 24
        path_bytes = sum(
            40 + len(path) * 8  # Dict overhead + list of ints
            for path in self.path_node_ids.values()
        )
        coords_bytes = len(self.id_to_coords) * 40

        return {
            'distances_kb': distance_bytes / 1024,
            'paths_kb': path_bytes / 1024,
            'coords_kb': coords_bytes / 1024,
            'total_kb': (distance_bytes + path_bytes + coords_bytes) / 1024
        }


# ============================================================================
# OPTION 3: SPARSE MATRIX (For Very Large Clusters)
# ============================================================================

class SparseDistanceMatrix:
    """
    Use scipy sparse matrix for distance storage.

    Memory: Only stores non-infinite distances
    Best for: Clusters with sparse connectivity (not fully connected)
    """

    def __init__(self, num_nodes: int):
        try:
            from scipy.sparse import lil_matrix
            self.matrix = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            self.node_id_to_idx: Dict[int, int] = {}
            self.idx_to_node_id: Dict[int, int] = {}
            self.scipy_available = True
        except ImportError:
            # Fallback to dict if scipy not available
            self.matrix = {}
            self.scipy_available = False

    def add_endpoint(self, node_id: int) -> int:
        """Add endpoint and return its matrix index"""
        if node_id not in self.node_id_to_idx:
            idx = len(self.node_id_to_idx)
            self.node_id_to_idx[node_id] = idx
            self.idx_to_node_id[idx] = node_id
        return self.node_id_to_idx[node_id]

    def set_distance(self, source_id: int, target_id: int, distance: float):
        """Set distance in sparse matrix"""
        src_idx = self.add_endpoint(source_id)
        tgt_idx = self.add_endpoint(target_id)

        if self.scipy_available:
            self.matrix[src_idx, tgt_idx] = distance
        else:
            self.matrix[(src_idx, tgt_idx)] = distance

    def get_distance(self, source_id: int, target_id: int) -> float:
        """Get distance from sparse matrix"""
        if source_id not in self.node_id_to_idx or target_id not in self.node_id_to_idx:
            return float('inf')

        src_idx = self.node_id_to_idx[source_id]
        tgt_idx = self.node_id_to_idx[target_id]

        if self.scipy_available:
            val = self.matrix[src_idx, tgt_idx]
            return float(val) if val != 0 else float('inf')
        else:
            return self.matrix.get((src_idx, tgt_idx), float('inf'))

    def to_csr(self):
        """Convert to CSR format for faster lookups (call after all distances added)"""
        if self.scipy_available:
            self.matrix = self.matrix.tocsr()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def reconstruct_path_ids_robust(prev_array, source_id: int, target_id: int) -> List[int]:
    """
    Robust path reconstruction handling multiple sentinel values.
    """
    path_ids = []
    cur = target_id
    max_iterations = len(prev_array)

    for _ in range(max_iterations):
        path_ids.append(cur)

        if cur == source_id:
            break

        prev = prev_array[cur]

        # Handle None, -1, or self-loop
        if prev is None or prev == -1 or prev == cur:
            return []

        cur = prev

    path_ids.reverse()
    return path_ids


# ============================================================================
# MEMORY COMPARISON
# ============================================================================

def compare_memory_options(num_endpoints: int, avg_path_length: int = 10,
                          graph_nodes: int = 10000):
    """
    Compare memory usage of different storage options.

    Args:
        num_endpoints: Number of unique endpoints in cluster
        avg_path_length: Average path length (nodes)
        graph_nodes: Total nodes in graph
    """

    # CURRENT APPROACH: Store (distance, path_coords)
    current_distance = num_endpoints ** 2 * 8  # float64
    current_path_coords = num_endpoints ** 2 * avg_path_length * 16  # 2 × float64 per coord
    current_dict_overhead = num_endpoints ** 2 * 40  # Python dict overhead
    current_total = current_distance + current_path_coords + current_dict_overhead

    # OPTION 1: Distance only + prev_arrays
    opt1_distance = num_endpoints ** 2 * 8
    opt1_prev_arrays = num_endpoints * graph_nodes * 4  # int32 per node
    opt1_dict_overhead = num_endpoints ** 2 * 24
    opt1_total = opt1_distance + opt1_prev_arrays + opt1_dict_overhead

    # OPTION 2: Distance + node ID paths
    opt2_distance = num_endpoints ** 2 * 8
    opt2_path_ids = num_endpoints ** 2 * avg_path_length * 8  # int64 per node
    opt2_dict_overhead = num_endpoints ** 2 * 40
    opt2_total = opt2_distance + opt2_path_ids + opt2_dict_overhead

    # OPTION 3: Sparse matrix
    opt3_sparse = num_endpoints ** 2 * 12  # Approximate CSR storage

    return {
        'current': {
            'total_mb': current_total / (1024 ** 2),
            'breakdown': {
                'distances_mb': current_distance / (1024 ** 2),
                'path_coords_mb': current_path_coords / (1024 ** 2),
                'overhead_mb': current_dict_overhead / (1024 ** 2)
            }
        },
        'option1_distance_only': {
            'total_mb': opt1_total / (1024 ** 2),
            'savings_ratio': current_total / opt1_total if opt1_total > 0 else 1,
            'breakdown': {
                'distances_mb': opt1_distance / (1024 ** 2),
                'prev_arrays_mb': opt1_prev_arrays / (1024 ** 2),
                'overhead_mb': opt1_dict_overhead / (1024 ** 2)
            }
        },
        'option2_node_ids': {
            'total_mb': opt2_total / (1024 ** 2),
            'savings_ratio': current_total / opt2_total if opt2_total > 0 else 1,
            'breakdown': {
                'distances_mb': opt2_distance / (1024 ** 2),
                'path_ids_mb': opt2_path_ids / (1024 ** 2),
                'overhead_mb': opt2_dict_overhead / (1024 ** 2)
            }
        },
        'option3_sparse': {
            'total_mb': opt3_sparse / (1024 ** 2),
            'savings_ratio': current_total / opt3_sparse if opt3_sparse > 0 else 1
        }
    }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def recommend_storage_strategy(num_endpoints: int, num_segments: int,
                               graph_nodes: int) -> str:
    """
    Recommend best storage strategy based on cluster size.

    Rules:
      - Small clusters (<50 endpoints): Current approach is fine
      - Medium clusters (50-500 endpoints): Use NodeIDPathMatrix
      - Large clusters (500+ endpoints): Use DistanceOnlyMatrix
      - Very sparse graphs: Consider SparseDistanceMatrix
    """
    if num_endpoints < 50:
        return "CURRENT (coordinates): Memory footprint acceptable for small clusters"

    elif num_endpoints < 500:
        return "OPTION2 (NodeIDPathMatrix): Best balance for medium clusters. " \
               "2x memory savings, no recomputation needed."

    else:
        return "OPTION1 (DistanceOnlyMatrix): Critical for large clusters. " \
               "5-10x memory savings, path reconstruction on-demand."


if __name__ == '__main__':
    # Example comparison for a large cluster
    print("="*70)
    print("MEMORY COMPARISON: 200 Endpoints, 10 Avg Path Length")
    print("="*70)

    comparison = compare_memory_options(
        num_endpoints=200,
        avg_path_length=10,
        graph_nodes=10000
    )

    print(f"\nCURRENT APPROACH (distance + path_coords):")
    print(f"  Total: {comparison['current']['total_mb']:.2f} MB")
    print(f"  - Distances: {comparison['current']['breakdown']['distances_mb']:.2f} MB")
    print(f"  - Path coords: {comparison['current']['breakdown']['path_coords_mb']:.2f} MB")
    print(f"  - Overhead: {comparison['current']['breakdown']['overhead_mb']:.2f} MB")

    print(f"\nOPTION 1 (distance only + prev_arrays):")
    print(f"  Total: {comparison['option1_distance_only']['total_mb']:.2f} MB")
    print(f"  Savings: {comparison['option1_distance_only']['savings_ratio']:.1f}x")
    print(f"  Trade-off: Path reconstruction on-demand")

    print(f"\nOPTION 2 (distance + node ID paths): ⭐ RECOMMENDED")
    print(f"  Total: {comparison['option2_node_ids']['total_mb']:.2f} MB")
    print(f"  Savings: {comparison['option2_node_ids']['savings_ratio']:.1f}x")
    print(f"  Trade-off: None - just convert IDs to coords when needed")

    print(f"\nOPTION 3 (sparse matrix):")
    print(f"  Total: {comparison['option3_sparse']['total_mb']:.2f} MB")
    print(f"  Savings: {comparison['option3_sparse']['savings_ratio']:.1f}x")
    print(f"  Best for: Sparse connectivity graphs")

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print(recommend_storage_strategy(200, 400, 10000))
    print("="*70)
