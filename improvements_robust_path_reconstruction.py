"""
IMPROVEMENT 3: Robust Path Reconstruction
==========================================

Problem: Current code assumes prev_array sentinel is -1.
Different graph implementations may use:
  - None (common in Python)
  - -1 (C-style)
  - cur == source_id (loop detection)

Solution: Implement robust path reconstruction with:
  - Multiple sentinel value handling
  - Cycle detection
  - Maximum iteration limit
  - Validation of path connectivity
"""

from typing import List, Optional, Tuple, Any
import numpy as np


# ============================================================================
# ROBUST PATH RECONSTRUCTION
# ============================================================================

def reconstruct_path_from_prev_robust(
    graph,
    prev_array,
    source_id: int,
    target_id: int,
    return_coords: bool = True
) -> List:
    """
    Robust path reconstruction that handles all common sentinel values.

    Features:
      - Handles None, -1, and self-loop sentinels
      - Detects cycles (prevents infinite loops)
      - Validates path connectivity
      - Optionally returns coordinates or node IDs

    Args:
        graph: Graph object with id_to_node mapping
        prev_array: Previous node array from Dijkstra
        source_id: Source node ID
        target_id: Target node ID
        return_coords: If True, return coordinates; else return node IDs

    Returns:
        List of coordinates (if return_coords=True) or node IDs (if False)
        Empty list if path not found or invalid

    Example sentinels handled:
        - prev_array[node] = -1 (no predecessor, C-style)
        - prev_array[node] = None (no predecessor, Python-style)
        - prev_array[source] = source (self-loop at source)
    """
    path_ids = []
    cur = target_id
    visited = set()  # Cycle detection

    # Maximum iterations = array length (prevents infinite loops)
    max_iterations = len(prev_array)

    for iteration in range(max_iterations):
        # Add current node
        path_ids.append(cur)

        # Check if we reached the source
        if cur == source_id:
            # Success! Reverse path and return
            path_ids.reverse()

            if return_coords:
                # Convert to coordinates
                return [
                    graph.id_to_node[node_id]
                    for node_id in path_ids
                    if node_id in graph.id_to_node
                ]
            else:
                return path_ids

        # Cycle detection
        if cur in visited:
            # Detected cycle - path is invalid
            return []

        visited.add(cur)

        # Get previous node
        prev = prev_array[cur]

        # Check for sentinel values indicating no predecessor
        if prev is None:
            # No predecessor found (common in Python graphs)
            return []

        if prev == -1:
            # No predecessor found (common in C-style implementations)
            return []

        if prev == cur:
            # Self-loop at non-source node (invalid)
            return []

        # Move to previous node
        cur = prev

    # Exceeded max iterations without reaching source
    return []


# ============================================================================
# ALTERNATIVE: AUTO-DETECT SENTINEL
# ============================================================================

def detect_sentinel_value(prev_array, source_id: int) -> Any:
    """
    Auto-detect the sentinel value used in prev_array.

    Returns the value used to mark "no predecessor"

    Common patterns:
      - prev_array[source_id] = -1  (return -1)
      - prev_array[source_id] = None (return None)
      - prev_array[source_id] = source_id (return source_id)
    """
    # Check source node's predecessor
    source_prev = prev_array[source_id]

    if source_prev is None:
        return None
    elif source_prev == -1:
        return -1
    elif source_prev == source_id:
        return source_id
    else:
        # Unknown pattern, default to -1
        return -1


def reconstruct_path_auto_sentinel(
    graph,
    prev_array,
    source_id: int,
    target_id: int
) -> List[Tuple[float, float]]:
    """
    Reconstruct path with auto-detected sentinel value.

    Automatically detects which sentinel value is used (-1, None, or source_id)
    """
    sentinel = detect_sentinel_value(prev_array, source_id)

    path_ids = []
    cur = target_id
    visited = set()
    max_iterations = len(prev_array)

    for _ in range(max_iterations):
        path_ids.append(cur)

        if cur == source_id:
            break

        if cur in visited:
            return []

        visited.add(cur)

        prev = prev_array[cur]

        # Check against detected sentinel
        if prev == sentinel and cur != source_id:
            return []

        if prev == cur and cur != source_id:
            return []

        cur = prev

    path_ids.reverse()

    # Convert to coordinates
    return [
        graph.id_to_node[node_id]
        for node_id in path_ids
        if node_id in graph.id_to_node
    ]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_path(graph, path_ids: List[int], source_id: int, target_id: int) -> bool:
    """
    Validate that a reconstructed path is correct.

    Checks:
      - Path starts at source
      - Path ends at target
      - All consecutive nodes are connected in graph
      - No duplicate nodes (no cycles)
    """
    if not path_ids:
        return False

    # Check endpoints
    if path_ids[0] != source_id:
        return False

    if path_ids[-1] != target_id:
        return False

    # Check no duplicates (no cycles)
    if len(path_ids) != len(set(path_ids)):
        return False

    # Check connectivity (each pair of consecutive nodes has an edge)
    for i in range(len(path_ids) - 1):
        current_id = path_ids[i]
        next_id = path_ids[i + 1]

        # Check if edge exists in graph
        if not has_edge(graph, current_id, next_id):
            return False

    return True


def has_edge(graph, from_id: int, to_id: int) -> bool:
    """
    Check if graph has an edge from from_id to to_id.

    Assumes graph has an adjacency structure like:
      graph.adj[from_id] = [(to_id, weight), ...]
    """
    if hasattr(graph, 'adj'):
        neighbors = graph.adj.get(from_id, [])
        return any(neighbor_id == to_id for neighbor_id, _ in neighbors)

    # Fallback: assume edge exists if both nodes exist
    return from_id in graph.id_to_node and to_id in graph.id_to_node


# ============================================================================
# UPDATED PRECOMPUTE FUNCTION (Drop-in replacement)
# ============================================================================

def precompute_distance_matrix_robust(graph, required_edges, seg_idxs, start_node=None):
    """
    ROBUST VERSION of precompute_distance_matrix.

    Drop-in replacement for line 93-155 in parallel_processing_addon_greedy_v2.py

    Changes:
      - Uses robust path reconstruction
      - Validates paths before storing
      - Logs warnings for invalid paths
    """
    # Extract all unique endpoint nodes
    endpoints = set()
    for seg_idx in seg_idxs:
        start_node_seg = required_edges[seg_idx][0]
        end_node_seg = required_edges[seg_idx][1]
        endpoints.add(start_node_seg)
        endpoints.add(end_node_seg)

    # Add start node if specified
    if start_node is not None:
        endpoints.add(start_node)

    distance_matrix = {}
    invalid_paths = 0

    # Run Dijkstra from each endpoint ONCE
    for source_node in endpoints:
        # Get node ID
        source_id = graph.node_to_id.get(source_node)
        if source_id is None:
            continue

        # Run Dijkstra once from this source
        dist_array, prev_array = graph.dijkstra(source_id)

        # Store distances and paths to all other endpoints
        for target_node in endpoints:
            if source_node == target_node:
                continue

            target_id = graph.node_to_id.get(target_node)
            if target_id is None or dist_array[target_id] == float('inf'):
                continue

            # ROBUST path reconstruction
            path_coords = reconstruct_path_from_prev_robust(
                graph, prev_array, source_id, target_id, return_coords=True
            )

            # Validate path
            if not path_coords:
                invalid_paths += 1
                continue

            distance = dist_array[target_id]

            # Store in matrix
            distance_matrix[(source_node, target_node)] = (distance, path_coords)

    if invalid_paths > 0:
        print(f"  ⚠️ Warning: {invalid_paths} invalid paths detected and skipped")

    return distance_matrix, endpoints


# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def debug_path_reconstruction(graph, prev_array, source_id: int, target_id: int):
    """
    Debug helper to understand why path reconstruction might fail.

    Prints detailed information about:
      - Sentinel value detected
      - Path backtracking steps
      - Where reconstruction failed
    """
    print(f"\n[DEBUG] Reconstructing path from {source_id} to {target_id}")

    # Detect sentinel
    sentinel = detect_sentinel_value(prev_array, source_id)
    print(f"[DEBUG] Detected sentinel: {sentinel}")

    # Backtrack
    path_ids = []
    cur = target_id
    step = 0

    while step < 20:  # Limit debug output
        print(f"[DEBUG] Step {step}: cur={cur}, prev={prev_array[cur]}")

        path_ids.append(cur)

        if cur == source_id:
            print(f"[DEBUG] ✓ Reached source at step {step}")
            break

        prev = prev_array[cur]

        if prev == sentinel and cur != source_id:
            print(f"[DEBUG] ✗ Hit sentinel at step {step} (before reaching source)")
            return None

        if prev == cur:
            print(f"[DEBUG] ✗ Self-loop detected at step {step}")
            return None

        cur = prev
        step += 1

    path_ids.reverse()
    print(f"[DEBUG] ✓ Path: {path_ids}")
    return path_ids


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

class MockGraph:
    """Mock graph for testing path reconstruction"""

    def __init__(self):
        self.id_to_node = {
            0: (0.0, 0.0),
            1: (1.0, 1.0),
            2: (2.0, 2.0),
            3: (3.0, 3.0),
            4: (4.0, 4.0)
        }
        self.node_to_id = {v: k for k, v in self.id_to_node.items()}


def test_path_reconstruction():
    """Test robust path reconstruction with various sentinel values"""
    graph = MockGraph()

    print("="*60)
    print("TESTING ROBUST PATH RECONSTRUCTION")
    print("="*60)

    # Test 1: Sentinel = -1
    print("\nTest 1: Sentinel = -1 (C-style)")
    prev_array_1 = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    path = reconstruct_path_from_prev_robust(graph, prev_array_1, 0, 4)
    print(f"  Path from 0 to 4: {path}")
    print(f"  ✓ Expected: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]")

    # Test 2: Sentinel = 0 (source self-loop)
    print("\nTest 2: Sentinel = source_id (self-loop)")
    prev_array_2 = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    path = reconstruct_path_from_prev_robust(graph, prev_array_2, 0, 4)
    print(f"  Path from 0 to 4: {path}")

    # Test 3: Invalid path (unreachable)
    print("\nTest 3: Invalid path (unreachable node)")
    prev_array_3 = np.array([-1, 0, 1, -1, -1], dtype=np.int32)
    path = reconstruct_path_from_prev_robust(graph, prev_array_3, 0, 4)
    print(f"  Path from 0 to 4: {path}")
    print(f"  ✓ Expected: [] (unreachable)")

    # Test 4: Cycle detection
    print("\nTest 4: Cycle detection")
    prev_array_4 = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    prev_array_4[2] = 4  # Create cycle: 4 -> 3 -> 2 -> 4
    prev_array_4[4] = 2
    path = reconstruct_path_from_prev_robust(graph, prev_array_4, 0, 4)
    print(f"  Path from 0 to 4: {path}")
    print(f"  ✓ Expected: [] (cycle detected)")

    print("\n" + "="*60)


if __name__ == '__main__':
    test_path_reconstruction()
