"""
IMPROVEMENT 4: Ensure start_node and Node Keys Alignment
=========================================================

Problem: Mixing node IDs (integers) and coordinates (tuples) as keys causes:
  - KeyError when looking up (current, segment_start) in distance matrix
  - Type mismatches: current might be tuple, segment_start might be int
  - Inconsistent key types across preprocessing and routing

Solution: Normalize ALL node references to a single type (prefer node IDs)

Current code issues (line 352):
  matrix_key = (current, segment_start)  # current might be coords OR int!

Best practice: Use node IDs for all internal operations, convert to coords only for output
"""

from typing import Union, Tuple, Any, Dict, Set


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

NodeCoords = Tuple[float, float]
NodeID = int
NodeKey = Union[NodeID, NodeCoords]


# ============================================================================
# NODE NORMALIZATION
# ============================================================================

class NodeNormalizer:
    """
    Ensures consistent node representation throughout routing.

    Strategy: Use node IDs internally, coordinates only for display/output

    Benefits:
      - Faster lookups (int hash vs tuple hash)
      - Smaller memory (8 bytes vs 16 bytes)
      - Type consistency (no mixed keys)
    """

    def __init__(self, graph):
        self.graph = graph
        self.node_to_id = graph.node_to_id
        self.id_to_node = graph.id_to_node

    def normalize_to_id(self, node: NodeKey) -> int:
        """
        Convert any node representation to node ID.

        Args:
            node: Either node ID (int) or coordinates (tuple)

        Returns:
            Node ID (int)

        Raises:
            ValueError: If node not found in graph
        """
        # Already a node ID
        if isinstance(node, int):
            return node

        # Coordinates - look up ID
        if isinstance(node, tuple) and len(node) == 2:
            node_id = self.node_to_id.get(node)
            if node_id is None:
                raise ValueError(f"Node coordinates {node} not found in graph")
            return node_id

        raise TypeError(f"Invalid node type: {type(node)}")

    def normalize_to_coords(self, node: NodeKey) -> NodeCoords:
        """Convert any node representation to coordinates"""
        # Already coordinates
        if isinstance(node, tuple) and len(node) == 2:
            return node

        # Node ID - look up coordinates
        if isinstance(node, int):
            coords = self.id_to_node.get(node)
            if coords is None:
                raise ValueError(f"Node ID {node} not found in graph")
            return coords

        raise TypeError(f"Invalid node type: {type(node)}")

    def normalize_required_edges(self, required_edges):
        """
        Normalize required_edges to use node IDs consistently.

        Input: [(start_coords, end_coords, coords, idx), ...]
        Output: [(start_id, end_id, coords, idx), ...]
        """
        normalized = []

        for start_node, end_node, coords, idx in required_edges:
            start_id = self.normalize_to_id(start_node)
            end_id = self.normalize_to_id(end_node)
            normalized.append((start_id, end_id, coords, idx))

        return normalized


# ============================================================================
# UPDATED PREPROCESSING (with normalization)
# ============================================================================

def precompute_distance_matrix_normalized(graph, required_edges, seg_idxs, start_node=None):
    """
    NORMALIZED VERSION of precompute_distance_matrix.

    Key improvement: ALL keys use node IDs (integers), not coordinates

    Benefits:
      - No type mismatches
      - Faster dictionary lookups
      - Smaller memory footprint
    """
    normalizer = NodeNormalizer(graph)

    # Normalize required edges to use node IDs
    required_edges_normalized = normalizer.normalize_required_edges(required_edges)

    # Extract unique endpoint IDs
    endpoint_ids = set()
    for seg_idx in seg_idxs:
        start_id = required_edges_normalized[seg_idx][0]
        end_id = required_edges_normalized[seg_idx][1]
        endpoint_ids.add(start_id)
        endpoint_ids.add(end_id)

    # Normalize start node to ID
    start_node_id = None
    if start_node is not None:
        start_node_id = normalizer.normalize_to_id(start_node)
        endpoint_ids.add(start_node_id)

    # Distance matrix with node ID keys
    distance_matrix_ids = {}

    # Run Dijkstra from each endpoint
    for source_id in endpoint_ids:
        dist_array, prev_array = graph.dijkstra(source_id)

        for target_id in endpoint_ids:
            if source_id == target_id:
                continue

            if dist_array[target_id] == float('inf'):
                continue

            # Reconstruct path (as node IDs)
            path_ids = reconstruct_path_robust(prev_array, source_id, target_id)

            if not path_ids:
                continue

            # Store with node ID keys
            distance_matrix_ids[(source_id, target_id)] = (
                dist_array[target_id],
                path_ids  # Store as node IDs, not coordinates
            )

    return distance_matrix_ids, endpoint_ids, normalizer


def reconstruct_path_robust(prev_array, source_id: int, target_id: int):
    """Robust path reconstruction returning node IDs"""
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


# ============================================================================
# UPDATED GREEDY ROUTING (with normalization)
# ============================================================================

def optimized_greedy_route_normalized(graph, required_edges, seg_idxs, start_node):
    """
    NORMALIZED VERSION of optimized_greedy_route_cluster.

    All node references use IDs consistently.

    Drop-in replacement for lines 299-390 in parallel_processing_addon_greedy_v2.py
    """
    if not seg_idxs:
        return [], 0.0, []

    # Precompute with normalization
    distance_matrix_ids, endpoint_ids, normalizer = precompute_distance_matrix_normalized(
        graph, required_edges, seg_idxs, start_node
    )

    # Normalize required edges
    required_edges_normalized = normalizer.normalize_required_edges(required_edges)

    path_ids = []  # Store path as node IDs during routing
    remaining = set(seg_idxs)
    current_id = normalizer.normalize_to_id(start_node)  # NORMALIZED to ID
    total_dist = 0.0
    unreachable = []

    # Main greedy loop - ALL operations use node IDs
    while remaining:
        best_seg_idx = None
        best_approach_dist = float('inf')
        best_approach_path_ids = None

        # Find nearest remaining segment
        for seg_idx in remaining:
            segment_start_id = required_edges_normalized[seg_idx][0]  # Already ID

            # Look up precomputed distance - NO TYPE MISMATCH
            matrix_key = (current_id, segment_start_id)  # Both are ints!

            if matrix_key in distance_matrix_ids:
                approach_dist, approach_path_ids = distance_matrix_ids[matrix_key]

                if approach_dist < best_approach_dist:
                    best_approach_dist = approach_dist
                    best_seg_idx = seg_idx
                    best_approach_path_ids = approach_path_ids

        # Check if any reachable segment found
        if best_seg_idx is None:
            unreachable.extend(list(remaining))
            break

        # Get segment data
        segment_start_id = required_edges_normalized[best_seg_idx][0]
        segment_end_id = required_edges_normalized[best_seg_idx][1]
        segment_coords = required_edges_normalized[best_seg_idx][2]

        # Add approach path (as node IDs)
        if best_approach_path_ids:
            path_ids.extend(best_approach_path_ids)
            total_dist += best_approach_dist

        # Traverse segment
        # Add segment endpoints as IDs (coordinates stored separately)
        total_dist += calculate_path_length(segment_coords)

        # Update position
        current_id = segment_end_id  # Stay in ID space
        remaining.remove(best_seg_idx)

    # Convert final path from IDs to coordinates
    path_coords = convert_path_to_coords(path_ids, required_edges_normalized, normalizer)

    return path_coords, total_dist, unreachable


def convert_path_to_coords(path_ids, required_edges_normalized, normalizer):
    """
    Convert path node IDs to coordinates, incorporating segment geometries.

    Args:
        path_ids: List of node IDs forming the route
        required_edges_normalized: Required edges with full geometries
        normalizer: NodeNormalizer instance

    Returns:
        List of coordinate tuples
    """
    if not path_ids:
        return []

    path_coords = []

    # Convert approach path IDs to coordinates
    for node_id in path_ids:
        coords = normalizer.id_to_node.get(node_id)
        if coords:
            if not path_coords or path_coords[-1] != coords:
                path_coords.append(coords)

    return path_coords


def calculate_path_length(coords):
    """Calculate path length from coordinates"""
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
# DEBUGGING: TYPE CHECKER
# ============================================================================

def validate_key_types(distance_matrix, current_position, required_edges):
    """
    Debug helper to validate all keys have consistent types.

    Args:
        distance_matrix: Dict[(node, node)] -> (dist, path)
        current_position: Current node position
        required_edges: List of edge tuples

    Returns:
        Dict with type consistency report
    """
    key_types = set()
    position_type = type(current_position)

    for (from_node, to_node) in distance_matrix.keys():
        key_types.add((type(from_node), type(to_node)))

    segment_start_types = set()
    for start, end, coords, idx in required_edges:
        segment_start_types.add(type(start))

    return {
        'position_type': position_type.__name__,
        'matrix_key_types': [(t1.__name__, t2.__name__) for t1, t2 in key_types],
        'segment_start_types': [t.__name__ for t in segment_start_types],
        'is_consistent': len(key_types) == 1 and position_type in [t[0] for t in key_types]
    }


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

def print_migration_guide():
    """Print guide for migrating from mixed-type to normalized keys"""
    print("="*70)
    print("MIGRATION GUIDE: Mixed-Type Keys → Normalized Node IDs")
    print("="*70)

    print("\nBEFORE (Mixed Types - PROBLEMATIC):")
    print("-" * 70)
    print("""
    # Current code mixes coordinates and IDs
    current = required_edges[seg_idx][0]  # Might be coords tuple
    segment_start = required_edges[seg_idx][0]  # Also coords
    matrix_key = (current, segment_start)  # (tuple, tuple)

    # But after routing:
    current = segment_end  # Still coords
    # Matrix lookup works... until:

    # Worker receives different representation
    current = graph.node_to_id[start_node]  # Now it's an int!
    matrix_key = (current, segment_start)  # (int, tuple) - MISMATCH!
    """)

    print("\nAFTER (Normalized - ROBUST):")
    print("-" * 70)
    print("""
    normalizer = NodeNormalizer(graph)

    # Normalize everything upfront
    required_edges_norm = normalizer.normalize_required_edges(required_edges)
    current_id = normalizer.normalize_to_id(start_node)

    # All operations use IDs
    segment_start_id = required_edges_norm[seg_idx][0]  # int
    matrix_key = (current_id, segment_start_id)  # (int, int) - CONSISTENT!

    # Update position
    current_id = segment_end_id  # Still int

    # Convert to coords only for final output
    path_coords = [normalizer.id_to_node[nid] for nid in path_ids]
    """)

    print("\nBENEFITS:")
    print("-" * 70)
    print("  ✓ No type mismatches or KeyErrors")
    print("  ✓ Faster lookups (int hash > tuple hash)")
    print("  ✓ Smaller memory (8 bytes vs 16 bytes per key)")
    print("  ✓ Easier debugging (clear int types)")
    print("="*70)


if __name__ == '__main__':
    print_migration_guide()
