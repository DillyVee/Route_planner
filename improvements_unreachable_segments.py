"""
IMPROVEMENT 6: Enhanced Unreachable Segment Handling
=====================================================

Problem (line 366-369):
  - If best_seg_idx is None, all remaining segments marked unreachable
  - No diagnostic information about WHY segments are unreachable
  - No fallback strategies attempted
  - Silent failure in workers

Solutions:
  1. Collect detailed unreachable segment metadata
  2. Attempt secondary fallback (Dijkstra recheck, nearest-neighbor)
  3. Log per-cluster unreachable segments with reasons
  4. Provide actionable diagnostics for graph connectivity issues
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import time


# ============================================================================
# UNREACHABLE SEGMENT METADATA
# ============================================================================

class UnreachableSegmentInfo:
    """
    Detailed information about why a segment is unreachable.
    """
    __slots__ = ['seg_idx', 'reason', 'from_node', 'to_node', 'attempted_from',
                 'min_distance', 'coords']

    def __init__(self, seg_idx: int, reason: str):
        self.seg_idx = seg_idx
        self.reason = reason
        self.from_node = None
        self.to_node = None
        self.attempted_from = None
        self.min_distance = float('inf')
        self.coords = None

    def to_dict(self):
        return {
            'seg_idx': self.seg_idx,
            'reason': self.reason,
            'from_node': self.from_node,
            'to_node': self.to_node,
            'attempted_from': self.attempted_from,
            'min_distance': self.min_distance
        }


# ============================================================================
# REASON CODES
# ============================================================================

class UnreachableReason:
    """Standard reason codes for unreachable segments"""
    NO_PATH_IN_MATRIX = "no_path_in_precomputed_matrix"
    DISCONNECTED_COMPONENT = "disconnected_graph_component"
    ALL_SEGMENTS_UNREACHABLE = "all_remaining_segments_unreachable"
    DIJKSTRA_FAILED = "dijkstra_failed"
    NODE_NOT_IN_GRAPH = "node_not_in_graph"
    INFINITE_DISTANCE = "infinite_distance"


# ============================================================================
# ENHANCED GREEDY WITH UNREACHABLE TRACKING
# ============================================================================

def optimized_greedy_route_with_tracking(
    graph, required_edges, seg_idxs, start_node,
    distance_matrix=None, endpoints=None,
    enable_fallback=True
):
    """
    Enhanced greedy routing with detailed unreachable segment tracking.

    New features:
      - Collects reason codes for unreachable segments
      - Attempts fallback strategies before marking unreachable
      - Provides diagnostic information

    Args:
        graph: DirectedGraph instance
        required_edges: List of (start, end, coords, idx) tuples
        seg_idxs: Segment indices to visit
        start_node: Starting position
        distance_matrix: Optional precomputed matrix
        endpoints: Optional set of endpoint nodes
        enable_fallback: Try fallback strategies for unreachable segments

    Returns:
        (path_coords, total_distance, unreachable_info_list)
        where unreachable_info_list is List[UnreachableSegmentInfo]
    """
    if not seg_idxs:
        return [], 0.0, []

    # Precompute distance matrix if needed
    if distance_matrix is None:
        distance_matrix, endpoints = precompute_distance_matrix_basic(
            graph, required_edges, seg_idxs, start_node
        )

    path = []
    remaining = set(seg_idxs)
    current = start_node
    total_dist = 0.0
    unreachable_info = []

    # Main greedy loop
    while remaining:
        best_seg_idx = None
        best_approach_dist = float('inf')
        best_approach_path = None

        # Track why segments weren't chosen
        segment_diagnostics = {}

        # Find nearest remaining segment
        for seg_idx in remaining:
            segment_start = required_edges[seg_idx][0]

            # Look up precomputed distance
            matrix_key = (current, segment_start)

            if matrix_key in distance_matrix:
                approach_dist, approach_path = distance_matrix[matrix_key]

                # Track diagnostic info
                segment_diagnostics[seg_idx] = {
                    'has_path': True,
                    'distance': approach_dist
                }

                if approach_dist < best_approach_dist:
                    best_approach_dist = approach_dist
                    best_seg_idx = seg_idx
                    best_approach_path = approach_path
            else:
                # Segment not reachable from current position
                segment_diagnostics[seg_idx] = {
                    'has_path': False,
                    'distance': float('inf')
                }

        # If no reachable segment found
        if best_seg_idx is None:
            # Try fallback strategies
            if enable_fallback:
                fallback_result = try_fallback_strategies(
                    graph, required_edges, remaining, current, distance_matrix
                )

                if fallback_result is not None:
                    best_seg_idx, best_approach_dist, best_approach_path = fallback_result
                else:
                    # Fallback failed - mark all remaining as unreachable
                    for seg_idx in remaining:
                        info = create_unreachable_info(
                            seg_idx, required_edges, current,
                            segment_diagnostics.get(seg_idx, {}),
                            UnreachableReason.ALL_SEGMENTS_UNREACHABLE
                        )
                        unreachable_info.append(info)
                    break
            else:
                # No fallback - mark all remaining as unreachable
                for seg_idx in remaining:
                    info = create_unreachable_info(
                        seg_idx, required_edges, current,
                        segment_diagnostics.get(seg_idx, {}),
                        UnreachableReason.NO_PATH_IN_MATRIX
                    )
                    unreachable_info.append(info)
                break

        # Route to the best segment
        segment_start = required_edges[best_seg_idx][0]
        segment_end = required_edges[best_seg_idx][1]
        segment_coords = required_edges[best_seg_idx][2]

        # Add approach path
        if best_approach_path:
            path = append_path(path, best_approach_path)
            total_dist += best_approach_dist

        # Traverse the segment itself
        path = append_path(path, segment_coords)
        segment_length = path_length(segment_coords)
        total_dist += segment_length

        # Update position
        current = segment_end
        remaining.remove(best_seg_idx)

    return path, total_dist, unreachable_info


def create_unreachable_info(seg_idx, required_edges, current_pos, diagnostics, reason):
    """Create detailed unreachable segment info"""
    info = UnreachableSegmentInfo(seg_idx, reason)

    if seg_idx < len(required_edges):
        info.from_node = required_edges[seg_idx][0]
        info.to_node = required_edges[seg_idx][1]
        info.coords = required_edges[seg_idx][2]

    info.attempted_from = current_pos
    info.min_distance = diagnostics.get('distance', float('inf'))

    return info


# ============================================================================
# FALLBACK STRATEGIES
# ============================================================================

def try_fallback_strategies(graph, required_edges, remaining, current, distance_matrix):
    """
    Try fallback strategies to reach unreachable segments.

    Strategies (in order):
      1. Nearest reachable endpoint (ignore approach distance)
      2. Re-run Dijkstra from current position
      3. Check alternative endpoints (segment end instead of start)

    Returns:
        (seg_idx, approach_dist, approach_path) or None
    """
    # STRATEGY 1: Find segment with minimum Euclidean distance (ignore graph distance)
    fallback_seg_idx = find_nearest_segment_euclidean(
        required_edges, remaining, current
    )

    if fallback_seg_idx is not None:
        # Try to find path via Dijkstra
        segment_start = required_edges[fallback_seg_idx][0]

        # Get node IDs
        current_id = graph.node_to_id.get(current)
        target_id = graph.node_to_id.get(segment_start)

        if current_id is not None and target_id is not None:
            # Run Dijkstra to recheck connectivity
            dist_array, prev_array = graph.dijkstra(current_id)

            if dist_array[target_id] != float('inf'):
                # Found a path!
                path = reconstruct_path(graph, prev_array, current_id, target_id)
                return (fallback_seg_idx, dist_array[target_id], path)

    # STRATEGY 2: Try alternative endpoints (segment end instead of start)
    for seg_idx in remaining:
        segment_end = required_edges[seg_idx][1]
        matrix_key = (current, segment_end)

        if matrix_key in distance_matrix:
            approach_dist, approach_path = distance_matrix[matrix_key]
            # Use segment end as entry point, then backtrack
            return (seg_idx, approach_dist, approach_path)

    # All strategies failed
    return None


def find_nearest_segment_euclidean(required_edges, remaining, current):
    """
    Find nearest segment by Euclidean distance (ignore graph connectivity).

    Fallback when no graph path exists.
    """
    best_seg_idx = None
    best_dist = float('inf')

    current_lat, current_lon = current

    for seg_idx in remaining:
        seg_start = required_edges[seg_idx][0]
        seg_lat, seg_lon = seg_start

        # Simple Euclidean distance in degrees (fast approximation)
        dist = ((seg_lat - current_lat)**2 + (seg_lon - current_lon)**2)**0.5

        if dist < best_dist:
            best_dist = dist
            best_seg_idx = seg_idx

    return best_seg_idx


# ============================================================================
# DIAGNOSTIC REPORTING
# ============================================================================

def analyze_unreachable_segments(unreachable_info: List[UnreachableSegmentInfo], cluster_id):
    """
    Analyze unreachable segments and generate diagnostic report.

    Returns:
        Dict with analysis results
    """
    if not unreachable_info:
        return {'total': 0}

    # Count by reason
    reason_counts = defaultdict(int)
    for info in unreachable_info:
        reason_counts[info.reason] += 1

    # Find unique nodes involved
    unique_nodes = set()
    for info in unreachable_info:
        if info.from_node:
            unique_nodes.add(info.from_node)
        if info.to_node:
            unique_nodes.add(info.to_node)

    analysis = {
        'total': len(unreachable_info),
        'cluster_id': cluster_id,
        'reason_breakdown': dict(reason_counts),
        'unique_nodes': len(unique_nodes),
        'segment_indices': [info.seg_idx for info in unreachable_info]
    }

    return analysis


def print_unreachable_report(unreachable_info: List[UnreachableSegmentInfo], cluster_id):
    """Print detailed report of unreachable segments"""
    if not unreachable_info:
        return

    print(f"\n⚠️ Unreachable Segments Report - Cluster {cluster_id}")
    print("="*70)

    analysis = analyze_unreachable_segments(unreachable_info, cluster_id)

    print(f"Total unreachable: {analysis['total']}")
    print(f"\nReason breakdown:")
    for reason, count in analysis['reason_breakdown'].items():
        print(f"  • {reason}: {count}")

    print(f"\nAffected segment indices: {analysis['segment_indices'][:10]}")
    if len(analysis['segment_indices']) > 10:
        print(f"  ... and {len(analysis['segment_indices']) - 10} more")

    print("\nRecommendations:")
    if UnreachableReason.DISCONNECTED_COMPONENT in analysis['reason_breakdown']:
        print("  → Graph may have disconnected components. Check OSM data quality.")
    if UnreachableReason.NO_PATH_IN_MATRIX in analysis['reason_breakdown']:
        print("  → Some segments not in distance matrix. Consider expanding endpoint set.")
    if analysis['total'] > 10:
        print("  → Many unreachable segments. Consider:")
        print("     - Different clustering strategy")
        print("     - Checking graph connectivity")
        print("     - Verifying OSM data coverage")

    print("="*70)


# ============================================================================
# UPDATED WORKER FUNCTION
# ============================================================================

def _route_cluster_worker_with_tracking(args):
    """
    Worker function with enhanced unreachable segment tracking.

    Returns detailed unreachable info for diagnostics.
    """
    cluster_idx, cid, seg_idxs, graph, required_edges, allow_return, start_node = args

    try:
        start_time = time.time()

        # Precompute distance matrix
        distance_matrix, endpoints = precompute_distance_matrix_basic(
            graph, required_edges, seg_idxs, start_node
        )

        # Run greedy with tracking
        cluster_path, cluster_dist, unreachable_info = optimized_greedy_route_with_tracking(
            graph, required_edges, seg_idxs, start_node,
            distance_matrix=distance_matrix,
            endpoints=endpoints,
            enable_fallback=True
        )

        elapsed = time.time() - start_time

        # Analyze unreachable segments
        analysis = analyze_unreachable_segments(unreachable_info, cid) if unreachable_info else {}

        return {
            'success': True,
            'cluster_idx': cluster_idx,
            'cid': cid,
            'path': cluster_path,
            'distance': cluster_dist,
            'num_segments': len(seg_idxs),
            'unreachable_count': len(unreachable_info),
            'unreachable_analysis': analysis,
            'time': elapsed
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'cluster_idx': cluster_idx,
            'cid': cid,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def precompute_distance_matrix_basic(graph, required_edges, seg_idxs, start_node):
    """Basic distance matrix precomputation (simplified)"""
    endpoints = set()
    for seg_idx in seg_idxs:
        endpoints.add(required_edges[seg_idx][0])
        endpoints.add(required_edges[seg_idx][1])

    if start_node is not None:
        endpoints.add(start_node)

    distance_matrix = {}

    for source_node in endpoints:
        source_id = graph.node_to_id.get(source_node)
        if source_id is None:
            continue

        dist_array, prev_array = graph.dijkstra(source_id)

        for target_node in endpoints:
            if source_node == target_node:
                continue

            target_id = graph.node_to_id.get(target_node)
            if target_id is None or dist_array[target_id] == float('inf'):
                continue

            path_coords = reconstruct_path(graph, prev_array, source_id, target_id)
            distance_matrix[(source_node, target_node)] = (dist_array[target_id], path_coords)

    return distance_matrix, endpoints


def reconstruct_path(graph, prev_array, source_id, target_id):
    """Reconstruct path from Dijkstra prev_array"""
    path_ids = []
    cur = target_id

    while cur != -1 and cur is not None:
        path_ids.append(cur)
        if cur == source_id:
            break
        cur = prev_array[cur]

    path_ids.reverse()
    return [graph.id_to_node[nid] for nid in path_ids if nid in graph.id_to_node]


def append_path(total_path, new_coords):
    """Append path coordinates, avoiding duplicates"""
    if not new_coords:
        return total_path
    if total_path and len(new_coords) > 0 and total_path[-1] == new_coords[0]:
        new_coords = new_coords[1:]
    total_path.extend(new_coords)
    return total_path


def path_length(coords):
    """Calculate path length in meters"""
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
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("ENHANCED UNREACHABLE SEGMENT HANDLING")
    print("="*70)

    # Example unreachable info
    unreachable = [
        UnreachableSegmentInfo(5, UnreachableReason.NO_PATH_IN_MATRIX),
        UnreachableSegmentInfo(12, UnreachableReason.DISCONNECTED_COMPONENT),
        UnreachableSegmentInfo(18, UnreachableReason.NO_PATH_IN_MATRIX),
    ]

    unreachable[0].attempted_from = (40.7128, -74.0060)
    unreachable[0].from_node = (40.7500, -73.9900)
    unreachable[0].min_distance = float('inf')

    print_unreachable_report(unreachable, cluster_id=3)
