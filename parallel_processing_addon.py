"""
Parallel Processing Add-on for Route Planner
Significantly speeds up optimization using multiprocessing.

Add this to your route planner for major performance improvements.

Usage:
    from parallel_processing_addon import (
        parallel_cluster_routing,
        parallel_osm_matching,
        estimate_optimal_workers
    )
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional, Callable
import time
from functools import partial


# ============================================================================
# PARALLEL OSM MATCHING
# ============================================================================

def _match_segment_worker(args):
    """
    Worker function for parallel segment-to-OSM matching.
    NOTE: index cannot be pickled, so we pass grid data and rebuild it
    """
    seg_idx, segment, grid_data, fallback_speeds, max_distance = args
    
    # Rebuild index in worker process
    from collections import defaultdict
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(a, b):
        """Calculate distance between two lat/lon points in meters"""
        lat1, lon1 = a
        lat2, lon2 = b
        lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        aa = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2*asin(sqrt(aa))
        return 6371000 * c
    
    # Simple grid index class for worker
    class SimpleGridIndex:
        def __init__(self, cell_size_deg=0.01):
            self.cell_size = cell_size_deg
            self.grid = defaultdict(list)
        
        def _get_cell(self, lat, lon):
            return (int(lat / self.cell_size), int(lon / self.cell_size))
        
        def find_nearest_way(self, point, search_radius_cells=2, max_distance_m=100.0):
            lat, lon = point
            center_cell = self._get_cell(lat, lon)
            
            candidates = []
            for dx in range(-search_radius_cells, search_radius_cells + 1):
                for dy in range(-search_radius_cells, search_radius_cells + 1):
                    cell = (center_cell[0] + dx, center_cell[1] + dy)
                    if cell in self.grid:
                        candidates.extend(self.grid[cell])
            
            if not candidates:
                return None
            
            best_way = None
            best_dist = float('inf')
            
            seen = set()
            unique_candidates = []
            for way in candidates:
                way_id = id(way)
                if way_id not in seen:
                    seen.add(way_id)
                    unique_candidates.append(way)
            
            for way in unique_candidates:
                geometry = way.get('geometry', [])
                if not geometry:
                    continue
                
                min_dist = min(haversine(point, way_pt) for way_pt in geometry)
                
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_way = way
            
            return best_way if best_dist <= max_distance_m else None
    
    # Rebuild index
    index = SimpleGridIndex()
    index.grid = grid_data
    
    coords = segment['coords']
    midpoint = coords[len(coords) // 2]
    
    nearest_way = index.find_nearest_way(midpoint, max_distance_m=max_distance)
    
    result = {
        'seg_idx': seg_idx,
        'matched': False,
        'speed_limit': None,
        'speed_source': 'default',
        'highway_type': None
    }
    
    if nearest_way:
        result['matched'] = True
        result['highway_type'] = nearest_way['highway']
        
        if nearest_way['maxspeed'] and nearest_way['maxspeed'] > 0:
            result['speed_limit'] = nearest_way['maxspeed']
            result['speed_source'] = 'osm_matched'
        else:
            # Use fallback speed dictionary
            result['speed_limit'] = fallback_speeds.get(nearest_way['highway'], 30.0)
            result['speed_source'] = 'osm_fallback'
    else:
        result['speed_limit'] = 30.0
        result['speed_source'] = 'default'
    
    return result


def parallel_osm_matching(segments, index, overpass_fetcher, 
                         max_distance=100.0, num_workers=None):
    """
    Match segments to OSM ways in parallel.
    
    Args:
        segments: List of segments to match
        index: SimpleGridIndex with OSM ways
        overpass_fetcher: OverpassSpeedFetcher instance
        max_distance: Max matching distance (meters)
        num_workers: Number of workers (default: CPU count - 1)
    
    Returns:
        segments with updated speed information
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Filter segments that need matching (no KML speed)
    segments_to_match = []
    for idx, seg in enumerate(segments):
        if not seg.get('speed_limit') or seg['speed_limit'] <= 0:
            segments_to_match.append((idx, seg))
    
    if not segments_to_match:
        print("[PARALLEL] No segments need OSM matching")
        return segments
    
    print(f"[PARALLEL] Matching {len(segments_to_match)} segments using {num_workers} workers...")
    
    # Extract grid data and fallback speeds (these CAN be pickled)
    grid_data = dict(index.grid)  # Convert defaultdict to dict for pickling
    fallback_speeds = {
        'motorway': 110, 'motorway_link': 80, 'trunk': 90, 'trunk_link': 70,
        'primary': 70, 'primary_link': 50, 'secondary': 60, 'secondary_link': 50,
        'tertiary': 50, 'tertiary_link': 40, 'unclassified': 40, 'residential': 30,
        'living_street': 20, 'service': 20, 'track': 15, 'path': 10,
        'footway': 5, 'cycleway': 20, 'unknown': 30
    }
    
    # Prepare work items - pass grid data instead of index object
    work_items = [
        (idx, seg, grid_data, fallback_speeds, max_distance)
        for idx, seg in segments_to_match
    ]
    
    # Process in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(_match_segment_worker, work_items)
    
    # Apply results back to segments
    stats = {'osm_matched': 0, 'osm_fallback': 0, 'default': 0}
    
    for result in results:
        seg = segments[result['seg_idx']]
        seg['speed_limit'] = result['speed_limit']
        seg['speed_source'] = result['speed_source']
        if result['highway_type']:
            seg['highway_type'] = result['highway_type']
        stats[result['speed_source']] += 1
    
    print(f"[PARALLEL] ✓ Matched: OSM={stats['osm_matched']}, "
          f"Fallback={stats['osm_fallback']}, Default={stats['default']}")
    
    return segments


# ============================================================================
# PARALLEL CLUSTER ROUTING (for future use)
# ============================================================================

def _route_cluster_worker(args):
    """
    Worker function for parallel cluster routing.
    Must be top-level function for pickling.
    
    FIXED: Removed segment_callback parameter
    """
    cluster_idx, cid, seg_idxs, graph, required_edges, allow_return, start_node = args
    
    # Import here to avoid pickling issues
    from Route_Planner import greedy_arc_route_with_hungarian
    
    try:
        # ✅ FIXED: No segment_callback parameter
        cluster_path, cluster_m = greedy_arc_route_with_hungarian(
            graph, 
            required_edges, 
            seg_idxs,
            start_node=start_node,
            allow_return_on_completed=allow_return,
            distance_cache=None
        )
        
        return {
            'success': True,
            'cluster_idx': cluster_idx,
            'cid': cid,
            'path': cluster_path,
            'distance': cluster_m,
            'num_segments': len(seg_idxs)
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


def parallel_cluster_routing(graph, required_edges, clusters, cluster_order,
                            allow_return=True, num_workers=None,
                            progress_callback=None):
    """
    Route all clusters in parallel using multiprocessing.
    
    FIXED: Removed segment_callback parameter (not supported in parallel mode)
    
    Args:
        graph: DirectedGraph instance
        required_edges: List of required edges
        clusters: Dict of cluster_id -> segment_indices
        cluster_order: Ordered list of cluster IDs
        allow_return: Allow return on completed segments
        num_workers: Number of parallel workers (default: CPU count - 1)
        progress_callback: Callback(current, total) for progress updates
    
    Returns:
        List of cluster results in order: [(path, distance, cid), ...]
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    # Don't use more workers than clusters
    num_workers = min(num_workers, len(cluster_order))
    
    print(f"\n[PARALLEL] Routing {len(cluster_order)} clusters using {num_workers} workers...")
    
    # Determine starting nodes for each cluster
    first_cid = cluster_order[0]
    first_seg_idx = clusters[first_cid][0]
    current_loc = required_edges[first_seg_idx][0]
    
    # Prepare work items
    work_items = []
    for idx, cid in enumerate(cluster_order):
        seg_idxs = clusters[cid]
        work_items.append((
            idx,                # cluster_idx
            cid,               # cluster_id
            seg_idxs,          # segment indices
            graph,             # graph (will be pickled)
            required_edges,    # required edges
            allow_return,      # allow_return flag
            current_loc        # start node (approximate for now)
        ))
    
    # Process clusters in parallel
    results = []
    completed = 0
    failed = 0
    
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_route_cluster_worker, work_items):
            completed += 1
            
            if progress_callback:
                progress_callback(completed, len(cluster_order))
            
            if result['success']:
                results.append(result)
                if completed % 10 == 0 or completed == len(cluster_order):
                    print(f"  ✓ Progress: {completed}/{len(cluster_order)} "
                          f"({100*completed/len(cluster_order):.1f}%)")
            else:
                failed += 1
                print(f"  ✗ Cluster {result['cluster_idx']+1} FAILED: {result['error']}")
                # Create empty result for failed cluster
                results.append({
                    'success': True,  # Mark as "success" but with empty path
                    'cluster_idx': result['cluster_idx'],
                    'cid': result['cid'],
                    'path': [],
                    'distance': 0.0,
                    'num_segments': 0
                })
    
    # Sort results back into original order
    results.sort(key=lambda x: x['cluster_idx'])
    
    # Return in format expected by main pipeline
    ordered_results = [
        (r['path'], r['distance'], r['cid']) 
        for r in results
    ]
    
    total_distance = sum(r['distance'] for r in results)
    print(f"[PARALLEL] ✓ Completed all clusters: {total_distance/1000:.1f} km total")
    if failed > 0:
        print(f"[PARALLEL] ⚠️ {failed} clusters failed and were skipped")
    
    return ordered_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_optimal_workers(num_clusters, num_segments):
    """
    Estimate optimal number of workers based on workload.
    
    More clusters = more parallelism benefit
    Very large clusters = diminishing returns
    """
    available_cpus = cpu_count()
    
    if num_clusters >= available_cpus:
        # Plenty of clusters to distribute
        return max(1, available_cpus - 1)
    elif num_clusters > 1:
        # Limited clusters, use fewer workers
        return min(num_clusters, max(1, available_cpus // 2))
    else:
        # Single cluster, no parallelism benefit
        return 1


def get_cpu_info():
    """Get CPU information for optimization recommendations"""
    cpu_count_logical = cpu_count()
    
    try:
        # Try to get physical core count
        import psutil
        cpu_count_physical = psutil.cpu_count(logical=False)
    except ImportError:
        cpu_count_physical = cpu_count_logical // 2  # Rough estimate
    
    return {
        'logical_cores': cpu_count_logical,
        'physical_cores': cpu_count_physical,
        'recommended_workers': max(1, cpu_count_logical - 1)
    }


class ParallelTimer:
    """Context manager for timing parallel operations"""
    
    def __init__(self, description):
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n[TIMER] {self.description}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[TIMER] {self.description} completed in {elapsed:.2f}s")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Show system info
    info = get_cpu_info()
    print("="*60)
    print("PARALLEL PROCESSING SYSTEM INFO")
    print("="*60)
    print(f"Logical CPU Cores: {info['logical_cores']}")
    print(f"Physical CPU Cores: {info['physical_cores']}")
    print(f"Recommended Workers: {info['recommended_workers']}")
    print("="*60)