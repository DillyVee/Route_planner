"""
OSM Speed Integration Module for Route Planner
Add this to your route planner to enable time-based routing with OSM speed limits.

Usage:
    from osm_speed_integration import OverpassSpeedFetcher, enrich_segments_with_osm_speeds
    from osm_speed_integration import snap_coord, build_graph_with_time_weights
    
    # In your parse_kml():
    lat, lon = snap_coord(lat, lon, precision=6)
    
    # After parsing:
    overpass = OverpassSpeedFetcher('cache.json')
    segments = enrich_segments_with_osm_speeds(segments, overpass)
    
    # Build graph:
    graph, required = build_graph_with_time_weights(segments)
"""

import os
import json
import time as time_module
import requests
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt


# ============================================================================
# COORDINATE SNAPPING
# ============================================================================

def snap_coord(lat: float, lon: float, precision: int = 6) -> Tuple[float, float]:
    """
    Snap coordinates to fixed precision to eliminate near-duplicates.
    
    precision=6: ~0.11m accuracy (recommended)
    precision=5: ~1.1m accuracy
    precision=7: ~0.01m accuracy (overkill)
    """
    return (round(lat, precision), round(lon, precision))


def snap_coords_list(coords: List[Tuple[float, float]], 
                    precision: int = 6) -> List[Tuple[float, float]]:
    """
    Snap all coordinates and remove consecutive duplicates.
    """
    if not coords:
        return []
    
    snapped = [snap_coord(lat, lon, precision) for lat, lon in coords]
    
    # Remove consecutive duplicates
    deduped = [snapped[0]]
    for coord in snapped[1:]:
        if coord != deduped[-1]:
            deduped.append(coord)
    
    return deduped


# ============================================================================
# OVERPASS API INTEGRATION
# ============================================================================

class OverpassSpeedFetcher:
    """
    Fetch speed limits from OpenStreetMap using Overpass API.
    
    Features:
    - Local caching (no repeated API calls)
    - Rate limiting (respectful of Overpass servers)
    - Bounding box queries (efficient)
    - Highway type fallbacks (when maxspeed missing)
    """
    
    def __init__(self, cache_file: str = 'overpass_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.api_url = 'https://overpass-api.de/api/interpreter'
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    def _load_cache(self) -> dict:
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  ⚠ Cache load error: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Persist cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"  ⚠ Cache save error: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting to be respectful of Overpass API"""
        elapsed = time_module.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time_module.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time_module.time()
    
    def fetch_speeds_for_bbox(self, min_lat: float, min_lon: float, 
                              max_lat: float, max_lon: float) -> Dict[str, dict]:
        """
        Fetch all ways with highway tags in bounding box.
        
        Returns:
            {way_id: {
                'maxspeed': speed_kmh or None,
                'highway': road_type,
                'geometry': [(lat,lon), ...],
                'name': road_name
            }}
        """
        cache_key = f"{min_lat:.5f},{min_lon:.5f},{max_lat:.5f},{max_lon:.5f}"
        
        if cache_key in self.cache:
            print(f"  ✓ Using cached Overpass data ({len(self.cache[cache_key])} ways)")
            return self.cache[cache_key]
        
        self._rate_limit()
        
        # Query for all highways (not just those with maxspeed)
        # We'll use highway type for fallback speeds
        query = f"""
        [out:json][timeout:25];
        (
          way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        
        try:
            print(f"  → Querying Overpass API (bbox: {max_lat-min_lat:.3f}° x {max_lon-min_lon:.3f}°)...")
            response = requests.post(self.api_url, data={'data': query}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            ways = {}
            for element in data.get('elements', []):
                if element['type'] == 'way':
                    way_id = str(element['id'])
                    tags = element.get('tags', {})
                    
                    # Parse maxspeed
                    maxspeed_raw = tags.get('maxspeed', '')
                    speed_kmh = self._parse_maxspeed(maxspeed_raw)
                    
                    # Extract geometry
                    geometry = []
                    if 'geometry' in element:
                        geometry = [(node['lat'], node['lon']) 
                                   for node in element['geometry']]
                    
                    ways[way_id] = {
                        'maxspeed': speed_kmh,
                        'highway': tags.get('highway', 'unknown'),
                        'geometry': geometry,
                        'name': tags.get('name', ''),
                        'oneway': tags.get('oneway', 'no')
                    }
            
            self.cache[cache_key] = ways
            self._save_cache()
            
            # Stats
            with_speed = sum(1 for w in ways.values() if w['maxspeed'] is not None)
            print(f"  ✓ Found {len(ways)} OSM ways ({with_speed} with maxspeed)")
            return ways
            
        except requests.exceptions.Timeout:
            print(f"  ⚠ Overpass API timeout (bbox too large?)")
            return {}
        except Exception as e:
            print(f"  ⚠ Overpass API error: {e}")
            return {}
    
    def _parse_maxspeed(self, maxspeed_str: str) -> Optional[float]:
        """
        Parse OSM maxspeed tag to km/h.
        
        Handles:
        - "50" -> 50 km/h
        - "30 mph" -> 48.3 km/h
        - "50 km/h" -> 50 km/h
        - "walk" -> None
        - "none" -> None
        """
        if not maxspeed_str:
            return None
        
        maxspeed_str = maxspeed_str.lower().strip()
        
        # Handle special values
        if maxspeed_str in ('walk', 'none', 'signals', 'variable'):
            return None
        
        # Handle mph
        if 'mph' in maxspeed_str:
            try:
                mph = float(maxspeed_str.replace('mph', '').strip())
                return mph * 1.60934
            except ValueError:
                return None
        
        # Handle km/h variants
        maxspeed_str = (maxspeed_str.replace('km/h', '')
                                    .replace('kmh', '')
                                    .replace('kph', '')
                                    .strip())
        
        try:
            return float(maxspeed_str)
        except ValueError:
            return None
    
    def get_fallback_speed(self, highway_type: str) -> float:
        """
        Default speed limits by highway classification (km/h).
        Based on typical speed limits in most countries.
        """
        defaults = {
            'motorway': 110,
            'motorway_link': 80,
            'trunk': 90,
            'trunk_link': 70,
            'primary': 70,
            'primary_link': 50,
            'secondary': 60,
            'secondary_link': 50,
            'tertiary': 50,
            'tertiary_link': 40,
            'unclassified': 40,
            'residential': 30,
            'living_street': 20,
            'service': 20,
            'track': 15,
            'path': 10,
            'footway': 5,
            'cycleway': 20,
            'unknown': 30
        }
        return defaults.get(highway_type, 30)


# ============================================================================
# SPATIAL INDEXING & MATCHING
# ============================================================================

def haversine(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate distance between two (lat, lon) points in meters"""
    lat1, lon1 = a
    lat2, lon2 = b
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(aa))
    return 6371000 * c


class SimpleGridIndex:
    """
    Lightweight spatial index for nearest-way matching.
    Uses grid-based bucketing for O(1) lookups in most cases.
    """
    
    def __init__(self, cell_size_deg: float = 0.01):
        """
        cell_size_deg: Grid cell size in degrees (~1.1km at equator for 0.01)
        """
        self.cell_size = cell_size_deg
        self.grid = defaultdict(list)  # {(grid_x, grid_y): [way_data, ...]}
    
    def _get_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert coordinate to grid cell"""
        return (int(lat / self.cell_size), int(lon / self.cell_size))
    
    def add_way(self, way_data: dict):
        """
        Add OSM way to spatial index.
        way_data must have 'geometry' key with list of (lat, lon) tuples.
        """
        geometry = way_data.get('geometry', [])
        if not geometry:
            return
        
        # Add to all cells the way passes through
        cells_covered = set()
        for lat, lon in geometry:
            cell = self._get_cell(lat, lon)
            cells_covered.add(cell)
        
        for cell in cells_covered:
            self.grid[cell].append(way_data)
    
    def find_nearest_way(self, point: Tuple[float, float], 
                         search_radius_cells: int = 2,
                         max_distance_m: float = 100.0) -> Optional[dict]:
        """
        Find nearest OSM way to a point.
        
        Args:
            point: (lat, lon)
            search_radius_cells: How many cells to search in each direction
            max_distance_m: Max matching distance (meters)
        
        Returns:
            way_data dict or None
        """
        lat, lon = point
        center_cell = self._get_cell(lat, lon)
        
        # Collect candidates from surrounding cells
        candidates = []
        for dx in range(-search_radius_cells, search_radius_cells + 1):
            for dy in range(-search_radius_cells, search_radius_cells + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    candidates.extend(self.grid[cell])
        
        if not candidates:
            return None
        
        # Find closest way
        best_way = None
        best_dist = float('inf')
        
        # Remove duplicates (same way might be in multiple cells)
        seen = set()
        unique_candidates = []
        for way in candidates:
            way_id = id(way)  # Use object identity
            if way_id not in seen:
                seen.add(way_id)
                unique_candidates.append(way)
        
        for way in unique_candidates:
            geometry = way.get('geometry', [])
            if not geometry:
                continue
            
            # Distance to closest point on way
            min_dist = min(haversine(point, way_pt) for way_pt in geometry)
            
            if min_dist < best_dist:
                best_dist = min_dist
                best_way = way
        
        # Only return if within max distance
        return best_way if best_dist <= max_distance_m else None


def enrich_segments_with_osm_speeds(segments: List[dict], 
                                    overpass_fetcher: OverpassSpeedFetcher,
                                    max_match_distance_m: float = 100.0) -> List[dict]:
    """
    Match KML segments to OSM ways and inherit speed limits.
    
    Process:
    1. Calculate bounding box of all segments
    2. Fetch OSM data for that bbox (cached)
    3. Build spatial index of OSM ways
    4. Match each segment to nearest OSM way
    5. Inherit speed limit (or fallback to highway type default)
    
    Args:
        segments: List of segment dicts from parse_kml()
        overpass_fetcher: OverpassSpeedFetcher instance
        max_match_distance_m: Max distance for segment-to-way matching
    
    Returns:
        segments with added fields:
        - speed_limit: Speed in km/h
        - speed_source: 'kml', 'osm_matched', 'osm_fallback', or 'default'
        - highway_type: OSM highway classification (if matched)
    """
    if not segments:
        return segments
    
    # Calculate bounding box
    all_lats = []
    all_lons = []
    for seg in segments:
        for lat, lon in seg['coords']:
            all_lats.append(lat)
            all_lons.append(lon)
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    
    # Add 1km padding to ensure we don't miss nearby ways
    pad = 0.01  # ~1.1km at equator
    min_lat -= pad
    max_lat += pad
    min_lon -= pad
    max_lon += pad
    
    print(f"\n[OSM] Fetching speed data for area...")
    print(f"  Bbox: ({min_lat:.5f}, {min_lon:.5f}) to ({max_lat:.5f}, {max_lon:.5f})")
    
    osm_ways = overpass_fetcher.fetch_speeds_for_bbox(min_lat, min_lon, max_lat, max_lon)
    
    if not osm_ways:
        print("  ⚠ No OSM data retrieved, segments will use KML speeds or defaults")
        # Mark all segments without speeds as needing defaults
        for seg in segments:
            if not seg.get('speed_limit'):
                seg['speed_source'] = 'default'
        return segments
    
    # Build spatial index
    print(f"[OSM] Building spatial index...")
    index = SimpleGridIndex(cell_size_deg=0.01)  # ~1km cells
    for way_data in osm_ways.values():
        index.add_way(way_data)
    print(f"  ✓ Indexed {len(osm_ways)} ways")
    
    # Match segments to OSM ways
    print(f"[OSM] Matching {len(segments)} segments to OSM ways...")
    
    stats = {
        'kml': 0,
        'osm_matched': 0,
        'osm_fallback': 0,
        'default': 0
    }
    
    # Count and mark KML segments
    kml_count = sum(1 for seg in segments 
                    if seg.get('speed_limit') and seg['speed_limit'] > 0)

    for seg in segments:
        if seg.get('speed_limit') and seg['speed_limit'] > 0:
            seg['speed_source'] = 'kml'

    # Parallel matching
    from parallel_processing_addon import parallel_osm_matching

    segments = parallel_osm_matching(
        segments=segments,
        index=index,
        overpass_fetcher=overpass_fetcher,
        max_distance=max_match_distance_m,
        num_workers=4
    )

    # Recalculate stats
    stats = {
        'kml': kml_count,
        'osm_matched': sum(1 for s in segments if s.get('speed_source') == 'osm_matched'),
        'osm_fallback': sum(1 for s in segments if s.get('speed_source') == 'osm_fallback'),
        'default': sum(1 for s in segments if s.get('speed_source') == 'default')
    }
    
    print(f"  ✓ Speed sources: KML={stats['kml']}, OSM={stats['osm_matched']}, "
          f"Fallback={stats['osm_fallback']}, Default={stats['default']}")
    
    return segments


# ============================================================================
# TIME-BASED GRAPH CONSTRUCTION
# ============================================================================

class DirectedGraph:
    """
    Directed graph with Dijkstra shortest path.
    Edge weights represent TRAVEL TIME (seconds).
    """
    
    def __init__(self):
        self.node_to_id = {}
        self.id_to_node = []
        self.adj = []
    
    def _ensure(self, node):
        if node in self.node_to_id:
            return self.node_to_id[node]
        idx = len(self.id_to_node)
        self.node_to_id[node] = idx
        self.id_to_node.append(node)
        self.adj.append([])
        return idx
    
    def add_edge(self, a, b, w):
        """Add edge from a to b with weight w (time in seconds)"""
        ia = self._ensure(a)
        ib = self._ensure(b)
        # Don't add duplicate edges
        if not any(v == ib for v, _ in self.adj[ia]):
            self.adj[ia].append((ib, w))
    
    def dijkstra(self, source_id):
        """Dijkstra's algorithm - finds shortest TIME paths"""
        n = len(self.id_to_node)
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[source_id] = 0.0
        
        import heapq
        h = [(0.0, source_id)]
        
        while h:
            d, u = heapq.heappop(h)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(h, (nd, v))
        
        return dist, prev
    
    def shortest_path(self, start_node, end_node):
        """
        Find shortest path from start to end.
        Returns: (path_coords, total_time_seconds)
        """
        if start_node not in self.node_to_id or end_node not in self.node_to_id:
            return None, float('inf')
        
        s = self.node_to_id[start_node]
        t = self.node_to_id[end_node]
        
        dist, prev = self.dijkstra(s)
        
        if dist[t] == float('inf'):
            return None, float('inf')
        
        # Reconstruct path
        cur = t
        rev = []
        while cur != -1:
            rev.append(cur)
            cur = prev[cur]
        rev.reverse()
        
        coords = [self.id_to_node[i] for i in rev]
        return coords, dist[t]


def build_graph_with_time_weights(segments: List[dict], 
                                  treat_unspecified_as_two_way: bool = True,
                                  default_speed_kmh: float = 30.0) -> Tuple[DirectedGraph, List]:
    """
    Build directed graph with TIME-based edge weights.
    
    Edge weight = travel_time_seconds = distance_meters / speed_mps
    
    Args:
        segments: List of segment dicts with 'coords', 'speed_limit', 'oneway'
        treat_unspecified_as_two_way: If True, segments with oneway=None are bidirectional
        default_speed_kmh: Fallback speed when segment has no speed limit
    
    Returns:
        (graph, required_edges)
        - graph: DirectedGraph with time-weighted edges
        - required_edges: [(start, end, coords, idx), ...] for each segment
    """
    g = DirectedGraph()
    required = []
    
    edges_without_speed = 0
    total_edges = 0
    
    for idx, s in enumerate(segments):
        coords = s['coords']
        oneway = s.get('oneway', None)
        
        # Get speed for this segment (km/h)
        speed_kmh = s.get('speed_limit', default_speed_kmh)
        if not speed_kmh or speed_kmh <= 0:
            speed_kmh = default_speed_kmh
            edges_without_speed += 1
        
        # Convert to m/s
        speed_mps = (speed_kmh * 1000.0) / 3600.0
        
        # Add edges between consecutive points
        for i in range(len(coords) - 1):
            a = coords[i]
            b = coords[i + 1]
            
            dist_m = haversine(a, b)
            time_s = dist_m / speed_mps  # ✅ TIME-BASED WEIGHT
            
            g.add_edge(a, b, time_s)
            total_edges += 1
            
            # Add reverse edge if two-way
            if oneway is False or (oneway is None and treat_unspecified_as_two_way):
                g.add_edge(b, a, time_s)
                total_edges += 1
        
        required.append((coords[0], coords[-1], coords, idx))
    
    if edges_without_speed > 0:
        pct = (edges_without_speed / len(segments)) * 100
        print(f"  ⚠ {edges_without_speed}/{len(segments)} segments ({pct:.1f}%) "
              f"using default speed {default_speed_kmh} km/h")
    
    print(f"  ✓ Created {total_edges:,} time-weighted edges")
    
    return g, required


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_average_speed(segments: List[dict]) -> float:
    """
    Calculate weighted average speed from segments.
    Weights by segment length for accuracy.
    """
    total_distance = 0.0
    weighted_speed = 0.0
    segments_with_speed = 0
    
    for s in segments:
        dist = s.get('length_m', 0)
        speed = s.get('speed_limit')
        
        if speed and speed > 0 and dist > 0:
            total_distance += dist
            weighted_speed += dist * speed
            segments_with_speed += 1
    
    if total_distance > 0:
        avg_speed = weighted_speed / total_distance
        print(f"  ✓ Calculated average speed: {avg_speed:.1f} km/h "
              f"from {segments_with_speed}/{len(segments)} segments")
        return avg_speed
    
    # Fallback
    print(f"  ⚠ No speed data found, using default 30 km/h")
    return 30.0


def path_length_meters(coords: List[Tuple[float, float]]) -> float:
    """Calculate total path length in meters"""
    if len(coords) < 2:
        return 0.0
    return sum(haversine(coords[i], coords[i+1]) for i in range(len(coords)-1))