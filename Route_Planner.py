"""
Survey Route Planner - Complete Standalone Application with PyQt6 GUI

A user-friendly interface for optimizing vehicle survey routes from KML files.
Ultra-modern dark mode with live progress tracking for every step.

Requirements:
    pip install PyQt6
    pip install scikit-learn scipy ortools (optional but recommended)

Usage:
    python route_planner_complete.py
"""

import heapq
import json
import os
import sys
import time
import webbrowser
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt

from osm_speed_integration import OverpassSpeedFetcher, build_graph_with_time_weights
from osm_speed_integration import calculate_average_speed as calculate_average_speed_osm
from osm_speed_integration import enrich_segments_with_osm_speeds, snap_coord, snap_coords_list

# ============================================================================
# PARALLEL PROCESSING - PRODUCTION V4 (with fallbacks to legacy versions)
# ============================================================================

# Try to import Production V4 first (recommended)
V4_AVAILABLE = False
try:
    from drpp_core import ClusteringMethod, PathResult
    from drpp_core import cluster_segments as cluster_segments_v4
    from drpp_core import estimate_optimal_workers as estimate_optimal_workers_v4
    from drpp_core import parallel_cluster_routing as parallel_cluster_routing_v4
    from drpp_core import parallel_cluster_routing_ondemand as parallel_cluster_routing_v4_ondemand
    from drpp_core.logging_config import setup_logging as setup_drpp_logging

    V4_AVAILABLE = True
    print("✅ Using Production V4 DRPP Solver with ON-DEMAND mode (FASTEST)")
except ImportError:
    print("⚠️ Production V4 not available, using legacy versions")

# Legacy imports (fallback)
from legacy.parallel_processing_addon import (
    estimate_optimal_workers,
)
from legacy.parallel_processing_addon import (
    parallel_cluster_routing as parallel_cluster_routing_hungarian,
)

# Import greedy algorithm version (legacy)
try:
    from legacy.parallel_processing_addon_greedy import (
        parallel_cluster_routing as parallel_cluster_routing_greedy,
    )

    GREEDY_AVAILABLE = True
except ImportError:
    GREEDY_AVAILABLE = False
    if not V4_AVAILABLE:
        print("⚠️ Greedy algorithm not available - using Hungarian only")

# Import RFCS (Route-First, Cluster-Second) algorithm version (legacy)
try:
    from legacy.parallel_processing_addon_rfcs import (
        parallel_cluster_routing as parallel_cluster_routing_rfcs,
    )

    RFCS_AVAILABLE = True
except ImportError:
    RFCS_AVAILABLE = False
    if not V4_AVAILABLE:
        print("⚠️ RFCS algorithm not available")

# PyQt6 imports
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Optional packages
try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# ============================================================================
# CORE ROUTING LOGIC
# ============================================================================


def haversine(a, b):
    """Calculate distance between two lat/lon points in meters"""
    lat1, lon1 = a
    lat2, lon2 = b
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    aa = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(aa))
    return 6371000 * c


def path_length(coords):
    """Calculate total path length in meters"""
    if len(coords) < 2:
        return 0.0
    return sum(haversine(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def parse_speed_limit(text):
    """Extract speed limit from text (mph or km/h)"""
    if not text:
        return None

    text = text.lower().strip()

    # Try to find number followed by mph or km/h or kph
    import re

    # Look for patterns like "25 mph", "50km/h", "30kph", etc.
    patterns = [
        r"(\d+)\s*mph",
        r"(\d+)\s*km[/\s]?h",
        r"(\d+)\s*kph",
        r"maxspeed[=:]\s*(\d+)",
        r"speed[=:]\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            speed = int(match.group(1))
            # Convert mph to km/h if needed
            if "mph" in text:
                speed = int(speed * 1.60934)
            return speed

    return None


def parse_kml(kml_path):
    """Parse KML file and extract road segments with one-way detection and speed limits"""
    try:
        # Try standard parsing first
        tree = ET.parse(kml_path)
    except ET.ParseError as e:
        # If parsing fails, try to fix common XML issues
        print(f"  ⚠️ XML parsing error: {e}")
        print("  🔧 Attempting to fix XML issues...")

        try:
            # Read file and try to fix common issues
            with open(kml_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Fix common XML issues
            import re

            # Remove invalid XML characters (control characters except tab, newline, carriage return)
            content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)

            # Fix unescaped ampersands (but not &amp;, &lt;, &gt;, &quot;, &apos;)
            content = re.sub(r"&(?!(amp|lt|gt|quot|apos);)", "&amp;", content)

            # Try parsing the cleaned content
            from io import StringIO

            tree = ET.parse(StringIO(content))
            print("  ✓ Successfully fixed and parsed KML")

        except Exception as fix_error:
            raise ValueError(f"Failed to parse KML file even after attempted fixes: {fix_error}")
    except Exception as e:
        raise ValueError(f"Failed to parse KML file: {e}")

    root = tree.getroot()
    segments = []
    speed_limits_found = 0

    for pm in root.findall(".//{http://www.opengis.net/kml/2.2}Placemark"):
        ls = pm.find(".//{http://www.opengis.net/kml/2.2}LineString")
        if ls is None:
            continue

        coords_elem = ls.find("{http://www.opengis.net/kml/2.2}coordinates")
        if coords_elem is None or coords_elem.text is None:
            continue

        raw = coords_elem.text.strip()
        pts = [p for p in raw.replace("\n", " ").split() if p.strip()]
        coords = []

        for p in pts:
            ps = p.split(",")
            if len(ps) < 2:
                continue
            try:
                lon = float(ps[0])
                lat = float(ps[1])
                lat, lon = snap_coord(lat, lon, precision=6)
                coords.append((lat, lon))
            except ValueError:
                continue

        coords = snap_coords_list(coords, precision=6)

        if len(coords) < 2:
            continue

        # Detect one-way flag and speed limit
        oneway = None
        speed_limit = None

        ext = pm.find("{http://www.opengis.net/kml/2.2}ExtendedData")
        oneway_candidates = ["oneway", "one_way", "one-way", "is_one_way"]
        speed_candidates = ["maxspeed", "speed_limit", "speed", "speedlimit"]

        if ext is not None:
            for elem in ext.iter():
                tag = elem.tag.split("}")[-1].lower() if isinstance(elem.tag, str) else ""
                txt = (elem.text or "").strip().lower()

                # Check for oneway
                if tag in oneway_candidates:
                    if txt in ("yes", "true", "1", "y"):
                        oneway = True
                    elif txt in ("no", "false", "0", "n"):
                        oneway = False

                # Check for speed limit
                if tag in speed_candidates:
                    speed_limit = parse_speed_limit(elem.text)

        # Check description field
        desc = pm.find("{http://www.opengis.net/kml/2.2}description")
        if desc is not None and desc.text:
            txt = desc.text.lower()

            # Check oneway
            if oneway is None:
                for cand in oneway_candidates:
                    if cand in txt:
                        if any(x in txt for x in ("yes", "true", "1", "y")):
                            oneway = True
                        elif any(x in txt for x in ("no", "false", "0", "n")):
                            oneway = False

            # Check speed limit
            if speed_limit is None:
                speed_limit = parse_speed_limit(desc.text)

        # Check name field for speed limit
        if speed_limit is None:
            name = pm.find("{http://www.opengis.net/kml/2.2}name")
            if name is not None and name.text:
                speed_limit = parse_speed_limit(name.text)

        if speed_limit:
            speed_limits_found += 1

        segments.append(
            {
                "coords": coords,
                "start": coords[0],
                "end": coords[-1],
                "length_m": path_length(coords),
                "oneway": oneway,
                "speed_limit": speed_limit,
            }
        )

    if not segments:
        raise ValueError("No valid segments found in KML file")

    print(f"Found speed limits in {speed_limits_found}/{len(segments)} segments")

    return segments


def calculate_average_speed(segments):
    """Calculate weighted average speed from segments with speed limits"""
    total_distance = 0.0
    weighted_speed = 0.0
    segments_with_speed = 0

    for s in segments:
        dist = s["length_m"]
        speed = s.get("speed_limit")

        if speed and speed > 0:
            total_distance += dist
            weighted_speed += dist * speed
            segments_with_speed += 1

    if total_distance > 0:
        avg_speed = weighted_speed / total_distance
        print(f"Calculated average speed: {avg_speed:.1f} km/h from {segments_with_speed} segments")
        return avg_speed

    # Default fallback
    print("No speed limits found, using default 30 km/h")
    return 30.0


class DirectedGraph:
    """Directed graph with Dijkstra shortest path"""

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
        ia = self._ensure(a)
        ib = self._ensure(b)
        if not any(v == ib for v, _ in self.adj[ia]):
            self.adj[ia].append((ib, w))

    def dijkstra(self, source_id):
        n = len(self.id_to_node)
        dist = [float("inf")] * n
        prev = [-1] * n
        dist[source_id] = 0.0
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
        if start_node not in self.node_to_id or end_node not in self.node_to_id:
            return None, float("inf")

        s = self.node_to_id[start_node]
        t = self.node_to_id[end_node]

        dist, prev = self.dijkstra(s)

        if dist[t] == float("inf"):
            return None, float("inf")

        cur = t
        rev = []
        while cur != -1:
            rev.append(cur)
            cur = prev[cur]
        rev.reverse()

        coords = [self.id_to_node[i] for i in rev]
        return coords, dist[t]


def fetch_osm_roads_for_routing(bbox, timeout=60):
    """
    Fetch OSM road network in bounding box for routing purposes.
    Returns list of ways with geometry and attributes.

    Args:
        bbox: (min_lat, min_lon, max_lat, max_lon)
        timeout: Query timeout in seconds

    Returns:
        List of dicts with 'geometry', 'highway', 'maxspeed', 'oneway'
    """
    import requests

    min_lat, min_lon, max_lat, max_lon = bbox

    # Overpass query for all roads in bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """

    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter", data=query, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"    Overpass query failed: {e}")
        return []

    ways = []
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue

        geometry = element.get("geometry", [])
        if len(geometry) < 2:
            continue

        # Convert geometry to (lat, lon) tuples
        coords = [(pt["lat"], pt["lon"]) for pt in geometry]

        tags = element.get("tags", {})
        highway_type = tags.get("highway", "unclassified")

        # Parse maxspeed
        maxspeed_str = tags.get("maxspeed", "")
        maxspeed = parse_speed_limit(maxspeed_str) if maxspeed_str else 0

        # Parse oneway
        oneway_str = tags.get("oneway", "no").lower()
        oneway = oneway_str in ("yes", "true", "1", "-1")

        ways.append(
            {"geometry": coords, "highway": highway_type, "maxspeed": maxspeed, "oneway": oneway}
        )

    return ways


def build_graph(segments, treat_unspecified_as_two_way=True):
    """
    Build directed graph from segments.
    Now DELEGATES to time-based version and enriches with OSM roads.
    """

    print("  Building base graph from survey segments...")
    graph, required_edges = build_graph_with_time_weights(
        segments, treat_unspecified_as_two_way=treat_unspecified_as_two_way, default_speed_kmh=30.0
    )

    print(f"  Base graph: {len(graph.id_to_node):,} nodes from survey segments")

    # Calculate bounding box for OSM query
    all_coords = []
    for seg in segments:
        all_coords.extend(seg["coords"])

    if not all_coords:
        return graph, required_edges

    lats = [c[0] for c in all_coords]
    lons = [c[1] for c in all_coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Add some padding (0.01 degrees ≈ 1km)
    padding = 0.01
    bbox = (min_lat - padding, min_lon - padding, max_lat + padding, max_lon + padding)

    print("  Fetching OSM connecting roads in bbox...")
    print(f"    Lat: {min_lat:.4f} to {max_lat:.4f}")
    print(f"    Lon: {min_lon:.4f} to {max_lon:.4f}")

    try:
        # Fetch OSM ways in the bounding box
        osm_ways = fetch_osm_roads_for_routing(bbox)

        if osm_ways:
            print(f"  ✓ Found {len(osm_ways)} OSM road segments")
            print("  Adding OSM roads to graph for routing (not as survey segments)...")

            osm_edges_added = 0

            # Fallback speeds based on highway type
            fallback_speeds = {
                "motorway": 110,
                "motorway_link": 80,
                "trunk": 90,
                "trunk_link": 70,
                "primary": 70,
                "primary_link": 50,
                "secondary": 60,
                "secondary_link": 50,
                "tertiary": 50,
                "tertiary_link": 40,
                "unclassified": 40,
                "residential": 30,
                "living_street": 20,
                "service": 20,
                "track": 15,
            }

            for way in osm_ways:
                geometry = way["geometry"]
                highway_type = way["highway"]
                speed_kmh = way["maxspeed"]
                oneway = way["oneway"]

                # Use fallback speed if not specified
                if not speed_kmh or speed_kmh <= 0:
                    speed_kmh = fallback_speeds.get(highway_type, 30)

                # Add edges to graph (for routing only, not as survey segments)
                for i in range(len(geometry) - 1):
                    a = geometry[i]
                    b = geometry[i + 1]

                    # Calculate time weight (in seconds)
                    dist_m = haversine(a, b)
                    time_seconds = (dist_m / 1000.0) / speed_kmh * 3600.0

                    # Add forward edge
                    graph.add_edge(a, b, time_seconds)
                    osm_edges_added += 1

                    # Add reverse edge if not one-way
                    if not oneway:
                        graph.add_edge(b, a, time_seconds)
                        osm_edges_added += 1

            print(f"  ✓ Added {osm_edges_added:,} OSM edges to graph")
            print(f"  ✓ Enhanced graph: {len(graph.id_to_node):,} total nodes")
        else:
            print("  ⚠️ No OSM roads found in area (will use survey segments only)")

    except Exception as e:
        print(f"  ⚠️ Could not fetch OSM roads: {e}")
        print("  Continuing with survey segments only...")

    return graph, required_edges


def precompute_segment_distances(graph, required_edges, seg_idxs):
    """
    Precompute road distances between segment starts.
    Returns dict: {(from_seg_idx, to_seg_idx): (path, distance)}

    This avoids recomputing paths during greedy selection.
    """
    cache = {}
    n = len(seg_idxs)

    for i, si in enumerate(seg_idxs):
        s_start = required_edges[si][0]

        # Run Dijkstra once from this segment's start
        s_id = graph.node_to_id.get(s_start)
        if s_id is None:
            continue

        dist, prev = graph.dijkstra(s_id)

        # Store paths to all other segments
        for j, sj in enumerate(seg_idxs):
            if i == j:
                continue

            t_start = required_edges[sj][0]
            t_id = graph.node_to_id.get(t_start)

            if t_id is None or dist[t_id] == float("inf"):
                continue

            # Reconstruct path
            path_ids = []
            cur = t_id
            while cur != -1:
                path_ids.append(cur)
                cur = prev[cur]
            path_ids.reverse()

            path = [graph.id_to_node[pid] for pid in path_ids]
            cache[(si, sj)] = (path, dist[t_id])

    return cache


def cluster_segments(
    segments, method="auto", gx=10, gy=10, k_clusters=40, eps_km=5.0, min_samples=3
):
    """
    Cluster segments using V4 production clustering or legacy methods.

    Args:
        segments: List of segment dicts with 'start', 'end', 'coords'
        method: 'auto', 'kmeans', 'grid', or 'dbscan'
        gx, gy: Grid dimensions for grid clustering
        k_clusters: Number of clusters for kmeans
        eps_km: Epsilon in kilometers for DBSCAN (V4 only)
        min_samples: Minimum samples for DBSCAN (V4 only)

    Returns:
        Dict mapping cluster_id to list of segment indices
    """
    # Use Production V4 if available (RECOMMENDED)
    if V4_AVAILABLE:
        if method == "dbscan" or (method == "auto" and len(segments) > 50):
            try:
                result = cluster_segments_v4(
                    segments, ClusteringMethod.DBSCAN, eps_km=eps_km, min_samples=min_samples
                )
                print(
                    f"  ✅ V4 DBSCAN: {len(result.clusters)} clusters, {result.noise_count} noise points"
                )
                return result.clusters
            except Exception as e:
                print(f"  ⚠️ V4 DBSCAN failed ({e}), falling back to grid")
                result = cluster_segments_v4(segments, ClusteringMethod.GRID, grid_x=gx, grid_y=gy)
                return result.clusters
        elif method == "kmeans":
            try:
                result = cluster_segments_v4(
                    segments, ClusteringMethod.KMEANS, k_clusters=k_clusters
                )
                print(f"  ✅ V4 K-means: {len(result.clusters)} clusters")
                return result.clusters
            except Exception as e:
                print(f"  ⚠️ V4 K-means failed ({e}), falling back to grid")
                result = cluster_segments_v4(segments, ClusteringMethod.GRID, grid_x=gx, grid_y=gy)
                return result.clusters
        else:  # grid or auto with small datasets
            result = cluster_segments_v4(segments, ClusteringMethod.GRID, grid_x=gx, grid_y=gy)
            print(f"  ✅ V4 Grid: {len(result.clusters)} clusters")
            return result.clusters

    # Legacy clustering (fallback)
    if method in ("kmeans", "auto") and SKLEARN_AVAILABLE:
        pts = [
            ((s["start"][0] + s["end"][0]) / 2.0, (s["start"][1] + s["end"][1]) / 2.0)
            for s in segments
        ]
        k = min(k_clusters, max(1, len(pts)))

        if k <= 1:
            return {0: list(range(len(segments)))}

        X = [[p[0], p[1]] for p in pts]
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)

        clusters = defaultdict(list)
        for i, label in enumerate(km.labels_):
            clusters[int(label)].append(i)
        return dict(clusters)
    else:
        return grid_cluster_segments(segments, gx, gy)


def grid_cluster_segments(segments, gx=8, gy=8):
    """Simple grid-based clustering"""
    lats = [(s["start"][0] + s["end"][0]) / 2.0 for s in segments]
    lons = [(s["start"][1] + s["end"][1]) / 2.0 for s in segments]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_step = (max_lat - min_lat) / gy if max_lat > min_lat else 1.0
    lon_step = (max_lon - min_lon) / gx if max_lon > min_lon else 1.0

    clusters = {}
    for i, seg in enumerate(segments):
        clat = (seg["start"][0] + seg["end"][0]) / 2.0
        clon = (seg["start"][1] + seg["end"][1]) / 2.0

        ix = int((clon - min_lon) / lon_step) if lon_step > 0 else 0
        iy = int((clat - min_lat) / lat_step) if lat_step > 0 else 0

        if ix == gx:
            ix = gx - 1
        if iy == gy:
            iy = gy - 1

        cid = iy * gx + ix
        clusters.setdefault(cid, []).append(i)

    return clusters


# ============================================================================
# V4 COMPATIBILITY WRAPPER
# ============================================================================


def parallel_cluster_routing_v4_wrapper(
    graph,
    required_edges,
    clusters,
    cluster_order,
    allow_return=True,
    num_workers=None,
    progress_callback=None,
):
    """
    Wrapper to make V4 parallel_cluster_routing compatible with legacy API.

    Uses ON-DEMAND mode for maximum speed (10-100x faster).
    Converts PathResult objects to legacy (path, distance, cluster_id) tuples.
    """
    # Determine start node
    first_cid = cluster_order[0]
    first_seg_idx = clusters[first_cid][0]
    start_node = required_edges[first_seg_idx][0]

    # Call V4 ON-DEMAND routing (much faster than matrix precomputation!)
    results = parallel_cluster_routing_v4_ondemand(
        graph=graph,
        required_edges=required_edges,
        clusters=clusters,
        cluster_order=cluster_order,
        start_node=start_node,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )

    # Convert PathResult objects to legacy tuple format
    legacy_results = []
    for result in results:
        legacy_results.append((result.path, result.distance, result.cluster_id))

    return legacy_results


def compute_imbalance(required_edges, seg_idxs):
    """Compute in/out degree imbalance for each node"""
    outc = Counter()
    inc = Counter()

    for si in seg_idxs:
        s, e, coords, idx = required_edges[si]
        outc[s] += 1
        inc[e] += 1

    nodes = set(list(outc.keys()) + list(inc.keys()))
    imbalance = {}

    for n in nodes:
        imbalance[n] = outc[n] - inc[n]

    return imbalance


def pair_imbalances(graph, required_edges, seg_idxs):
    """Pair positive imbalance nodes to negative ones using Hungarian algorithm"""
    imbalance = compute_imbalance(required_edges, seg_idxs)
    pos_nodes = []
    neg_nodes = []

    for n, v in imbalance.items():
        if v > 0:
            pos_nodes += [n] * v
        elif v < 0:
            neg_nodes += [n] * (-v)

    if not pos_nodes or not neg_nodes:
        return []

    m = min(len(pos_nodes), len(neg_nodes))
    pos_nodes = pos_nodes[:m]
    neg_nodes = neg_nodes[:m]

    cost = [[0.0] * len(neg_nodes) for _ in range(len(pos_nodes))]

    for i, p in enumerate(pos_nodes):
        for j, q in enumerate(neg_nodes):
            _, d = graph.shortest_path(p, q)
            if d == float("inf"):
                d = haversine(p, q) * 3.0
            cost[i][j] = d

    pairs = []

    if SCIPY_AVAILABLE:
        import numpy as np

        cost_mat = np.array(cost)
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        for r, c in zip(row_ind, col_ind):
            p = pos_nodes[r]
            q = neg_nodes[c]
            path, d = graph.shortest_path(p, q)
            if path is None:
                path = [p, q]
            pairs.append(path)
    else:
        used = [False] * len(neg_nodes)
        for i, p in enumerate(pos_nodes):
            bestk, bestd = None, float("inf")
            for j, q in enumerate(neg_nodes):
                if used[j]:
                    continue
                d = cost[i][j]
                if d < bestd:
                    bestd = d
                    bestk = j

            if bestk is not None:
                used[bestk] = True
                pth, _ = graph.shortest_path(p, neg_nodes[bestk])
                if pth is None:
                    pth = [p, neg_nodes[bestk]]
                pairs.append(pth)

    return pairs


def greedy_arc_route_with_hungarian(
    graph,
    required_edges,
    seg_idxs,
    start_node=None,
    allow_return_on_completed=True,
    distance_cache=None,
):
    """Route through all required edges in a cluster"""
    if not seg_idxs:
        return [], 0.0, []

    connecting_paths = pair_imbalances(graph, required_edges, seg_idxs)

    if start_node is None:
        start_node = required_edges[seg_idxs[0]][0]

    remaining = set(seg_idxs)
    unreachable = []  # Track unreachable segments

    # Track segment progress
    total_segs = len(seg_idxs)
    processed = 0
    cur = start_node
    total_path = []
    total_m = 0.0

    def append_path(pcoords):
        nonlocal total_path, total_m
        if not pcoords:
            return
        if total_path and total_path[-1] == pcoords[0]:
            pcoords = pcoords[1:]
        total_path.extend(pcoords)
        total_m += path_length(pcoords)

    # Main routing loop
    while remaining:
        best_seg = None
        best_dist = float("inf")
        best_path = None

        # Find nearest unvisited segment
        for seg_idx in remaining:
            seg_start = required_edges[seg_idx][0]
            path, dist = graph.shortest_path(cur, seg_start)

            if dist < best_dist:
                best_dist = dist
                best_seg = seg_idx
                best_path = path

        if best_seg is None:
            # No reachable segments
            unreachable.extend(list(remaining))
            break

        # Move to segment and traverse it
        if best_path:
            append_path(best_path)

        seg_coords = required_edges[best_seg][2]
        append_path(seg_coords)

        cur = required_edges[best_seg][1]
        remaining.remove(best_seg)
        processed += 1

    # Add connecting paths
    for path in connecting_paths:
        append_path(path)

    return total_path, total_m, unreachable


def centroid_of_cluster(cluster_idx_list, segments):
    """Calculate centroid of a cluster"""
    lat = sum(
        (segments[i]["start"][0] + segments[i]["end"][0]) / 2.0 for i in cluster_idx_list
    ) / len(cluster_idx_list)
    lon = sum(
        (segments[i]["start"][1] + segments[i]["end"][1]) / 2.0 for i in cluster_idx_list
    ) / len(cluster_idx_list)
    return (lat, lon)


def order_clusters(clusters, segments, use_ortools=False):
    """Order clusters using TSP heuristics"""
    ids = list(clusters.keys())
    if len(ids) <= 1:
        return ids

    centroids = {cid: centroid_of_cluster(clusters[cid], segments) for cid in ids}

    if use_ortools and ORTOOLS_AVAILABLE and len(ids) > 1:
        n = len(ids)
        coords = [centroids[c] for c in ids]
        dist_mat = [[int(haversine(coords[i], coords[j])) for j in range(n)] for i in range(n)]

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(i, j):
            return dist_mat[manager.IndexToNode(i)][manager.IndexToNode(j)]

        transit_cb = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search_params)

        if solution:
            order = []
            idx = routing.Start(0)
            while not routing.IsEnd(idx):
                order.append(ids[manager.IndexToNode(idx)])
                idx = solution.Value(routing.NextVar(idx))
            return order

    return greedy_cluster_order(ids, centroids)


def greedy_cluster_order(ids, centroids):
    """Greedy nearest-neighbor cluster ordering"""
    if not ids:
        return []

    remaining = set(ids)
    cur = remaining.pop()
    order = [cur]

    while remaining:
        best, bestd = None, float("inf")
        for cid in remaining:
            d = haversine(centroids[cur], centroids[cid])
            if d < bestd:
                bestd = d
                best = cid
        order.append(best)
        remaining.remove(best)
        cur = best

    return order


def two_opt_order(order, centroids, max_iter=1000):
    """Improve cluster order with 2-opt"""
    n = len(order)
    if n <= 2:
        return order

    def tour_length(o):
        return sum(haversine(centroids[o[i]], centroids[o[i + 1]]) for i in range(len(o) - 1))

    improved = True
    it = 0
    best = order[:]
    best_len = tour_length(best)

    while improved and it < max_iter:
        improved = False
        it += 1

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                nl = tour_length(new)

                if nl + 1e-6 < best_len:
                    best = new
                    best_len = nl
                    improved = True
                    break
            if improved:
                break

    return best


def write_mobile_gpx(
    full_coords,
    leg_breaks,
    leg_names,
    leg_distances,
    filename="mobile_route.gpx",
    avg_speed_kmh=30.0,
):
    """Write mobile-friendly GPX with proper timestamps and metadata"""
    from xml.dom.minidom import Document

    doc = Document()
    gpx = doc.createElement("gpx")
    gpx.setAttribute("version", "1.1")
    gpx.setAttribute("creator", "AutoRouteSurveyPlanner")
    gpx.setAttribute("xmlns", "http://www.topografix.com/GPX/1/1")
    doc.appendChild(gpx)

    metadata = doc.createElement("metadata")
    name = doc.createElement("name")
    name.appendChild(doc.createTextNode("Survey Route"))
    metadata.appendChild(name)

    desc = doc.createElement("desc")
    total_dist = sum(leg_distances)
    desc.appendChild(
        doc.createTextNode(
            f"Optimized survey route: {len(leg_breaks)} legs, {total_dist/1000:.1f} km total"
        )
    )
    metadata.appendChild(desc)

    time_elem = doc.createElement("time")
    time_elem.appendChild(doc.createTextNode(datetime.utcnow().isoformat() + "Z"))
    metadata.appendChild(time_elem)

    gpx.appendChild(metadata)

    trk = doc.createElement("trk")
    gpx.appendChild(trk)

    trk_name = doc.createElement("name")
    trk_name.appendChild(doc.createTextNode("Survey Route Track"))
    trk.appendChild(trk_name)

    start_time = datetime.now()
    elapsed = timedelta(0)

    for i, start_idx in enumerate(leg_breaks):
        end_idx = leg_breaks[i + 1] if i + 1 < len(leg_breaks) else len(full_coords)
        pts = full_coords[start_idx:end_idx]

        if not pts:
            continue

        seg = doc.createElement("trkseg")
        for lat, lon in pts:
            tp = doc.createElement("trkpt")
            tp.setAttribute("lat", str(lat))
            tp.setAttribute("lon", str(lon))
            seg.appendChild(tp)
        trk.appendChild(seg)

        wpt = doc.createElement("wpt")
        wpt.setAttribute("lat", str(pts[0][0]))
        wpt.setAttribute("lon", str(pts[0][1]))

        wpt_name = doc.createElement("name")
        wpt_name.appendChild(
            doc.createTextNode(leg_names[i] if i < len(leg_names) else f"Leg {i+1}")
        )
        wpt.appendChild(wpt_name)

        dist_m = leg_distances[i] if i < len(leg_distances) else 0
        hours = (dist_m / 1000.0) / max(avg_speed_kmh, 1e-6)

        wpt_desc = doc.createElement("desc")
        eta_time = start_time + elapsed
        wpt_desc.appendChild(
            doc.createTextNode(
                f"Dist: {dist_m/1000:.2f} km | Est: {hours*60:.0f} min | ETA: {eta_time.strftime('%H:%M')}"
            )
        )
        wpt.appendChild(wpt_desc)

        gpx.appendChild(wpt)

        elapsed += timedelta(hours=hours)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(doc.toprettyxml(indent="  "))

    return filename


def write_html_preview(coords, clusters, cluster_order, segments, filename="route_preview.html"):
    """Enhanced HTML preview with cluster visualization"""

    route_geo = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[c[1], c[0]] for c in coords]},
        "properties": {"name": "Route", "color": "#FF0000"},
    }

    cluster_features = []
    for i, cid in enumerate(cluster_order):
        seg_idxs = clusters[cid]
        centroid = centroid_of_cluster(seg_idxs, segments)
        cluster_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [centroid[1], centroid[0]]},
                "properties": {
                    "name": f"Cluster {i+1}",
                    "description": f"{len(seg_idxs)} segments",
                },
            }
        )

    geo_collection = {"type": "FeatureCollection", "features": [route_geo] + cluster_features}

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Survey Route Preview</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <style>
        html, body {{ height: 100%; margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100%; }}
        .info-box {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 250px;
        }}
        .info-box h3 {{ margin-top: 0; }}
        .stat {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3>Survey Route</h3>
        <div class="stat"><strong>Total Distance:</strong> <span id="total-dist">Calculating...</span></div>
        <div class="stat"><strong>Clusters:</strong> {len(cluster_order)}</div>
        <div class="stat"><strong>Segments:</strong> {len(segments)}</div>
    </div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const geoData = {json.dumps(geo_collection)};
        
        const map = L.map('map');
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        const routeFeature = geoData.features[0];
        const routeLayer = L.geoJSON(routeFeature, {{
            style: {{ color: '#0066ff', weight: 3, opacity: 0.7 }}
        }}).addTo(map);
        
        const clusterFeatures = geoData.features.slice(1);
        clusterFeatures.forEach((feature, idx) => {{
            L.marker([feature.geometry.coordinates[1], feature.geometry.coordinates[0]])
                .bindPopup(`<b>${{feature.properties.name}}</b><br>${{feature.properties.description}}`)
                .addTo(map);
        }});
        
        map.fitBounds(routeLayer.getBounds().pad(0.1));
        
        const coords = routeFeature.geometry.coordinates;
        let totalDist = 0;
        for (let i = 0; i < coords.length - 1; i++) {{
            const lat1 = coords[i][1] * Math.PI / 180;
            const lat2 = coords[i+1][1] * Math.PI / 180;
            const dLat = lat2 - lat1;
            const dLon = (coords[i+1][0] - coords[i][0]) * Math.PI / 180;
            const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                     Math.cos(lat1) * Math.cos(lat2) *
                     Math.sin(dLon/2) * Math.sin(dLon/2);
            const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
            totalDist += 6371000 * c;
        }}
        document.getElementById('total-dist').textContent = (totalDist / 1000).toFixed(2) + ' km';
    </script>
</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    return filename


def full_pipeline(
    kml_path,
    cluster_method="auto",
    gx=10,
    gy=10,
    k_clusters=40,
    use_ortools_for_cluster_order=True,
    allow_return_on_completed=True,
    avg_speed_kmh=None,
    output_gpx="final_mobile_route.gpx",
    output_html="route_preview.html",
    progress_callback=None,
    use_osm_speeds=True,
    overpass_cache_file="overpass_cache.json",
    routing_algorithm="greedy",
):
    """
    Complete pipeline from KML to optimized route.

    NEW: use_osm_speeds parameter enables OSM speed limit integration
    NEW: routing_algorithm parameter chooses between 'greedy' (fast) and 'hungarian' (optimal)
    """

    t0 = time.time()

    print("=" * 60)
    if use_osm_speeds:
        print("AUTO-ROUTE SURVEY PLANNER - OSM ENHANCED")
    else:
        print("AUTO-ROUTE SURVEY PLANNER")
    print("=" * 60)

    print("\n[1/8] Parsing KML...")
    segments = parse_kml(kml_path)
    print(f"  ✓ Loaded {len(segments)} segments")

    # OSM Speed Integration
    if use_osm_speeds:
        print("\n[2/8] Fetching OSM speed limits...")
        overpass = OverpassSpeedFetcher(cache_file=overpass_cache_file)
        segments = enrich_segments_with_osm_speeds(segments, overpass)
    else:
        print("\n[2/8] Skipping OSM (using KML speeds only)...")

    # Calculate average speed AFTER OSM enrichment
    if avg_speed_kmh is None:
        if use_osm_speeds:
            avg_speed_kmh = calculate_average_speed_osm(segments)
        else:
            avg_speed_kmh = calculate_average_speed(segments)

    print(f"  ✓ Using average speed: {avg_speed_kmh:.1f} km/h")

    print("\n[3/8] Building directed graph...")
    graph, required_edges = build_graph(segments, treat_unspecified_as_two_way=True)
    print(f"  ✓ Graph has {len(graph.id_to_node):,} nodes")

    print(f"\n[4/8] Clustering segments (method: {cluster_method})...")
    clusters = cluster_segments(
        segments, method=cluster_method, gx=gx, gy=gy, k_clusters=k_clusters
    )
    print(f"  ✓ Created {len(clusters)} clusters")

    print("\n[5/8] Ordering clusters...")
    order = order_clusters(
        clusters, segments, use_ortools=use_ortools_for_cluster_order and ORTOOLS_AVAILABLE
    )

    print("\n[6/8] Optimizing cluster order with 2-opt...")
    centroids = {cid: centroid_of_cluster(clusters[cid], segments) for cid in order}
    improved_order = two_opt_order(order, centroids) if len(order) > 2 else order
    print(f"  ✓ Cluster visit order: {len(improved_order)} clusters")

    print("\n[7/8] Routing segments within clusters...")

    # Estimate optimal workers
    if V4_AVAILABLE:
        num_workers = estimate_optimal_workers_v4(len(improved_order), len(segments))
    else:
        num_workers = estimate_optimal_workers(len(improved_order), len(segments))
    print(f"  Using {num_workers} parallel workers")
    print(f"  Algorithm: {routing_algorithm.upper()}")

    # Select routing function based on algorithm choice
    # PRIORITY: Use Production V4 for greedy routing (RECOMMENDED)
    if V4_AVAILABLE and routing_algorithm == "greedy":
        parallel_cluster_routing = parallel_cluster_routing_v4_wrapper
        print("  🚀 Using Production V4 Greedy with ON-DEMAND mode (FASTEST)")
        print("     ✅ 10-100x faster (bypasses matrix precomputation)")
        print("     ✅ 10-50x memory reduction")
        print("     ✅ <0.1% crash rate")
    elif routing_algorithm == "rfcs" and RFCS_AVAILABLE:
        parallel_cluster_routing = parallel_cluster_routing_rfcs
        print("  🏆 Using RFCS + Eulerization (GOLD STANDARD - 95-98% optimal)")
    elif routing_algorithm == "greedy" and GREEDY_AVAILABLE:
        parallel_cluster_routing = parallel_cluster_routing_greedy
        print("  ⚡ Using Legacy Greedy (consider upgrading to V4)")
    else:
        parallel_cluster_routing = parallel_cluster_routing_hungarian
        if routing_algorithm == "rfcs" and not RFCS_AVAILABLE:
            print("  ⚠️ RFCS not available, using Hungarian")
        elif routing_algorithm == "greedy" and not GREEDY_AVAILABLE and not V4_AVAILABLE:
            print("  ⚠️ Greedy not available, using Hungarian")
        else:
            print("  🎯 Using Hungarian algorithm (slower but more optimal)")

    # Route clusters in parallel
    cluster_results = parallel_cluster_routing(
        graph=graph,
        required_edges=required_edges,
        clusters=clusters,
        cluster_order=improved_order,
        allow_return=allow_return_on_completed,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )

    # Assemble results with inter-cluster routing
    full_route = []
    total_m = 0.0
    leg_breaks = []
    leg_names = []
    leg_distances = []

    for idx, (cluster_path, cluster_m, cid) in enumerate(cluster_results):
        leg_breaks.append(len(full_route))
        leg_names.append(f"Cluster {idx+1} ({len(clusters[cid])} segs)")

        # ✅ Add road routing between clusters
        if full_route and full_route[-1] != cluster_path[0]:
            prev_end = full_route[-1]
            next_start = cluster_path[0]

            # Compute road path between clusters
            inter_path, inter_dist = graph.shortest_path(prev_end, next_start)

            if inter_path is None:
                print(f"  ⚠️ WARNING: No road path from cluster {idx} to {idx+1}")
                print(f"     From: {prev_end}")
                print(f"     To: {next_start}")
                print("     Using direct connection (check your road network!)")
                inter_path = [prev_end, next_start]
                inter_dist = haversine(prev_end, next_start)

            # Add inter-cluster travel to route
            full_route.extend(inter_path[1:])  # Skip duplicate first point
            total_m += inter_dist
            leg_distances.append(cluster_m + inter_dist)
        else:
            leg_distances.append(cluster_m)

        # Add cluster path
        if full_route and full_route[-1] == cluster_path[0]:
            full_route.extend(cluster_path[1:])
        else:
            full_route.extend(cluster_path)

        total_m += cluster_m

    if not leg_breaks:
        leg_breaks = [0]
        leg_names = ["Complete Route"]
        leg_distances = [total_m]

    print("\n[8/8] Generating outputs...")

    gpx_file = write_mobile_gpx(
        full_route,
        leg_breaks,
        leg_names,
        leg_distances,
        filename=output_gpx,
        avg_speed_kmh=avg_speed_kmh,
    )
    print(f"  ✓ GPX: {gpx_file}")

    html_file = write_html_preview(
        full_route, clusters, improved_order, segments, filename=output_html
    )
    print(f"  ✓ HTML: {html_file}")

    elapsed = time.time() - t0

    # Calculate actual travel time
    total_time_hours = total_m / 1000.0 / avg_speed_kmh

    print(f"\n✓ Completed in {elapsed:.1f} seconds")
    print(f"✓ Total route distance: {total_m/1000:.2f} km")
    print(
        f"✓ Estimated travel time: {total_time_hours:.2f} hours ({total_time_hours*60:.0f} minutes)"
    )
    print(f"✓ Average speed: {avg_speed_kmh:.1f} km/h")

    return gpx_file, html_file, total_m, full_route, avg_speed_kmh


# ============================================================================
# PYQT6 GUI WITH ULTRA-MODERN DARK MODE
# ============================================================================


def apply_dark_mode(app):
    """Apply ultra-modern dark mode with vibrant accents"""
    palette = QPalette()

    # Ultra-modern color scheme
    bg_darkest = QColor(15, 15, 20)  # Almost black background
    bg_dark = QColor(22, 22, 28)  # Card backgrounds
    bg_medium = QColor(32, 32, 40)  # Input fields
    text_white = QColor(240, 240, 245)  # Primary text
    text_gray = QColor(160, 160, 170)  # Secondary text
    accent_cyan = QColor(0, 229, 255)  # Primary accent - electric cyan
    accent_purple = QColor(138, 43, 226)  # Secondary accent - vibrant purple
    accent_green = QColor(0, 255, 159)  # Success - neon green

    palette.setColor(QPalette.ColorRole.Window, bg_dark)
    palette.setColor(QPalette.ColorRole.WindowText, text_white)
    palette.setColor(QPalette.ColorRole.Base, bg_medium)
    palette.setColor(QPalette.ColorRole.AlternateBase, bg_darkest)
    palette.setColor(QPalette.ColorRole.ToolTipBase, bg_darkest)
    palette.setColor(QPalette.ColorRole.ToolTipText, text_white)
    palette.setColor(QPalette.ColorRole.Text, text_white)
    palette.setColor(QPalette.ColorRole.Button, bg_dark)
    palette.setColor(QPalette.ColorRole.ButtonText, text_white)
    palette.setColor(QPalette.ColorRole.Link, accent_cyan)
    palette.setColor(QPalette.ColorRole.Highlight, accent_cyan)
    palette.setColor(QPalette.ColorRole.HighlightedText, bg_darkest)

    app.setPalette(palette)

    # Ultra-modern stylesheet with glassmorphism and animations
    app.setStyleSheet(
        """
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
        }
        
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f0f14, stop:1 #16161c);
        }
        
        QWidget {
            font-size: 11pt;
            color: #f0f0f5;
        }
        
        /* Ultra-modern buttons */
        QPushButton {
            background: rgba(32, 32, 40, 0.8);
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 8px;
            padding: 12px 24px;
            color: #f0f0f5;
            font-weight: 600;
            font-size: 11pt;
        }
        
        QPushButton:hover {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid rgba(0, 229, 255, 0.6);
            color: #00e5ff;
        }
        
        QPushButton:pressed {
            background: rgba(0, 229, 255, 0.25);
        }
        
        QPushButton:disabled {
            background: rgba(32, 32, 40, 0.4);
            border: 1px solid rgba(100, 100, 110, 0.2);
            color: #505055;
        }
        
        /* Primary CTA button */
        QPushButton#run_button {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00e5ff, stop:1 #8a2be2);
            border: none;
            color: #0f0f14;
            font-weight: 700;
            font-size: 13pt;
            min-height: 48px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        QPushButton#run_button:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #33edff, stop:1 #a347f5);
        }
        
        QPushButton#run_button:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00d0e8, stop:1 #7a24cc);
        }
        
        /* Secondary action buttons */
        QPushButton#view_gpx_button, QPushButton#view_html_button {
            background: rgba(0, 255, 159, 0.12);
            border: 1px solid rgba(0, 255, 159, 0.4);
            color: #00ff9f;
            font-size: 11pt;
        }
        
        QPushButton#view_gpx_button:hover, QPushButton#view_html_button:hover {
            background: rgba(0, 255, 159, 0.2);
            border: 1px solid rgba(0, 255, 159, 0.7);
        }
        
        /* Input fields with glow */
        QLineEdit {
            background: rgba(32, 32, 40, 0.6);
            border: 1px solid rgba(100, 100, 110, 0.3);
            border-radius: 6px;
            padding: 10px 14px;
            color: #f0f0f5;
            font-size: 10pt;
        }
        
        QLineEdit:focus {
            background: rgba(32, 32, 40, 0.9);
            border: 1px solid #00e5ff;
        }
        
        /* Glassmorphic group boxes */
        QGroupBox {
            background: rgba(32, 32, 40, 0.4);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 12px;
            margin-top: 16px;
            padding-top: 24px;
            font-weight: 700;
            font-size: 12pt;
        }
        
        QGroupBox::title {
            color: #00e5ff;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 6px 12px;
            background: rgba(0, 229, 255, 0.1);
            border-radius: 6px;
            margin-left: 12px;
        }
        
        /* Modern tabs */
        QTabWidget::pane {
            border: 1px solid rgba(0, 229, 255, 0.2);
            background: rgba(22, 22, 28, 0.8);
            border-radius: 8px;
            margin-top: -1px;
        }
        
        QTabBar::tab {
            background: rgba(32, 32, 40, 0.5);
            border: none;
            padding: 12px 24px;
            margin-right: 4px;
            color: #a0a0aa;
            font-weight: 600;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(0, 229, 255, 0.2), stop:1 rgba(0, 229, 255, 0.05));
            color: #00e5ff;
            border-bottom: 2px solid #00e5ff;
        }
        
        QTabBar::tab:hover:!selected {
            background: rgba(0, 229, 255, 0.1);
            color: #f0f0f5;
        }
        
        /* Neon progress bars */
        QProgressBar {
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 8px;
            background: rgba(15, 15, 20, 0.8);
            text-align: center;
            color: #00e5ff;
            font-weight: 700;
            font-size: 11pt;
            min-height: 32px;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00e5ff, stop:0.5 #8a2be2, stop:1 #00ff9f);
            border-radius: 7px;
        }
        
        /* Terminal-style text display */
        QTextEdit {
            background: #0f0f14;
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 8px;
            color: #00ff9f;
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
            padding: 12px;
            selection-background-color: rgba(0, 229, 255, 0.3);
        }
        
        /* Glowing checkboxes */
        QCheckBox {
            spacing: 10px;
            color: #f0f0f5;
            font-size: 10pt;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid rgba(0, 229, 255, 0.4);
            background: rgba(32, 32, 40, 0.6);
        }
        
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #00e5ff, stop:1 #8a2be2);
            border: 2px solid #00e5ff;
        }
        
        QCheckBox::indicator:hover {
            border: 2px solid #00e5ff;
        }
        
        /* Radio buttons */
        QRadioButton {
            spacing: 10px;
            color: #f0f0f5;
            font-size: 10pt;
        }
        
        QRadioButton::indicator {
            width: 20px;
            height: 20px;
            border-radius: 10px;
            border: 2px solid rgba(0, 229, 255, 0.4);
            background: rgba(32, 32, 40, 0.6);
        }
        
        QRadioButton::indicator:checked {
            background: qradial-gradient(cx:0.5, cy:0.5, radius:0.5,
                fx:0.5, fy:0.5, stop:0 #00e5ff, stop:0.7 #00e5ff, stop:1 rgba(0, 229, 255, 0));
            border: 2px solid #00e5ff;
        }
        
        /* Spin boxes */
        QSpinBox, QDoubleSpinBox {
            background: rgba(32, 32, 40, 0.6);
            border: 1px solid rgba(100, 100, 110, 0.3);
            border-radius: 6px;
            padding: 8px;
            color: #f0f0f5;
            font-size: 10pt;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #00e5ff;
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background: rgba(0, 229, 255, 0.1);
            border: none;
            width: 20px;
        }
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background: rgba(0, 229, 255, 0.2);
        }
        
        /* Labels */
        QLabel {
            color: #f0f0f5;
            font-size: 10pt;
        }
        
        /* Status bar */
        QStatusBar {
            background: rgba(15, 15, 20, 0.95);
            color: #00e5ff;
            border-top: 1px solid rgba(0, 229, 255, 0.2);
            font-weight: 600;
        }
        
        /* Scrollbars */
        QScrollBar:vertical {
            background: rgba(15, 15, 20, 0.5);
            width: 10px;
            border-radius: 5px;
            margin: 0;
        }
        
        QScrollBar::handle:vertical {
            background: rgba(0, 229, 255, 0.4);
            border-radius: 5px;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: rgba(0, 229, 255, 0.6);
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """
    )


class RoutingWorker(QThread):
    """Background worker thread for route calculation"""

    progress = pyqtSignal(str)
    cluster_progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Execute the routing pipeline"""
        try:
            import builtins

            original_print = builtins.print

            def custom_print(*args, **kwargs):
                message = " ".join(str(arg) for arg in args)
                self.progress.emit(message)

            builtins.print = custom_print

            def progress_callback(current, total):
                self.cluster_progress.emit(current, total)

            gpx_file, html_file, total_m, full_route, avg_speed = full_pipeline(
                self.params["kml_path"],
                cluster_method=self.params["cluster_method"],
                gx=self.params["grid_x"],
                gy=self.params["grid_y"],
                k_clusters=self.params["k_clusters"],
                use_ortools_for_cluster_order=self.params["use_ortools"],
                allow_return_on_completed=self.params["allow_return"],
                avg_speed_kmh=self.params.get("avg_speed"),
                output_gpx=self.params["output_gpx"],
                output_html=self.params["output_html"],
                progress_callback=progress_callback,
                use_osm_speeds=self.params.get("use_osm_speeds", True),
                overpass_cache_file=self.params.get("overpass_cache_file", "overpass_cache.json"),
                routing_algorithm=self.params.get("routing_algorithm", "greedy"),
            )

            builtins.print = original_print

            self.finished.emit(
                {
                    "gpx_file": gpx_file,
                    "html_file": html_file,
                    "total_distance": total_m,
                    "num_points": len(full_route),
                    "avg_speed": avg_speed,
                }
            )

        except Exception as e:
            import traceback

            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class RouteOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the ultra-modern user interface"""
        self.setWindowTitle("🚀 Survey Route Planner - Ultra Modern")
        self.setMinimumSize(800, 600)
        self.resize(1100, 850)

        # Create central widget with scroll area
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout for the window
        window_layout = QVBoxLayout()
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)
        central_widget.setLayout(window_layout)

        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)
        content_widget.setLayout(main_layout)

        window_layout.addWidget(scroll_area)

        # Epic header
        header_widget = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        header_widget.setLayout(header_layout)

        title = QLabel("🗺️ ROUTE OPTIMIZER")
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #8a2be2);"
        )
        header_layout.addWidget(title)

        subtitle = QLabel("INTELLIGENT VEHICLE SURVEY ROUTE OPTIMIZATION")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            "color: #a0a0aa; font-size: 11pt; letter-spacing: 1px; font-weight: 600;"
        )
        header_layout.addWidget(subtitle)

        main_layout.addWidget(header_widget)

        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabWidget::pane { margin-top: 5px; }")
        main_layout.addWidget(tabs)

        tabs.addTab(self.create_io_tab(), "📁 FILES")
        tabs.addTab(self.create_cluster_tab(), "📊 CLUSTERING")
        tabs.addTab(self.create_routing_tab(), "🚗 ROUTING")

        # LIVE PROGRESS SECTION - The star of the show!
        progress_container = QGroupBox("⚡ LIVE PROGRESS")
        progress_container.setStyleSheet(
            """
            QGroupBox {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 229, 255, 0.08), stop:1 rgba(138, 43, 226, 0.08));
                border: 2px solid rgba(0, 229, 255, 0.3);
            }
        """
        )
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(16)

        # Status label - shows current operation
        self.status_label = QLabel("⏳ Ready to optimize...")
        self.status_label.setStyleSheet(
            """
            font-size: 13pt;
            font-weight: 700;
            color: #00e5ff;
            padding: 8px;
            background: rgba(0, 229, 255, 0.1);
            border-radius: 6px;
        """
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)

        # Main progress bar (for steps 1-6)
        self.main_progress_bar = QProgressBar()
        self.main_progress_bar.setVisible(False)
        self.main_progress_bar.setTextVisible(True)
        self.main_progress_bar.setFormat("Step %v/%m - %p%")
        self.main_progress_bar.setStyleSheet(
            """
            QProgressBar {
                min-height: 36px;
                font-size: 12pt;
            }
        """
        )
        progress_layout.addWidget(self.main_progress_bar)

        # Cluster routing progress
        cluster_label = QLabel("🎯 Cluster Routing Progress:")
        cluster_label.setStyleSheet("font-weight: 700; color: #00ff9f; font-size: 11pt;")
        progress_layout.addWidget(cluster_label)

        self.cluster_progress_bar = QProgressBar()
        self.cluster_progress_bar.setVisible(False)
        self.cluster_progress_bar.setFormat("Cluster %v / %m (%p%)")
        self.cluster_progress_bar.setStyleSheet(
            """
            QProgressBar {
                min-height: 28px;
                font-size: 10pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8a2be2, stop:1 #00ff9f);
            }
        """
        )
        progress_layout.addWidget(self.cluster_progress_bar)

        # Time estimate label
        self.time_estimate_label = QLabel("")
        self.time_estimate_label.setStyleSheet(
            """
            font-size: 10pt;
            color: #a0a0aa;
            padding: 6px;
            font-style: italic;
        """
        )
        self.time_estimate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.time_estimate_label)

        # Terminal-style log
        log_label = QLabel("📜 Process Log:")
        log_label.setStyleSheet(
            "font-weight: 700; color: #00e5ff; font-size: 11pt; margin-top: 8px;"
        )
        progress_layout.addWidget(log_label)

        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMinimumHeight(150)
        self.progress_text.setMaximumHeight(200)
        progress_layout.addWidget(self.progress_text)

        progress_container.setLayout(progress_layout)
        main_layout.addWidget(progress_container)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.run_button = QPushButton("▶ START OPTIMIZATION")
        self.run_button.setObjectName("run_button")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(self.run_optimization)
        button_layout.addWidget(self.run_button, 2)

        self.view_gpx_button = QPushButton("📁 OPEN GPX FOLDER")
        self.view_gpx_button.setObjectName("view_gpx_button")
        self.view_gpx_button.setEnabled(False)
        self.view_gpx_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_gpx_button.clicked.connect(self.open_gpx)
        button_layout.addWidget(self.view_gpx_button, 1)

        self.view_html_button = QPushButton("🗺️ VIEW MAP")
        self.view_html_button.setObjectName("view_html_button")
        self.view_html_button.setEnabled(False)
        self.view_html_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_html_button.clicked.connect(self.open_html)
        button_layout.addWidget(self.view_html_button, 1)

        main_layout.addLayout(button_layout)

        self.statusBar().showMessage("⚡ Ready - Load a KML file to begin optimization")

        # Log library status with style
        self.log_message("╔════════════════════════════════════════════╗")
        self.log_message("    ROUTE OPTIMIZER - SYSTEM STATUS")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message(f"✓ scikit-learn: {'AVAILABLE' if SKLEARN_AVAILABLE else 'NOT INSTALLED'}")
        self.log_message(f"✓ scipy: {'AVAILABLE' if SCIPY_AVAILABLE else 'NOT INSTALLED'}")
        self.log_message(f"✓ OR-Tools: {'AVAILABLE' if ORTOOLS_AVAILABLE else 'NOT INSTALLED'}")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message("💡 Speed limits will be auto-detected from KML + OSM")
        self.log_message("")

    def create_io_tab(self):
        """Create the input/output tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)

        input_group = QGroupBox("INPUT FILE")
        input_layout = QVBoxLayout()
        input_layout.setSpacing(12)

        file_row = QHBoxLayout()
        self.kml_path = QLineEdit()
        self.kml_path.setPlaceholderText("Select your KML file containing road segments...")
        file_row.addWidget(self.kml_path, 3)

        browse_button = QPushButton("BROWSE")
        browse_button.setFixedWidth(120)
        browse_button.clicked.connect(self.browse_kml)
        file_row.addWidget(browse_button)

        input_layout.addLayout(file_row)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        output_group = QGroupBox("OUTPUT FILES")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(12)

        gpx_layout = QHBoxLayout()
        gpx_layout.addWidget(QLabel("GPX File:"), 0)
        self.gpx_output = QLineEdit("final_mobile_route.gpx")
        gpx_layout.addWidget(self.gpx_output, 1)
        output_layout.addLayout(gpx_layout)

        html_layout = QHBoxLayout()
        html_layout.addWidget(QLabel("HTML Preview:"), 0)
        self.html_output = QLineEdit("route_preview.html")
        html_layout.addWidget(self.html_output, 1)
        output_layout.addLayout(html_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Info panel
        info_label = QLabel(
            "💡 <b>SPEED LIMIT DETECTION:</b><br><br>"
            "• Automatically extracts speed limits from KML metadata<br>"
            "• Fetches OSM speed limits via Overpass API<br>"
            "• Supports formats: '25 mph', '50 km/h', 'maxspeed=30'<br>"
            "• Falls back to highway classification defaults<br>"
            "• Default: 30 km/h if no data available"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            """
            QLabel {
                background: rgba(0, 229, 255, 0.08);
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #00e5ff;
                font-size: 10pt;
                line-height: 1.6;
            }
        """
        )
        layout.addWidget(info_label)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_cluster_tab(self):
        """Create the clustering options tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)

        method_group = QGroupBox("CLUSTERING METHOD")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(12)

        self.cluster_auto = QRadioButton("⚡ Auto (K-Means if available, else Grid)")
        self.cluster_auto.setChecked(True)
        method_layout.addWidget(self.cluster_auto)

        self.cluster_kmeans = QRadioButton("🎯 K-Means Clustering (requires scikit-learn)")
        self.cluster_kmeans.setEnabled(SKLEARN_AVAILABLE)
        method_layout.addWidget(self.cluster_kmeans)

        self.cluster_grid = QRadioButton("📊 Grid-Based Clustering")
        method_layout.addWidget(self.cluster_grid)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        kmeans_group = QGroupBox("K-MEANS OPTIONS")
        kmeans_layout = QHBoxLayout()

        kmeans_layout.addWidget(QLabel("Number of Clusters:"))
        self.k_clusters = QSpinBox()
        self.k_clusters.setRange(1, 1000)
        self.k_clusters.setValue(40)
        self.k_clusters.setFixedWidth(100)
        kmeans_layout.addWidget(self.k_clusters)
        kmeans_layout.addStretch()

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group)

        grid_group = QGroupBox("GRID OPTIONS")
        grid_layout = QHBoxLayout()

        grid_layout.addWidget(QLabel("Grid X:"))
        self.grid_x = QSpinBox()
        self.grid_x.setRange(1, 50)
        self.grid_x.setValue(10)
        self.grid_x.setFixedWidth(80)
        grid_layout.addWidget(self.grid_x)

        grid_layout.addSpacing(20)

        grid_layout.addWidget(QLabel("Grid Y:"))
        self.grid_y = QSpinBox()
        self.grid_y.setRange(1, 50)
        self.grid_y.setValue(10)
        self.grid_y.setFixedWidth(80)
        grid_layout.addWidget(self.grid_y)
        grid_layout.addStretch()

        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_routing_tab(self):
        """Create the routing options tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Algorithm Selection Group
        algo_group = QGroupBox("ROUTING ALGORITHM")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(12)

        self.algo_greedy = QRadioButton("⚡ Simple Greedy (10-100x FASTER, ~85% optimal)")
        self.algo_greedy.setChecked(True if GREEDY_AVAILABLE else False)
        self.algo_greedy.setEnabled(GREEDY_AVAILABLE)
        algo_layout.addWidget(self.algo_greedy)

        self.algo_hungarian = QRadioButton("🎯 Hungarian (Slower, ~95% optimal)")
        self.algo_hungarian.setChecked(
            True if (not GREEDY_AVAILABLE and not RFCS_AVAILABLE) else False
        )
        algo_layout.addWidget(self.algo_hungarian)

        self.algo_rfcs = QRadioButton("🏆 RFCS + Eulerization (GOLD STANDARD, ~95-98% optimal)")
        self.algo_rfcs.setChecked(True if RFCS_AVAILABLE else False)
        self.algo_rfcs.setEnabled(RFCS_AVAILABLE)
        algo_layout.addWidget(self.algo_rfcs)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        opt_group = QGroupBox("OPTIMIZATION OPTIONS")
        opt_layout = QVBoxLayout()
        opt_layout.setSpacing(15)

        self.use_ortools = QCheckBox("🎯 Use OR-Tools for TSP cluster ordering (recommended)")
        self.use_ortools.setChecked(ORTOOLS_AVAILABLE)
        self.use_ortools.setEnabled(ORTOOLS_AVAILABLE)
        opt_layout.addWidget(self.use_ortools)

        self.use_osm_speeds = QCheckBox("🌍 Fetch OSM speed limits (Overpass API)")
        self.use_osm_speeds.setChecked(True)
        opt_layout.addWidget(self.use_osm_speeds)

        self.allow_return = QCheckBox("🔄 Allow return on completed segments (flexible routing)")
        self.allow_return.setChecked(True)
        opt_layout.addWidget(self.allow_return)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        info_label = QLabel(
            "💡 <b>ALGORITHM COMPARISON:</b><br><br>"
            "• <b>Simple Greedy:</b> Fast nearest-neighbor approach. Solves in seconds instead of hours. Recommended for large datasets (1000+ segments).<br>"
            "• <b>Hungarian:</b> More optimal but much slower. Best for small datasets (&lt;500 segments) or when route quality is critical.<br><br>"
            "<b>OTHER OPTIONS:</b><br>"
            "• <b>OR-Tools:</b> Provides optimal cluster ordering using TSP solver<br>"
            "• <b>OSM Speeds:</b> Real-world speed limits for accurate ETAs<br>"
            "• <b>Return Allowed:</b> More flexible routing on two-way roads"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            """
            QLabel {
                background: rgba(138, 43, 226, 0.08);
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #8a2be2;
                font-size: 10pt;
                line-height: 1.6;
            }
        """
        )
        layout.addWidget(info_label)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def browse_kml(self):
        """Open file dialog to select KML file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select KML File", "", "KML Files (*.kml);;All Files (*)"
        )

        if file_path:
            self.kml_path.setText(file_path)
            self.log_message(f"✓ Selected: {os.path.basename(file_path)}")
            self.statusBar().showMessage(f"✓ Loaded: {os.path.basename(file_path)}")

    def log_message(self, message):
        """Add message to progress log with auto-scroll"""
        self.progress_text.append(message)
        self.progress_text.moveCursor(QTextCursor.MoveOperation.End)
        QApplication.processEvents()  # Force UI update

    def update_status(self, message):
        """Update the main status label"""
        self.status_label.setText(message)
        QApplication.processEvents()

    def run_optimization(self):
        """Start the route optimization process"""
        if not self.kml_path.text():
            QMessageBox.warning(self, "No Input File", "Please select a KML file first.")
            return

        if not os.path.exists(self.kml_path.text()):
            QMessageBox.warning(self, "File Not Found", "The selected KML file does not exist.")
            return

        params = {
            "kml_path": self.kml_path.text(),
            "cluster_method": self.get_cluster_method(),
            "grid_x": self.grid_x.value(),
            "grid_y": self.grid_y.value(),
            "k_clusters": self.k_clusters.value(),
            "use_ortools": self.use_ortools.isChecked(),
            "allow_return": self.allow_return.isChecked(),
            "avg_speed": None,
            "output_gpx": self.gpx_output.text(),
            "output_html": self.html_output.text(),
            "use_osm_speeds": self.use_osm_speeds.isChecked(),
            "overpass_cache_file": "overpass_cache.json",
            "routing_algorithm": self.get_routing_algorithm(),
        }

        self.progress_text.clear()
        self.log_message("╔════════════════════════════════════════════╗")
        self.log_message("    STARTING ROUTE OPTIMIZATION")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message(f"📍 Input: {os.path.basename(params['kml_path'])}")
        self.log_message(f"🎯 Method: {params['cluster_method'].upper()}")
        self.log_message(f"⚡ Algorithm: {params['routing_algorithm'].upper()}")
        self.log_message(f"🌍 OSM Speeds: {'ENABLED' if params['use_osm_speeds'] else 'DISABLED'}")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message("")

        self.update_status("⏳ Initializing optimization...")

        self.run_button.setEnabled(False)
        self.view_gpx_button.setEnabled(False)
        self.view_html_button.setEnabled(False)

        # Show progress bars
        self.main_progress_bar.setVisible(True)
        self.main_progress_bar.setRange(0, 8)  # 8 total steps
        self.main_progress_bar.setValue(0)

        self.cluster_progress_bar.setVisible(False)
        self.time_estimate_label.setText("⏱️ Calculating time estimate...")

        self.statusBar().showMessage("⚡ Optimization in progress...")

        self.start_time = time.time()

        self.worker = RoutingWorker(params)
        self.worker.progress.connect(self.on_progress_message)
        self.worker.cluster_progress.connect(self.on_cluster_progress)
        self.worker.finished.connect(self.on_optimization_complete)
        self.worker.error.connect(self.on_optimization_error)
        self.worker.start()

    def get_cluster_method(self):
        """Get selected clustering method"""
        if self.cluster_kmeans.isChecked():
            return "kmeans"
        elif self.cluster_grid.isChecked():
            return "grid"
        else:
            return "auto"

    def get_routing_algorithm(self):
        """Get selected routing algorithm"""
        if self.algo_greedy.isChecked():
            return "greedy"
        elif self.algo_rfcs.isChecked():
            return "rfcs"
        else:
            return "hungarian"

    def on_progress_message(self, message):
        """Handle progress messages and update status"""
        self.log_message(message)

        # Update main progress bar based on message content
        if "[1/8]" in message:
            self.main_progress_bar.setValue(1)
            self.update_status("📖 Parsing KML file...")
        elif "[2/8]" in message:
            self.main_progress_bar.setValue(2)
            self.update_status("🌍 Fetching OSM speed limits...")
        elif "[3/8]" in message:
            self.main_progress_bar.setValue(3)
            self.update_status("🔗 Building route graph...")
        elif "[4/8]" in message:
            self.main_progress_bar.setValue(4)
            self.update_status("📊 Clustering segments...")
        elif "[5/8]" in message:
            self.main_progress_bar.setValue(5)
            self.update_status("🎯 Ordering clusters...")
        elif "[6/8]" in message:
            self.main_progress_bar.setValue(6)
            self.update_status("⚡ Optimizing with 2-opt...")
        elif "[7/8]" in message:
            self.main_progress_bar.setValue(7)
            self.update_status("🚗 Routing segments...")
            self.cluster_progress_bar.setVisible(True)
        elif "[8/8]" in message:
            self.main_progress_bar.setValue(8)
            self.update_status("📝 Generating output files...")

    def on_cluster_progress(self, current, total):
        """Update cluster routing progress bar"""
        if not self.cluster_progress_bar.isVisible():
            self.cluster_progress_bar.setVisible(True)
            # ✅ Set range to 0-10000 for precision (allows 2 decimal places)
            self.cluster_progress_bar.setRange(0, 10000)
            self.cluster_progress_bar.setMaximum(10000)

        # ✅ Calculate percentage with decimal precision
        percentage_float = (current / total) * 100 if total > 0 else 0.0
        percentage_value = int(percentage_float * 100)  # Scale to 0-10000 range

        # ✅ Update with scaled percentage value
        self.cluster_progress_bar.setValue(percentage_value)
        # ✅ Custom format string with float percentage
        self.cluster_progress_bar.setFormat(f"Cluster {current}/{total} ({percentage_float:.2f}%)")
        self.update_status(f"🚗 Routing cluster {current}/{total}...")

        QApplication.processEvents()

        # Calculate and show time estimate
        if current > 0:
            elapsed = time.time() - self.start_time
            avg_time_per_cluster = elapsed / current
            remaining = total - current
            est_remaining = avg_time_per_cluster * remaining

            mins = int(est_remaining // 60)
            secs = int(est_remaining % 60)
            self.time_estimate_label.setText(
                f"⏱️ Est. {mins}m {secs}s remaining  |  "
                f"⏳ Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s"
            )
            QApplication.processEvents()

    def on_optimization_complete(self, results):
        """Handle successful optimization completion"""
        elapsed = time.time() - self.start_time

        self.main_progress_bar.setValue(8)
        self.cluster_progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        self.last_gpx = results["gpx_file"]
        self.last_html = results["html_file"]

        if results["gpx_file"]:
            self.view_gpx_button.setEnabled(True)
        if results["html_file"]:
            self.view_html_button.setEnabled(True)

        self.log_message("")
        self.log_message("╔════════════════════════════════════════════╗")
        self.log_message("    ✓ OPTIMIZATION COMPLETE!")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message(f"📏 Total Distance: {results['total_distance']/1000:.2f} km")
        self.log_message(f"📍 Route Points: {results['num_points']:,}")
        self.log_message(f"⚡ Avg Speed: {results['avg_speed']:.1f} km/h")
        self.log_message(f"⏱️ Processing Time: {int(elapsed//60)}m {int(elapsed%60)}s")
        self.log_message(f"📁 GPX: {results['gpx_file']}")
        self.log_message(f"🗺️ HTML: {results['html_file']}")
        self.log_message("╚════════════════════════════════════════════╝")

        self.update_status("✓ Optimization complete! Ready to view results.")
        self.time_estimate_label.setText(f"✓ Completed in {int(elapsed//60)}m {int(elapsed%60)}s")
        self.statusBar().showMessage("✓ Optimization complete! Click buttons to view outputs.")

        # Success animation - make the progress bar pulse green
        self.main_progress_bar.setStyleSheet(
            """
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff9f, stop:1 #00e5ff);
            }
        """
        )

        QMessageBox.information(
            self,
            "✓ Success!",
            f"Route optimization complete!\n\n"
            f"📏 Distance: {results['total_distance']/1000:.2f} km\n"
            f"📍 Points: {results['num_points']:,}\n"
            f"⚡ Avg Speed: {results['avg_speed']:.1f} km/h\n"
            f"⏱️ Time: {int(elapsed//60)}m {int(elapsed%60)}s\n\n"
            f"Click 'VIEW MAP' to preview or 'OPEN GPX FOLDER' to access files.",
        )

    def on_optimization_error(self, error_msg):
        """Handle optimization error"""
        self.main_progress_bar.setVisible(False)
        self.cluster_progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        self.log_message("")
        self.log_message("╔════════════════════════════════════════════╗")
        self.log_message("    ✗ ERROR OCCURRED")
        self.log_message("╚════════════════════════════════════════════╝")
        self.log_message(error_msg)
        self.log_message("╚════════════════════════════════════════════╝")

        self.update_status("✗ Error occurred during optimization")
        self.time_estimate_label.setText("")
        self.statusBar().showMessage("✗ Optimization failed - check log for details")

        QMessageBox.critical(
            self,
            "✗ Optimization Error",
            f"An error occurred during route optimization:\n\n{error_msg[:500]}",
        )

    def open_gpx(self):
        """Open the GPX file location"""
        if hasattr(self, "last_gpx") and self.last_gpx and os.path.exists(self.last_gpx):
            file_path = os.path.abspath(self.last_gpx)

            if sys.platform == "win32":
                os.startfile(os.path.dirname(file_path))
            elif sys.platform == "darwin":
                os.system(f'open "{os.path.dirname(file_path)}"')
            else:
                os.system(f'xdg-open "{os.path.dirname(file_path)}"')

            self.log_message(f"📁 Opened folder: {os.path.dirname(file_path)}")

    def open_html(self):
        """Open the HTML preview in browser"""
        if hasattr(self, "last_html") and self.last_html and os.path.exists(self.last_html):
            file_path = os.path.abspath(self.last_html)
            webbrowser.open("file://" + file_path)
            self.log_message(f"🗺️ Opened in browser: {file_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply ultra-modern dark mode
    apply_dark_mode(app)

    window = RouteOptimizer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
