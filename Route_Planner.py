#!/usr/bin/env python3
"""
Survey Route Planner
====================

Plans an efficient driving route that covers every road section in a KML file
(e.g. MapPlus/Duweis roadway-survey exports), then writes a mobile-friendly
GPX track and an interactive HTML map preview.

How it works
------------
1. Parse the KML into required road sections (direction-aware).
2. Merge endpoints that are within a few meters of each other.
3. Chain sections that run end-to-start into single continuous runs, so
   "chopped" road sections are driven in one pass and the solver has far
   fewer pieces to order.
4. Optionally fetch the surrounding OSM road network once (cached) for
   realistic deadhead routing and speed limits.
5. Greedy nearest-section ordering: from the current position, a single
   Dijkstra search runs only until it touches the closest remaining section
   (zero cost when the next section starts where the last one ended).

Usage
-----
    python Route_Planner.py                      # GUI (requires PyQt6)
    python Route_Planner.py sections.kml         # headless CLI
    python Route_Planner.py sections.kml --no-osm --gpx out.gpx

Dependencies: none required. Optional: requests (OSM data), PyQt6 (GUI).
"""

import argparse
import heapq
import json
import os
import re
import sys
import time
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

KML_NS = "{http://www.opengis.net/kml/2.2}"
DEFAULT_SPEED_KMH = 30.0
ENDPOINT_MERGE_TOLERANCE_M = 15.0
OSM_LINK_TOLERANCE_M = 50.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_CACHE_FILE = "overpass_cache.json"

# Fallback speeds (km/h) by OSM highway classification.
HIGHWAY_SPEEDS = {
    "motorway": 110, "motorway_link": 80, "trunk": 90, "trunk_link": 70,
    "primary": 70, "primary_link": 50, "secondary": 60, "secondary_link": 50,
    "tertiary": 50, "tertiary_link": 40, "unclassified": 40, "residential": 30,
    "living_street": 20, "service": 20, "track": 15,
}


# ============================================================================
# Geometry
# ============================================================================

def haversine(a, b):
    """Distance between two (lat, lon) points in meters."""
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * asin(sqrt(h))


def path_length(coords):
    """Total polyline length in meters."""
    return sum(haversine(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def snap_coord(lat, lon, precision=6):
    """Round coordinates (~0.11 m at precision 6) to kill float noise."""
    return (round(lat, precision), round(lon, precision))


# ============================================================================
# KML parsing
# ============================================================================

def parse_speed_limit(text):
    """Extract a speed limit in km/h from free text ('25 mph', '50 km/h', ...)."""
    if not text:
        return None
    text = text.lower().strip()
    if text in ("walk", "none", "signals", "variable"):
        return None
    for pattern in (r"(\d+(?:\.\d+)?)\s*mph", r"(\d+(?:\.\d+)?)\s*k[mp][/\s]?h?",
                    r"maxspeed[=:]\s*(\d+)", r"speed[=:]\s*(\d+)", r"^(\d+(?:\.\d+)?)$"):
        m = re.search(pattern, text)
        if m:
            speed = float(m.group(1))
            if "mph" in text:
                speed *= 1.60934
            return speed
    return None


def _parse_kml_tree(kml_path):
    """Parse KML, repairing common XML problems if the strict parse fails."""
    try:
        return ET.parse(kml_path)
    except ET.ParseError:
        with open(kml_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # Strip control characters and escape bare ampersands.
        content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)
        content = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", content)
        from io import StringIO
        return ET.parse(StringIO(content))


def _extract_extended_data(placemark):
    """Return {lowercased field name: text} from ExtendedData (both schemas)."""
    fields = {}
    ext = placemark.find(f"{KML_NS}ExtendedData")
    if ext is None:
        return fields
    for elem in ext.iter():
        tag = elem.tag.split("}")[-1].lower() if isinstance(elem.tag, str) else ""
        if tag in ("data", "simpledata"):
            name = (elem.get("name") or "").strip()
            if tag == "data":
                value_elem = elem.find(f"{KML_NS}value")
                value = value_elem.text if value_elem is not None else None
            else:
                value = elem.text
            if name and value:
                fields[name.lower()] = value.strip()
    return fields


def parse_kml(kml_path, log=print):
    """Parse a KML file into a list of section dicts.

    Each section: {coords, start, end, length_m, oneway, speed_limit, name}
    - oneway True  -> must and can only be driven in coords order
    - oneway False -> two-way, may be surveyed in either direction
    - oneway None  -> unspecified (treated as two-way)
    """
    tree = _parse_kml_tree(kml_path)
    sections = []

    for pm in tree.getroot().iter(f"{KML_NS}Placemark"):
        ls = pm.find(f".//{KML_NS}LineString")
        if ls is None:
            continue
        coords_elem = ls.find(f"{KML_NS}coordinates")
        if coords_elem is None or not coords_elem.text:
            continue

        coords = []
        for token in coords_elem.text.split():
            parts = token.split(",")
            if len(parts) < 2:
                continue
            try:
                pt = snap_coord(float(parts[1]), float(parts[0]))
            except ValueError:
                continue
            if not coords or pt != coords[-1]:
                coords.append(pt)
        if len(coords) < 2:
            continue

        fields = _extract_extended_data(pm)
        name_elem = pm.find(f"{KML_NS}name")
        name = fields.get("collid") or (name_elem.text.strip() if name_elem is not None
                                        and name_elem.text else f"Section {len(sections) + 1}")
        route = fields.get("routename")
        if route:
            name = f"{route} {name}"

        # Direction: MapPlus 'Dir' field, else generic oneway flags.
        oneway = None
        dir_code = fields.get("dir", "").upper()
        if dir_code:
            if dir_code in ("B", "T", "BOTH"):
                oneway = False
            elif dir_code == "D":  # decreasing: drive opposite to digitized order
                coords.reverse()
                oneway = True
            else:  # 'I', NB/SB/EB/WB, N/S/E/W ...
                oneway = True
        else:
            for key in ("oneway", "one_way", "one-way", "is_one_way"):
                value = fields.get(key, "").lower()
                if value in ("yes", "true", "1", "y"):
                    oneway = True
                elif value in ("no", "false", "0", "n"):
                    oneway = False

        # Speed limit from metadata, description, or name.
        speed = None
        for key in ("maxspeed", "speed_limit", "speedlimit", "speed"):
            if key in fields:
                speed = parse_speed_limit(fields[key])
                if speed:
                    break
        if not speed:
            desc = pm.find(f"{KML_NS}description")
            if desc is not None and desc.text:
                speed = parse_speed_limit(desc.text)

        sections.append({
            "coords": coords,
            "start": coords[0],
            "end": coords[-1],
            "length_m": path_length(coords),
            "oneway": oneway,
            "speed_limit": speed,
            "name": name,
        })

    if not sections:
        raise ValueError("No LineString sections found in KML file")
    log(f"  Parsed {len(sections)} road sections "
        f"({sum(s['length_m'] for s in sections) / 1000:.1f} km to survey)")
    return sections


# ============================================================================
# Endpoint merging + section chaining
# ============================================================================

def merge_endpoints(sections, tolerance_m=ENDPOINT_MERGE_TOLERANCE_M, log=print):
    """Snap section endpoints that lie within tolerance_m onto a shared point.

    The state often chops one road into sections whose endpoints differ by a
    few meters; without this the graph is disconnected and nothing chains.
    """
    cell = tolerance_m / 111320.0  # degrees latitude per meter
    grid = defaultdict(list)      # cell -> [canonical points]
    canonical = {}                # original point -> canonical point
    merged = 0

    def resolve(pt):
        nonlocal merged
        if pt in canonical:
            return canonical[pt]
        cx, cy = int(pt[0] / cell), int(pt[1] / cell)
        best, best_d = None, tolerance_m
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cand in grid.get((cx + dx, cy + dy), ()):
                    d = haversine(pt, cand)
                    if d < best_d:
                        best, best_d = cand, d
        if best is None:
            grid[(cx, cy)].append(pt)
            canonical[pt] = pt
            return pt
        canonical[pt] = best
        merged += 1
        return best

    for s in sections:
        start, end = resolve(s["start"]), resolve(s["end"])
        coords = s["coords"]
        if start != coords[0]:
            coords[0] = start
        if end != coords[-1]:
            coords[-1] = end
        # Guard against a tiny section collapsing onto itself.
        if coords[0] == coords[-1] and len(coords) == 2:
            coords[-1] = s["end"]
            end = s["end"]
        s["start"], s["end"] = coords[0], coords[-1]

    if merged:
        log(f"  Merged {merged} near-coincident endpoints (<{tolerance_m:.0f} m apart)")
    return sections


def chain_sections(sections, log=print):
    """Join sections that run end-to-start into single continuous chains.

    A join happens only where exactly two section endpoints meet (so no
    routing choice is lost) and driving directions are compatible. Returns
    chain dicts: {parts: [(section, flipped)], coords, oneway, name,
    length_m, time_s}.
    """
    incident = defaultdict(list)  # point -> [(section_idx, is_start)]
    for i, s in enumerate(sections):
        incident[s["start"]].append((i, True))
        incident[s["end"]].append((i, False))

    def continuation(node, current_idx, used):
        """Section that can be entered at `node`, or None. Returns (idx, flipped)."""
        entries = [(j, at_start) for j, at_start in incident[node] if j != current_idx]
        if len(entries) != 1:
            return None
        j, at_start = entries[0]
        if j in used:
            return None
        if at_start:
            return (j, False)
        if sections[j]["oneway"] is not True:  # two-way: may be flipped
            return (j, True)
        return None

    used = set()
    chains = []
    for i in range(len(sections)):
        if i in used:
            continue
        used.add(i)
        parts = [(i, False)]

        # Extend forward from the chain's end.
        while True:
            idx, flipped = parts[-1]
            node = sections[idx]["start" if flipped else "end"]
            nxt = continuation(node, idx, used)
            if nxt is None:
                break
            used.add(nxt[0])
            parts.append(nxt)

        # Extend backward from the chain's start.
        while True:
            idx, flipped = parts[0]
            node = sections[idx]["end" if flipped else "start"]
            entries = [(j, at_start) for j, at_start in incident[node] if j != idx]
            if len(entries) != 1:
                break
            j, at_start = entries[0]
            if j in used:
                break
            if not at_start:
                parts.insert(0, (j, False))
            elif sections[j]["oneway"] is not True:
                parts.insert(0, (j, True))
            else:
                break
            used.add(j)

        # Assemble chain geometry and stats.
        coords = []
        length_m = 0.0
        time_s = 0.0
        oneway = False
        for idx, flipped in parts:
            s = sections[idx]
            pc = s["coords"][::-1] if flipped else s["coords"]
            coords.extend(pc if not coords else pc[1:])
            length_m += s["length_m"]
            speed = s["speed_limit"] or DEFAULT_SPEED_KMH
            time_s += s["length_m"] / (speed * 1000.0 / 3600.0)
            if s["oneway"] is True:
                oneway = True

        first = sections[parts[0][0]]["name"]
        last = sections[parts[-1][0]]["name"]
        name = first if len(parts) == 1 else f"{first} … {last} ({len(parts)} sections)"
        chains.append({
            "parts": [(sections[idx], flipped) for idx, flipped in parts],
            "coords": coords,
            "oneway": oneway,
            "name": name,
            "length_m": length_m,
            "time_s": time_s,
        })

    joined = len(sections) - len(chains)
    log(f"  Chained {len(sections)} sections into {len(chains)} continuous runs "
        f"({joined} end-to-start joins)")
    return chains


# ============================================================================
# OSM road network (optional, cached)
# ============================================================================

def fetch_osm_ways(bbox, cache_file=OVERPASS_CACHE_FILE, timeout=90, log=print):
    """Fetch all roads in bbox from Overpass, with a persistent JSON cache.

    Returns a list of {geometry: [(lat, lon)...], speed_kmh, oneway} dicts,
    or [] if the network/dependency is unavailable.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    cache_key = f"{min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f}"

    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    if cache_key in cache:
        log(f"  Using cached OSM data ({len(cache[cache_key])} ways)")
        return [
            {"geometry": [tuple(pt) for pt in w["geometry"]],
             "speed_kmh": w["speed_kmh"], "oneway": w["oneway"]}
            for w in cache[cache_key]
        ]

    try:
        import requests
    except ImportError:
        log("  'requests' not installed - skipping OSM data")
        return []

    query = (f'[out:json][timeout:{timeout}];'
             f'way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});out geom;')
    try:
        log("  Querying Overpass API for the road network...")
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=timeout,
                             headers={"User-Agent": "SurveyRoutePlanner/1.0"})
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception as e:
        log(f"  OSM fetch failed ({e}) - continuing with survey sections only")
        return []

    ways = []
    for el in elements:
        geometry = el.get("geometry") or []
        if el.get("type") != "way" or len(geometry) < 2:
            continue
        tags = el.get("tags", {})
        if tags.get("highway") in ("footway", "path", "cycleway", "steps", "pedestrian"):
            continue
        speed = parse_speed_limit(tags.get("maxspeed", ""))
        if not speed:
            speed = HIGHWAY_SPEEDS.get(tags.get("highway"), DEFAULT_SPEED_KMH)
        ways.append({
            "geometry": [snap_coord(pt["lat"], pt["lon"]) for pt in geometry],
            "speed_kmh": speed,
            "oneway": tags.get("oneway", "no").lower() in ("yes", "true", "1", "-1"),
        })

    cache[cache_key] = [
        {"geometry": [list(pt) for pt in w["geometry"]],
         "speed_kmh": w["speed_kmh"], "oneway": w["oneway"]}
        for w in ways
    ]
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f)
    except OSError:
        pass
    log(f"  Fetched {len(ways)} OSM ways (cached for next run)")
    return ways


def apply_osm_speeds(sections, osm_ways, max_match_m=100.0, log=print):
    """Fill in missing section speed limits from the nearest OSM way."""
    if not osm_ways:
        return
    cell = 0.01  # ~1.1 km grid cells
    grid = defaultdict(list)
    for way in osm_ways:
        for cell_key in {(int(lat / cell), int(lon / cell)) for lat, lon in way["geometry"]}:
            grid[cell_key].append(way)

    matched = 0
    for s in sections:
        if s["speed_limit"]:
            continue
        mid = s["coords"][len(s["coords"]) // 2]
        cx, cy = int(mid[0] / cell), int(mid[1] / cell)
        best_d, best_speed = max_match_m, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for way in grid.get((cx + dx, cy + dy), ()):
                    d = min(haversine(mid, pt) for pt in way["geometry"])
                    if d < best_d:
                        best_d, best_speed = d, way["speed_kmh"]
        if best_speed:
            s["speed_limit"] = best_speed
            matched += 1
    log(f"  Matched OSM speed limits to {matched} sections")


# ============================================================================
# Graph
# ============================================================================

class Graph:
    """Directed graph over (lat, lon) nodes; edge weights are seconds."""

    def __init__(self):
        self.node_id = {}
        self.nodes = []
        self.adj = []  # node id -> {neighbor id: seconds}
        self.meters = []  # node id -> {neighbor id: meters}

    def _ensure(self, node):
        idx = self.node_id.get(node)
        if idx is None:
            idx = len(self.nodes)
            self.node_id[node] = idx
            self.nodes.append(node)
            self.adj.append({})
            self.meters.append({})
        return idx

    def add_edge(self, a, b, seconds, dist_m):
        ia, ib = self._ensure(a), self._ensure(b)
        if seconds < self.adj[ia].get(ib, float("inf")):
            self.adj[ia][ib] = seconds
            self.meters[ia][ib] = dist_m

    def add_polyline(self, coords, speed_kmh, oneway):
        mps = speed_kmh * 1000.0 / 3600.0
        for i in range(len(coords) - 1):
            d = haversine(coords[i], coords[i + 1])
            self.add_edge(coords[i], coords[i + 1], d / mps, d)
            if not oneway:
                self.add_edge(coords[i + 1], coords[i], d / mps, d)

    def nearest_target(self, source, targets):
        """Dijkstra from source, stopping at the first reached target node.

        Returns (target_id, seconds, meters, path_coords) or None. This is
        the hot path: when the next section starts at the current position
        it returns immediately.
        """
        source_id = self.node_id.get(source)
        if source_id is None:
            return None
        if source_id in targets:
            return (source_id, 0.0, 0.0, [source])

        n = len(self.nodes)
        dist = [float("inf")] * n
        dist_m = [0.0] * n
        prev = [-1] * n
        dist[source_id] = 0.0
        heap = [(0.0, source_id)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u in targets:
                path = []
                cur, steps = u, 0
                while cur != -1 and steps <= n:
                    path.append(self.nodes[cur])
                    cur = prev[cur]
                    steps += 1
                path.reverse()
                return (u, d, dist_m[u], path)
            meters_u = self.meters[u]
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    dist_m[v] = dist_m[u] + meters_u[v]
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None


def build_graph(chains, osm_ways, log=print):
    """Build the routing graph from survey chains plus OSM deadhead roads."""
    graph = Graph()

    for chain in chains:
        for section, flipped in chain["parts"]:
            coords = section["coords"][::-1] if flipped else section["coords"]
            speed = section["speed_limit"] or DEFAULT_SPEED_KMH
            graph.add_polyline(coords, speed, section["oneway"] is True)

    survey_nodes = len(graph.nodes)
    for way in osm_ways:
        graph.add_polyline(way["geometry"], way["speed_kmh"], way["oneway"])

    # Survey coordinates rarely coincide with OSM node coordinates, so link
    # each chain endpoint to nearby OSM nodes or the two graphs stay disjoint.
    if len(graph.nodes) > survey_nodes:
        cell = OSM_LINK_TOLERANCE_M / 111320.0
        grid = defaultdict(list)
        for idx in range(survey_nodes, len(graph.nodes)):
            lat, lon = graph.nodes[idx]
            grid[(int(lat / cell), int(lon / cell))].append(idx)

        links = 0
        endpoints = set()
        for chain in chains:
            endpoints.add(chain["coords"][0])
            endpoints.add(chain["coords"][-1])
        for pt in endpoints:
            cx, cy = int(pt[0] / cell), int(pt[1] / cell)
            best, best_d = None, OSM_LINK_TOLERANCE_M
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for idx in grid.get((cx + dx, cy + dy), ()):
                        d = haversine(pt, graph.nodes[idx])
                        if d < best_d:
                            best, best_d = idx, d
            if best is not None:
                osm_pt = graph.nodes[best]
                seconds = best_d / (DEFAULT_SPEED_KMH * 1000.0 / 3600.0)
                graph.add_edge(pt, osm_pt, seconds, best_d)
                graph.add_edge(osm_pt, pt, seconds, best_d)
                links += 1
        log(f"  Linked {links}/{len(endpoints)} chain endpoints to the OSM network")

    log(f"  Graph: {len(graph.nodes):,} nodes")
    return graph


# ============================================================================
# Route solver
# ============================================================================

def solve_route(graph, chains, start=None, log=print, progress=None):
    """Order and connect all chains with greedy nearest-section routing.

    Returns (route_coords, legs, totals) where each leg is
    {name, dist_m, time_s, start_index} and totals has survey/deadhead splits.
    """
    # Target node -> list of (chain index, reversed?). Two-way chains can be
    # entered from either end.
    entries = defaultdict(list)
    for ci, chain in enumerate(chains):
        entries[graph.node_id[chain["coords"][0]]].append((ci, False))
        if not chain["oneway"] and chain["coords"][-1] != chain["coords"][0]:
            entries[graph.node_id[chain["coords"][-1]]].append((ci, True))

    remaining = set(range(len(chains)))
    targets = {nid for nid, lst in entries.items()
               if any(ci in remaining for ci, _ in lst)}

    if start is None:
        current = chains[0]["coords"][0]
    else:
        current = min(graph.nodes, key=lambda nd: haversine(nd, start))

    route = []
    legs = []
    total = {"survey_m": 0.0, "deadhead_m": 0.0, "survey_s": 0.0,
             "deadhead_s": 0.0, "jumps": 0}

    def extend(coords):
        if route and coords and route[-1] == coords[0]:
            coords = coords[1:]
        route.extend(coords)

    while remaining:
        found = graph.nearest_target(current, targets)
        if found is not None:
            target_id, dh_s, dh_m, dh_path = found
            ci, flipped = next((ci, fl) for ci, fl in entries[target_id] if ci in remaining)
        else:
            # Unreachable by road: jump straight-line to the closest chain start.
            ci, flipped = min(
                ((ci, fl) for nid in targets for ci, fl in entries[nid] if ci in remaining),
                key=lambda e: haversine(
                    current,
                    chains[e[0]]["coords"][-1 if e[1] else 0]),
            )
            entry_pt = chains[ci]["coords"][-1 if flipped else 0]
            dh_m = haversine(current, entry_pt)
            dh_s = dh_m / (DEFAULT_SPEED_KMH * 1000.0 / 3600.0)
            dh_path = [current, entry_pt]
            total["jumps"] += 1

        chain = chains[ci]
        coords = chain["coords"][::-1] if flipped else chain["coords"]

        legs.append({
            "name": chain["name"],
            "dist_m": dh_m + chain["length_m"],
            "time_s": dh_s + chain["time_s"],
            "start_index": max(0, len(route) - 1) if route else 0,
        })
        extend(dh_path)
        extend(coords)
        current = coords[-1]

        remaining.discard(ci)
        total["survey_m"] += chain["length_m"]
        total["survey_s"] += chain["time_s"]
        total["deadhead_m"] += dh_m
        total["deadhead_s"] += dh_s

        targets = {nid for nid, lst in entries.items()
                   if any(c in remaining for c, _ in lst)}
        if progress:
            progress(len(chains) - len(remaining), len(chains))

    if total["jumps"]:
        log(f"  WARNING: {total['jumps']} chains had no road connection; "
            f"straight-line jumps were used (check the map)")
    return route, legs, total


# ============================================================================
# Output: GPX + HTML map
# ============================================================================

def write_gpx(route, legs, total, filename):
    """Write a GPX file: one track per leg plus a named waypoint with ETA."""
    now = datetime.now(timezone.utc)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="SurveyRoutePlanner" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        "<metadata>",
        "<name>Survey Route</name>",
        f"<desc>{len(legs)} runs, "
        f"{(total['survey_m'] + total['deadhead_m']) / 1000:.1f} km total "
        f"({total['deadhead_m'] / 1000:.1f} km deadhead)</desc>",
        f"<time>{now.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>",
        "</metadata>",
    ]

    elapsed = 0.0
    for i, leg in enumerate(legs):
        end = legs[i + 1]["start_index"] if i + 1 < len(legs) else len(route)
        pts = route[leg["start_index"]:end + 1]
        if not pts:
            continue
        eta = now + timedelta(seconds=elapsed)
        lines.append(f'<wpt lat="{pts[0][0]}" lon="{pts[0][1]}">'
                     f"<name>{i + 1}. {escape(leg['name'])}</name>"
                     f"<desc>{leg['dist_m'] / 1000:.2f} km | "
                     f"{leg['time_s'] / 60:.0f} min | ETA +{elapsed / 3600:.1f} h "
                     f"({eta.strftime('%H:%M')} UTC)</desc></wpt>")
        elapsed += leg["time_s"]

    lines.append("<trk><name>Survey Route</name><trkseg>")
    lines.extend(f'<trkpt lat="{lat}" lon="{lon}"/>' for lat, lon in route)
    lines.append("</trkseg></trk></gpx>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return filename


def write_html(route, legs, total, filename):
    """Write an interactive Leaflet map preview of the route."""
    waypoints = []
    for i, leg in enumerate(legs):
        lat, lon = route[min(leg["start_index"], len(route) - 1)]
        waypoints.append({"lat": lat, "lon": lon,
                          "label": f"{i + 1}. {leg['name']}",
                          "info": f"{leg['dist_m'] / 1000:.2f} km, "
                                  f"{leg['time_s'] / 60:.0f} min"})
    data = {
        "route": [[lat, lon] for lat, lon in route],
        "waypoints": waypoints,
        "stats": {
            "km": (total["survey_m"] + total["deadhead_m"]) / 1000,
            "survey_km": total["survey_m"] / 1000,
            "deadhead_km": total["deadhead_m"] / 1000,
            "hours": (total["survey_s"] + total["deadhead_s"]) / 3600,
            "runs": len(legs),
        },
    }

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Survey Route</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
 html, body, #map { height: 100%; margin: 0; font-family: system-ui, sans-serif; }
 .info { position: absolute; top: 10px; right: 10px; z-index: 1000; background: #fff;
         padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.3);
         font-size: 14px; line-height: 1.6; }
</style></head><body>
<div id="map"></div>
<div class="info">
 <b>Survey Route</b><br>
 Total: <b>__KM__ km</b> (~__H__ h)<br>
 Survey: __SKM__ km | Deadhead: __DKM__ km<br>
 Runs: __RUNS__
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
const line = L.polyline(data.route, {color: '#0066ff', weight: 3, opacity: .8}).addTo(map);
data.waypoints.forEach(w => L.circleMarker([w.lat, w.lon],
  {radius: 5, color: '#d00', fillColor: '#ff5252', fillOpacity: .9})
  .bindPopup('<b>' + w.label + '</b><br>' + w.info).addTo(map));
if (data.route.length) {
  L.marker(data.route[0]).bindPopup('<b>START</b>').addTo(map);
  L.marker(data.route[data.route.length - 1]).bindPopup('<b>END</b>').addTo(map);
}
map.fitBounds(line.getBounds().pad(0.05));
</script></body></html>"""
    html = (html.replace("__DATA__", json.dumps(data))
                .replace("__KM__", f"{data['stats']['km']:.1f}")
                .replace("__H__", f"{data['stats']['hours']:.1f}")
                .replace("__SKM__", f"{data['stats']['survey_km']:.1f}")
                .replace("__DKM__", f"{data['stats']['deadhead_km']:.1f}")
                .replace("__RUNS__", str(data["stats"]["runs"])))
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename


# ============================================================================
# Pipeline
# ============================================================================

def plan_route(kml_path, use_osm=True, start=None, output_gpx="survey_route.gpx",
               output_html="route_preview.html", cache_file=OVERPASS_CACHE_FILE,
               log=print, step=None, progress=None):
    """Full pipeline: KML -> optimized route -> GPX + HTML. Returns results dict."""
    t0 = time.time()

    def stage(n, msg):
        log(f"\n[{n}/5] {msg}")
        if step:
            step(n)

    stage(1, "Parsing KML...")
    sections = parse_kml(kml_path, log=log)
    merge_endpoints(sections, log=log)

    stage(2, "Fetching OSM road network..." if use_osm else "Skipping OSM (offline mode)")
    osm_ways = []
    if use_osm:
        lats = [lat for s in sections for lat, _ in (s["start"], s["end"])]
        lons = [lon for s in sections for _, lon in (s["start"], s["end"])]
        pad = 0.01
        bbox = (min(lats) - pad, min(lons) - pad, max(lats) + pad, max(lons) + pad)
        osm_ways = fetch_osm_ways(bbox, cache_file=cache_file, log=log)
        apply_osm_speeds(sections, osm_ways, log=log)

    stage(3, "Chaining contiguous sections...")
    chains = chain_sections(sections, log=log)

    stage(4, "Building graph and solving route...")
    graph = build_graph(chains, osm_ways, log=log)
    route, legs, total = solve_route(graph, chains, start=start, log=log, progress=progress)

    stage(5, "Writing outputs...")
    write_gpx(route, legs, total, output_gpx)
    write_html(route, legs, total, output_html)
    log(f"  GPX: {output_gpx}")
    log(f"  Map: {output_html}")

    km = (total["survey_m"] + total["deadhead_m"]) / 1000
    hours = (total["survey_s"] + total["deadhead_s"]) / 3600
    log(f"\nDone in {time.time() - t0:.1f} s")
    log(f"Route: {km:.1f} km total ({total['survey_m'] / 1000:.1f} survey + "
        f"{total['deadhead_m'] / 1000:.1f} deadhead), est. {hours:.1f} h driving")

    return {"gpx": output_gpx, "html": output_html, "route": route, "legs": legs,
            "total_m": total["survey_m"] + total["deadhead_m"],
            "total_s": total["survey_s"] + total["deadhead_s"],
            "survey_m": total["survey_m"], "deadhead_m": total["deadhead_m"],
            "sections": len(sections), "runs": len(chains), "jumps": total["jumps"]}


# ============================================================================
# GUI (PyQt6, optional)
# ============================================================================

def run_gui():
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
        QMainWindow, QMessageBox, QProgressBar, QPushButton, QTextEdit,
        QVBoxLayout, QWidget,
    )

    class Worker(QThread):
        message = pyqtSignal(str)
        stepped = pyqtSignal(int)
        solved = pyqtSignal(int, int)
        done = pyqtSignal(dict)
        failed = pyqtSignal(str)

        def __init__(self, params):
            super().__init__()
            self.params = params

        def run(self):
            try:
                results = plan_route(
                    self.params["kml"],
                    use_osm=self.params["use_osm"],
                    output_gpx=self.params["gpx"],
                    output_html=self.params["html"],
                    log=self.message.emit,
                    step=self.stepped.emit,
                    progress=self.solved.emit,
                )
                self.done.emit(results)
            except Exception as e:
                import traceback
                self.failed.emit(f"{e}\n\n{traceback.format_exc()}")

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.worker = None
            self.results = None
            self.setWindowTitle("Survey Route Planner")
            self.resize(760, 640)

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setSpacing(10)
            layout.setContentsMargins(16, 16, 16, 16)

            row = QHBoxLayout()
            self.kml_edit = QLineEdit()
            self.kml_edit.setPlaceholderText("Select the KML file with your road sections...")
            browse = QPushButton("Browse…")
            browse.clicked.connect(self.browse)
            row.addWidget(QLabel("KML:"))
            row.addWidget(self.kml_edit, 1)
            row.addWidget(browse)
            layout.addLayout(row)

            out_row = QHBoxLayout()
            self.gpx_edit = QLineEdit("survey_route.gpx")
            self.html_edit = QLineEdit("route_preview.html")
            out_row.addWidget(QLabel("GPX:"))
            out_row.addWidget(self.gpx_edit, 1)
            out_row.addWidget(QLabel("Map:"))
            out_row.addWidget(self.html_edit, 1)
            layout.addLayout(out_row)

            self.osm_check = QCheckBox(
                "Use OpenStreetMap for connecting roads and speed limits (recommended)")
            self.osm_check.setChecked(True)
            layout.addWidget(self.osm_check)

            self.run_btn = QPushButton("Plan Route")
            self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
            self.run_btn.clicked.connect(self.start)
            layout.addWidget(self.run_btn)

            self.bar = QProgressBar()
            self.bar.setRange(0, 5)
            self.bar.setFormat("Step %v / %m")
            layout.addWidget(self.bar)

            self.solve_bar = QProgressBar()
            self.solve_bar.setFormat("Routing run %v / %m")
            self.solve_bar.setVisible(False)
            layout.addWidget(self.solve_bar)

            self.log_box = QTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setStyleSheet(
                "background:#111; color:#8f8; font-family:monospace; font-size:11px;")
            layout.addWidget(self.log_box, 1)

            btns = QHBoxLayout()
            self.gpx_btn = QPushButton("Open GPX Folder")
            self.map_btn = QPushButton("View Map")
            self.gpx_btn.setEnabled(False)
            self.map_btn.setEnabled(False)
            self.gpx_btn.clicked.connect(self.open_gpx_folder)
            self.map_btn.clicked.connect(self.open_map)
            btns.addWidget(self.gpx_btn)
            btns.addWidget(self.map_btn)
            layout.addLayout(btns)

        def browse(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select KML File", "",
                                                  "KML Files (*.kml);;All Files (*)")
            if path:
                self.kml_edit.setText(path)

        def log(self, msg):
            self.log_box.append(msg)
            self.log_box.moveCursor(QTextCursor.MoveOperation.End)

        def start(self):
            kml = self.kml_edit.text().strip()
            if not kml or not os.path.exists(kml):
                QMessageBox.warning(self, "No input", "Please select an existing KML file.")
                return
            self.run_btn.setEnabled(False)
            self.gpx_btn.setEnabled(False)
            self.map_btn.setEnabled(False)
            self.log_box.clear()
            self.bar.setValue(0)
            self.solve_bar.setVisible(False)
            self.worker = Worker({
                "kml": kml,
                "use_osm": self.osm_check.isChecked(),
                "gpx": self.gpx_edit.text().strip() or "survey_route.gpx",
                "html": self.html_edit.text().strip() or "route_preview.html",
            })
            self.worker.message.connect(self.log)
            self.worker.stepped.connect(self.bar.setValue)
            self.worker.solved.connect(self.on_solve_progress)
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

        def on_solve_progress(self, current, total_runs):
            self.solve_bar.setVisible(True)
            self.solve_bar.setRange(0, total_runs)
            self.solve_bar.setValue(current)

        def on_done(self, results):
            self.results = results
            self.run_btn.setEnabled(True)
            self.gpx_btn.setEnabled(True)
            self.map_btn.setEnabled(True)
            self.bar.setValue(5)
            QMessageBox.information(
                self, "Route ready",
                f"Total: {results['total_m'] / 1000:.1f} km "
                f"(deadhead {results['deadhead_m'] / 1000:.1f} km)\n"
                f"Estimated driving time: {results['total_s'] / 3600:.1f} h\n"
                f"{results['sections']} sections in {results['runs']} continuous runs")

        def on_failed(self, msg):
            self.run_btn.setEnabled(True)
            self.log(f"\nERROR: {msg}")
            QMessageBox.critical(self, "Error", msg[:800])

        def open_gpx_folder(self):
            if self.results:
                folder = os.path.dirname(os.path.abspath(self.results["gpx"])) or "."
                webbrowser.open("file://" + folder)

        def open_map(self):
            if self.results:
                webbrowser.open("file://" + os.path.abspath(self.results["html"]))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = Window()
    win.show()
    sys.exit(app.exec())


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plan an efficient survey driving route from a KML file.")
    parser.add_argument("kml", nargs="?", help="KML file (omit to launch the GUI)")
    parser.add_argument("--gpx", default="survey_route.gpx", help="output GPX file")
    parser.add_argument("--html", default="route_preview.html", help="output HTML map")
    parser.add_argument("--no-osm", action="store_true",
                        help="skip OpenStreetMap fetch (offline mode)")
    parser.add_argument("--start", metavar="LAT,LON",
                        help="start position, e.g. 40.44,-79.99")
    args = parser.parse_args()

    if args.kml is None:
        try:
            run_gui()
        except ImportError:
            parser.error("PyQt6 is not installed - pass a KML file to run headless, "
                         "or `pip install PyQt6` for the GUI")
        return

    start = None
    if args.start:
        try:
            lat, lon = (float(x) for x in args.start.split(","))
            start = (lat, lon)
        except ValueError:
            parser.error("--start must be LAT,LON")

    plan_route(args.kml, use_osm=not args.no_osm, start=start,
               output_gpx=args.gpx, output_html=args.html)


if __name__ == "__main__":
    main()
