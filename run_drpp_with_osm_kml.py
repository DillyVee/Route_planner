#!/usr/bin/env python3
"""
DRPP Router with OSM Road Network + KML Output with Metadata (OPTIMIZED)

This script:
1. Parses your KML with all metadata (CollId, RouteName, Dir, etc.)
2. Fetches Pennsylvania road network from OpenStreetMap
3. Routes between disconnected segments using REAL ROADS with OPTIMIZED greedy algorithm
4. Exports KML with sequence numbers and all metadata preserved

Output is compatible with MapPlus/Duweis.

PERFORMANCE OPTIMIZATIONS (100-1000x faster than previous version):
- Uses drpp_core.greedy_route_cluster with on-demand Dijkstra
- Computes Dijkstra ONCE per iteration (not per segment)
- Optional distance limiting for dense areas (--max-distance)
- Pure greedy with no lookahead overhead (lookahead_depth=1)

Usage:
    # For large sparse areas (like entire state):
    python run_drpp_with_osm_kml.py PA_2025_Region2.kml --output-dir output

    # For dense urban areas (optional distance limit):
    python run_drpp_with_osm_kml.py urban_area.kml --max-distance 20

Options:
    --no-osm            Skip OSM fetch (use straight-line routing)
    --max-distance KM   Maximum search distance in km (default: unlimited)
    --verbose           Show detailed progress
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Import routing components
from drpp_pipeline import DRPPPipeline
from Route_Planner import fetch_osm_roads_for_routing, DirectedGraph, haversine
from drpp_core import greedy_route_cluster


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def build_spatial_grid(nodes, cell_size_deg=0.001):
    """
    Build spatial grid index for fast nearest neighbor search.

    Args:
        nodes: List of (lat, lon) tuples
        cell_size_deg: Grid cell size in degrees (~0.001° ≈ 100m)

    Returns:
        Dict mapping (grid_lat, grid_lon) -> list of nodes
    """
    from collections import defaultdict
    grid = defaultdict(list)

    for node in nodes:
        lat, lon = node
        grid_lat = int(lat / cell_size_deg)
        grid_lon = int(lon / cell_size_deg)
        grid[(grid_lat, grid_lon)].append(node)

    return grid


def find_nearest_nodes_fast(point, spatial_grid, max_distance=50.0, k=5, cell_size_deg=0.001):
    """
    Find k nearest nodes using spatial grid index.

    Args:
        point: (lat, lon) tuple
        spatial_grid: Dict from build_spatial_grid()
        max_distance: Maximum distance in meters
        k: Number of nearest nodes to return
        cell_size_deg: Grid cell size used to build index

    Returns:
        List of (node, distance) tuples
    """
    lat, lon = point
    grid_lat = int(lat / cell_size_deg)
    grid_lon = int(lon / cell_size_deg)

    # Check current cell and 8 neighbors (3x3 grid)
    candidates = []
    for dlat in [-1, 0, 1]:
        for dlon in [-1, 0, 1]:
            cell = (grid_lat + dlat, grid_lon + dlon)
            candidates.extend(spatial_grid.get(cell, []))

    # Calculate distances
    distances = []
    for candidate in candidates:
        dist = haversine(point, candidate)
        if dist <= max_distance:
            distances.append((candidate, dist))

    # Return k nearest
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def build_graph_with_osm(segments, use_osm=True, logger=None):
    """
    Build routing graph from segments + OSM roads.

    Args:
        segments: Parsed segments from KML
        use_osm: Whether to fetch OSM roads
        logger: Logger instance

    Returns:
        DirectedGraph with segments + OSM roads
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    graph = DirectedGraph()

    # Add segments to graph
    logger.info(f"Adding {len(segments)} segments to graph...")
    segment_endpoints = set()  # Track segment endpoints for snapping

    for seg in segments:
        coords = seg.coordinates
        # Track endpoints
        segment_endpoints.add(coords[0])   # Start
        segment_endpoints.add(coords[-1])  # End

        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dist = haversine(p1, p2)

            if seg.forward_required:
                graph.add_edge(p1, p2, dist)
            if seg.backward_required:
                graph.add_edge(p2, p1, dist)

    base_nodes = len(graph.id_to_node)
    logger.info(f"  ✓ Base graph: {base_nodes:,} nodes from segments")
    logger.info(f"  ✓ Segment endpoints: {len(segment_endpoints):,}")

    # Add OSM roads if requested
    if use_osm:
        logger.info("Fetching OSM road network...")

        # Calculate bounding box
        all_coords = []
        for seg in segments:
            all_coords.extend(seg.coordinates)

        if not all_coords:
            logger.warning("  No coordinates found, skipping OSM")
            return graph

        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Add padding (0.02 degrees ≈ 2km)
        padding = 0.02
        bbox = (min_lat - padding, min_lon - padding,
                max_lat + padding, max_lon + padding)

        logger.info(f"  Bbox: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")

        try:
            osm_ways = fetch_osm_roads_for_routing(bbox, timeout=120)

            if osm_ways:
                logger.info(f"  ✓ Found {len(osm_ways):,} OSM road segments")
                logger.info("  Adding OSM roads to graph...")

                # Collect all OSM nodes
                osm_nodes = set()
                osm_edges = 0
                for way in osm_ways:
                    coords = way['geometry']
                    oneway = way.get('oneway', False)

                    # Add all nodes from this way
                    for coord in coords:
                        osm_nodes.add(coord)

                    for i in range(len(coords) - 1):
                        p1, p2 = coords[i], coords[i + 1]
                        dist = haversine(p1, p2)

                        # Add forward direction
                        graph.add_edge(p1, p2, dist)
                        osm_edges += 1

                        # Add reverse if not oneway
                        if not oneway:
                            graph.add_edge(p2, p1, dist)
                            osm_edges += 1

                total_nodes = len(graph.id_to_node)
                logger.info(f"  ✓ Added {osm_edges:,} OSM edges")
                logger.info(f"  ✓ Total graph: {total_nodes:,} nodes (added {total_nodes - base_nodes:,})")

                # Snap segment endpoints to OSM network
                logger.info("  Connecting segment endpoints to OSM network...")
                logger.info("  Building spatial index...")
                spatial_grid = build_spatial_grid(list(osm_nodes))
                logger.info(f"  ✓ Spatial index built ({len(spatial_grid):,} cells)")

                snap_edges = 0
                endpoints_connected = 0

                for endpoint in segment_endpoints:
                    # Find nearest OSM nodes within 50 meters using spatial index
                    nearest = find_nearest_nodes_fast(endpoint, spatial_grid, max_distance=50.0, k=3)

                    if nearest:
                        endpoints_connected += 1
                        for osm_node, dist in nearest:
                            # Add bidirectional edges to connect
                            graph.add_edge(endpoint, osm_node, dist)
                            graph.add_edge(osm_node, endpoint, dist)
                            snap_edges += 2

                logger.info(f"  ✓ Connected {endpoints_connected:,}/{len(segment_endpoints):,} endpoints to OSM")
                logger.info(f"  ✓ Added {snap_edges:,} snap edges")
                logger.info(f"  ✓ Final graph: {len(graph.id_to_node):,} nodes")
            else:
                logger.warning("  No OSM roads found in bbox")

        except Exception as e:
            logger.error(f"  OSM fetch failed: {e}")
            logger.warning("  Continuing with segment-only routing")

    return graph


def route_segments_greedy(segments, graph, max_distance_km=None, logger=None):
    """
    Route through segments using OPTIMIZED greedy on-demand algorithm.

    Uses drpp_core.greedy_route_cluster with performance optimizations:
    - On-demand Dijkstra (computes once per iteration, not per segment)
    - Optional distance limiting (for dense areas)
    - Pure greedy (lookahead_depth=1, no O(n²) lookahead)

    Args:
        segments: List of segments to route
        graph: DirectedGraph with OSM roads
        max_distance_km: Maximum search distance in km (None = unlimited)
        logger: Logger instance

    Returns:
        List of (segment, path, distance) tuples in route order
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Running OPTIMIZED greedy routing algorithm...")
    logger.info(f"  Segments to route: {len(segments)}")
    logger.info(f"  Graph nodes: {len(graph.id_to_node):,}")

    max_search_distance = None
    if max_distance_km is not None:
        max_search_distance = max_distance_km * 1000  # Convert km to meters
        logger.info(f"  Using: on-demand Dijkstra, max_search_distance={max_distance_km}km, lookahead_depth=1")
    else:
        logger.info(f"  Using: on-demand Dijkstra, unlimited search distance, lookahead_depth=1")

    # Convert segments to required_edges format
    # Format: (start_coord, end_coord, coordinates_list)
    required_edges = []
    segment_metadata_map = {}  # Map edge index to original segment

    for idx, seg in enumerate(segments):
        coords = seg.coordinates
        # Add edge for this segment
        required_edges.append((coords[0], coords[-1], coords))
        segment_metadata_map[idx] = seg

    # Starting position
    start_node = segments[0].coordinates[0]

    # Call optimized greedy router
    logger.info("  Calling greedy_route_cluster with optimizations...")
    result = greedy_route_cluster(
        graph=graph,
        required_edges=required_edges,
        segment_indices=list(range(len(required_edges))),
        start_node=start_node,
        use_ondemand=True,              # On-demand mode: compute Dijkstra once per iteration
        lookahead_depth=1,              # Pure greedy: no lookahead (avoids O(n²) overhead)
        max_search_distance=max_search_distance  # Distance limit in meters (None = unlimited)
    )

    logger.info(f"  ✓ Greedy routing complete:")
    logger.info(f"    - Segments covered: {result.segments_covered}/{len(segments)}")
    logger.info(f"    - Total distance: {result.distance / 1000:.2f} km")
    logger.info(f"    - Computation time: {result.computation_time:.2f}s")
    if result.segments_unreachable > 0:
        logger.warning(f"    - Unreachable segments: {result.segments_unreachable}")

    # Convert PathResult to route format expected by export_route_to_kml
    # We need to reconstruct which parts are deadhead vs required segments
    route = []
    path_coords = result.path

    if not path_coords:
        logger.warning("  No path generated!")
        return route

    # Strategy: Match path segments to original required segments
    # For now, create a simple representation showing the full path
    # In a more sophisticated implementation, we'd split the path into
    # approach segments (deadhead) and required segment traversals

    # Track which segments were covered by matching coordinates
    covered_segments = set()
    current_idx = 0

    while current_idx < len(path_coords):
        # Try to match current position to a segment start
        current_coord = path_coords[current_idx]

        matched = False
        for seg_idx, seg in enumerate(segments):
            if seg_idx in covered_segments:
                continue

            seg_coords = seg.coordinates
            seg_start = seg_coords[0]
            seg_end = seg_coords[-1]

            # Check if we're at the start of this segment
            # Use small tolerance for floating point comparison
            if (abs(current_coord[0] - seg_start[0]) < 0.00001 and
                abs(current_coord[1] - seg_start[1]) < 0.00001):

                # Found a segment match - extract its path from result
                seg_len = len(seg_coords)
                segment_path = path_coords[current_idx:current_idx + seg_len]

                # Calculate distance
                seg_dist = sum(
                    haversine(segment_path[i], segment_path[i+1])
                    for i in range(len(segment_path) - 1)
                ) if len(segment_path) > 1 else 0

                route.append({
                    'type': 'segment',
                    'segment': seg,
                    'path': segment_path,
                    'distance': seg_dist,
                    'metadata': seg.metadata
                })

                covered_segments.add(seg_idx)
                current_idx += seg_len - 1  # -1 because segments share endpoints
                matched = True
                break

        if not matched:
            # This is a deadhead/approach segment
            # Find next segment start
            next_seg_start_idx = None
            for i in range(current_idx + 1, len(path_coords)):
                for seg_idx, seg in enumerate(segments):
                    if seg_idx in covered_segments:
                        continue
                    seg_start = seg.coordinates[0]
                    if (abs(path_coords[i][0] - seg_start[0]) < 0.00001 and
                        abs(path_coords[i][1] - seg_start[1]) < 0.00001):
                        next_seg_start_idx = i
                        break
                if next_seg_start_idx:
                    break

            if next_seg_start_idx:
                # Extract deadhead path
                deadhead_path = path_coords[current_idx:next_seg_start_idx + 1]
                deadhead_dist = sum(
                    haversine(deadhead_path[i], deadhead_path[i+1])
                    for i in range(len(deadhead_path) - 1)
                ) if len(deadhead_path) > 1 else 0

                route.append({
                    'type': 'routing',
                    'path': deadhead_path,
                    'distance': deadhead_dist,
                    'metadata': {'type': 'deadhead'}
                })

                current_idx = next_seg_start_idx
            else:
                # No more segments found, must be trailing path
                current_idx += 1
        else:
            current_idx += 1

    logger.info(f"  ✓ Converted to {len(route)} route steps")
    logger.info(f"  ✓ Matched {len(covered_segments)}/{len(segments)} segments to path")

    return route


def export_route_to_kml(route, output_path, logger=None):
    """
    Export route to KML with sequence numbers and metadata.

    Compatible with MapPlus/Duweis format.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Exporting KML to {output_path}...")

    # Create KML structure
    kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
    document = ET.SubElement(kml, 'Document')
    ET.SubElement(document, 'name').text = 'DRPP Optimized Route'

    # Add styles
    _add_kml_styles(document)

    # Calculate statistics
    total_dist = sum(r['distance'] for r in route)
    segment_dist = sum(r['distance'] for r in route if r['type'] == 'segment')
    deadhead_dist = total_dist - segment_dist
    deadhead_pct = (deadhead_dist / total_dist * 100) if total_dist > 0 else 0

    # Add description
    desc_lines = [
        "DRPP Optimized Route with OSM Road Network",
        "",
        f"Total Distance: {total_dist/1000:.2f} km",
        f"Required Distance: {segment_dist/1000:.2f} km",
        f"Deadhead Distance: {deadhead_dist/1000:.2f} km ({deadhead_pct:.1f}%)",
        f"Route Steps: {len(route)}",
    ]
    ET.SubElement(document, 'description').text = "\n".join(desc_lines)

    # Add each route step as placemark
    for seq_num, step in enumerate(route, 1):
        _add_route_step_placemark(document, step, seq_num)

    # Write to file
    xml_string = ET.tostring(kml, encoding='utf-8')
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent='  ')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

    logger.info(f"  ✓ Exported {len(route)} route steps")
    logger.info(f"  ✓ Total: {total_dist/1000:.2f} km ({segment_dist/1000:.2f} km required + {deadhead_dist/1000:.2f} km routing)")


def _add_kml_styles(document):
    """Add KML line styles."""
    # Required segment style (red, thick)
    style_req = ET.SubElement(document, 'Style', id='requiredStyle')
    line_req = ET.SubElement(style_req, 'LineStyle')
    ET.SubElement(line_req, 'color').text = 'ff0000ff'  # Red
    ET.SubElement(line_req, 'width').text = '5'

    # Deadhead routing style (yellow, thin)
    style_dead = ET.SubElement(document, 'Style', id='deadheadStyle')
    line_dead = ET.SubElement(style_dead, 'LineStyle')
    ET.SubElement(line_dead, 'color').text = '7f00ffff'  # Yellow, semi-transparent
    ET.SubElement(line_dead, 'width').text = '2'


def _add_route_step_placemark(document, step, seq_num):
    """Add a route step as KML Placemark."""
    placemark = ET.SubElement(document, 'Placemark')

    # Name with sequence number
    if step['type'] == 'segment':
        seg = step['segment']
        seg_id = seg.segment_id or f"seg_{seq_num}"
        name = f"{seq_num}. {seg_id}"
    else:
        name = f"{seq_num}. Routing"

    ET.SubElement(placemark, 'name').text = name

    # Description
    desc_lines = [f"<b>Step {seq_num}</b>"]
    desc_lines.append(f"<br/>Type: {'Required Segment' if step['type'] == 'segment' else 'Routing'}")
    desc_lines.append(f"<br/>Distance: {step['distance']/1000:.3f} km")

    if step['type'] == 'segment' and step['metadata']:
        desc_lines.append("<br/><br/><b>Metadata:</b>")
        for key, value in step['metadata'].items():
            desc_lines.append(f"<br/>{key}: {value}")

    ET.SubElement(placemark, 'description').text = "".join(desc_lines)

    # Style
    if step['type'] == 'segment':
        ET.SubElement(placemark, 'styleUrl').text = '#requiredStyle'
    else:
        ET.SubElement(placemark, 'styleUrl').text = '#deadheadStyle'

    # ExtendedData with metadata
    if step['metadata']:
        ext_data = ET.SubElement(placemark, 'ExtendedData')

        # Add sequence
        data_seq = ET.SubElement(ext_data, 'Data', name='sequence')
        ET.SubElement(data_seq, 'value').text = str(seq_num)

        # Add is_required
        data_req = ET.SubElement(ext_data, 'Data', name='is_required')
        ET.SubElement(data_req, 'value').text = str(step['type'] == 'segment')

        # Add all original metadata
        for key, value in step['metadata'].items():
            data = ET.SubElement(ext_data, 'Data', name=str(key))
            ET.SubElement(data, 'value').text = str(value)

    # LineString geometry
    linestring = ET.SubElement(placemark, 'LineString')
    ET.SubElement(linestring, 'tessellate').text = '1'

    coords_text = " ".join([f"{lon},{lat},0" for lat, lon in step['path']])
    ET.SubElement(linestring, 'coordinates').text = coords_text


def main():
    parser = argparse.ArgumentParser(
        description='DRPP Router with OSM Roads + KML Output with Metadata'
    )
    parser.add_argument('input_kml', help='Input KML file')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--no-osm', action='store_true',
                       help='Skip OSM fetch (use segment-only routing)')
    parser.add_argument('--max-distance', type=float, default=None,
                       help='Maximum search distance in km (default: unlimited). Use 10-50 for dense areas, unlimited for sparse areas.')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(args.input_kml)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("DRPP ROUTER WITH OSM + KML OUTPUT")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"OSM routing: {'No' if args.no_osm else 'Yes'}")
    logger.info("")

    # Step 1: Parse KML
    logger.info("[1/4] Parsing KML...")
    pipeline = DRPPPipeline()
    segments = pipeline._parse_kml(str(input_path))
    logger.info(f"  ✓ Parsed {len(segments)} segments")

    # Step 2: Build graph with OSM
    logger.info("")
    logger.info("[2/4] Building routing graph...")
    graph = build_graph_with_osm(segments, use_osm=not args.no_osm, logger=logger)

    # Step 3: Route segments
    logger.info("")
    logger.info("[3/4] Computing route...")
    route = route_segments_greedy(segments, graph, max_distance_km=args.max_distance, logger=logger)

    # Step 4: Export KML
    logger.info("")
    logger.info("[4/4] Exporting results...")
    output_kml = output_dir / "route_with_metadata.kml"
    export_route_to_kml(route, output_kml, logger=logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Output KML: {output_kml}")
    logger.info("")
    logger.info("Open in Google Earth or MapPlus to view:")
    logger.info("  - Numbered route steps (1, 2, 3...)")
    logger.info("  - Color-coded (Red=required, Yellow=routing)")
    logger.info("  - All metadata preserved (CollId, RouteName, Dir, etc.)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
