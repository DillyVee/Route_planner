#!/usr/bin/env python3
"""
DRPP Router with OSM Road Network + KML Output with Metadata

This script:
1. Parses your KML with all metadata (CollId, RouteName, Dir, etc.)
2. Fetches Pennsylvania road network from OpenStreetMap
3. Routes between disconnected segments using REAL ROADS
4. Exports KML with sequence numbers and all metadata preserved

Output is compatible with MapPlus/Duweis.

Usage:
    python run_drpp_with_osm_kml.py PA_2025_Region2.kml --output-dir output

Options:
    --no-osm         Skip OSM fetch (use straight-line routing)
    --verbose        Show detailed progress
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


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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
    for seg in segments:
        coords = seg.coordinates
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dist = haversine(p1, p2)

            if seg.forward_required:
                graph.add_edge(p1, p2, dist)
            if seg.backward_required:
                graph.add_edge(p2, p1, dist)

    base_nodes = len(graph.id_to_node)
    logger.info(f"  ✓ Base graph: {base_nodes:,} nodes from segments")

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

                osm_edges = 0
                for way in osm_ways:
                    coords = way['geometry']
                    oneway = way.get('oneway', False)

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
            else:
                logger.warning("  No OSM roads found in bbox")

        except Exception as e:
            logger.error(f"  OSM fetch failed: {e}")
            logger.warning("  Continuing with segment-only routing")

    return graph


def route_segments_greedy(segments, graph, logger=None):
    """
    Route through segments using greedy on-demand algorithm.

    Returns:
        List of (segment, path, distance) tuples in route order
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Running greedy routing algorithm...")

    route = []
    covered = set()
    current_pos = segments[0].coordinates[0]  # Start at first segment

    iteration = 0
    while len(covered) < len(segments):
        iteration += 1

        if iteration % 100 == 0:
            logger.info(f"  Progress: {len(covered)}/{len(segments)} segments covered")

        # Find nearest uncovered segment
        best_seg = None
        best_dist = float('inf')
        best_path = None

        for idx, seg in enumerate(segments):
            if idx in covered:
                continue

            seg_start = seg.coordinates[0]

            # Find shortest path from current position to segment start
            try:
                path, dist = graph.shortest_path(current_pos, seg_start)

                if dist < best_dist:
                    best_dist = dist
                    best_seg = (idx, seg)
                    best_path = path
            except:
                # No path found
                continue

        if best_seg is None:
            # No more reachable segments
            unreachable = len(segments) - len(covered)
            logger.warning(f"  {unreachable} segments unreachable from current position")
            break

        idx, seg = best_seg

        # Add routing to segment (if not already there)
        if best_dist > 0.001:  # More than 1 meter away
            route.append({
                'type': 'routing',
                'path': best_path,
                'distance': best_dist,
                'metadata': {'type': 'deadhead'}
            })

        # Add segment itself
        seg_dist = sum(
            haversine(seg.coordinates[i], seg.coordinates[i+1])
            for i in range(len(seg.coordinates) - 1)
        )

        route.append({
            'type': 'segment',
            'segment': seg,
            'path': seg.coordinates,
            'distance': seg_dist,
            'metadata': seg.metadata
        })

        covered.add(idx)
        current_pos = seg.coordinates[-1]  # End of segment

    logger.info(f"  ✓ Routed {len(covered)}/{len(segments)} segments")

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
    route = route_segments_greedy(segments, graph, logger=logger)

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
