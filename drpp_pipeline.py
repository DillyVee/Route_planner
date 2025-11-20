"""
Complete DRPP Visualization and Solving Pipeline

This module provides a comprehensive solution for:
1. Parsing KML with segment IDs and directionality
2. Solving DRPP with multiple algorithms (RFCS, Greedy, Hungarian)
3. Generating rich visualizations with:
   - Color-coded segments by requirement type
   - Segment ID labels
   - Route step numbering
   - Multiple output formats (SVG, PNG, HTML, GeoJSON)

Usage:
    from drpp_pipeline import DRPPPipeline

    pipeline = DRPPPipeline()
    results = pipeline.run(
        kml_file='your_segments.kml',
        algorithm='rfcs',  # or 'greedy', 'v4', 'hungarian'
        output_formats=['html', 'geojson', 'png']
    )
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SegmentRequirement:
    """Defines traversal requirements for a segment.

    Attributes:
        segment_id: Unique segment identifier from KML
        forward_required: Must traverse start->end
        backward_required: Must traverse end->start
        one_way: Only one direction is allowed
        coordinates: List of (lat, lon) points
        metadata: Additional KML data (speed, name, etc.)
    """

    segment_id: str
    forward_required: bool
    backward_required: bool
    one_way: bool = False
    coordinates: List[Tuple[float, float]] = None
    metadata: Dict = None

    @property
    def is_two_way_required(self) -> bool:
        """Both directions must be traversed."""
        return self.forward_required and self.backward_required

    @property
    def required_traversals(self) -> int:
        """Number of required traversals (0, 1, or 2)."""
        return int(self.forward_required) + int(self.backward_required)


@dataclass
class RouteStep:
    """A single step in the computed route.

    Attributes:
        step_number: Sequential step number (1, 2, 3...)
        segment_id: Which segment is being traversed
        direction: 'forward' or 'backward'
        is_deadhead: Not covering a required direction (routing between segments)
        coordinates: Path coordinates for this step
        distance_meters: Length of this step
    """

    step_number: int
    segment_id: Optional[str]
    direction: str
    is_deadhead: bool
    coordinates: List[Tuple[float, float]]
    distance_meters: float


class DRPPPipeline:
    """Complete pipeline for DRPP solving and visualization."""

    def __init__(self, log_level=logging.INFO):
        """Initialize pipeline.

        Args:
            log_level: Logging level for pipeline operations
        """
        logging.basicConfig(level=log_level)
        self.logger = logger

        self.segments: List[SegmentRequirement] = []
        self.graph = None
        self.route_steps: List[RouteStep] = []
        self.routing_results = []  # Store actual routing results from algorithms

    def run(
        self,
        kml_file: Path,
        algorithm: str = "rfcs",
        output_dir: Path = Path("./output"),
        output_formats: List[str] = ["html", "geojson"],
        visualization_config: Optional[Dict] = None,
    ) -> Dict:
        """Run complete DRPP pipeline.

        Args:
            kml_file: Path to KML input file
            algorithm: 'rfcs', 'greedy', 'v4', or 'hungarian'
            output_dir: Directory for output files
            output_formats: List of formats: 'html', 'svg', 'png', 'geojson'
            visualization_config: Optional visualization settings

        Returns:
            Dict with:
                - 'route_steps': List of RouteStep objects
                - 'total_distance': Total route distance in meters
                - 'coverage': Percentage of required segments covered
                - 'output_files': Dict of generated output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 80)
        self.logger.info("DRPP PIPELINE - Complete Visualization & Solving System")
        self.logger.info("=" * 80)

        # Phase 1: Parse KML
        self.logger.info("\n[1/5] Parsing KML and extracting segments...")
        self.segments = self._parse_kml(kml_file)
        self.logger.info(f"  ✓ Loaded {len(self.segments)} segments")
        self.logger.info(
            f"  ✓ Forward-only: {sum(1 for s in self.segments if s.forward_required and not s.backward_required)}"
        )
        self.logger.info(
            f"  ✓ Backward-only: {sum(1 for s in self.segments if s.backward_required and not s.forward_required)}"
        )
        self.logger.info(
            f"  ✓ Two-way required: {sum(1 for s in self.segments if s.is_two_way_required)}"
        )

        # Phase 2: Build graph
        self.logger.info("\n[2/5] Building directed graph...")
        self.graph = self._build_graph()
        self.logger.info(f"  ✓ Graph has {len(self.graph.id_to_node)} nodes")

        # Phase 3: Solve DRPP
        self.logger.info(f"\n[3/5] Solving DRPP with algorithm: {algorithm.upper()}...")
        self.route_steps = self._solve_drpp(algorithm)
        self.logger.info(f"  ✓ Generated route with {len(self.route_steps)} steps")

        # Phase 4: Generate visualizations
        self.logger.info("\n[4/5] Generating visualizations...")
        output_files = self._generate_visualizations(
            output_dir, output_formats, visualization_config
        )
        for fmt, path in output_files.items():
            self.logger.info(f"  ✓ {fmt.upper()}: {path}")

        # Phase 5: Validate and report
        self.logger.info("\n[5/5] Validation and summary...")
        stats = self._compute_statistics()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total distance: {stats['total_distance'] / 1000:.1f} km")
        self.logger.info(f"Required coverage: {stats['coverage']:.1f}%")
        self.logger.info(
            f"Deadhead distance: {stats['deadhead_distance'] / 1000:.1f} km ({stats['deadhead_percent']:.1f}%)"
        )
        self.logger.info(f"Output files: {len(output_files)}")
        self.logger.info("=" * 80)

        return {
            "route_steps": self.route_steps,
            "total_distance": stats["total_distance"],
            "coverage": stats["coverage"],
            "statistics": stats,
            "output_files": output_files,
        }

    def _parse_kml(self, kml_file: Path) -> List[SegmentRequirement]:
        """Parse KML and extract segment requirements.

        Reads KML format and extracts:
        - Segment IDs (from CollId or name)
        - Directionality (one-way vs two-way)
        - Required traversals for each direction
        - Coordinates
        - MapPlus/Duweis roadway metadata

        Supports MapPlus format with:
        - MapPlusCustomFeatureClass schema (CollId, RouteName, Dir, etc.)
        - MapPlusSystemData schema (LABEL_EXPR, LABEL_TEXT)
        """
        import xml.etree.ElementTree as ET

        from osm_speed_integration import snap_coord, snap_coords_list

        self.logger.info(f"  Parsing {kml_file}...")

        try:
            tree = ET.parse(kml_file)
        except ET.ParseError as e:
            # Try to fix common XML issues
            self.logger.warning(f"XML parsing error: {e}, attempting fixes...")
            import re

            with open(kml_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)
            content = re.sub(r"&(?!(amp|lt|gt|quot|apos);)", "&amp;", content)
            from io import StringIO

            tree = ET.parse(StringIO(content))

        root = tree.getroot()
        segments = []
        mapplus_format_detected = False

        for idx, pm in enumerate(root.findall(".//{http://www.opengis.net/kml/2.2}Placemark")):
            ls = pm.find(".//{http://www.opengis.net/kml/2.2}LineString")
            if ls is None:
                continue

            # Extract coordinates
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

            # Extract metadata from ExtendedData
            metadata = {}
            segment_id = None
            oneway = None

            ext = pm.find("{http://www.opengis.net/kml/2.2}ExtendedData")
            if ext is not None:
                # Look for MapPlus schema data
                for schema_data in ext.findall("{http://www.opengis.net/kml/2.2}SchemaData"):
                    schema_url = schema_data.get("schemaUrl", "")

                    # MapPlusCustomFeatureClass - main road segment attributes
                    if "MapPlusCustomFeatureClass" in schema_url:
                        mapplus_format_detected = True
                        for simple_data in schema_data.findall(
                            "{http://www.opengis.net/kml/2.2}SimpleData"
                        ):
                            field_name = simple_data.get("name", "")
                            field_value = simple_data.text or ""

                            # Store all fields in metadata
                            metadata[field_name] = field_value

                            # Extract specific fields for routing logic
                            if field_name == "CollId":
                                segment_id = field_value
                            elif field_name == "Dir":
                                # Dir field might indicate direction
                                # Common values: 'N', 'S', 'E', 'W', 'NB', 'SB', 'EB', 'WB'
                                # For now, preserve in metadata
                                metadata["direction_code"] = field_value
                            elif field_name == "RouteName":
                                metadata["route_name"] = field_value
                            elif field_name == "LengthFt":
                                try:
                                    metadata["length_ft"] = float(field_value)
                                except (ValueError, TypeError):
                                    pass
                            elif field_name in [
                                "Region",
                                "Juris",
                                "CntyCode",
                                "StRtNo",
                                "SegNo",
                                "BegM",
                                "EndM",
                                "IsPilot",
                                "Collected",
                            ]:
                                # Store roadway metadata
                                metadata[field_name.lower()] = field_value

                    # MapPlusSystemData - label information
                    elif "MapPlusSystemData" in schema_url:
                        for simple_data in schema_data.findall(
                            "{http://www.opengis.net/kml/2.2}SimpleData"
                        ):
                            field_name = simple_data.get("name", "")
                            field_value = simple_data.text or ""
                            metadata[f"label_{field_name.lower()}"] = field_value

                # Fallback: check for generic extended data (non-schema)
                if not mapplus_format_detected:
                    for elem in ext.iter():
                        tag = elem.tag.split("}")[-1].lower() if isinstance(elem.tag, str) else ""
                        txt = (elem.text or "").strip()

                        # Check for oneway indicators
                        if tag in ["oneway", "one_way", "one-way", "is_one_way"]:
                            txt_lower = txt.lower()
                            if txt_lower in ("yes", "true", "1", "y"):
                                oneway = True
                            elif txt_lower in ("no", "false", "0", "n"):
                                oneway = False

                        # Store all metadata
                        if txt:
                            metadata[tag] = txt

            # Use name as fallback for segment_id
            if segment_id is None:
                name = pm.find("{http://www.opengis.net/kml/2.2}name")
                segment_id = (
                    name.text.strip() if name is not None and name.text else f"seg_{idx:05d}"
                )

            # Determine required traversals
            # For roadway surveys, typically:
            # - All segments are required in at least one direction
            # - Most are two-way unless specifically marked one-way
            # - Use 'Dir' field or 'oneway' field if available

            if oneway is True:
                # Explicitly one-way
                forward_required = True
                backward_required = False
            elif oneway is False:
                # Explicitly two-way
                forward_required = True
                backward_required = True
            else:
                # Default behavior for roadway surveys:
                # Assume both directions required unless specified otherwise
                # This ensures complete coverage for survey routes
                forward_required = True
                backward_required = True

            segments.append(
                SegmentRequirement(
                    segment_id=segment_id,
                    forward_required=forward_required,
                    backward_required=backward_required,
                    one_way=(oneway is True),
                    coordinates=coords,
                    metadata=metadata,
                )
            )

        if not segments:
            raise ValueError("No valid segments found in KML file")

        self.logger.info(f"  Parsed {len(segments)} segments")
        if mapplus_format_detected:
            self.logger.info("  ✓ MapPlus/Duweis format detected - preserving roadway metadata")
            # Show sample of what was extracted
            if segments and segments[0].metadata:
                sample_fields = list(segments[0].metadata.keys())[:5]
                self.logger.info(f"  ✓ Extracted fields: {', '.join(sample_fields)}...")

        return segments

    def _build_graph(self):
        """Build directed graph from segments."""
        # Import from standalone module to avoid PyQt6 dependency
        try:
            from graph_core import DirectedGraph, haversine
        except ImportError:
            # Fallback to Route_Planner (requires PyQt6)
            from Route_Planner import DirectedGraph, haversine

        graph = DirectedGraph()

        # Build graph from all segment coordinates
        for segment in self.segments:
            coords = segment.coordinates
            if not coords or len(coords) < 2:
                continue

            # Add edges between consecutive points
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]

                # Calculate distance between points
                dist = haversine(start, end)

                # Add forward edge
                if segment.forward_required or not segment.one_way:
                    graph.add_edge(start, end, dist)

                # Add backward edge for two-way segments
                if segment.backward_required or not segment.one_way:
                    graph.add_edge(end, start, dist)

        # CRITICAL FIX: Add deadhead edges between all segment endpoints
        # This ensures disconnected segments can be reached
        self._add_deadhead_edges_to_graph(graph, haversine)

        self.logger.info(f"  Built graph with {len(graph.id_to_node)} nodes")
        return graph

    def _add_deadhead_edges_to_graph(self, graph, haversine_func):
        """Add connecting edges between all segment endpoints for reachability.

        This is essential for DRPP: segments may not be directly connected,
        so we need to add shortest-path edges between them.

        Args:
            graph: DirectedGraph to add edges to
            haversine_func: Function to calculate distance
        """
        # Get all unique segment endpoints
        endpoints = set()
        for segment in self.segments:
            if segment.coordinates and len(segment.coordinates) >= 2:
                endpoints.add(segment.coordinates[0])  # Start
                endpoints.add(segment.coordinates[-1])  # End

        endpoints_list = list(endpoints)
        num_endpoints = len(endpoints_list)

        if num_endpoints <= 1:
            return  # Nothing to connect

        self.logger.info(f"  Adding deadhead edges between {num_endpoints} segment endpoints...")

        # For each pair of endpoints, add direct edge (straight-line distance)
        # This ensures all segments are reachable
        # Limit to reasonable number to avoid explosion
        max_connections = min(20, num_endpoints - 1)

        deadhead_count = 0
        for i, point1 in enumerate(endpoints_list):
            # Find nearest neighbors for this endpoint
            neighbors = []
            for j, point2 in enumerate(endpoints_list):
                if i == j:
                    continue
                dist = haversine_func(point1, point2)
                neighbors.append((dist, point2))

            # Sort by distance and connect to nearest N
            neighbors.sort()
            for dist, point2 in neighbors[:max_connections]:
                # Add edge (DirectedGraph.add_edge avoids duplicates)
                graph.add_edge(point1, point2, dist)
                deadhead_count += 1

        self.logger.info(f"  Added {deadhead_count} deadhead edges for full connectivity")

    def _solve_drpp(self, algorithm: str) -> List[RouteStep]:
        """Solve DRPP with specified algorithm."""
        # Check algorithm availability
        V4_AVAILABLE = False
        GREEDY_AVAILABLE = False
        RFCS_AVAILABLE = False

        try:
            from drpp_core import parallel_cluster_routing
            V4_AVAILABLE = True
        except ImportError:
            pass

        try:
            from legacy.parallel_processing_addon_greedy import parallel_cluster_routing as parallel_cluster_routing_greedy
            GREEDY_AVAILABLE = True
        except ImportError:
            pass

        try:
            from legacy.parallel_processing_addon_rfcs import parallel_cluster_routing as parallel_cluster_routing_rfcs
            RFCS_AVAILABLE = True
        except ImportError:
            pass

        # Build required edges list from segments
        required_edges = []
        segment_id_map = {}  # Map edge index to segment ID

        for segment in self.segments:
            coords = segment.coordinates
            if not coords or len(coords) < 2:
                continue

            # Add forward edge if required
            if segment.forward_required:
                edge_idx = len(required_edges)
                required_edges.append((coords[0], coords[-1], coords, "forward"))
                segment_id_map[edge_idx] = (segment.segment_id, "forward")

            # Add backward edge if required
            if segment.backward_required:
                edge_idx = len(required_edges)
                reversed_coords = list(reversed(coords))
                required_edges.append((coords[-1], coords[0], reversed_coords, "backward"))
                segment_id_map[edge_idx] = (segment.segment_id, "backward")

        self.logger.info(f"  Required edges: {len(required_edges)}")

        # Select algorithm
        if algorithm == "rfcs" and RFCS_AVAILABLE:
            self.logger.info("  Using RFCS algorithm (Route-First, Cluster-Second)")
            from legacy.parallel_processing_addon_rfcs import parallel_cluster_routing
        elif algorithm == "v4" and V4_AVAILABLE:
            self.logger.info("  Using V4 greedy algorithm (Production)")
            from drpp_core import parallel_cluster_routing
        elif algorithm == "greedy" and GREEDY_AVAILABLE:
            self.logger.info("  Using legacy greedy algorithm")
            from legacy.parallel_processing_addon_greedy import parallel_cluster_routing
        else:
            self.logger.info("  Using Hungarian algorithm (default)")
            from legacy.parallel_processing_addon import parallel_cluster_routing

        # For simplicity, use single cluster for now (full route)
        clusters = {0: list(range(len(required_edges)))}
        cluster_order = [0]

        # Get start node
        start_node = self.segments[0].coordinates[0] if self.segments else None
        if start_node is None:
            raise ValueError("No starting point available")

        # Route through all segments
        if V4_AVAILABLE and algorithm == "v4":
            # Use V4 with on-demand mode for large datasets
            from drpp_core import greedy_route_cluster

            result = greedy_route_cluster(
                graph=self.graph,
                required_edges=required_edges,
                segment_indices=list(range(len(required_edges))),
                start_node=start_node,
                use_ondemand=True,  # Enable on-demand mode
            )
            results = [result]
        else:
            # Use parallel routing
            results = parallel_cluster_routing(
                graph=self.graph,
                required_edges=required_edges,
                clusters=clusters,
                cluster_order=cluster_order,
                start_node=start_node,
                num_workers=1,
            )

        # Store the actual routing results for statistics computation
        # The results contain the actual path and distance information
        self.routing_results = results

        # Log actual results for verification
        total_dist = 0
        total_covered = 0
        for result in results:
            if hasattr(result, "distance"):
                # V4 PathResult format
                total_dist += result.distance
                total_covered += result.segments_covered
                self.logger.info(
                    f"  Result: {result.distance / 1000:.1f}km, "
                    f"{result.segments_covered} segments covered"
                )
            else:
                # Legacy format: (path, distance, cluster_id)
                path, dist, cid = result
                total_dist += dist
                self.logger.info(f"  Result: {dist / 1000:.1f}km")

        self.logger.info(f"  Total: {total_dist / 1000:.1f}km, {total_covered} segments")

        # Convert routing_results to RouteSteps for visualization
        route_steps = self._convert_results_to_steps(results)
        return route_steps

    def _convert_results_to_steps(self, results: List) -> List[RouteStep]:
        """Convert routing results to RouteStep objects for visualization.

        Args:
            results: List of PathResult objects or legacy tuples

        Returns:
            List of RouteStep objects with full path information
        """
        route_steps = []
        step_number = 1

        for result in results:
            if hasattr(result, "path"):
                # V4 PathResult format - has full path coordinates
                path_coords = result.path
                total_distance = result.distance

                # For now, create a single step per result
                # In a more detailed implementation, this could break down the path
                # into individual segment traversals
                if path_coords and len(path_coords) > 0:
                    step = RouteStep(
                        step_number=step_number,
                        segment_id=f"cluster_{result.cluster_id}",
                        direction="forward",
                        is_deadhead=False,
                        coordinates=path_coords,
                        distance_meters=total_distance
                    )
                    route_steps.append(step)
                    step_number += 1

            else:
                # Legacy format: (path, distance, cluster_id)
                path, dist, cid = result
                if path and len(path) > 0:
                    step = RouteStep(
                        step_number=step_number,
                        segment_id=f"cluster_{cid}",
                        direction="forward",
                        is_deadhead=False,
                        coordinates=path,
                        distance_meters=dist
                    )
                    route_steps.append(step)
                    step_number += 1

        return route_steps

    def _generate_visualizations(
        self, output_dir: Path, formats: List[str], config: Optional[Dict]
    ) -> Dict[str, Path]:
        """Generate all requested visualization formats."""
        from drpp_visualization import DRPPVisualizer, VisualizationConfig

        # Create visualizer
        viz_config = VisualizationConfig(**(config or {}))
        visualizer = DRPPVisualizer(viz_config)

        output_files = {}

        # Generate HTML map
        if "html" in formats:
            html_path = output_dir / "route_map.html"
            visualizer.generate_html_map(self.segments, self.route_steps, html_path)
            output_files["html"] = html_path

        # Generate GeoJSON
        if "geojson" in formats:
            geojson_path = output_dir / "route_data.geojson"
            visualizer.generate_geojson(self.segments, self.route_steps, geojson_path)
            output_files["geojson"] = geojson_path

        # Generate SVG
        if "svg" in formats:
            svg_path = output_dir / "route_map.svg"
            visualizer.generate_svg(self.segments, self.route_steps, svg_path)
            output_files["svg"] = svg_path

        # PNG would require additional rendering (e.g., via CairoSVG or similar)
        if "png" in formats:
            self.logger.warning(
                "PNG generation requires additional dependencies (not yet implemented)"
            )

        return output_files

    def _compute_statistics(self) -> Dict:
        """Compute route statistics."""
        # Use routing_results instead of route_steps
        if not self.routing_results:
            return {
                "total_distance": 0,
                "coverage": 0,
                "deadhead_distance": 0,
                "deadhead_percent": 0,
            }

        # Extract statistics from routing results
        total_distance = 0
        total_segments_covered = 0
        total_segments_unreachable = 0

        for result in self.routing_results:
            if hasattr(result, "distance"):
                # V4 PathResult format
                total_distance += result.distance
                total_segments_covered += result.segments_covered
                total_segments_unreachable += result.segments_unreachable
            else:
                # Legacy format: (path, distance, cluster_id)
                path, dist, cid = result
                total_distance += dist
                # For legacy format, we don't have coverage info easily available

        # Calculate coverage based on segments
        # Count required traversals (forward + backward)
        required_count = sum(s.required_traversals for s in self.segments)
        coverage = (
            (total_segments_covered / required_count * 100) if required_count > 0 else 0
        )

        # Note: Deadhead calculation requires detailed path tracing
        # For now, we estimate it based on unreachable segments
        # In a full implementation, this would come from route_steps
        return {
            "total_distance": total_distance,
            "coverage": coverage,
            "deadhead_distance": 0,  # TODO: Implement with full path tracing
            "deadhead_percent": 0,  # TODO: Implement with full path tracing
            "segments_covered": total_segments_covered,
            "segments_unreachable": total_segments_unreachable,
            "required_count": required_count,
        }
