"""
KML/KMZ export with full metadata preservation.

This module exports DRPP solutions to KML format with:
- All original ExtendedData from input KML
- Route sequencing and statistics
- Color-coding by segment type (required/deadhead)
- Direction arrows and labels
"""

from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .industry_drpp_solver import DRPPSolution
from .topology import TopologyEdge, Coordinate
from .eulerian_solver import EulerianTour


class KMLExporter:
    """
    Export DRPP solution to KML format with full metadata preservation.
    """

    def __init__(self, solution: DRPPSolution):
        """
        Initialize KML exporter.

        Args:
            solution: Complete DRPP solution
        """
        self.solution = solution

    def export_kml(self, output_path: str, include_styles: bool = True) -> None:
        """
        Export solution to KML file.

        Args:
            output_path: Path to output KML file
            include_styles: Whether to include color styles
        """
        # Create KML root
        kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
        document = ET.SubElement(kml, 'Document')

        # Add document metadata
        ET.SubElement(document, 'name').text = 'DRPP Optimized Route'
        ET.SubElement(document, 'description').text = self._generate_description()

        # Add styles if requested
        if include_styles:
            self._add_styles(document)

        # Add route segments as Placemarks
        for i, edge in enumerate(self.solution.tour.edges):
            self._add_placemark(document, edge, i + 1)

        # Add summary statistics placemark
        self._add_summary_placemark(document)

        # Write to file
        xml_string = ET.tostring(kml, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_string).toprettyxml(indent='  ')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

    def _generate_description(self) -> str:
        """Generate document description with statistics."""
        lines = [
            "Industry-Standard DRPP Optimized Route",
            "",
            f"Total Distance: {self.solution.total_distance_km:.2f} km",
            f"Required Distance: {self.solution.required_distance_km:.2f} km",
            f"Deadhead Distance: {self.solution.deadhead_distance_km:.2f} km",
            f"Deadhead Percentage: {self.solution.deadhead_percentage:.1f}%",
            f"Number of Segments: {self.solution.num_segments}",
            "",
            "Route Messages:",
        ]

        for msg in self.solution.messages:
            lines.append(f"  - {msg}")

        return "\n".join(lines)

    def _add_styles(self, document: ET.Element) -> None:
        """Add KML styles for different segment types."""
        # Style for required segments
        style_required = ET.SubElement(document, 'Style', id='requiredStyle')
        line_style_req = ET.SubElement(style_required, 'LineStyle')
        ET.SubElement(line_style_req, 'color').text = 'ff0000ff'  # Red (AABBGGRR)
        ET.SubElement(line_style_req, 'width').text = '4'

        # Style for deadhead segments
        style_deadhead = ET.SubElement(document, 'Style', id='deadheadStyle')
        line_style_dead = ET.SubElement(style_deadhead, 'LineStyle')
        ET.SubElement(line_style_dead, 'color').text = '7f00ffff'  # Yellow, semi-transparent
        ET.SubElement(line_style_dead, 'width').text = '2'

        # Style for balancing edges
        style_balance = ET.SubElement(document, 'Style', id='balancingStyle')
        line_style_bal = ET.SubElement(style_balance, 'LineStyle')
        ET.SubElement(line_style_bal, 'color').text = '7fff0000'  # Blue, semi-transparent
        ET.SubElement(line_style_bal, 'width').text = '2'

    def _add_placemark(self, document: ET.Element, edge: TopologyEdge, sequence: int) -> None:
        """
        Add a Placemark for a route edge.

        Args:
            document: KML Document element
            edge: Topology edge
            sequence: Sequence number in route
        """
        placemark = ET.SubElement(document, 'Placemark')

        # Name (sequence + segment ID)
        name = f"{sequence}. "
        if edge.segment_id:
            name += edge.segment_id
        else:
            name += f"{'Required' if edge.required else 'Deadhead'}"

        ET.SubElement(placemark, 'name').text = name

        # Description with metadata
        description = self._generate_edge_description(edge, sequence)
        ET.SubElement(placemark, 'description').text = description

        # Style reference
        if edge.required:
            ET.SubElement(placemark, 'styleUrl').text = '#requiredStyle'
        elif edge.metadata.get('type') == 'balancing_edge':
            ET.SubElement(placemark, 'styleUrl').text = '#balancingStyle'
        else:
            ET.SubElement(placemark, 'styleUrl').text = '#deadheadStyle'

        # ExtendedData with all metadata
        if edge.metadata:
            extended_data = ET.SubElement(placemark, 'ExtendedData')

            # Add all original metadata
            for key, value in edge.metadata.items():
                data = ET.SubElement(extended_data, 'Data', name=str(key))
                ET.SubElement(data, 'value').text = str(value)

            # Add computed fields
            data_seq = ET.SubElement(extended_data, 'Data', name='sequence')
            ET.SubElement(data_seq, 'value').text = str(sequence)

            data_req = ET.SubElement(extended_data, 'Data', name='is_required')
            ET.SubElement(data_req, 'value').text = str(edge.required)

            data_dist = ET.SubElement(extended_data, 'Data', name='distance_m')
            ET.SubElement(data_dist, 'value').text = f"{edge.cost:.2f}"

        # LineString geometry
        linestring = ET.SubElement(placemark, 'LineString')
        ET.SubElement(linestring, 'tessellate').text = '1'

        coordinates_text = self._format_coordinates(edge.coordinates)
        ET.SubElement(linestring, 'coordinates').text = coordinates_text

    def _generate_edge_description(self, edge: TopologyEdge, sequence: int) -> str:
        """Generate HTML description for edge."""
        lines = [
            f"<h3>Segment {sequence}</h3>",
            f"<p><b>Type:</b> {'Required' if edge.required else 'Deadhead'}</p>",
            f"<p><b>Distance:</b> {edge.cost:.2f} m ({edge.cost / 1000:.3f} km)</p>",
        ]

        # Add metadata fields
        if edge.metadata:
            lines.append("<h4>Metadata:</h4>")
            lines.append("<ul>")

            # Common fields first
            common_fields = ['CollId', 'RouteName', 'Dir', 'LengthFt', 'Region', 'Juris']
            for field in common_fields:
                if field in edge.metadata:
                    lines.append(f"<li><b>{field}:</b> {edge.metadata[field]}</li>")

            # Other fields
            for key, value in edge.metadata.items():
                if key not in common_fields:
                    lines.append(f"<li><b>{key}:</b> {value}</li>")

            lines.append("</ul>")

        return "\n".join(lines)

    def _format_coordinates(self, coordinates: List[Coordinate]) -> str:
        """
        Format coordinates for KML LineString.

        KML format: lon,lat,alt (space-separated)

        Args:
            coordinates: List of (lat, lon) tuples

        Returns:
            Formatted coordinate string
        """
        coord_strings = []
        for lat, lon in coordinates:
            coord_strings.append(f"{lon},{lat},0")

        return " ".join(coord_strings)

    def _add_summary_placemark(self, document: ET.Element) -> None:
        """Add a summary Placemark with route statistics."""
        placemark = ET.SubElement(document, 'Placemark')
        ET.SubElement(placemark, 'name').text = 'Route Summary'
        ET.SubElement(placemark, 'description').text = self._generate_description()

        # Add point at start of route (if available)
        if self.solution.route_coordinates:
            lat, lon = self.solution.route_coordinates[0]
            point = ET.SubElement(placemark, 'Point')
            ET.SubElement(point, 'coordinates').text = f"{lon},{lat},0"


def export_drpp_to_kml(solution: DRPPSolution, output_path: str) -> None:
    """
    Convenience function to export DRPP solution to KML.

    Args:
        solution: DRPP solution
        output_path: Path to output KML file
    """
    exporter = KMLExporter(solution)
    exporter.export_kml(output_path)
