"""
DRPP Visualization Module

Generates rich, color-coded visualizations of DRPP solutions with:
- Segment ID labels
- Route step numbering
- Direction-based color coding
- Multiple output formats (HTML, SVG, PNG, GeoJSON)
- Support for very large datasets

Author: Production-ready visualization system
Version: 1.0.0
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import colorsys


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance.

    Attributes:
        colors: Dict mapping requirement types to colors
        segment_label_size: Font size for segment IDs
        step_label_size: Font size for route step numbers
        line_width: Width of segment lines
        show_segment_ids: Whether to display segment IDs
        show_step_numbers: Whether to display route step numbers
        show_direction_arrows: Whether to show arrows for direction
    """
    colors: Dict[str, str] = None
    segment_label_size: int = 10
    step_label_size: int = 12
    line_width: int = 3
    show_segment_ids: bool = True
    show_step_numbers: bool = True
    show_direction_arrows: bool = True

    def __post_init__(self):
        """Set default colors if not provided."""
        if self.colors is None:
            self.colors = {
                'forward_required': '#FF0000',      # Red
                'backward_required': '#0000FF',     # Blue
                'both_required': '#9900FF',         # Purple
                'not_required': '#CCCCCC',          # Light gray
                'deadhead': '#FFA500',              # Orange
                'route_path': '#00FF00'             # Green
            }


class DRPPVisualizer:
    """Generates visualizations for DRPP solutions."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer.

        Args:
            config: Visualization configuration (uses defaults if None)
        """
        self.config = config or VisualizationConfig()

    def generate_html_map(self,
                         segments: List,
                         route_steps: List,
                         output_file: Path) -> Path:
        """Generate interactive HTML map using Folium/Leaflet.

        This creates a zoomable, pannable map that can handle large datasets.

        Args:
            segments: List of SegmentRequirement objects
            route_steps: List of RouteStep objects
            output_file: Path to save HTML file

        Returns:
            Path to generated HTML file
        """
        try:
            import folium
            from folium import plugins
        except ImportError:
            raise ImportError(
                "Folium required for HTML maps. Install with: pip install folium"
            )

        # Calculate map center
        all_coords = []
        for segment in segments:
            if segment.coordinates:
                all_coords.extend(segment.coordinates)

        if not all_coords:
            raise ValueError("No coordinates found in segments")

        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )

        # Add all segments with color coding
        segment_group = folium.FeatureGroup(name='Required Segments')

        for segment in segments:
            if not segment.coordinates:
                continue

            # Determine color based on requirement type
            if segment.is_two_way_required:
                color = self.config.colors['both_required']
                tooltip = f"Segment {segment.segment_id}: BOTH DIRECTIONS REQUIRED"
            elif segment.forward_required:
                color = self.config.colors['forward_required']
                tooltip = f"Segment {segment.segment_id}: FORWARD REQUIRED →"
            elif segment.backward_required:
                color = self.config.colors['backward_required']
                tooltip = f"Segment {segment.segment_id}: BACKWARD REQUIRED ←"
            else:
                color = self.config.colors['not_required']
                tooltip = f"Segment {segment.segment_id}: Not required"

            # Add segment line
            folium.PolyLine(
                segment.coordinates,
                color=color,
                weight=self.config.line_width,
                opacity=0.7,
                tooltip=tooltip
            ).add_to(segment_group)

            # Add segment ID label at midpoint
            if self.config.show_segment_ids and segment.coordinates:
                mid_idx = len(segment.coordinates) // 2
                mid_point = segment.coordinates[mid_idx]

                folium.Marker(
                    mid_point,
                    icon=folium.DivIcon(html=f'''
                        <div style="
                            font-size: {self.config.segment_label_size}px;
                            color: black;
                            background: white;
                            border: 1px solid black;
                            padding: 2px;
                            border-radius: 3px;
                            font-weight: bold;
                        ">{segment.segment_id}</div>
                    ''')
                ).add_to(segment_group)

        segment_group.add_to(m)

        # Add route path with step numbers
        if route_steps:
            route_group = folium.FeatureGroup(name='Computed Route')

            # Draw route path
            route_coords = []
            for step in route_steps:
                if step.coordinates:
                    route_coords.extend(step.coordinates)

            if route_coords:
                folium.PolyLine(
                    route_coords,
                    color=self.config.colors['route_path'],
                    weight=self.config.line_width + 2,
                    opacity=0.8,
                    tooltip="Computed Route Path"
                ).add_to(route_group)

            # Add step numbers
            if self.config.show_step_numbers:
                for step in route_steps:
                    if step.coordinates:
                        start_point = step.coordinates[0]

                        # Color code by deadhead
                        number_color = self.config.colors['deadhead'] if step.is_deadhead else '#00AA00'

                        folium.Marker(
                            start_point,
                            icon=folium.DivIcon(html=f'''
                                <div style="
                                    font-size: {self.config.step_label_size}px;
                                    color: white;
                                    background: {number_color};
                                    border: 2px solid white;
                                    padding: 4px;
                                    border-radius: 50%;
                                    width: 24px;
                                    height: 24px;
                                    text-align: center;
                                    font-weight: bold;
                                    line-height: 24px;
                                ">{step.step_number}</div>
                            '''),
                            tooltip=f"Step {step.step_number}: Segment {step.segment_id} ({step.direction})"
                        ).add_to(route_group)

            route_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Add legend
        legend_html = self._create_html_legend()
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save to file
        m.save(str(output_file))
        return output_file

    def generate_geojson(self,
                        segments: List,
                        route_steps: List,
                        output_file: Path) -> Path:
        """Generate GeoJSON with all segments and route.

        Args:
            segments: List of SegmentRequirement objects
            route_steps: List of RouteStep objects
            output_file: Path to save GeoJSON file

        Returns:
            Path to generated GeoJSON file
        """
        features = []

        # Add segments
        for segment in segments:
            if not segment.coordinates:
                continue

            properties = {
                'segment_id': segment.segment_id,
                'forward_required': segment.forward_required,
                'backward_required': segment.backward_required,
                'is_two_way_required': segment.is_two_way_required,
                'one_way': segment.one_way,
                'feature_type': 'segment'
            }

            # Add metadata if available
            if segment.metadata:
                properties.update(segment.metadata)

            feature = {
                'type': 'Feature',
                'properties': properties,
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon, lat] for lat, lon in segment.coordinates]
                }
            }
            features.append(feature)

        # Add route steps
        for step in route_steps:
            if not step.coordinates:
                continue

            properties = {
                'step_number': step.step_number,
                'segment_id': step.segment_id,
                'direction': step.direction,
                'is_deadhead': step.is_deadhead,
                'distance_meters': step.distance_meters,
                'feature_type': 'route_step'
            }

            feature = {
                'type': 'Feature',
                'properties': properties,
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon, lat] for lat, lon in step.coordinates]
                }
            }
            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)

        return output_file

    def generate_svg(self,
                    segments: List,
                    route_steps: List,
                    output_file: Path,
                    width: int = 2000,
                    height: int = 2000) -> Path:
        """Generate SVG visualization.

        Args:
            segments: List of SegmentRequirement objects
            route_steps: List of RouteStep objects
            output_file: Path to save SVG file
            width: SVG width in pixels
            height: SVG height in pixels

        Returns:
            Path to generated SVG file
        """
        # Calculate bounding box
        all_coords = []
        for segment in segments:
            if segment.coordinates:
                all_coords.extend(segment.coordinates)

        if not all_coords:
            raise ValueError("No coordinates found")

        min_lat = min(c[0] for c in all_coords)
        max_lat = max(c[0] for c in all_coords)
        min_lon = min(c[1] for c in all_coords)
        max_lon = max(c[1] for c in all_coords)

        # Convert lat/lon to SVG coordinates
        def to_svg(lat, lon):
            x = ((lon - min_lon) / (max_lon - min_lon)) * (width - 100) + 50
            y = height - (((lat - min_lat) / (max_lat - min_lat)) * (height - 100) + 50)
            return x, y

        # Start SVG
        svg_lines = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
            '<defs>',
            '  <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">',
            '    <path d="M 0 0 L 10 5 L 0 10 z" fill="black"/>',
            '  </marker>',
            '</defs>'
        ]

        # Add segments
        for segment in segments:
            if not segment.coordinates:
                continue

            # Determine color
            if segment.is_two_way_required:
                color = self.config.colors['both_required']
            elif segment.forward_required:
                color = self.config.colors['forward_required']
            elif segment.backward_required:
                color = self.config.colors['backward_required']
            else:
                color = self.config.colors['not_required']

            # Draw polyline
            points = ' '.join(f'{to_svg(lat, lon)[0]},{to_svg(lat, lon)[1]}'
                            for lat, lon in segment.coordinates)

            svg_lines.append(
                f'<polyline points="{points}" '
                f'stroke="{color}" stroke-width="{self.config.line_width}" '
                f'fill="none" opacity="0.7"/>'
            )

            # Add segment ID label
            if self.config.show_segment_ids:
                mid_idx = len(segment.coordinates) // 2
                mid_lat, mid_lon = segment.coordinates[mid_idx]
                x, y = to_svg(mid_lat, mid_lon)

                svg_lines.append(
                    f'<text x="{x}" y="{y}" font-size="{self.config.segment_label_size}" '
                    f'fill="black" text-anchor="middle" '
                    f'stroke="white" stroke-width="0.5">{segment.segment_id}</text>'
                )

        # Add route steps
        if route_steps and self.config.show_step_numbers:
            for step in route_steps:
                if step.coordinates:
                    start_lat, start_lon = step.coordinates[0]
                    x, y = to_svg(start_lat, start_lon)

                    color = self.config.colors['deadhead'] if step.is_deadhead else '#00AA00'

                    svg_lines.append(
                        f'<circle cx="{x}" cy="{y}" r="12" fill="{color}" '
                        f'stroke="white" stroke-width="2"/>'
                    )
                    svg_lines.append(
                        f'<text x="{x}" y="{y+5}" font-size="{self.config.step_label_size}" '
                        f'fill="white" text-anchor="middle" font-weight="bold">{step.step_number}</text>'
                    )

        # Add legend
        svg_lines.append(self._create_svg_legend(width, height))

        svg_lines.append('</svg>')

        # Save to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(svg_lines))

        return output_file

    def _create_html_legend(self) -> str:
        """Create HTML legend for map."""
        return f'''
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            background: white;
            padding: 10px;
            border: 2px solid black;
            border-radius: 5px;
            z-index: 9999;
            font-family: Arial;
        ">
            <h4 style="margin: 0 0 10px 0;">Legend</h4>
            <div><span style="color: {self.config.colors['forward_required']};">■</span> Forward Required →</div>
            <div><span style="color: {self.config.colors['backward_required']};">■</span> Backward Required ←</div>
            <div><span style="color: {self.config.colors['both_required']};">■</span> Both Directions Required ↔</div>
            <div><span style="color: {self.config.colors['not_required']};">■</span> Not Required</div>
            <div><span style="color: {self.config.colors['route_path']};">■</span> Route Path</div>
            <div><span style="color: {self.config.colors['deadhead']};">●</span> Deadhead (routing between)</div>
        </div>
        '''

    def _create_svg_legend(self, width: int, height: int) -> str:
        """Create SVG legend."""
        legend_x = width - 200
        legend_y = 50

        return f'''
        <g>
            <rect x="{legend_x}" y="{legend_y}" width="180" height="150"
                  fill="white" stroke="black" stroke-width="2"/>
            <text x="{legend_x + 10}" y="{legend_y + 20}" font-weight="bold">Legend</text>
            <line x1="{legend_x + 10}" y1="{legend_y + 35}" x2="{legend_x + 40}" y2="{legend_y + 35}"
                  stroke="{self.config.colors['forward_required']}" stroke-width="3"/>
            <text x="{legend_x + 50}" y="{legend_y + 40}" font-size="12">Forward Required →</text>
            <line x1="{legend_x + 10}" y1="{legend_y + 55}" x2="{legend_x + 40}" y2="{legend_y + 55}"
                  stroke="{self.config.colors['backward_required']}" stroke-width="3"/>
            <text x="{legend_x + 50}" y="{legend_y + 60}" font-size="12">Backward Required ←</text>
            <line x1="{legend_x + 10}" y1="{legend_y + 75}" x2="{legend_x + 40}" y2="{legend_y + 75}"
                  stroke="{self.config.colors['both_required']}" stroke-width="3"/>
            <text x="{legend_x + 50}" y="{legend_y + 80}" font-size="12">Both Required ↔</text>
            <line x1="{legend_x + 10}" y1="{legend_y + 95}" x2="{legend_x + 40}" y2="{legend_y + 95}"
                  stroke="{self.config.colors['not_required']}" stroke-width="3"/>
            <text x="{legend_x + 50}" y="{legend_y + 100}" font-size="12">Not Required</text>
            <circle cx="{legend_x + 25}" cy="{legend_y + 120}" r="8" fill="{self.config.colors['deadhead']}"/>
            <text x="{legend_x + 50}" y="{legend_y + 125}" font-size="12">Route Steps</text>
        </g>
        '''
