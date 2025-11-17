# Route Planner - DRPP Solver

**Production-ready Directed Rural Postman Problem (DRPP) solver with advanced optimization and visualization capabilities.**

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/DillyVee/Route_planner/actions/workflows/ci.yml/badge.svg)](https://github.com/DillyVee/Route_planner/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Overview

Route Planner is an industrial-strength solver for the Directed Rural Postman Problem (DRPP), designed to optimize routes that must cover required road segments while minimizing total distance traveled. Perfect for roadway surveys, infrastructure inspection, snow plowing, street sweeping, and delivery route optimization.

### Key Features

- **🚀 High Performance** - Optimized V4 greedy algorithm handles 10,000+ segments efficiently with on-demand routing
- **🗺️ MapPlus/Duweis Support** - Full support for roadway survey KML format with automatic metadata extraction
- **📊 Rich Visualizations** - Interactive HTML maps with segment IDs, route steps, and color-coded requirements
- **🔄 Multiple Algorithms** - Choose from V4 Greedy, RFCS, Legacy Greedy, or Hungarian based on your needs
- **⚡ Parallel Processing** - Multi-core support for large datasets with geographic clustering
- **📦 Production Ready** - Type hints, comprehensive tests, logging, error handling, and CI/CD
- **🛡️ Robust Error Handling** - Custom exception hierarchy with clear error messages
- **🌍 Geographic Utilities** - Haversine distance, coordinate snapping, and bearing calculations

## Quick Start

### Installation

#### For End Users (GUI Application)

```bash
# Clone the repository
git clone https://github.com/DillyVee/Route_planner.git
cd Route_planner

# Install with GUI and all features
pip install -e ".[all]"
```

#### For Developers

```bash
# Clone the repository
git clone https://github.com/DillyVee/Route_planner.git
cd Route_planner

# Install with development tools
pip install -e ".[dev,all]"

# Install pre-commit hooks (recommended)
pre-commit install
```

#### Core Library Only

```bash
# Install minimal dependencies (no GUI, no visualization)
pip install -e .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup instructions.

### Basic Usage

```bash
# Run the complete pipeline on your KML file
python run_drpp_pipeline.py your_segments.kml v4
```

This will:
1. Parse your KML file and extract segment requirements
2. Build a directed graph from the segments
3. Solve the DRPP using the V4 greedy algorithm
4. Generate interactive visualizations (HTML, GeoJSON, SVG)

### View Results

Open `output/route_map.html` in your browser to see:
- Color-coded segments by requirement type (forward, backward, both, none)
- Segment ID labels on every segment
- Route step numbering showing the optimized path
- Interactive map with zoom, pan, and layer controls

## Supported Input Formats

### MapPlus/Duweis Roadway Survey KML

Fully supports MapPlus roadway survey format with automatic extraction of:
- `CollId` - Segment ID
- `RouteName` - Route name (e.g., "PA-981")
- `Dir` - Direction code (NB, SB, EB, WB, etc.)
- `LengthFt` - Segment length (auto-converted to meters)
- `Region`, `Juris`, `CntyCode`, `StRtNo`, `SegNo`, `BegM`, `EndM`, and more

All metadata is preserved, displayed in visualizations, and exported to GeoJSON.

### Standard KML Format

Generic KML with LineString geometries is also supported:
- Segment IDs from `<name>` tags
- Directionality from `oneway` extended data
- Custom metadata preserved

## Algorithm Comparison

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| **V4 Greedy** | ⚡⚡⚡ Very Fast | ⭐⭐ Good | Large datasets (1000+ segments) |
| **RFCS** | ⚡⚡ Moderate | ⭐⭐⭐ Excellent | High-quality routes |
| **Legacy Greedy** | ⚡⚡ Fast | ⭐⭐ Good | Medium datasets |
| **Hungarian** | ⚡ Slow | ⭐⭐⭐ Excellent | Small datasets (<500 segments) |

## Python API

```python
from pathlib import Path
from drpp_pipeline import DRPPPipeline

# Create pipeline
pipeline = DRPPPipeline()

# Run complete pipeline
results = pipeline.run(
    kml_file=Path('your_file.kml'),
    algorithm='v4',  # or 'rfcs', 'greedy', 'hungarian'
    output_dir=Path('./output'),
    output_formats=['html', 'geojson', 'svg']
)

# Check results
print(f"Total distance: {results['total_distance'] / 1000:.1f} km")
print(f"Coverage: {results['coverage']:.1f}%")
print(f"Segments covered: {results['segments_covered']}/{results['total_segments']}")
```

### Advanced Usage

```python
from drpp_core import (
    greedy_route_cluster,
    cluster_segments,
    ClusteringMethod,
    compute_distance_matrix,
    haversine,
    snap_coordinate
)

# Use V4 core API directly
from drpp_core import greedy_route_cluster

result = greedy_route_cluster(
    graph=graph,
    required_edges=edges,
    segment_indices=segment_list,
    start_node=start_coords,
    use_ondemand=True  # Enable on-demand mode for large datasets
)

# Geographic clustering for parallel processing
clusters = cluster_segments(
    segments,
    method=ClusteringMethod.DBSCAN,
    eps_km=2.0,  # 2km radius
    min_samples=3
)

# Geographic utilities
from drpp_core import haversine, snap_coordinate, calculate_bearing

# Calculate distance between two points
distance_m = haversine((37.7749, -122.4194), (34.0522, -118.2437))
print(f"Distance: {distance_m / 1000:.1f} km")  # ~559.1 km

# Snap coordinates to eliminate floating-point errors
coord = snap_coordinate(37.774929999, -122.419415001, precision=6)

# Calculate bearing
bearing = calculate_bearing((37.7749, -122.4194), (34.0522, -118.2437))
print(f"Bearing: {bearing:.1f}°")  # ~126.5°

# Robust error handling with custom exceptions
from drpp_core import KMLParseError, RoutingError, handle_parse_error

try:
    result = greedy_route_cluster(...)
except RoutingError as e:
    print(f"Routing failed: {e}")
```

## Performance

### V4 Greedy On-Demand Optimization

For large datasets, V4 automatically switches to on-demand routing:
- **Before**: 11,060 × 11,060 = 122M distance computations (very slow)
- **After**: ~11,060 single-source Dijkstra calls (much faster)
- **Speedup**: 10-100x faster for datasets with 1000+ segments

The algorithm automatically detects when a cluster has >1000 segment endpoints and switches to on-demand mode.

## Output Formats

### HTML Interactive Map
- Folium/Leaflet-based interactive visualization
- Layer controls to toggle segments and route
- Click segments for detailed metadata
- Legend explaining color codes

### GeoJSON
- Machine-readable geographic data format
- Import into QGIS, ArcGIS, or other GIS software
- All segment metadata and route properties preserved

### SVG
- Scalable vector graphics for documents/presentations
- Static image with all route details
- High-quality print output

## Documentation

- **[Pipeline Guide](docs/PIPELINE_GUIDE.md)** - Complete pipeline usage and customization
- **[V4 Production Guide](docs/PRODUCTION_REFACTOR_GUIDE.md)** - V4 core API reference
- **[V4 Integration Summary](docs/V4_INTEGRATION_SUMMARY.md)** - V4 architecture and design

## Architecture

```
Route_planner/
├── drpp_core/              # V4 production core
│   ├── greedy_router.py    # Optimized greedy routing
│   ├── distance_matrix.py  # Memory-efficient storage
│   ├── clustering.py       # Geographic clustering
│   ├── path_reconstruction.py
│   ├── parallel_executor.py
│   └── types.py            # Type definitions
├── drpp_pipeline.py        # Main pipeline orchestrator
├── drpp_visualization.py   # Visualization generation
├── run_drpp_pipeline.py    # CLI script
├── tests/                  # Unit tests
├── legacy/                 # Historical implementations
└── docs/                   # Documentation
```

## Requirements

- Python 3.9 or higher
- NumPy
- SciPy
- NetworkX
- Folium
- lxml (for KML parsing)

See `requirements_production.txt` for complete list.

## Testing

```bash
# Run all tests
python -m unittest discover tests -v

# Test on your own KML file
python run_drpp_pipeline.py your_file.kml v4
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use Cases

- **Roadway Survey Planning** - Optimize routes for highway data collection
- **Infrastructure Inspection** - Minimize travel time for bridge/road inspections
- **Snow Plowing Routes** - Efficient coverage of required streets
- **Street Sweeping** - Optimize municipal cleaning routes
- **Delivery Optimization** - Plan routes with required stops and directional constraints

## Support

For bugs, feature requests, or questions:
- Open an issue on GitHub
- See documentation in the `docs/` directory
- Check `example_production_usage.py` for code examples

---

**Version**: 4.0.0
**Status**: Production Ready ✅
**Last Updated**: 2025-11-17
