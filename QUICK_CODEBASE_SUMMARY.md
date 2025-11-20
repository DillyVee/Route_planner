# Route Planner - Quick Codebase Reference

## 📋 At a Glance

| Aspect | Status | Details |
|--------|--------|---------|
| **Project Status** | ✅ Production Ready (v4.0.0) | Type-safe, well-tested, documented |
| **Primary Use** | 🗺️ DRPP Solving | Directed Rural Postman Problem (roadway surveys) |
| **Main Language** | 🐍 Python 3.9+ | Type hints, dataclasses, modern patterns |
| **Core Algorithm** | ⚡ V4 Greedy | On-demand Dijkstra, scales to 10K+ segments |
| **Architecture** | 🏗️ Modular | drpp_core + pipeline + visualization |
| **Code Quality** | ✨ High | Black formatted, Ruff linted, pre-commit hooks |

---

## 🎯 Key Components (What Exists)

### 1. **KML Parsing** ✅
```
File: drpp_pipeline.py::DRPPPipeline._parse_kml()
├─ Standard KML 2.2 support (LineString/Polygon)
├─ MapPlus/Duweis format (full metadata extraction)
├─ Robust error recovery (corrupt XML handling)
└─ Output: List[SegmentRequirement] with metadata
```

### 2. **Graph Building** ✅
```
File: Route_Planner.py::DirectedGraph
├─ Directed graph with weighted edges
├─ Haversine distance calculation
├─ Dijkstra's shortest paths
├─ On-demand computation (large graphs)
└─ Time-based weights (via OSM integration)
```

### 3. **Route Optimization** ✅
```
Four algorithms available:
├─ V4 Greedy (⚡⚡⚡ fast, practical) [drpp_core/greedy_router.py]
├─ RFCS (⚡⚡ moderate, high quality) [legacy/parallel_processing_addon_rfcs.py]
├─ Hungarian (⚡ slow, optimal) [legacy/parallel_processing_addon.py]
└─ Legacy Greedy (⚡⚡ fast) [legacy/parallel_processing_addon_greedy.py]
```

### 4. **Output Generation** ✅
```
File: drpp_visualization.py::DRPPVisualizer
├─ HTML Interactive Maps (Folium/Leaflet)
├─ GeoJSON Export (all metadata preserved)
├─ SVG Graphics (for documents)
└─ Statistics (distance, coverage, deadhead)
```

### 5. **Geographic Utilities** ✅
```
File: drpp_core/geo.py
├─ Haversine distance (accurate spherical)
├─ Coordinate snapping (eliminate duplicates)
├─ Bearing calculation
├─ Clustering (DBSCAN, KMeans, Grid)
└─ Path reconstruction (with cycle detection)
```

---

## 🚀 Entry Points

| Interface | File | Use Case |
|-----------|------|----------|
| **CLI** | `run_drpp_pipeline.py` | Quick command-line usage |
| **Python API** | `drpp_pipeline.DRPPPipeline` | Integration into other tools |
| **GUI** | `Route_Planner.py` | User-friendly desktop app |
| **Core Library** | `drpp_core/` | Advanced algorithmic use |

**Example - CLI**:
```bash
python run_drpp_pipeline.py segments.kml v4
# → output/route_map.html, route_data.geojson, route_map.svg
```

**Example - Python API**:
```python
from drpp_pipeline import DRPPPipeline
pipeline = DRPPPipeline()
results = pipeline.run(
    kml_file=Path('segments.kml'),
    algorithm='v4',
    output_formats=['html', 'geojson']
)
print(f"Distance: {results['total_distance']/1000:.1f}km")
print(f"Coverage: {results['coverage']:.1f}%")
```

---

## 📊 Data Structures

### SegmentRequirement (Input)
```python
@dataclass
class SegmentRequirement:
    segment_id: str                              # KML CollId
    forward_required: bool                       # Traverse →
    backward_required: bool                      # Traverse ←
    one_way: bool                               # One-way only
    coordinates: List[Tuple[float, float]]      # (lat, lon) points
    metadata: Dict[str, Any]                    # MapPlus fields, etc
```

### PathResult (Output)
```python
class PathResult(NamedTuple):
    path: List[Coordinate]                      # Route coordinates
    distance: float                             # Total in meters
    cluster_id: int                             # Cluster ID
    segments_covered: int                       # Count
    segments_unreachable: int                   # Count
    computation_time: float                     # Seconds
```

---

## 🔄 Pipeline Flow

```
    KML File
       ↓
┌──────────────────────────────────────┐
│ Parse KML & Extract Segments         │ → 1. Coordinates (lat, lon)
│ (drpp_pipeline._parse_kml)           │ → 2. Segment IDs
│                                      │ → 3. Directionality
└──────────────────────────────────────┘ → 4. Metadata (MapPlus)
       ↓
┌──────────────────────────────────────┐
│ Build Directed Graph                 │ → Nodes: Coordinates
│ (drpp_pipeline._build_graph)         │ → Edges: Haversine distance
└──────────────────────────────────────┘ → Weights: Time/Distance
       ↓
┌──────────────────────────────────────┐
│ Solve DRPP                           │ → Algorithm selection
│ (drpp_pipeline._solve_drpp)          │ → Route computation
│ - Select algorithm (v4/rfcs/hung)    │ → Path reconstruction
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ Generate Visualizations              │ → HTML Map (Folium)
│ (drpp_visualization)                 │ → GeoJSON (RFC 7946)
│                                      │ → SVG Vector graphic
└──────────────────────────────────────┘ → Statistics
       ↓
    Results
   (distance, coverage %, output files)
```

---

## 🎨 Metadata Handling

### MapPlus Fields (Extracted from KML)
```python
metadata = {
    'CollId': 'segment_001',           # Segment ID
    'RouteName': 'PA-981',             # Route
    'Dir': 'NB',                       # Direction (N/S/E/W)
    'LengthFt': 1234.5,                # Length (auto-converted to m)
    'Region': '...',                   # Administrative region
    'Juris': '...',                    # Jurisdiction
    'CntyCode': '36',                  # County code
    'StRtNo': 'PA-981',                # State route number
    'SegNo': '001',                    # Segment number
    'BegM': '1.0',                     # Begin measure
    'EndM': '2.5',                     # End measure
    'IsPilot': '1',                    # Pilot project flag
    'Collected': '2024-11-20'          # Collection date
}
```

### Usage in Visualization
```
HTML Tooltip:
├─ Segment ID
├─ Direction requirement (→/←/↔)
├─ Route name
├─ Direction code
├─ Length (ft & m)
├─ Region, County, State Route
└─ Collection status

GeoJSON Properties:
└─ All fields preserved + computed (length_km, etc)
```

---

## ⚙️ V4 Greedy Algorithm (Best for Large Datasets)

### How It Works
```
1. Start at initial position
2. While segments remain:
   a) Compute Dijkstra from current position
      - On-demand (vs all-pairs precomputation)
      - O(n log n) per iteration, not O(n²) upfront
   b) Find nearest unreachable segment
      - Within max_search_distance if specified
   c) Route to it and traverse
   d) Update position
   e) Remove from remaining
3. Return complete path + statistics
```

### Key Optimization: On-Demand Mode
```
Large Cluster Detection:
├─ If >500 segment endpoints
├─ Switch from O(n²) precomputation → on-demand Dijkstra
├─ Result: 10-100x speedup for 1000+ segments
│
Example: 11,060 nodes
├─ All-pairs: 122 million distance computations (slow)
└─ On-demand: 11,060 Dijkstra calls (much faster)
```

### Parameters
```python
greedy_route_cluster(
    graph,                          # Required graph object
    required_edges,                 # Edges to traverse
    segment_indices,                # Which segments to route
    start_node,                     # (lat, lon) or node ID
    use_ondemand=True,             # Auto-detects large clusters
    lookahead_depth=1,             # 1=greedy, 3=smart scoring
    max_search_distance=None        # Radius limit (meters)
)
```

---

## ❌ Known Gaps (For Industry DRPP)

| Gap | Impact | Priority |
|-----|--------|----------|
| No time-based routing | Can't use OSM speed in costs | **High** |
| No multi-objective | Can't balance distance + time | **High** |
| No real-world constraints | Can't handle traffic, hours, etc | **High** |
| No advanced heuristics | Greedy-only (no LK, SA, genetic) | **Medium** |
| No REST API | Can't use as service | **Low** |
| No distributed computing | Single machine only | **Low** |

### What's NOT There
```
❌ Time-windowed VRP (multiple time constraints)
❌ Vehicle capacity constraints
❌ Traffic patterns / turn penalties
❌ Bridge/tunnel handling
❌ Chinese Postman Problem (all edges, not just required)
❌ Advanced heuristics (Lin-Kernighan, Christofides)
❌ Feasibility analysis (connectivity check upfront)
❌ REST/GraphQL API
❌ Database integration (PostGIS)
❌ GPU acceleration
```

---

## 📁 Important Files Quick Lookup

| What You Need | File(s) |
|---------------|---------|
| **Run everything** | `run_drpp_pipeline.py` |
| **Use as library** | `drpp_pipeline.py` (DRPPPipeline class) |
| **Fast routing** | `drpp_core/greedy_router.py` |
| **Geographic math** | `drpp_core/geo.py` (haversine, bearing) |
| **Clustering** | `drpp_core/clustering.py` |
| **Distance matrix** | `drpp_core/distance_matrix.py` |
| **Visualization** | `drpp_visualization.py` |
| **Graph structure** | `Route_Planner.py::DirectedGraph` |
| **Legacy algorithms** | `legacy/parallel_processing_addon*.py` |
| **GUI app** | `Route_Planner.py` (main + PyQt6) |
| **OSM integration** | `osm_speed_integration.py` |
| **Types/dataclasses** | `drpp_core/types.py` |
| **Error handling** | `drpp_core/exceptions.py` |

---

## 🔗 Key Classes & Functions

### Main Classes
- **DRPPPipeline** - Orchestrator (parse → build → solve → visualize)
- **DirectedGraph** - Graph representation with Dijkstra
- **DRPPVisualizer** - Output generator (HTML/GeoJSON/SVG)
- **DistanceMatrix** - Memory-efficient storage (dict/numpy)
- **OverpassSpeedFetcher** - OSM speed data fetching

### Main Functions
- **greedy_route_cluster()** - V4 greedy routing algorithm
- **cluster_segments()** - Geographic clustering (DBSCAN/KMeans/Grid)
- **haversine()** - Distance between coordinates
- **reconstruct_path()** - Dijkstra path reconstruction
- **snap_coordinate()** - Precision snapping for duplicates

---

## 💾 Output Formats

| Format | File | Use Case |
|--------|------|----------|
| **HTML** | `route_map.html` | Interactive web map (Folium) |
| **GeoJSON** | `route_data.geojson` | Import to QGIS/ArcGIS |
| **SVG** | `route_map.svg` | Print documents/presentations |
| **Console** | stdout | Statistics + progress |

### HTML Map Features
```
✅ Zoomable/pannable
✅ Layer toggles (segments/route)
✅ Color-coded by requirement type
✅ Segment ID labels
✅ Rich tooltips with metadata
✅ Route step numbering
✅ Legend
```

### GeoJSON Features
```
✅ All metadata preserved
✅ Computed fields (length_m, length_km)
✅ Both segments and route steps
✅ Compatible with any GIS software
```

---

## 🧪 Testing

Location: `tests/`
- `test_clustering.py` - Haversine, clustering methods
- `test_distance_matrix.py` - Matrix operations
- `test_path_reconstruction.py` - Dijkstra path recovery

Run: `python -m pytest tests/ -v`

---

## 📚 Documentation

- **CODEBASE_EXPLORATION.md** - This detailed report
- **README.md** - User-facing overview
- **CHANGELOG.md** - Version history
- **CONTRIBUTING.md** - Development guide
- **docs/** - 13 detailed markdown files
  - PIPELINE_GUIDE.md
  - V4_INTEGRATION_SUMMARY.md
  - PRODUCTION_REFACTOR_GUIDE.md
  - etc.

---

## ⚡ Performance Notes

### Scaling (V4 Greedy)
- **Small** (100 segments): < 1 second
- **Medium** (1,000 segments): 5-10 seconds
- **Large** (10,000 segments): 30-60 seconds
- **Very Large** (100K segments): Requires distributed computing

### Memory Usage
- **Distance Matrix**: O(n²) if precomputed, O(n) if on-demand
- **Graph**: O(n+m) where n=nodes, m=edges
- **On-demand Dijkstra**: O(n log n) per iteration

### Optimization Tips
```
1. Use on-demand mode (auto-detects >500 nodes)
2. Use lookahead_depth=1 for speed (greedy), =3 for quality
3. Set max_search_distance to limit search radius
4. Use geographic clustering for very large datasets
5. Consider parallel processing via drpp_core.parallel_cluster_routing
```

---

**Version**: 4.0.0  
**Last Updated**: 2025-11-20  
**Status**: Production Ready ✅
