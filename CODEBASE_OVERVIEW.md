# Route Planner Codebase - Comprehensive Overview

## Executive Summary
This is a production-ready Directed Rural Postman Problem (DRPP) solver designed for roadway surveys, infrastructure inspection, and similar applications. The codebase has ~15,000 lines of Python across multiple modules, with a modern V4 architecture and legacy implementations.

---

## 1. KML IMPORT/PROCESSING LOGIC

### Primary Entry Points
- **`drpp_pipeline.py::DRPPPipeline._parse_kml()`** (lines 187-385)
  - Main KML parser for the modern pipeline
  - Supports MapPlus/Duweis roadway survey format
  - Returns `List[SegmentRequirement]` objects

- **`Route_Planner.py::parse_kml()`** (lines 189-325)
  - Legacy KML parser for older applications
  - Basic LineString parsing with metadata extraction

### KML Parsing Features

**Format Support:**
1. **MapPlus/Duweis Format** (Modern):
   - Extracts `CollId` as segment identifier
   - Reads `Dir` field for direction information
   - Parses `LengthFt`, `RouteName`, `Region`, `Juris`, `CntyCode`, `StRtNo`, `SegNo`, `BegM`, `EndM`
   - Supports MapPlusCustomFeatureClass and MapPlusSystemData schemas
   - All metadata preserved in `segment.metadata` dict

2. **Standard KML Format**:
   - Extracts coordinates from LineString geometries
   - Reads `<name>` for segment IDs (fallback: `seg_XXXXX`)
   - Checks extended data for `oneway` indicators
   - Supports standard speed limit fields

**Coordinate Processing:**
```python
# From osm_speed_integration.py
snap_coord(lat, lon, precision=6)       # Snap to fixed precision
snap_coords_list(coords, precision=6)   # Remove consecutive duplicates
```

**Data Structures:**
```python
# Modern format (drpp_pipeline.py)
@dataclass
class SegmentRequirement:
    segment_id: str                      # Unique identifier
    forward_required: bool               # Must traverse start->end
    backward_required: bool              # Must traverse end->start
    one_way: bool                        # Only one direction allowed
    coordinates: List[Tuple[float, float]]  # [(lat, lon), ...]
    metadata: Dict                       # All KML extended data

# Legacy format (Route_Planner.py)
segment = {
    'coords': [(lat, lon), ...],
    'start': (lat, lon),
    'end': (lat, lon),
    'length_m': float,
    'oneway': bool or None,
    'speed_limit': int or None,
}
```

**Error Handling:**
- Robust XML parsing with automatic fixes for:
  - Invalid XML characters (control characters)
  - Unescaped ampersands in extended data
  - Malformed coordinate strings

**One-Way Detection Logic:**
```python
if oneway is True:
    forward_required = True
    backward_required = False
elif oneway is False:
    forward_required = True
    backward_required = True
else:
    # Default: assume both directions required
    forward_required = True
    backward_required = True
```

---

## 2. ROUTE OPTIMIZATION IMPLEMENTATION

### Algorithm Variants

**V4 Greedy (Production, Recommended)** - `drpp_core/greedy_router.py`
- **Best for**: Large datasets (1000+ segments)
- **Performance**: ~10-100x faster than legacy algorithms
- **Features**:
  - Nearest-neighbor greedy with dynamic routing
  - On-demand Dijkstra mode for large endpoint sets
  - Memory-efficient distance matrix storage
  - Robust error handling with fallback strategies

**RFCS (Route-First, Cluster-Second)** - `legacy/parallel_processing_addon_rfcs.py`
- **Best for**: High-quality routes
- **Quality**: Excellent (⭐⭐⭐)
- **Speed**: Moderate (⚡⚡)

**Legacy Greedy** - `legacy/parallel_processing_addon_greedy.py`
- **Best for**: Medium datasets
- **Quality**: Good (⭐⭐)
- **Speed**: Fast (⚡⚡)

**Hungarian** - `legacy/parallel_processing_addon.py`
- **Best for**: Small datasets (<500 segments)
- **Quality**: Excellent (⭐⭐⭐)
- **Speed**: Slow (⚡)

### V4 Greedy Algorithm Details

**Core Algorithm**: `greedy_route_cluster()` in `drpp_core/greedy_router.py`

```python
def greedy_route_cluster(
    graph: Any,
    required_edges: List[Tuple],
    segment_indices: List[SegmentIndex],
    start_node: Coordinate,
    use_ondemand: bool = True,
) -> PathResult:
    """
    Greedy nearest-neighbor routing with on-demand Dijkstra.
    
    Algorithm:
    1. Start at start_node
    2. While segments remaining:
       a. Find nearest uncovered segment
       b. Route to nearest endpoint (or closest point)
       c. Traverse required direction(s)
       d. Mark as covered
    3. Return complete path with statistics
    """
```

**On-Demand Mode Trigger:**
```python
# Automatic switch when cluster has >1000 endpoint pairs
if num_endpoints > 1000:
    use_dijkstra_on_demand = True  # Single-source Dijkstra per step
else:
    use_full_distance_matrix = True  # Precompute all pairs
```

**Key Features:**
1. **Distance Matrix Mode**:
   - Precomputes shortest paths between all required segment endpoints
   - Memory efficient: stores only node IDs + distances
   - Supports both dict and numpy backends

2. **On-Demand Mode**:
   - Computes Dijkstra from current node only when needed
   - No matrix storage required
   - Falls back to matrix if available

3. **Unreachable Segment Handling**:
   - Tracks segments that couldn't be reached
   - Records reason (no path, disconnected component, etc.)
   - Returns count in PathResult

### Segment Covering Logic

**Traversal Requirements:**
```python
# For each segment, algorithm decides optimal coverage:
if segment.is_two_way_required:
    # Must traverse both directions
    required_traversals = 2
elif segment.forward_required:
    # Can traverse in either direction, must cover forward pass
    required_traversals = 1
elif segment.backward_required:
    # Must cover backward pass
    required_traversals = 1
```

**Greedy Nearest-Neighbor Selection:**
1. Find nearest uncovered segment endpoint from current position
2. Route to that endpoint (might traverse other segments as deadhead)
3. If required direction not yet traversed, continue on that segment
4. Move current position to end of segment
5. Repeat

### Parallel Processing

**Framework**: `drpp_core/parallel_executor.py`

```python
class ClusterTask:
    """Lightweight task for worker process"""
    cluster_id: ClusterID
    segment_indices: List[SegmentIndex]
    distance_matrix: DistanceMatrix  # Precomputed in parent
    normalizer: NodeNormalizer
    start_node_id: NodeID
```

**Execution Pattern:**
```python
def parallel_cluster_routing(
    graph: DirectedGraph,
    required_edges: List[Tuple],
    clusters: Dict[ClusterID, List[SegmentIndex]],
    cluster_order: List[ClusterID],
    start_node: Coordinate,
    num_workers: int,
) -> List[PathResult]:
    """
    1. Parent process precomputes distance matrix
    2. Creates lightweight ClusterTask for each cluster
    3. Distributes to ProcessPoolExecutor workers
    4. Workers route clusters using precomputed matrix
    5. Collect PathResult from each worker
    """
```

**Worker Isolation:**
- No full graph passed to workers (pickle overhead reduction)
- Only precomputed distance matrix + metadata
- Graph.dijkstra() not available in workers (only for fallback)

---

## 3. GRAPH STRUCTURES

### Primary Graph Implementation: `DirectedGraph`

**Location**: `Route_Planner.py` lines 353-453

```python
class DirectedGraph:
    """Weighted directed graph with Dijkstra shortest path"""
    
    node_to_id: Dict[Coordinate, NodeID]    # (lat,lon) -> int
    id_to_node: List[Coordinate]            # int -> (lat,lon)
    adj: List[List[Tuple[NodeID, float]]]   # Adjacency list with weights
    
    def add_edge(a: Coordinate, b: Coordinate, weight: float) -> None:
        """Add weighted directed edge from a to b"""
    
    def dijkstra(source_id: NodeID, max_distance: Optional[float] = None) 
        -> Tuple[List[float], List[int]]:
        """
        Returns (distances, predecessors) from source.
        Optional max_distance for early termination.
        """
    
    def shortest_path(start: Coordinate, end: Coordinate) 
        -> Tuple[List[Coordinate], float]:
        """Returns path coordinates and distance"""
```

**Graph Building Logic:**

```python
def build_graph(segments, treat_unspecified_as_two_way=True):
    """Build graph from segments with three-level approach"""
    
    # Level 1: Add survey segments
    for segment in segments:
        coords = segment['coords']
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i+1]
            dist = haversine(start, end)
            
            if treat_as_two_way:
                graph.add_edge(start, end, dist)  # Forward
                graph.add_edge(end, start, dist)  # Backward
            else:
                graph.add_edge(start, end, dist)  # One-way only
    
    # Level 2: Add OSM roads (for better connectivity)
    osm_ways = fetch_osm_roads_for_routing(bbox)
    for way in osm_ways:
        coords = way['geometry']
        for i in range(len(coords) - 1):
            start, end = coords[i], coords[i+1]
            # Add with speed-based weights (optional)
    
    # Level 3: Return graph + required_edges list
    required_edges = extract_required_edges(segments)
    return graph, required_edges
```

**Dijkstra Implementation:**

```python
def dijkstra(source_id: NodeID) -> Tuple[List[float], List[int]]:
    n = len(self.id_to_node)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[source_id] = 0.0
    
    # Min-heap: (distance, node_id)
    heap = [(0.0, source_id)]
    
    while heap:
        d, u = heapq.heappop(heap)
        
        if d > dist[u]:
            continue
        
        # Relax edges
        for v, weight in self.adj[u]:
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
    
    return dist, prev
```

**Path Reconstruction:**
- Robust implementation in `drpp_core/path_reconstruction.py`
- Handles sentinel values: -1, None, self-loops
- Detects cycles and maximum iteration limits
- Returns empty list on error (safe fallback)

### Modern Graph Interface: `GraphInterface`

**Location**: `drpp_core/types.py` lines 114-141

```python
@dataclass
class GraphInterface:
    """Required interface for any graph used with DRPP"""
    
    node_to_id: Dict[Coordinate, NodeID]
    id_to_node: Dict[NodeID, Coordinate]
    
    def dijkstra(source_id: NodeID) 
        -> Tuple[List[Distance], List[Optional[NodeID]]]:
        """Compute shortest paths from source"""
        raise NotImplementedError()
```

This allows plugging in different graph implementations (NetworkX, custom, etc.).

### Required Edges Representation

```python
# Format used throughout the codebase
required_edges: List[Tuple] = [
    (start_coord, end_coord, full_coords, direction),
    (start_coord, end_coord, full_coords, direction),
    ...
]

# Or in extended format with metadata:
required_edges: List[Tuple[Coordinate, Coordinate, List[Coordinate], str]]
```

---

## 4. METADATA HANDLING

### Segment Metadata Storage

**Modern Format** (SegmentRequirement):
```python
@dataclass
class SegmentRequirement:
    segment_id: str
    forward_required: bool
    backward_required: bool
    one_way: bool = False
    coordinates: List[Tuple[float, float]] = None
    metadata: Dict = None  # All KML extended data preserved
```

**Metadata Preservation:**
```python
# MapPlus specific fields extracted:
metadata = {
    'CollId': 'SEGMENT_001',           # Original segment ID
    'route_name': 'PA-981',
    'direction_code': 'NB',             # From Dir field
    'length_ft': 1234.5,
    'length_m': 376.0,                 # Auto-converted
    'region': 'Northeast',
    'cntycode': 'PA001',
    'cntyname': 'Adams County',
    'strtno': 981,
    'segno': 123,
    'begm': 0.0,                       # Begin milepost
    'endm': 0.234,                     # End milepost
    'ispilot': 'Y',
    'collected': '2024-03-15',
    # ... and all other fields from ExtendedData
}
```

### Metadata Extraction Process

**Priority Order:**
1. **MapPlus SchemaData** (MapPlusCustomFeatureClass) - Primary
2. **MapPlus SystemData** (MapPlusSystemData) - Secondary labels
3. **Generic ExtendedData** - Fallback for other KML formats
4. **Description field** - Additional context
5. **Name field** - Speed limits or alternate identifiers

### Speed Limit Handling

**Parsing Logic:**
```python
def parse_speed_limit(text: str) -> Optional[int]:
    """Extract speed from various formats"""
    patterns = [
        r'(\d+)\s*mph',         # "25 mph"
        r'(\d+)\s*km[/\s]?h',   # "50 km/h", "50kph"
        r'maxspeed[=:]\s*(\d+)', # "maxspeed:55"
    ]
    # If mph found, convert to km/h (x1.60934)
```

**Storage:**
- In segment metadata as `speed_limit` (int, km/h)
- Used in time-based routing weights

### OSM Speed Integration

**Location**: `osm_speed_integration.py`

**Highway Type Speed Mapping:**
```python
DEFAULT_SPEEDS = {
    'motorway': 110,
    'trunk': 100,
    'primary': 80,
    'secondary': 60,
    'tertiary': 50,
    'residential': 30,
    'living_street': 20,
    'unclassified': 40,
}
```

**Integration Process:**
1. Fetch OSM ways in bounding box via Overpass API
2. Extract maxspeed or use highway type fallback
3. Enrich segments with speed data
4. Build time-weighted graph instead of distance-only

**Caching:**
```python
class OverpassSpeedFetcher:
    cache_file: str = "overpass_cache.json"
    rate_limit_delay: float = 1.0  # seconds between requests
```

### Metadata in Visualizations

**HTML Map** (`drpp_visualization.py`):
```python
# Rich tooltips showing:
- Segment ID and direction requirements
- Route name, direction code
- Length in ft and m
- Region, county code
- State route number, segment number
- Collection date
```

**GeoJSON Output:**
```json
{
  "type": "Feature",
  "properties": {
    "segment_id": "SEG_001",
    "forward_required": true,
    "backward_required": true,
    "CollId": "...",
    "RouteName": "...",
    // All metadata fields included
  },
  "geometry": {...}
}
```

---

## 5. OPTIMIZATION LIBRARIES USED

### Current Dependencies

**Core Libraries** (required):
```
numpy>=1.21.0,<2.0.0       # Numerical operations
scipy>=1.7.0,<2.0.0       # Linear sum assignment (Hungarian)
networkx>=2.6.0,<4.0.0   # Graph algorithms
lxml>=4.6.0,<6.0.0       # KML XML parsing
psutil>=5.8.0            # Process/memory monitoring
```

**Optional Libraries**:
```
scikit-learn>=0.24.0     # DBSCAN/KMeans clustering
requests>=2.25.0         # Overpass API queries
folium>=0.12.0           # Interactive map visualization
geopy>=2.2.0             # Geolocation services
PyQt6>=6.4.0             # GUI application
```

### Library Usage

**NumPy**:
- Distance matrix storage (for large graphs)
- Coordinate array operations
- Radians conversion for haversine

**SciPy**:
- `scipy.optimize.linear_sum_assignment` - Hungarian algorithm
- Used in `legacy/parallel_processing_addon.py`

**NetworkX**:
- Primarily used in older code paths
- Potential for future integration of TSP algorithms
- Currently not heavily utilized in V4

**scikit-learn**:
- DBSCAN clustering (geographic density-based)
- KMeans clustering (centroid-based)
- Haversine metric for geographic distance
- Falls back to grid clustering if sklearn unavailable

**Folium**:
- Interactive Leaflet map generation
- Layer controls, plugins
- HTML-based visualization

### Algorithm NOT Used (Intentionally)

**OR-Tools (Google Operations Research):**
- Available but not integrated in V4
- V4 chose pure greedy to avoid external dependencies
- Could be integrated for TSP/VRP variants
- Check: `ORTOOLS_AVAILABLE` flag (Route_Planner.py line 129)

**Why not OR-Tools?**
1. Additional dependency (licensing considerations)
2. Greedy algorithm adequate for survey use cases
3. Simpler debugging and understanding
4. Avoids binary compilation issues

---

## 6. OUTPUT/EXPORT FUNCTIONALITY

### Output Formats Supported

**1. Interactive HTML Map** (Primary)
- **Location**: `drpp_visualization.py::DRPPVisualizer.generate_html_map()`
- **Technology**: Folium + Leaflet.js
- **File**: `output/route_map.html`

**Features:**
```python
- Color-coded segments by requirement type:
  Red (#FF0000)         - Forward required only
  Blue (#0000FF)        - Backward required only
  Purple (#9900FF)      - Both required
  Gray (#CCCCCC)        - Not required
  Orange (#FFA500)      - Deadhead (routing between)
  Green (#00FF00)       - Route path overlay

- Segment ID labels (white boxes at midpoints)
- Route step numbers (colored circles showing order)
- Interactive tooltips with full metadata
- Layer controls (toggle segments, route, layers)
- Zoomable, pannable map
- OpenStreetMap base layer
```

**HTML Generation Flow:**
```python
def generate_html_map(segments, route_steps, output_file):
    # 1. Calculate center from all coordinates
    center_lat = avg(all_lats)
    center_lon = avg(all_lons)
    
    # 2. Create base Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # 3. Add segment layer with color coding
    for segment in segments:
        color = determine_color(segment)
        folium.PolyLine(segment.coordinates, color=color, ...).add_to(m)
        
        # Add ID label at midpoint
        folium.Marker(midpoint, icon=DivIcon(html=label)).add_to(m)
    
    # 4. Add route overlay if available
    if route_steps:
        route_polyline = PolyLine(all_route_coords, color=green, ...).add_to(m)
        
        # Add step numbers as numbered circles
        for step in route_steps:
            folium.Marker(start, icon=DivIcon(number)).add_to(m)
    
    # 5. Add legend
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 6. Save to file
    m.save(str(output_file))
```

**2. GeoJSON Export**
- **Location**: `drpp_visualization.py::DRPPVisualizer.generate_geojson()`
- **File**: `output/route_data.geojson`
- **Import**: QGIS, ArcGIS, Mapbox, or any GIS software

**GeoJSON Structure:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "segment_id": "SEG_001",
        "forward_required": true,
        "backward_required": true,
        "is_two_way_required": true,
        "one_way": false,
        "feature_type": "segment",
        // ... all metadata fields
        "length_m": 376.0,
        "length_km": 0.376
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [[lon, lat], [lon, lat], ...]  // Note: LON first!
      }
    },
    // ... route_steps also included as features
  ]
}
```

**3. SVG Visualization**
- **Location**: `drpp_visualization.py::DRPPVisualizer.generate_svg()`
- **File**: `output/route_map.svg`
- **Use**: Printing, presentations, documents

**SVG Features:**
```xml
<svg width="2000" height="2000">
  <!-- Segments as colored polylines -->
  <polyline points="x1,y1 x2,y2..." stroke="#FF0000" stroke-width="3"/>
  
  <!-- Segment IDs as text labels -->
  <text x="..." y="..." font-size="10">SEG_001</text>
  
  <!-- Route step numbers as circles -->
  <circle cx="..." cy="..." r="12" fill="#00AA00"/>
  <text x="..." y="..." fill="white">1</text>
  
  <!-- Legend -->
  <rect x="..." y="..." width="180" height="150" fill="white" stroke="black"/>
</svg>
```

### Pipeline Output Structure

```
output/
├── route_map.html          # Interactive map (primary output)
├── route_data.geojson      # Geographic data format
└── route_map.svg           # Vector graphics
```

### Pipeline Return Value

```python
def run(kml_file, algorithm='v4', output_dir='./output', ...):
    return {
        'route_steps': List[RouteStep],
        'total_distance': float,           # meters
        'coverage': float,                 # percentage
        'statistics': {
            'total_distance': float,
            'coverage': float,
            'deadhead_distance': float,    # Non-required segments
            'deadhead_percent': float,
            'segments_covered': int,
            'segments_unreachable': int,
            'required_count': int,
        },
        'output_files': {
            'html': Path,
            'geojson': Path,
            'svg': Path,
        }
    }
```

### Statistics Computed

**Total Distance:**
```python
total_distance = sum(result.distance for result in routing_results)
```

**Coverage Percentage:**
```python
required_count = sum(segment.required_traversals for segment in segments)
segments_covered = sum(result.segments_covered for result in routing_results)
coverage = (segments_covered / required_count) * 100 if required_count > 0 else 0
```

**Deadhead Distance:**
- Distance traveled that doesn't cover required segments
- Computed from route_steps with `is_deadhead` flag
- Shows efficiency of solution (lower is better)

### Logging & Progress Tracking

**Pipeline Logging:**
```
[1/5] Parsing KML and extracting segments...
      ✓ Loaded 347 segments
      ✓ Forward-only: 12
      ✓ Backward-only: 8
      ✓ Two-way required: 327

[2/5] Building directed graph...
      ✓ Graph has 892 nodes

[3/5] Solving DRPP with algorithm: V4...
      ✓ Generated route with 347 steps
      Result: 234.5km, 347 segments covered

[4/5] Generating visualizations...
      ✓ HTML: output/route_map.html
      ✓ GEOJSON: output/route_data.geojson
      ✓ SVG: output/route_map.svg

[5/5] Validation and summary...
      Total distance: 234.5 km
      Required coverage: 100.0%
      Deadhead distance: 23.4 km (10.0%)
      Output files: 3
```

### Visualization Configuration

```python
@dataclass
class VisualizationConfig:
    colors: Dict[str, str] = None  # Color schemes
    segment_label_size: int = 10
    step_label_size: int = 12
    line_width: int = 3
    show_segment_ids: bool = True
    show_step_numbers: bool = True
    show_direction_arrows: bool = True

# Usage
config = VisualizationConfig(
    colors={
        'forward_required': '#FF0000',
        'backward_required': '#0000FF',
        'both_required': '#9900FF',
    }
)
visualizer = DRPPVisualizer(config)
```

---

## Key Code Locations Summary

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| **KML Parsing** | `drpp_pipeline.py` | `DRPPPipeline._parse_kml()` |
| **Graph Building** | `Route_Planner.py` | `DirectedGraph`, `build_graph()` |
| **V4 Routing** | `drpp_core/greedy_router.py` | `greedy_route_cluster()` |
| **Distance Matrix** | `drpp_core/distance_matrix.py` | `DistanceMatrix` |
| **Clustering** | `drpp_core/clustering.py` | `cluster_segments()` |
| **Path Reconstruction** | `drpp_core/path_reconstruction.py` | `reconstruct_path()` |
| **Parallel Processing** | `drpp_core/parallel_executor.py` | `parallel_cluster_routing()` |
| **Visualization** | `drpp_visualization.py` | `DRPPVisualizer` |
| **OSM Integration** | `osm_speed_integration.py` | `OverpassSpeedFetcher` |

---

## Technology Stack Summary

- **Language**: Python 3.9+
- **Routing Algorithm**: Greedy Nearest-Neighbor (V4)
- **Graph Structure**: Custom DirectedGraph (adjacency list)
- **Distance Computation**: Haversine formula
- **Parallel Execution**: Python ProcessPoolExecutor
- **Clustering**: DBSCAN/KMeans (optional) or Grid-based
- **Visualization**: Folium (HTML) + SVG
- **Data Format**: KML input, GeoJSON/SVG output
- **External APIs**: Overpass API (OSM speeds)
- **Optional Advanced**: OR-Tools (not currently integrated)

