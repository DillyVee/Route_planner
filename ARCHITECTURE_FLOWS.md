# Architecture & Data Flow Diagrams

## Complete Pipeline Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DRPP Pipeline.run()                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [PHASE 1] Parse KML                    (drpp_pipeline.py:187)       │
├──────────────────────────────────────────────────────────────────────┤
│ 1. XML.parse(kml_file)                                              │
│    └─ Handle XML errors, remove control chars, fix ampersands      │
│                                                                      │
│ 2. For each Placemark:                                              │
│    ├─ Extract LineString coordinates                               │
│    ├─ snap_coord(lat, lon, precision=6) → deduplicate             │
│    ├─ Extract ExtendedData (MapPlus format)                        │
│    │  ├─ MapPlusCustomFeatureClass → CollId, RouteName, Dir...   │
│    │  └─ MapPlusSystemData → label information                    │
│    └─ Determine one_way flag + required_traversals                │
│                                                                      │
│ Returns: List[SegmentRequirement]                                   │
│   ├─ segment_id: str                                               │
│   ├─ forward_required: bool                                        │
│   ├─ backward_required: bool                                       │
│   ├─ coordinates: List[(lat, lon)]                                 │
│   └─ metadata: Dict[all_kml_fields]                                │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [PHASE 2] Build Directed Graph         (Route_Planner.py:353)      │
├──────────────────────────────────────────────────────────────────────┤
│ DirectedGraph Setup:                                                 │
│   ├─ node_to_id: Dict[Coordinate, NodeID]                          │
│   ├─ id_to_node: List[Coordinate]                                  │
│   └─ adj: List[List[(NodeID, weight)]]                             │
│                                                                      │
│ For each segment in segments:                                       │
│   ├─ For each consecutive point pair (p1, p2):                    │
│   │   ├─ dist = haversine(p1, p2)  [meters]                       │
│   │   ├─ graph.add_edge(p1, p2, dist)  [forward]                  │
│   │   └─ graph.add_edge(p2, p1, dist)  [backward]                 │
│   └─ Creates nodes on-demand via _ensure(node)                    │
│                                                                      │
│ Optional: Fetch OSM roads in bounding box                           │
│   └─ fetch_osm_roads_for_routing(bbox)                             │
│                                                                      │
│ Returns: DirectedGraph with |V| nodes, weighted edges               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [PHASE 3] Solve DRPP                                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Algorithm Selection:                                                 │
│   ├─ v4       → drpp_core/greedy_router.py (RECOMMENDED)           │
│   ├─ rfcs     → legacy/parallel_processing_addon_rfcs.py           │
│   ├─ greedy   → legacy/parallel_processing_addon_greedy.py         │
│   └─ hungarian → legacy/parallel_processing_addon.py               │
│                                                                      │
│ V4 Greedy Algorithm Flow:                                           │
│                                                                      │
│   greedy_route_cluster(                                             │
│     graph, required_edges, segment_indices, start_node             │
│   )                                                                  │
│   │                                                                  │
│   ├─ Extract endpoints from required_edges                         │
│   │  └─ endpoints = [(seg[0], seg[1]) for seg in required_edges]  │
│   │                                                                  │
│   ├─ IF num_endpoints > 1000:                                      │
│   │   ├─ Enable ON-DEMAND MODE                                    │
│   │   └─ Use live Dijkstra per step                                │
│   │ ELSE:                                                           │
│   │   ├─ Precompute distance matrix                               │
│   │   ├─ compute_all_pairs_shortest_paths()                       │
│   │   └─ Store in DistanceMatrix(dict or numpy)                   │
│   │                                                                  │
│   ├─ GREEDY MAIN LOOP:                                             │
│   │  current_pos = start_node                                      │
│   │  remaining = set(range(num_segments))                          │
│   │  path = []                                                      │
│   │                                                                  │
│   │  while remaining:                                               │
│   │    1. Find nearest endpoint from current_pos                   │
│   │       └─ nearest_seg = argmin(dist[current][endpoint[i]])     │
│   │                                                                  │
│   │    2. Route to nearest endpoint                                │
│   │       ├─ IF on-demand: Dijkstra(current_pos)                  │
│   │       ├─ ELSE: Use precomputed matrix                         │
│   │       └─ path.extend(shortest_path_coords)                    │
│   │                                                                  │
│   │    3. Traverse required segment direction                      │
│   │       ├─ IF forward_required:                                  │
│   │       │   └─ path.extend(segment.coordinates)                 │
│   │       ├─ IF backward_required:                                 │
│   │       │   └─ path.extend(segment.coordinates[::-1])           │
│   │       └─ Mark as covered                                       │
│   │                                                                  │
│   │    4. Update current_pos = segment.end_node                   │
│   │                                                                  │
│   └─ Return PathResult(                                             │
│       path, total_distance, segments_covered, ...                  │
│     )                                                               │
│                                                                      │
│ Parallel Mode (if multiple clusters):                               │
│   ├─ Parent: compute_distance_matrix(all_endpoints)               │
│   ├─ Create ClusterTask for each cluster                          │
│   │  └─ ClusterTask = {                                            │
│   │     cluster_id, segment_indices,                              │
│   │     distance_matrix (precomputed!),                           │
│   │     normalizer, start_node_id                                  │
│   │    }                                                            │
│   ├─ Spawn ProcessPoolExecutor(num_workers)                        │
│   ├─ Each worker: greedy_route_cluster(no_graph, matrix_only)    │
│   └─ Collect PathResult from each worker                          │
│                                                                      │
│ Returns: List[PathResult] with coverage statistics                  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [PHASE 4] Generate Visualizations                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ DRPPVisualizer.generate_html_map():                                 │
│   ├─ Calculate map center from all coordinates                     │
│   ├─ Create Folium base map                                        │
│   ├─ Add segment layer:                                            │
│   │  ├─ Color by requirement: Red|Blue|Purple|Gray                │
│   │  └─ Add ID labels at midpoints                                │
│   ├─ Add route overlay (if route_steps available)                 │
│   │  ├─ Green polyline showing full path                         │
│   │  └─ Numbered circles at step starts                          │
│   ├─ Add legend with color meanings                               │
│   └─ Save to output/route_map.html                                │
│                                                                      │
│ DRPPVisualizer.generate_geojson():                                  │
│   ├─ For each segment:                                             │
│   │  └─ Create Feature with all metadata in properties            │
│   ├─ For each route_step:                                         │
│   │  └─ Create Feature with step info                             │
│   └─ Save as FeatureCollection to output/route_data.geojson       │
│                                                                      │
│ DRPPVisualizer.generate_svg():                                      │
│   ├─ Calculate lat/lon to SVG coordinate transform                │
│   ├─ Draw segments as colored polylines                           │
│   ├─ Add text labels for segment IDs                              │
│   ├─ Add legend and title                                         │
│   └─ Save to output/route_map.svg                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ [PHASE 5] Compute Statistics & Return                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Stats Calculation:                                                   │
│   ├─ total_distance = sum(result.distance)                         │
│   ├─ segments_covered = sum(result.segments_covered)               │
│   ├─ required_count = sum(segment.required_traversals)             │
│   ├─ coverage% = (segments_covered / required_count) × 100         │
│   ├─ deadhead_distance = (traveled - covered) distance             │
│   └─ deadhead% = (deadhead / total) × 100                          │
│                                                                      │
│ Return:                                                              │
│ {                                                                    │
│   'route_steps': List[RouteStep],                                   │
│   'total_distance': float,          # meters                        │
│   'coverage': float,                # percentage                    │
│   'statistics': {...},              # detailed stats                │
│   'output_files': {                 # paths to outputs              │
│     'html': Path,                                                    │
│     'geojson': Path,                                                │
│     'svg': Path,                                                    │
│   }                                                                  │
│ }                                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure Hierarchy

```
INPUT: KML File
  │
  ├─[Placemark 1]
  │  ├─Coordinates: (lat, lon), (lat, lon), ...
  │  ├─ExtendedData
  │  │  ├─SchemaData: MapPlusCustomFeatureClass
  │  │  │  ├─CollId: "SEG_001"
  │  │  │  ├─Dir: "NB"
  │  │  │  ├─RouteName: "PA-981"
  │  │  │  └─... more fields
  │  │  └─SchemaData: MapPlusSystemData
  │  │     └─LABEL_TEXT: ...
  │  └─Description, Name: speed limits, etc.
  │
  └─[Placemark N]
     └─...
         │
         ▼
PARSE_KML()
    │
    ▼
SegmentRequirement(
    segment_id="SEG_001",
    forward_required=True,
    backward_required=True,
    one_way=False,
    coordinates=[(lat,lon), ...],
    metadata={
        'CollId': 'SEG_001',
        'RouteName': 'PA-981',
        'Dir': 'NB',
        'length_ft': 1234.5,
        ...all_extended_data...
    }
)
    │
    ▼
BUILD_GRAPH()
    │
    ├─ Create DirectedGraph
    │  ├─node_to_id: {(lat,lon): node_id, ...}
    │  ├─id_to_node: [(lat,lon), ...]
    │  └─adj: [[(node_id,weight), ...], ...]
    │
    └─ Extract required_edges: [
        (start_coord, end_coord, full_coords, "forward"),
        (start_coord, end_coord, full_coords, "backward"),
        ...
       ]
         │
         ▼
SOLVE_DRPP()
    │
    ├─ [On-Demand] Dijkstra per step
    │  └─ (distances, predecessors)
    │
    └─ [Distance Matrix] Precompute all pairs
       ├─ DistanceMatrix(use_numpy=False)  # dict-based
       │  ├─distances: {(src,dst): distance}
       │  └─path_node_ids: {(src,dst): [node_ids]}
       │
       └─ OR DistanceMatrix(use_numpy=True)   # numpy-based
          ├─distance_matrix: np.array (nxn)
          └─path_matrix: List[List[node_ids]]
          
    Greedy Loop generates:
    path: [(lat,lon), (lat,lon), ...] all route coordinates
         │
         ▼
PathResult(
    path=[(lat,lon), ...],
    distance=12345.5,  # meters
    cluster_id=0,
    segments_covered=347,
    segments_unreachable=0,
    computation_time=1.23
)
         │
         ▼
VISUALIZE()
    │
    ├─ HTML Map (Folium)
    │  ├─Base: OpenStreetMap
    │  ├─Layer: Segments (color-coded)
    │  ├─Layer: Route overlay
    │  ├─Markers: Segment IDs + step numbers
    │  └─Legend + Interactive controls
    │
    ├─ GeoJSON
    │  ├─Feature per segment with metadata
    │  ├─Feature per route step
    │  └─FeatureCollection wrapper
    │
    └─ SVG
       ├─Colored polylines for segments
       ├─Text labels for IDs
       ├─Circles for step numbers
       └─Legend
```

---

## Distance Matrix Organization

### Dict-Based (Default for small graphs)

```
DistanceMatrix:
  distances: {
    (src_id, dst_id): 1234.5,  # shortest distance in meters
    (src_id, dst_id): 5678.9,
    ...
  }
  
  path_node_ids: {
    (src_id, dst_id): [node_0, node_1, node_2, ...],  # path as node IDs
    ...
  }
  
  id_to_coords: {
    node_id: (lat, lon),
    ...
  }
```

### NumPy-Based (For large graphs >1000 nodes)

```
DistanceMatrix:
  distance_matrix: np.array of shape (n_nodes, n_nodes)
    └─ [i, j] = shortest distance from node i to node j
                or inf if no path
  
  path_matrix: List[List[List[NodeID]]]
    └─ [i][j] = [node_0, node_1, ...] path from i to j
```

### Lifecycle in V4 Greedy

```
1. PARENT PROCESS:
   ├─ Graph loaded entirely
   ├─ compute_distance_matrix() called
   │  ├─ For each source_id:
   │  │  ├─ dijkstra(source_id) from graph
   │  │  ├─ Store shortest distances & paths
   │  │  └─ Free intermediate structures
   │  └─ Returns DistanceMatrix(dict or numpy)
   │
   └─ Create ClusterTask with matrix
      ├─ NO graph object (pickle too large!)
      ├─ Only lightweight matrix + metadata
      └─ Send to workers via ProcessPoolExecutor
   
2. WORKER PROCESS:
   ├─ Receive ClusterTask with precomputed matrix
   ├─ Call greedy_route_cluster(
   │    graph=None,
   │    distance_matrix=task.distance_matrix,
   │    ...
   │  )
   ├─ Use matrix for all distance lookups
   │  └─ matrix.get(source_id, target_id)
   │
   └─ If no path in matrix:
      ├─ Try dijkstra_fallback() if graph available
      └─ Otherwise: mark unreachable
```

---

## Memory Efficiency Strategy

### Why Not Just Pass Graph to Workers?

```
Graph Pickling Problem:
  
  DirectedGraph with 10,000 nodes:
    ├─node_to_id: Dict[10k entries] → large pickle
    ├─id_to_node: List[10k coords] → large pickle
    └─adj: List[List[...]] → VERY large pickle!
           └─Could have 100k+ edges
  
  Serialization Cost:
    ├─ Large binary pickle files
    ├─ Network transfer between processes
    ├─ Deserialization in worker
    └─ Memory duplication in each worker (num_workers × pickle_size)

Distance Matrix Alternative:
  
  Only store needed paths:
    ├─ DistanceMatrix size ≈ k² where k = #segments
    ├─ For 1000 segments: ~1000 distances (not 10k²!)
    ├─ Much smaller pickle
    └─ Linear memory per worker (not graph duplication)
```

---

## Clustering & Parallel Execution

### Geographic Clustering

```
Clustering Methods (optional, for large datasets):

1. DBSCAN (Density-based):
   ├─ Input: List of segment centroids (lat, lon)
   ├─ eps_km: Geographic radius (e.g., 2.0 km)
   ├─ min_samples: Minimum points to form cluster
   │
   └─ Use sklearn.cluster.DBSCAN
      └─ metric='haversine' with radians conversion
   
   Output:
   ├─ Cluster assignments for each segment
   └─ Noise points as separate cluster

2. K-Means:
   ├─ Input: Centroids + desired number of clusters (k)
   ├─ sklearn.cluster.KMeans(n_clusters=k)
   │
   └─ Output: Cluster assignments

3. Grid-based (Fallback):
   ├─ No sklearn required
   ├─ Divide bbox into grid cells
   ├─ Assign segments to cells
   │
   └─ Always available, no dependencies
```

### Parallel Processing Flow

```
Sequential:
  clusters = {0: [seg0, seg1, ..., seg_N]}
  └─ Single cluster = full route

Parallel:
  clusters = {
    0: [seg0, seg1, ..., seg_10],     # Geographic cluster 1
    1: [seg11, seg12, ..., seg_20],   # Geographic cluster 2
    ...
    n: [seg_K, ...]                   # Geographic cluster N
  }
  
  cluster_order = [0, 1, 2, ..., n]  # Route between clusters
  
  ├─ Parent:
  │  ├─ Precompute distance matrix for ALL segments
  │  ├─ Create ClusterTask for each cluster
  │  └─ Submit to ProcessPoolExecutor(num_workers)
  │
  └─ Workers (in parallel):
     ├─ Worker 0 routes cluster 0
     ├─ Worker 1 routes cluster 1
     ├─ Worker 2 routes cluster 2
     └─ ... (up to num_workers workers)
  
  ├─ Collect results as each cluster completes
  └─ Combine into full route
```

---

## Key Decision Points in V4 Algorithm

```
┌─ Start at start_node
│
├─ DO:
│  │
│  ├─[1] Check num_endpoints
│  │    ├─ IF > 1000 → ON-DEMAND MODE
│  │    └─ ELSE → USE MATRIX MODE
│  │
│  ├─[2] Find nearest uncovered segment endpoint
│  │    ├─ IN ON-DEMAND: Dijkstra(current_pos) → find min distance
│  │    └─ IN MATRIX: matrix.get(current_id, *) → find min
│  │
│  ├─[3] Route to that endpoint
│  │    ├─ Get path from matrix OR Dijkstra
│  │    └─ Check if path exists (not unreachable)
│  │
│  ├─[4] Traverse required direction(s)
│  │    ├─ IF forward_required: traverse A→B
│  │    └─ IF backward_required: traverse B→A
│  │
│  ├─[5] Mark segment as covered
│  │    └─ Remove from remaining set
│  │
│  └─[6] Update current position
│       └─ Now at end of last traversed segment
│
└─ WHILE segments remaining
   └─ Exit when all covered or unreachable
```

---

## Error Handling Hierarchy

```
Exception Types (drpp_core/exceptions.py):

DRPPError (base)
├─ ParseError
│  ├─ KMLParseError
│  └─ XMLParseError
│
├─ ValidationError
│  └─ SegmentError
│
├─ GraphError
│  ├─ GraphBuildError
│  ├─ DisconnectedGraphError
│  └─ InvalidNodeError
│
├─ RoutingError
│  ├─ NoPathError
│  └─ UnreachableSegmentError
│
├─ OptimizationError
│  └─ ClusteringError
│
├─ OSMError
│  ├─ OverpassAPIError
│  └─ OSMMatchingError
│
└─ VisualizationError


Recovery Strategies:

1. XML Parse Error → Try to fix:
   ├─ Remove control characters
   ├─ Escape unescaped ampersands
   └─ Retry with fixed content

2. Missing Node ID → Fallback:
   ├─ Use alternative node representation
   └─ Generate synthetic ID

3. No Path Exists → Record as unreachable:
   ├─ Log reason
   ├─ Continue with remaining segments
   └─ Return count in statistics

4. Unrecoverable Error → Bubble up:
   └─ Return empty result or raise
```

---

## Metadata Flow Through Pipeline

```
KML File
  └─ ExtendedData fields
     └─[PARSE] SegmentRequirement.metadata dict
        ├─ All original fields preserved
        ├─ CollId → segment_id (primary)
        ├─ Dir → direction_code (for reference)
        ├─ LengthFt → length_ft (original) + length_m (converted)
        ├─ RouteName, Region, Juris, etc. → stored as-is
        └─ [SPEED] If maxspeed/speed_limit → parse_speed_limit()
           └─ Convert mph to km/h if needed
           
        [BUILD GRAPH]
        └─ Metadata passed through but not used for routing
        
        [VISUALIZE - HTML Map]
        └─ Display in interactive tooltips
           ├─ Segment ID
           ├─ Route name
           ├─ Direction code
           ├─ Length (ft and m)
           ├─ Region, county, state route
           └─ Collection date
        
        [VISUALIZE - GeoJSON]
        └─ Include all metadata in Feature properties
           ├─ segment_id
           ├─ forward_required / backward_required
           ├─ is_two_way_required
           ├─ one_way
           ├─ All original KML fields
           └─ Computed fields (length_km, etc.)
```

---

## OSM Speed Integration

```
[OPTIONAL] Enrich segments with OSM speeds:

1. Fetch OSM data:
   ├─ Calculate bounding box from segments
   ├─ Query Overpass API: way["highway"](bbox)
   ├─ Cache results in overpass_cache.json
   └─ Rate limit: 1 sec between requests

2. Extract speeds:
   ├─ For each way:
   │  ├─ Try maxspeed tag
   │  └─ If missing: use highway type default
   │     ├─ motorway: 110 km/h
   │     ├─ trunk: 100
   │     ├─ primary: 80
   │     ├─ secondary: 60
   │     ├─ residential: 30
   │     └─ etc.
   │
   └─ Store in segment metadata

3. Build time-weighted graph:
   ├─ Instead of distance-only weights
   ├─ Weight = distance / speed_kmh (converts to time)
   ├─ Routing then minimizes time instead of distance
   └─ Useful for delivery optimization

Result: segments enriched with speed_limit,
        routing can use time-based weights
```

---

## Parallel Executor Architecture

```
ProcessPoolExecutor Pattern:

┌─ MAIN PROCESS
│  ├─[1] Load graph
│  ├─[2] Precompute distance matrix
│  │     └─ DistanceMatrix(dict or numpy)
│  │
│  ├─[3] For each cluster:
│  │     └─ Create lightweight ClusterTask
│  │        ├─ cluster_id
│  │        ├─ segment_indices
│  │        ├─ distance_matrix (SHARED!)
│  │        ├─ normalizer
│  │        ├─ start_node_id
│  │        └─ enable_fallback=False
│  │
│  ├─[4] Submit to ProcessPoolExecutor.submit(
│  │        _route_cluster_worker, task
│  │     )
│  │
│  └─[5] Collect ClusterTaskResult from as_completed()
│
├─ WORKER PROCESS 0
│  ├─ Receive ClusterTask(cluster_0)
│  ├─ Call greedy_route_cluster(
│  │    graph=None,  # Not available!
│  │    distance_matrix=task.distance_matrix,
│  │    ...
│  │  )
│  └─ Return ClusterTaskResult
│
├─ WORKER PROCESS 1
│  ├─ Receive ClusterTask(cluster_1)
│  └─ ... same as worker 0 ...
│
└─ WORKER PROCESS N
   └─ ... same ...

Key: Workers receive NO graph object
     Only precomputed matrix + metadata
     Minimizes pickle/network overhead
     Reduces memory duplication
```

