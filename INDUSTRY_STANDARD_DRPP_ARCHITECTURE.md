# Industry-Standard DRPP Architecture

## Overview

This document describes the industry-standard Directed Rural Postman Problem (DRPP) implementation following the methodology used by DOTs, mapping companies, and AV data-collection firms (Esri, HERE, TomTom, Trimble).

## Pipeline Stages

### 1. Data Ingestion & Normalization

**Input**: KML file with Placemarks (road segments)

**Process**:
- Import KML using existing `DRPPPipeline._parse_kml()`
- Extract all metadata from ExtendedData (CollId, RouteName, Dir, LengthFt, etc.)
- Convert each Placemark → directed edge feature
- Preserve all attributes in metadata dictionary

**Output**: List of `SegmentRequirement` objects with full metadata

### 2. Graph Topology Construction

**Process**:
- **Snap endpoints** within tolerance (1-5 meters) to create proper node topology
- **Clean geometry** - remove duplicates, fix ordering
- **Build nodes** - create unique node IDs from snapped endpoints
- **Build directed edges** - create directed arcs with metadata
- **Ensure directionality** - respect Dir field (NB/SB/EB/WB/I/D)

**Output**: Directed multigraph with:
- Nodes: unique points (lat, lon)
- Edges: directed arcs with metadata and costs
- Edge types: REQUIRED (must traverse) and OPTIONAL (travel edges)

### 3. Required vs Optional Edge Classification

**Required Edges**:
- All road segments from KML that must be driven for data collection
- Marked with `required=True`
- Include full metadata from KML

**Optional Edges (Deadhead)**:
- Shortest-path connections between required edges
- Allow travel between disconnected segments
- Computed using Dijkstra's algorithm
- Lower priority than required edges

### 4. Strongly Connected Components Analysis

**Purpose**: Ensure graph is traversable

**Process**:
- Identify isolated components using Kosaraju's or Tarjan's algorithm
- Check if all required edges are reachable from start point
- Report connectivity issues before solving

**Output**: List of strongly connected components, reachability matrix

### 5. DRPP Solving via Eulerian Augmentation

This is the **core industry-standard methodology**:

#### Step 5.1: Node Balance Analysis

For a directed Eulerian tour to exist, every node must have:
- **in-degree = out-degree** (balanced node)

**Process**:
- Calculate degree balance for each node: `balance = out_degree - in_degree`
- Identify imbalanced nodes:
  - Positive balance: more outgoing than incoming (source)
  - Negative balance: more incoming than outgoing (sink)

#### Step 5.2: Minimum-Cost Matching

**Goal**: Add minimum-cost duplicate edges to balance all nodes

**Algorithm**:
1. Identify all source nodes (balance > 0) and sink nodes (balance < 0)
2. Compute shortest paths between all source-sink pairs
3. Solve minimum-cost flow problem:
   - Each source must send |balance| units
   - Each sink must receive |balance| units
   - Minimize total cost of flow
4. Add duplicate edges along shortest paths according to flow solution

**Implementation Options**:
- **Network Simplex** (scipy.optimize.linprog)
- **Hungarian algorithm** for balanced assignment (scipy.optimize.linear_sum_assignment)
- **OR-Tools** min-cost flow solver
- **NetworkX** min_cost_flow

#### Step 5.3: Hierholzer's Algorithm (Eulerian Tour)

Once graph is balanced (Eulerian), construct tour:

**Algorithm**:
1. Start at specified start node (or any node with edges)
2. Follow edges, removing each as traversed
3. When stuck, backtrack to node with unused edges
4. Merge sub-cycles into main tour
5. Result: Single tour visiting all edges exactly once

**Output**: Ordered sequence of edges (Eulerian tour)

#### Step 5.4: Tour Simplification

**Process**:
- Merge consecutive edges on same road segment
- Collapse duplicate traversals
- Retain metadata for each segment
- Add sequence numbers

### 6. Metadata Preservation

**Throughout pipeline**:
- Node IDs track back to original coordinates
- Edge IDs link to original segment_id (CollId)
- Metadata dictionary preserved in edge attributes
- Final route includes all original KML metadata

**Metadata fields preserved**:
- CollId, RouteName, Dir, LengthFt
- Region, Juris, CntyCode, StRtNo, SegNo
- BegM, EndM, IsPilot, Collected
- Custom ExtendedData fields
- Computed fields: sequence_number, traversal_count, is_deadhead

### 7. Export Formats

#### 7.1 KML/KMZ Output
- Ordered LineStrings with full ExtendedData
- Color-coded by:
  - Required vs deadhead
  - Route sequence
  - Traversal count
- Direction arrows
- Labels with metadata

#### 7.2 GeoJSON Output
- Feature collection with properties:
  - All original metadata
  - sequence_number
  - is_required, is_deadhead
  - cumulative_distance
- Machine-readable for GIS

#### 7.3 Turn-by-Turn Directions
- CSV/JSON format
- Fields:
  - Step number
  - Instruction (e.g., "Continue on PA-981 NB")
  - Distance
  - Cumulative distance
  - Segment metadata

#### 7.4 HTML Visualization
- Interactive map with Leaflet
- Route animation
- Metadata tooltips
- Progress tracking

### 8. Field Deployment

**Navigation formats**:
- KML for Google Earth / Trimble devices
- GPX for GPS navigation
- GeoJSON for web dashboards
- CSV with coordinates + metadata

## Implementation Modules

### Module: `drpp_core/topology.py`
- Endpoint snapping
- Geometry cleaning
- Node/edge construction

### Module: `drpp_core/connectivity.py`
- Strongly connected components (Tarjan's algorithm)
- Reachability analysis
- Component visualization

### Module: `drpp_core/eulerian_solver.py`
- Node balance calculation
- Minimum-cost matching
- Hierholzer's Eulerian tour
- Tour simplification

### Module: `drpp_core/drpp_solver.py`
- Main DRPP solver orchestrator
- Combines topology + connectivity + eulerian solver
- Metadata preservation
- Quality metrics

### Module: `drpp_core/export.py`
- KML writer with ExtendedData
- Turn-by-turn generator
- Multi-format export

## Algorithm Complexity

**Time Complexity**:
- Snapping: O(n log n) with spatial index
- SCC detection: O(V + E)
- Shortest paths: O((V + E) log V) with Dijkstra
- Min-cost matching: O(n³) with Hungarian, O(n²m) with network simplex
- Eulerian tour: O(E)
- **Total**: O(n³) for min-cost matching dominates

**Space Complexity**:
- Graph storage: O(V + E)
- Distance matrix: O(n²) where n = number of imbalanced nodes
- **Total**: O(V + E + n²)

**Scalability**:
- Efficient for 1,000-10,000 segments (typical DOT projects)
- For larger datasets: hierarchical decomposition or heuristic matching

## Comparison with Current Implementation

| Aspect | Current (Greedy) | Industry-Standard (Eulerian) |
|--------|------------------|------------------------------|
| **Algorithm** | Nearest-neighbor greedy | Eulerian augmentation + Hierholzer |
| **Optimality** | Heuristic (good solutions) | Optimal for Eulerian, near-optimal for matching |
| **Coverage** | All required edges | All required edges (guaranteed) |
| **Deadhead** | Minimized locally | Minimized globally |
| **Balance** | Not guaranteed | Guaranteed (Eulerian property) |
| **Use case** | Fast approximation | Production DOT work |

## Benefits of Industry-Standard Approach

1. **Theoretical guarantee**: Eulerian tour visits all required edges exactly once
2. **Optimal deadhead**: Global optimization vs local greedy decisions
3. **Production-proven**: Used by Esri, HERE, TomTom, state DOTs
4. **Balanced routes**: No node visited excessively
5. **Reproducible**: Deterministic solution (vs greedy randomness)
6. **Metadata integrity**: Full preservation through pipeline

## References

- Edmonds & Johnson (1973): Matching, Euler tours and the Chinese postman
- Eiselt et al. (1995): Arc routing problems, part I & II
- Ford & Fulkerson (1962): Flows in Networks
- pgRouting documentation: Rural Postman Problem
- ESRI Network Analyst: Route optimization methodology
