# Changelog

All notable changes to the Route Planner DRPP project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-11-17

### Major Refactor - Production V4 Release

Complete production-ready refactor with professional organization and packaging.

### Added

#### Repository Organization
- Professional README.md with quick start guide and feature overview
- MIT LICENSE file
- Modern `pyproject.toml` for PEP 517/518 packaging
- CHANGELOG.md for version tracking
- Organized `docs/` directory with all documentation
- `legacy/` directory for historical implementations
- Updated `.gitignore` for Python best practices

#### MapPlus/Duweis Format Support
- Full support for roadway survey KML format
- Automatic extraction of MapPlusCustomFeatureClass fields:
  - CollId, RouteName, Dir, LengthFt, Region, Juris, CntyCode
  - StRtNo, SegNo, BegM, EndM, IsPilot, Collected
- Rich tooltips in visualizations with all roadway metadata
- GeoJSON export with computed metrics (length_m, length_km)
- Automatic feet-to-meters conversion

#### V4 Performance Optimization
- On-demand Dijkstra routing mode for large datasets
- Automatic switching when clusters have >1000 endpoints
- 10-100x performance improvement for large datasets
- Memory-efficient distance computation
- Progress: Before = 122M all-pairs computations, After = ~11k single-source calls

#### Complete DRPP Pipeline
- `drpp_pipeline.py` - Main pipeline orchestrator
- `drpp_visualization.py` - Rich visualization generation
- `run_drpp_pipeline.py` - Command-line interface
- Support for multiple algorithms: V4, RFCS, Greedy, Hungarian
- Multiple output formats: HTML, GeoJSON, SVG
- Interactive HTML maps with Folium/Leaflet
- Color-coded segments by requirement type
- Segment ID labels and route step numbering

#### V4 Core Modules
- `drpp_core/greedy_router.py` - Optimized greedy routing with on-demand mode
- `drpp_core/distance_matrix.py` - Memory-efficient matrix storage
- `drpp_core/clustering.py` - Geographic-aware clustering (DBSCAN)
- `drpp_core/path_reconstruction.py` - Robust path handling
- `drpp_core/parallel_executor.py` - Multi-core processing support
- `drpp_core/types.py` - Type definitions and interfaces
- `drpp_core/logging_config.py` - Structured logging
- `drpp_core/profiling.py` - Performance profiling

#### Documentation
- Comprehensive PIPELINE_GUIDE.md
- V4_INTEGRATION_SUMMARY.md with architecture details
- PRODUCTION_REFACTOR_GUIDE.md with complete API reference
- Example code in `example_production_usage.py`

### Changed

#### Performance Improvements
- Greedy router now auto-detects large clusters and uses on-demand mode
- No more all-pairs distance matrix for large datasets
- Reduced memory footprint for massive route problems

#### Code Quality
- Comprehensive type hints across all modules
- Detailed docstrings with examples
- Separation of concerns (routing, clustering, visualization)
- Professional error handling and logging
- 50+ unit tests across 4 test files

### Fixed

#### V4 Compatibility Issues
- Non-contiguous node ID support (no longer requires IDs 0, 1, 2, ...)
- List-based graph support (nodes can be list indices or dict keys)
- Robust coordinate-to-ID mapping
- Fallback mechanisms for malformed KML files

#### Performance Issues
- V4 greedy taking forever on 11,060+ segments (SOLVED with on-demand mode)
- Memory issues with massive all-pairs distance matrices (SOLVED with on-demand)
- 2-opt optimization bottlenecks (addressed with auto-detection)

### Removed

- Graph pickling dependencies (no longer needed in V4)
- Redundant distance matrix computations
- Legacy code moved to `legacy/` directory for historical reference

---

## [3.0.0] - 2025-11-15

### Added
- V4 greedy algorithm with production-ready implementation
- Memory-efficient distance matrix computation
- Geographic clustering with DBSCAN
- Parallel processing support
- Comprehensive error handling

### Changed
- Refactored from monolithic Route_Planner.py to modular architecture
- Improved type safety with type hints
- Enhanced logging and profiling

---

## [2.0.0] - Earlier

### Added
- Greedy routing implementation
- RFCS algorithm support
- Basic KML parsing
- Simple visualization

---

## [1.0.0] - Earlier

### Added
- Initial DRPP solver implementation
- Basic route optimization
- KML file support

---

## Upgrade Notes

### Upgrading to 4.0.0

**Breaking Changes**: None - V4 API is fully backward compatible with V3

**Recommended Migration**:
1. Update `requirements_production.txt` dependencies
2. Use `drpp_pipeline.DRPPPipeline` for new projects
3. Switch to V4 greedy for large datasets (>1000 segments)
4. Take advantage of MapPlus format if applicable

**New Features You Should Use**:
- Set `algorithm='v4'` for best performance on large datasets
- Use `output_formats=['html', 'geojson', 'svg']` for rich visualizations
- Enable on-demand mode explicitly with `use_ondemand=True` if needed

### Upgrading to 3.0.0

See `docs/V2_TO_V3_MIGRATION.md` for detailed migration guide.

---

## Contributing

When making changes:
1. Update this CHANGELOG with your changes under "Unreleased" section
2. Follow [Keep a Changelog](https://keepachangelog.com/) format
3. Use semantic versioning for version numbers

## Version History Legend

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Now removed features
- **Fixed** - Bug fixes
- **Security** - Vulnerability fixes

[4.0.0]: https://github.com/DillyVee/Route_planner/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/DillyVee/Route_planner/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/DillyVee/Route_planner/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/DillyVee/Route_planner/releases/tag/v1.0.0
