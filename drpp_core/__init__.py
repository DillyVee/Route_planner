"""
DRPP Core - Production-Ready Directed Rural Postman Problem Solver

This package provides industrial-strength components for solving the DRPP with:
- Memory-efficient distance matrix computation
- Robust path reconstruction
- Geographic-aware clustering
- Parallel processing with minimal memory overhead
- Comprehensive error handling and logging

Author: Refactored to production standards
Version: 4.0.0
"""

from .clustering import ClusteringMethod, cluster_segments
from .distance_matrix import DistanceMatrix, compute_distance_matrix
from .exceptions import (
    ClusteringError,
    ConfigurationError,
    DisconnectedGraphError,
    DRPPError,
    GraphBuildError,
    GraphError,
    KMLParseError,
    NoPathError,
    OptimizationError,
    OSMError,
    OSMMatchingError,
    OverpassAPIError,
    ParseError,
    RoutingError,
    UnreachableSegmentError,
    ValidationError,
    VisualizationError,
)
from .geo import calculate_bearing, haversine, snap_coordinate
from .greedy_router import greedy_route_cluster
from .parallel_executor import (
    estimate_optimal_workers,
    parallel_cluster_routing,
    sequential_cluster_routing_ondemand,
)
from .path_reconstruction import reconstruct_path
from .types import ClusterResult, Coordinate, NodeID, PathResult, SegmentIndex, SegmentRequirement

# Industry-Standard DRPP Solver (NEW)
from .topology import TopologyBuilder, TopologyNode, TopologyEdge, build_adjacency_list, get_node_degrees
from .connectivity import ConnectivityAnalyzer, ConnectedComponent, check_graph_connectivity
from .eulerian_solver import EulerianSolver, EulerianTour, NodeBalance, solve_drpp_eulerian
from .industry_drpp_solver import IndustryDRPPSolver, DRPPSolution, solve_drpp_industry_standard
from .kml_export import KMLExporter, export_drpp_to_kml
from .turn_by_turn import TurnByTurnGenerator, TurnInstruction, generate_turn_by_turn

# Backward compatibility alias
parallel_cluster_routing_ondemand = sequential_cluster_routing_ondemand

__all__ = [
    # Types
    "Coordinate",
    "NodeID",
    "SegmentIndex",
    "PathResult",
    "ClusterResult",
    "SegmentRequirement",
    # Distance Matrix
    "DistanceMatrix",
    "compute_distance_matrix",
    # Path Reconstruction
    "reconstruct_path",
    # Clustering
    "cluster_segments",
    "ClusteringMethod",
    # Routing
    "greedy_route_cluster",
    # Parallel Processing
    "parallel_cluster_routing",
    "sequential_cluster_routing_ondemand",
    "parallel_cluster_routing_ondemand",  # Backward compatibility alias
    "estimate_optimal_workers",
    # Geographic Utilities
    "haversine",
    "snap_coordinate",
    "calculate_bearing",
    # Industry-Standard DRPP Solver
    "TopologyBuilder",
    "TopologyNode",
    "TopologyEdge",
    "build_adjacency_list",
    "get_node_degrees",
    "ConnectivityAnalyzer",
    "ConnectedComponent",
    "check_graph_connectivity",
    "EulerianSolver",
    "EulerianTour",
    "NodeBalance",
    "solve_drpp_eulerian",
    "IndustryDRPPSolver",
    "DRPPSolution",
    "solve_drpp_industry_standard",
    "KMLExporter",
    "export_drpp_to_kml",
    "TurnByTurnGenerator",
    "TurnInstruction",
    "generate_turn_by_turn",
    # Exceptions
    "DRPPError",
    "ParseError",
    "KMLParseError",
    "ValidationError",
    "GraphError",
    "GraphBuildError",
    "DisconnectedGraphError",
    "RoutingError",
    "NoPathError",
    "UnreachableSegmentError",
    "OptimizationError",
    "ClusteringError",
    "OSMError",
    "OverpassAPIError",
    "OSMMatchingError",
    "VisualizationError",
    "ConfigurationError",
]

__version__ = "4.0.0"
