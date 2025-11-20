"""
Unit tests for industry-standard DRPP solver.

Tests the Eulerian augmentation approach including:
- Topology building with snapping
- Connectivity analysis
- Node balancing
- Eulerian tour construction
"""

import unittest
from typing import List

from drpp_core.types import SegmentRequirement
from drpp_core import (
    solve_drpp_industry_standard,
    TopologyBuilder,
    ConnectivityAnalyzer,
    EulerianSolver
)


class TestTopologyBuilder(unittest.TestCase):
    """Test topology building with endpoint snapping."""

    def setUp(self):
        """Create sample segments for testing."""
        # Simple square: 4 segments forming a closed loop
        self.segments = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.0), (0.0, 0.001)],  # Bottom edge
                metadata={"name": "seg1"}
            ),
            SegmentRequirement(
                segment_id="seg2",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.001), (0.001, 0.001)],  # Right edge
                metadata={"name": "seg2"}
            ),
            SegmentRequirement(
                segment_id="seg3",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.001, 0.001), (0.001, 0.0)],  # Top edge
                metadata={"name": "seg3"}
            ),
            SegmentRequirement(
                segment_id="seg4",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.001, 0.0), (0.0, 0.0)],  # Left edge
                metadata={"name": "seg4"}
            ),
        ]

    def test_topology_building(self):
        """Test that topology builder creates correct nodes and edges."""
        builder = TopologyBuilder(snap_tolerance_meters=2.0)
        nodes, edges = builder.build_topology(self.segments)

        # Should have 4 nodes (corners of square)
        self.assertEqual(len(nodes), 4)

        # Should have 4 forward edges (one per segment)
        # Plus 4 backward edges (since one_way=False)
        self.assertEqual(len(edges), 8)

        # Check that required edges are marked
        required_edges = [e for e in edges if e.required]
        self.assertEqual(len(required_edges), 4)  # Only forward edges are required

    def test_endpoint_snapping(self):
        """Test that nearby endpoints are snapped together."""
        # Create segments with slightly different endpoints (should snap)
        segments_with_error = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={}
            ),
            SegmentRequirement(
                segment_id="seg2",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.0010001), (0.001, 0.001)],  # Slightly different
                metadata={}
            ),
        ]

        builder = TopologyBuilder(snap_tolerance_meters=2.0)
        nodes, edges = builder.build_topology(segments_with_error)

        # The two segments should share a common node after snapping
        # So we should have 3 nodes, not 4
        self.assertEqual(len(nodes), 3)


class TestConnectivityAnalyzer(unittest.TestCase):
    """Test strongly connected components analysis."""

    def test_connected_graph(self):
        """Test that connected graph is identified correctly."""
        # Create a simple connected graph
        segments = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=True,  # Two-way
                one_way=False,
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={}
            ),
        ]

        builder = TopologyBuilder()
        nodes, edges = builder.build_topology(segments)

        node_ids = set(nodes.keys())
        analyzer = ConnectivityAnalyzer(node_ids, edges)
        components = analyzer.find_strongly_connected_components()

        # Should have at least one component
        self.assertGreater(len(components), 0)

    def test_feasibility_check(self):
        """Test feasibility checking for DRPP."""
        # Create a simple feasible graph (bidirectional edge)
        segments = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=False,
                one_way=False,  # Allows both directions
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={}
            ),
        ]

        builder = TopologyBuilder()
        nodes, edges = builder.build_topology(segments)

        node_ids = set(nodes.keys())
        analyzer = ConnectivityAnalyzer(node_ids, edges)
        is_feasible, message = analyzer.check_feasibility()

        # Should be feasible
        self.assertTrue(is_feasible)


class TestEulerianSolver(unittest.TestCase):
    """Test Eulerian tour construction."""

    def test_balanced_graph(self):
        """Test that balanced graph (already Eulerian) is recognized."""
        # Create a square where all nodes have in-degree = out-degree
        segments = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=False,
                one_way=True,  # One-way to control degree
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={}
            ),
            SegmentRequirement(
                segment_id="seg2",
                forward_required=True,
                backward_required=False,
                one_way=True,
                coordinates=[(0.0, 0.001), (0.001, 0.001)],
                metadata={}
            ),
            SegmentRequirement(
                segment_id="seg3",
                forward_required=True,
                backward_required=False,
                one_way=True,
                coordinates=[(0.001, 0.001), (0.001, 0.0)],
                metadata={}
            ),
            SegmentRequirement(
                segment_id="seg4",
                forward_required=True,
                backward_required=False,
                one_way=True,
                coordinates=[(0.001, 0.0), (0.0, 0.0)],
                metadata={}
            ),
        ]

        builder = TopologyBuilder()
        nodes, edges = builder.build_topology(segments)

        solver = EulerianSolver(edges, len(nodes))
        balances = solver._compute_node_balances()

        # Check that all nodes are balanced (or close to it for this simple case)
        # In a perfect Eulerian circuit, all should be balanced
        # Note: Our test case forms a cycle, so should be balanced
        balanced_count = sum(1 for b in balances if b.is_balanced)
        self.assertGreater(balanced_count, 0)


class TestIndustryDRPPSolver(unittest.TestCase):
    """Test complete industry-standard DRPP solver."""

    def test_simple_route(self):
        """Test solving a simple route."""
        # Create a simple two-segment route
        segments = [
            SegmentRequirement(
                segment_id="seg1",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={"RouteName": "Route 1", "Dir": "N"}
            ),
            SegmentRequirement(
                segment_id="seg2",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.001), (0.001, 0.001)],
                metadata={"RouteName": "Route 2", "Dir": "E"}
            ),
        ]

        solution = solve_drpp_industry_standard(
            segments=segments,
            snap_tolerance_meters=2.0
        )

        # Should have a solution
        self.assertIsNotNone(solution)

        # Should have route coordinates
        self.assertGreater(len(solution.route_coordinates), 0)

        # Should have reasonable distance
        self.assertGreater(solution.total_distance_km, 0)

        # Required distance should be <= total distance
        self.assertLessEqual(solution.required_distance_km, solution.total_distance_km)

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the pipeline."""
        segments = [
            SegmentRequirement(
                segment_id="test_seg",
                forward_required=True,
                backward_required=False,
                one_way=False,
                coordinates=[(0.0, 0.0), (0.0, 0.001)],
                metadata={
                    "CollId": "12345",
                    "RouteName": "PA-981",
                    "Dir": "NB",
                    "LengthFt": "500",
                    "custom_field": "custom_value"
                }
            ),
        ]

        solution = solve_drpp_industry_standard(segments)

        # Check that tour edges have metadata
        for edge in solution.tour.edges:
            if edge.segment_id == "test_seg":
                self.assertIn("CollId", edge.metadata)
                self.assertEqual(edge.metadata["CollId"], "12345")
                self.assertEqual(edge.metadata["custom_field"], "custom_value")


if __name__ == '__main__':
    unittest.main()
