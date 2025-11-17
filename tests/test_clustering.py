"""
Unit tests for clustering module.

Tests geographic distance calculations and clustering methods.
"""

import sys
from pathlib import Path

# Add parent directory to path before importing drpp_core
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest  # noqa: E402

from drpp_core.clustering import (  # noqa: E402
    ClusteringMethod,
    cluster_segments_grid,
    haversine,
)


class TestHaversine(unittest.TestCase):
    """Test haversine distance calculation."""

    def test_same_point(self):
        """Test distance between same point is zero."""
        coord = (40.7128, -74.0060)
        dist = haversine(coord, coord)
        self.assertAlmostEqual(dist, 0.0, places=0)

    def test_known_distance(self):
        """Test known distance (NYC to LA approx 3944 km)."""
        nyc = (40.7128, -74.0060)
        la = (34.0522, -118.2437)
        dist = haversine(nyc, la)

        # Should be approximately 3944 km
        expected = 3944000  # meters
        # Allow 1% error margin
        self.assertAlmostEqual(dist, expected, delta=expected * 0.01)

    def test_equator_distance(self):
        """Test distance along equator."""
        # 1 degree longitude at equator ≈ 111 km
        coord1 = (0.0, 0.0)
        coord2 = (0.0, 1.0)
        dist = haversine(coord1, coord2)

        expected = 111000  # meters
        self.assertAlmostEqual(dist, expected, delta=expected * 0.01)

    def test_meridian_distance(self):
        """Test distance along meridian."""
        # 1 degree latitude ≈ 111 km everywhere
        coord1 = (0.0, 0.0)
        coord2 = (1.0, 0.0)
        dist = haversine(coord1, coord2)

        expected = 111000  # meters
        self.assertAlmostEqual(dist, expected, delta=expected * 0.01)

    def test_antipodal_points(self):
        """Test distance between antipodal points (opposite sides of Earth)."""
        coord1 = (0.0, 0.0)
        coord2 = (0.0, 180.0)
        dist = haversine(coord1, coord2)

        # Half Earth's circumference ≈ 20,000 km
        expected = 20000000  # meters
        self.assertAlmostEqual(dist, expected, delta=expected * 0.05)


class TestGridClustering(unittest.TestCase):
    """Test grid-based clustering."""

    def setUp(self):
        """Create sample segments for testing."""
        self.segments = [
            {
                "start": (40.0, -74.0),
                "end": (40.01, -74.01),
                "coords": [(40.0, -74.0), (40.01, -74.01)],
            },
            {
                "start": (40.02, -74.02),
                "end": (40.03, -74.03),
                "coords": [(40.02, -74.02), (40.03, -74.03)],
            },
            {
                "start": (40.5, -74.5),
                "end": (40.51, -74.51),
                "coords": [(40.5, -74.5), (40.51, -74.51)],
            },
        ]

    def test_basic_grid_clustering(self):
        """Test basic grid clustering."""
        result = cluster_segments_grid(self.segments, grid_x=2, grid_y=2)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.clusters), 0)
        self.assertEqual(result.method_used, "grid")

    def test_single_cluster(self):
        """Test grid with single cell."""
        result = cluster_segments_grid(self.segments, grid_x=1, grid_y=1)

        # All segments should be in one cluster
        self.assertEqual(len(result.clusters), 1)
        total_segments = sum(len(segs) for segs in result.clusters.values())
        self.assertEqual(total_segments, len(self.segments))

    def test_empty_segments(self):
        """Test grid clustering with no segments."""
        result = cluster_segments_grid([], grid_x=5, grid_y=5)

        self.assertEqual(len(result.clusters), 0)
        self.assertEqual(result.noise_count, 0)

    def test_segment_assignment(self):
        """Test that all segments are assigned to clusters."""
        result = cluster_segments_grid(self.segments, grid_x=3, grid_y=3)

        # Count total segments in clusters
        total_assigned = sum(len(segs) for segs in result.clusters.values())
        self.assertEqual(total_assigned, len(self.segments))

    def test_cluster_ids(self):
        """Test that cluster IDs are valid."""
        result = cluster_segments_grid(self.segments, grid_x=4, grid_y=4)

        # All cluster IDs should be non-negative integers
        for cluster_id in result.clusters.keys():
            self.assertIsInstance(cluster_id, int)
            self.assertGreaterEqual(cluster_id, 0)

    def test_no_duplicate_segments(self):
        """Test that no segment appears in multiple clusters."""
        result = cluster_segments_grid(self.segments, grid_x=5, grid_y=5)

        all_segments = []
        for segs in result.clusters.values():
            all_segments.extend(segs)

        # Check for duplicates
        self.assertEqual(len(all_segments), len(set(all_segments)))


class TestClusteringMethodEnum(unittest.TestCase):
    """Test ClusteringMethod enum."""

    def test_enum_values(self):
        """Test that all expected methods are defined."""
        self.assertEqual(ClusteringMethod.DBSCAN.value, "dbscan")
        self.assertEqual(ClusteringMethod.KMEANS.value, "kmeans")
        self.assertEqual(ClusteringMethod.GRID.value, "grid")


class TestClusteringIntegration(unittest.TestCase):
    """Integration tests for clustering methods."""

    def setUp(self):
        """Create sample segments."""
        # Create two clear clusters
        cluster1 = [
            {
                "start": (40.0 + i * 0.01, -74.0 + i * 0.01),
                "end": (40.0 + i * 0.01 + 0.001, -74.0 + i * 0.01 + 0.001),
                "coords": [
                    (40.0 + i * 0.01, -74.0 + i * 0.01),
                    (40.0 + i * 0.01 + 0.001, -74.0 + i * 0.01 + 0.001),
                ],
            }
            for i in range(5)
        ]

        cluster2 = [
            {
                "start": (41.0 + i * 0.01, -75.0 + i * 0.01),
                "end": (41.0 + i * 0.01 + 0.001, -75.0 + i * 0.01 + 0.001),
                "coords": [
                    (41.0 + i * 0.01, -75.0 + i * 0.01),
                    (41.0 + i * 0.01 + 0.001, -75.0 + i * 0.01 + 0.001),
                ],
            }
            for i in range(5)
        ]

        self.segments = cluster1 + cluster2

    def test_grid_separates_clusters(self):
        """Test that grid clustering separates distant clusters."""
        result = cluster_segments_grid(self.segments, grid_x=10, grid_y=10)

        # Should create at least 2 clusters
        self.assertGreaterEqual(len(result.clusters), 2)


if __name__ == "__main__":
    unittest.main()
