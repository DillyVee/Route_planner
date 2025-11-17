"""
Unit tests for distance matrix module.

Tests both dict-based and numpy-based storage.
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from drpp_core.distance_matrix import DistanceMatrix, MatrixStats


class TestDistanceMatrixDict(unittest.TestCase):
    """Test dict-based distance matrix."""

    def setUp(self):
        """Create a matrix for testing."""
        self.matrix = DistanceMatrix(use_numpy=False)
        self.matrix.id_to_coords = {
            0: (40.0, -74.0),
            1: (40.1, -74.1),
            2: (40.2, -74.2),
        }

    def test_set_and_get_distance(self):
        """Test storing and retrieving distances."""
        self.matrix.set(0, 1, 1234.5, [0, 1])
        dist = self.matrix.get_distance(0, 1)
        self.assertAlmostEqual(dist, 1234.5, places=1)

    def test_get_nonexistent_distance(self):
        """Test retrieving non-existent distance."""
        dist = self.matrix.get_distance(0, 2)
        self.assertEqual(dist, float('inf'))

    def test_set_and_get_path_ids(self):
        """Test storing and retrieving paths as IDs."""
        path = [0, 5, 10, 1]
        self.matrix.set(0, 1, 100.0, path)
        retrieved = self.matrix.get_path_ids(0, 1)
        self.assertEqual(retrieved, path)

    def test_get_nonexistent_path(self):
        """Test retrieving non-existent path."""
        path = self.matrix.get_path_ids(0, 2)
        self.assertEqual(path, [])

    def test_has_path(self):
        """Test checking if path exists."""
        self.matrix.set(0, 1, 100.0, [0, 1])
        self.assertTrue(self.matrix.has_path(0, 1))
        self.assertFalse(self.matrix.has_path(0, 2))

    def test_get_path_coords(self):
        """Test retrieving path as coordinates."""
        self.matrix.set(0, 2, 200.0, [0, 1, 2])
        coords = self.matrix.get_path_coords(0, 2)
        expected = [(40.0, -74.0), (40.1, -74.1), (40.2, -74.2)]
        self.assertEqual(coords, expected)

    def test_get_stats(self):
        """Test getting matrix statistics."""
        self.matrix.set(0, 1, 100.0, [0, 1])
        self.matrix.set(0, 2, 200.0, [0, 1, 2])
        self.matrix.set(1, 2, 150.0, [1, 2])

        stats = self.matrix.get_stats()
        self.assertIsInstance(stats, MatrixStats)
        self.assertEqual(stats.num_paths, 3)
        self.assertEqual(stats.storage_type, 'dict')
        self.assertGreater(stats.memory_bytes, 0)


class TestDistanceMatrixNumpy(unittest.TestCase):
    """Test numpy-based distance matrix."""

    def setUp(self):
        """Create a numpy matrix for testing."""
        try:
            import numpy as np
            self.numpy_available = True
            self.matrix = DistanceMatrix(use_numpy=True, num_nodes=3)
            self.matrix.id_to_coords = {
                0: (40.0, -74.0),
                1: (40.1, -74.1),
                2: (40.2, -74.2),
            }
        except (ImportError, RuntimeError):
            self.numpy_available = False

    def test_set_and_get_distance(self):
        """Test storing and retrieving distances with numpy."""
        if not self.numpy_available:
            self.skipTest("NumPy not available")

        self.matrix.set(0, 1, 1234.5, [0, 1])
        dist = self.matrix.get_distance(0, 1)
        self.assertAlmostEqual(dist, 1234.5, places=1)

    def test_get_nonexistent_distance(self):
        """Test retrieving non-existent distance with numpy."""
        if not self.numpy_available:
            self.skipTest("NumPy not available")

        dist = self.matrix.get_distance(0, 2)
        self.assertEqual(dist, float('inf'))

    def test_has_path(self):
        """Test checking if path exists with numpy."""
        if not self.numpy_available:
            self.skipTest("NumPy not available")

        self.matrix.set(0, 1, 100.0, [0, 1])
        self.assertTrue(self.matrix.has_path(0, 1))
        self.assertFalse(self.matrix.has_path(0, 2))

    def test_get_stats(self):
        """Test getting matrix statistics for numpy."""
        if not self.numpy_available:
            self.skipTest("NumPy not available")

        self.matrix.set(0, 1, 100.0, [0, 1])
        self.matrix.set(0, 2, 200.0, [0, 1, 2])

        stats = self.matrix.get_stats()
        self.assertEqual(stats.storage_type, 'numpy')
        self.assertGreater(stats.memory_bytes, 0)


class TestDistanceMatrixEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_create_numpy_without_numpy(self):
        """Test creating numpy matrix when numpy not available."""
        # This test depends on whether numpy is installed
        # If numpy is available, this should work; if not, should raise RuntimeError
        try:
            import numpy as np
            # NumPy available - should work
            matrix = DistanceMatrix(use_numpy=True, num_nodes=5)
            self.assertEqual(matrix.storage_type, 'numpy')
        except ImportError:
            # NumPy not available - should raise error
            with self.assertRaises(RuntimeError):
                matrix = DistanceMatrix(use_numpy=True, num_nodes=5)

    def test_path_with_missing_coords(self):
        """Test getting coordinates for path with missing nodes."""
        matrix = DistanceMatrix(use_numpy=False)
        matrix.id_to_coords = {0: (40.0, -74.0), 2: (40.2, -74.2)}
        matrix.set(0, 2, 100.0, [0, 1, 2])  # Node 1 has no coords

        coords = matrix.get_path_coords(0, 2)
        # Should skip node 1
        expected = [(40.0, -74.0), (40.2, -74.2)]
        self.assertEqual(coords, expected)

    def test_empty_matrix(self):
        """Test operations on empty matrix."""
        matrix = DistanceMatrix()

        self.assertEqual(matrix.get_distance(0, 1), float('inf'))
        self.assertEqual(matrix.get_path_ids(0, 1), [])
        self.assertFalse(matrix.has_path(0, 1))

        stats = matrix.get_stats()
        self.assertEqual(stats.num_paths, 0)


if __name__ == '__main__':
    unittest.main()
