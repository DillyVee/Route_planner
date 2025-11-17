"""
Unit tests for path reconstruction module.

Tests all sentinel value handling, edge cases, and error conditions.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drpp_core.path_reconstruction import (
    reconstruct_path,
    validate_path,
    reconstruct_path_safe
)


class TestPathReconstruction(unittest.TestCase):
    """Test cases for robust path reconstruction."""

    def test_simple_path(self):
        """Test basic path reconstruction."""
        # Path: 0 -> 2 -> 4
        # predecessors[4] = 2, predecessors[2] = 0, predecessors[0] = None
        predecessors = [None, None, 0, None, 2]
        path = reconstruct_path(predecessors, source_id=0, target_id=4)
        self.assertEqual(path, [0, 2, 4])

    def test_self_loop_source(self):
        """Test source with self-loop sentinel."""
        # Some implementations use predecessors[source] = source
        predecessors = [0, 0, 0, 1, 2]
        path = reconstruct_path(predecessors, source_id=0, target_id=4)
        # Should still work - source self-loop is valid
        self.assertEqual(path, [0, 2, 4])

    def test_no_path_none_sentinel(self):
        """Test unreachable target with None sentinel."""
        predecessors = [None, 0, 0, None, None]
        path = reconstruct_path(predecessors, source_id=0, target_id=4)
        self.assertEqual(path, [])  # No path

    def test_no_path_negative_one_sentinel(self):
        """Test unreachable target with -1 sentinel."""
        predecessors = [-1, 0, 0, -1, -1]
        path = reconstruct_path(predecessors, source_id=0, target_id=4)
        self.assertEqual(path, [])  # No path

    def test_cycle_detection(self):
        """Test cycle detection in predecessor array."""
        # Invalid cycle: 2 -> 3 -> 2
        predecessors = [None, 0, 3, 2, 2]
        path = reconstruct_path(predecessors, source_id=0, target_id=4)
        self.assertEqual(path, [])  # Cycle detected

    def test_invalid_self_loop(self):
        """Test invalid self-loop at non-source node."""
        # predecessors[2] = 2 but 2 is not the source
        predecessors = [None, 0, 2, 0, 2]
        path = reconstruct_path(predecessors, source_id=0, target_id=2)
        self.assertEqual(path, [])  # Invalid self-loop

    def test_source_equals_target(self):
        """Test path from node to itself."""
        predecessors = [None, 0, 0, 1]
        path = reconstruct_path(predecessors, source_id=2, target_id=2)
        self.assertEqual(path, [2])  # Single-node path

    def test_direct_connection(self):
        """Test direct connection (one edge)."""
        predecessors = [None, 0, None, None]
        path = reconstruct_path(predecessors, source_id=0, target_id=1)
        self.assertEqual(path, [0, 1])

    def test_long_path(self):
        """Test longer path reconstruction."""
        # Path: 0 -> 1 -> 2 -> 3 -> 4 -> 5
        predecessors = [None, 0, 1, 2, 3, 4]
        path = reconstruct_path(predecessors, source_id=0, target_id=5)
        self.assertEqual(path, [0, 1, 2, 3, 4, 5])

    def test_out_of_bounds_source(self):
        """Test invalid source ID."""
        predecessors = [None, 0, 0, 1]
        with self.assertRaises(ValueError):
            reconstruct_path(predecessors, source_id=10, target_id=2)

    def test_out_of_bounds_target(self):
        """Test invalid target ID."""
        predecessors = [None, 0, 0, 1]
        with self.assertRaises(ValueError):
            reconstruct_path(predecessors, source_id=0, target_id=10)

    def test_negative_source(self):
        """Test negative source ID."""
        predecessors = [None, 0, 0, 1]
        with self.assertRaises(ValueError):
            reconstruct_path(predecessors, source_id=-1, target_id=2)

    def test_max_iterations(self):
        """Test maximum iteration limit."""
        # Create very long predecessor array
        predecessors = list(range(-1, 999))
        predecessors[0] = None

        # With custom max_iterations
        path = reconstruct_path(
            predecessors,
            source_id=0,
            target_id=500,
            max_iterations=10
        )
        self.assertEqual(path, [])  # Exceeded max iterations


class TestValidatePath(unittest.TestCase):
    """Test cases for path validation."""

    def test_valid_path(self):
        """Test validation of correct path."""
        path = [0, 2, 4]
        self.assertTrue(validate_path(path, source_id=0, target_id=4))

    def test_empty_path(self):
        """Test empty path is invalid."""
        self.assertFalse(validate_path([], source_id=0, target_id=4))

    def test_wrong_start(self):
        """Test path with wrong starting node."""
        path = [1, 2, 4]
        self.assertFalse(validate_path(path, source_id=0, target_id=4))

    def test_wrong_end(self):
        """Test path with wrong ending node."""
        path = [0, 2, 3]
        self.assertFalse(validate_path(path, source_id=0, target_id=4))

    def test_path_with_cycle(self):
        """Test path containing duplicate nodes."""
        path = [0, 2, 3, 2, 4]  # 2 appears twice
        self.assertFalse(validate_path(path, source_id=0, target_id=4))

    def test_single_node_path(self):
        """Test single-node path (source = target)."""
        path = [5]
        self.assertTrue(validate_path(path, source_id=5, target_id=5))


class TestReconstructPathSafe(unittest.TestCase):
    """Test cases for safe path reconstruction wrapper."""

    def test_successful_reconstruction(self):
        """Test successful safe reconstruction."""
        predecessors = [None, 0, 0, 1, 2]
        path = reconstruct_path_safe(predecessors, source_id=0, target_id=4)
        self.assertIsNotNone(path)
        self.assertEqual(path, [0, 2, 4])

    def test_failed_reconstruction(self):
        """Test failed reconstruction returns None."""
        predecessors = [None, 0, 0, -1, -1]
        path = reconstruct_path_safe(predecessors, source_id=0, target_id=4)
        self.assertIsNone(path)  # No path available

    def test_exception_handling(self):
        """Test exception handling in safe wrapper."""
        # Invalid arguments should be caught
        predecessors = [None, 0, 0]
        path = reconstruct_path_safe(predecessors, source_id=0, target_id=10)
        self.assertIsNone(path)  # Exception caught, returns None

    def test_invalid_path_validation(self):
        """Test that invalid paths are rejected."""
        # Create a predecessor array that creates a cycle
        predecessors = [None, 0, 3, 2, 2]
        path = reconstruct_path_safe(predecessors, source_id=0, target_id=4)
        self.assertIsNone(path)  # Cycle detected, returns None


if __name__ == '__main__':
    unittest.main()
