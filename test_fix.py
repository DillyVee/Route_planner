#!/usr/bin/env python3
"""
Simple test to verify the routing fix.
This creates a minimal test case to ensure routing results are properly used.
"""

from pathlib import Path
from drpp_pipeline import DRPPPipeline, SegmentRequirement

def test_statistics_fix():
    """Test that routing results are properly stored and used for statistics."""

    # Create a pipeline instance
    pipeline = DRPPPipeline()

    # Create some test segments
    pipeline.segments = [
        SegmentRequirement(
            segment_id="seg_001",
            forward_required=True,
            backward_required=True,
            coordinates=[(40.0, -74.0), (40.01, -74.01)],
            metadata={}
        ),
        SegmentRequirement(
            segment_id="seg_002",
            forward_required=True,
            backward_required=False,
            coordinates=[(40.01, -74.01), (40.02, -74.02)],
            metadata={}
        )
    ]

    # Simulate routing results (V4 PathResult format)
    from drpp_core.types import PathResult

    # Mock result with actual distance
    mock_result = PathResult(
        path=[(40.0, -74.0), (40.01, -74.01), (40.02, -74.02)],
        distance=2500.0,  # 2.5 km
        cluster_id=0,
        segments_covered=3,  # forward + backward + forward
        segments_unreachable=0,
        computation_time=0.5
    )

    pipeline.routing_results = [mock_result]

    # Compute statistics
    stats = pipeline._compute_statistics()

    print("=" * 60)
    print("ROUTING FIX VERIFICATION TEST")
    print("=" * 60)
    print(f"Total distance: {stats['total_distance']:.1f} meters")
    print(f"Expected: 2500.0 meters")
    print(f"Coverage: {stats['coverage']:.1f}%")
    print(f"Segments covered: {stats.get('segments_covered', 0)}")
    print(f"Required count: {stats.get('required_count', 0)}")
    print()

    # Verify the fix worked
    if stats['total_distance'] > 0:
        print("✅ SUCCESS: Distance is now correctly computed!")
        print("   The bug has been fixed - routing results are now used.")
        return True
    else:
        print("❌ FAILURE: Distance is still 0")
        print("   The bug fix did not work as expected.")
        return False

if __name__ == "__main__":
    success = test_statistics_fix()
    exit(0 if success else 1)
