#!/usr/bin/env python3
"""
Diagnostic script to check route calculation issues.
Tests both the existing solver and the new industry-standard solver.
"""

from pathlib import Path
from drpp_pipeline import DRPPPipeline, SegmentRequirement

def create_test_segments():
    """Create simple test segments for diagnosis."""
    return [
        SegmentRequirement(
            segment_id="seg_001",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.0, -74.0), (40.01, -74.01)],
            metadata={"RouteName": "Test Route 1"}
        ),
        SegmentRequirement(
            segment_id="seg_002",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.01, -74.01), (40.02, -74.02)],
            metadata={"RouteName": "Test Route 2"}
        ),
        SegmentRequirement(
            segment_id="seg_003",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.02, -74.02), (40.03, -74.03)],
            metadata={"RouteName": "Test Route 3"}
        ),
    ]

def test_existing_pipeline():
    """Test the existing DRPPPipeline."""
    print("=" * 80)
    print("TESTING EXISTING DRPP PIPELINE")
    print("=" * 80)

    pipeline = DRPPPipeline()
    pipeline.segments = create_test_segments()

    # Build graph
    print("\n1. Building graph...")
    try:
        pipeline.graph = pipeline._build_graph()
        print(f"   ✓ Graph built: {len(pipeline.graph.id_to_node)} nodes")
    except Exception as e:
        print(f"   ✗ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Solve DRPP
    print("\n2. Solving DRPP...")
    try:
        pipeline._solve_drpp(algorithm="v4")
        print(f"   ✓ DRPP solved")
        print(f"   - Routing results: {len(pipeline.routing_results)}")
        if pipeline.routing_results:
            for i, result in enumerate(pipeline.routing_results):
                if hasattr(result, "distance"):
                    print(f"     Result {i}: {result.distance:.1f}m, {result.segments_covered} segments covered")
                else:
                    print(f"     Result {i}: {result}")
    except Exception as e:
        print(f"   ✗ Error solving DRPP: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute statistics
    print("\n3. Computing statistics...")
    try:
        stats = pipeline._compute_statistics()
        print(f"   ✓ Statistics computed:")
        print(f"     - Total distance: {stats['total_distance']:.1f}m ({stats['total_distance']/1000:.2f}km)")
        print(f"     - Coverage: {stats['coverage']:.1f}%")
        print(f"     - Required count: {stats.get('required_count', 'N/A')}")
        print(f"     - Segments covered: {stats.get('segments_covered', 'N/A')}")

        if stats['total_distance'] == 0:
            print("\n   ⚠️  WARNING: Distance is 0! This is the bug.")
            return False
        else:
            print("\n   ✓ Distance is non-zero - working correctly!")
            return True

    except Exception as e:
        print(f"   ✗ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_industry_standard_solver():
    """Test the new industry-standard DRPP solver."""
    print("\n\n" + "=" * 80)
    print("TESTING INDUSTRY-STANDARD DRPP SOLVER")
    print("=" * 80)

    try:
        from drpp_core import solve_drpp_industry_standard

        segments = create_test_segments()

        print("\n1. Solving with industry-standard DRPP...")
        solution = solve_drpp_industry_standard(
            segments=segments,
            snap_tolerance_meters=2.0
        )

        print(f"   ✓ Solution computed:")
        print(f"     - Total distance: {solution.total_distance_km:.2f}km")
        print(f"     - Required distance: {solution.required_distance_km:.2f}km")
        print(f"     - Deadhead distance: {solution.deadhead_distance_km:.2f}km ({solution.deadhead_percentage:.1f}%)")
        print(f"     - Number of edges in tour: {len(solution.tour.edges)}")
        print(f"     - Route coordinates: {len(solution.route_coordinates)} points")
        print(f"     - Valid: {solution.is_valid}")

        if solution.total_distance_km == 0:
            print("\n   ⚠️  WARNING: Distance is 0! This is the bug.")
            return False
        else:
            print("\n   ✓ Distance is non-zero - working correctly!")
            return True

    except Exception as e:
        print(f"   ✗ Error with industry-standard solver: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DRPP ROUTE CALCULATION DIAGNOSTICS" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test existing pipeline
    existing_ok = test_existing_pipeline()

    # Test new industry-standard solver
    industry_ok = test_industry_standard_solver()

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Existing pipeline: {'✓ PASS' if existing_ok else '✗ FAIL'}")
    print(f"Industry-standard solver: {'✓ PASS' if industry_ok else '✗ FAIL'}")

    if existing_ok and industry_ok:
        print("\n✓ All tests passed! Route calculation is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Route calculation has issues.")
        return 1

if __name__ == "__main__":
    exit(main())
