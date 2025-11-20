#!/usr/bin/env python3
"""
Detailed trace of DRPP routing to verify calculation logic.
This creates disconnected segments that REQUIRE routing between them.
"""

from drpp_pipeline import DRPPPipeline, SegmentRequirement
import logging

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

def create_disconnected_segments():
    """Create segments that are NOT connected - requires routing between them."""
    return [
        # Segment 1: Far north
        SegmentRequirement(
            segment_id="seg_001",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.10, -74.0), (40.11, -74.0)],  # ~1.11 km north
            metadata={"RouteName": "North Street"}
        ),
        # Segment 2: Far south (NOT connected to segment 1)
        SegmentRequirement(
            segment_id="seg_002",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.00, -74.0), (40.01, -74.0)],  # ~1.11 km south
            metadata={"RouteName": "South Street"}
        ),
        # Segment 3: Far east (NOT connected to segments 1 or 2)
        SegmentRequirement(
            segment_id="seg_003",
            forward_required=True,
            backward_required=False,
            one_way=False,
            coordinates=[(40.05, -73.90), (40.05, -73.91)],  # ~0.97 km east
            metadata={"RouteName": "East Street"}
        ),
    ]

def main():
    print("=" * 80)
    print("DETAILED ROUTING TRACE - VERIFYING CALCULATION LOGIC")
    print("=" * 80)
    print()
    print("This test uses DISCONNECTED segments that require routing between them.")
    print("If the algorithm just sums segment lengths, we'll get ~3.2 km")
    print("If it actually routes between segments, we'll get MUCH more.")
    print()

    pipeline = DRPPPipeline()
    pipeline.segments = create_disconnected_segments()

    # Calculate expected segment lengths
    from drpp_core.geo import haversine

    expected_segment_lengths = 0
    for seg in pipeline.segments:
        if len(seg.coordinates) >= 2:
            dist = haversine(seg.coordinates[0], seg.coordinates[-1])
            expected_segment_lengths += dist
            print(f"Segment {seg.segment_id}: {dist/1000:.2f} km")

    print(f"\nTotal segment lengths (no routing): {expected_segment_lengths/1000:.2f} km")
    print()

    # Build graph
    print("Building graph...")
    pipeline.graph = pipeline._build_graph()
    print(f"  → Graph has {len(pipeline.graph.id_to_node)} nodes")

    # Check graph connectivity manually
    print("\nChecking if segments are connected in graph...")
    for i, seg1 in enumerate(pipeline.segments):
        for j, seg2 in enumerate(pipeline.segments):
            if i >= j:
                continue
            end1 = seg1.coordinates[-1]
            start2 = seg2.coordinates[0]

            path, dist = pipeline.graph.shortest_path(end1, start2)
            if path:
                print(f"  → {seg1.segment_id} to {seg2.segment_id}: {dist/1000:.2f} km")
            else:
                print(f"  → {seg1.segment_id} to {seg2.segment_id}: NOT CONNECTED!")

    print("\nSolving DRPP...")
    pipeline.route_steps = pipeline._solve_drpp(algorithm="v4")

    print("\nComputing statistics...")
    stats = pipeline._compute_statistics()

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total distance: {stats['total_distance']/1000:.2f} km")
    print(f"Expected (segments only): {expected_segment_lengths/1000:.2f} km")
    print(f"Routing overhead: {(stats['total_distance'] - expected_segment_lengths)/1000:.2f} km")
    print()

    if stats['total_distance'] <= expected_segment_lengths + 100:  # Allow 100m tolerance
        print("❌ FAIL: Distance is basically just segment lengths!")
        print("   The algorithm is NOT routing between segments.")
        print("   It's just summing segment lengths.")
        return 1
    else:
        print("✅ PASS: Distance includes routing between segments!")
        print("   The algorithm IS calculating shortest paths.")
        overhead_pct = (stats['total_distance'] - expected_segment_lengths) / expected_segment_lengths * 100
        print(f"   Routing adds {overhead_pct:.1f}% overhead (this is expected for disconnected segments)")
        return 0

if __name__ == "__main__":
    exit(main())
