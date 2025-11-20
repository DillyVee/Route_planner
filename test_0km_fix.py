#!/usr/bin/env python3
"""
Test the DRPP pipeline to verify 0km bug is fixed.
This test runs WITHOUT visualization to avoid folium dependency.
"""

from pathlib import Path
from drpp_pipeline import DRPPPipeline

def main():
    print("=" * 80)
    print("TESTING DRPP PIPELINE - VERIFYING 0KM BUG IS FIXED")
    print("=" * 80)
    print()

    # Test with the test KML
    kml_file = Path("test_segments.kml")

    if not kml_file.exists():
        print(f"❌ Test KML not found: {kml_file}")
        return 1

    pipeline = DRPPPipeline()

    print("1. Parsing KML...")
    pipeline.segments = pipeline._parse_kml(kml_file)
    print(f"   ✓ Parsed {len(pipeline.segments)} segments")

    print("\n2. Building graph...")
    pipeline.graph = pipeline._build_graph()
    print(f"   ✓ Built graph with {len(pipeline.graph.id_to_node)} nodes")

    print("\n3. Solving DRPP...")
    pipeline.route_steps = pipeline._solve_drpp(algorithm="v4")
    print(f"   ✓ Generated {len(pipeline.route_steps)} route steps")

    print("\n4. Computing statistics...")
    stats = pipeline._compute_statistics()

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total distance: {stats['total_distance'] / 1000:.2f} km ({stats['total_distance']:.0f} m)")
    print(f"Coverage: {stats['coverage']:.1f}%")
    print(f"Segments covered: {stats.get('segments_covered', 'N/A')}")
    print(f"Required count: {stats.get('required_count', 'N/A')}")
    print(f"Route steps generated: {len(pipeline.route_steps)}")
    print()

    # Verify the fix
    if stats['total_distance'] == 0:
        print("❌ FAIL: Distance is still 0km!")
        print("   The bug is NOT fixed.")
        return 1
    elif len(pipeline.route_steps) == 0:
        print("⚠️  WARNING: Distance calculated but route_steps is empty!")
        print(f"   Distance: {stats['total_distance'] / 1000:.2f} km")
        print("   This means statistics work but visualization won't.")
        return 1
    else:
        print("✅ SUCCESS: Bug is fixed!")
        print(f"   Distance: {stats['total_distance'] / 1000:.2f} km")
        print(f"   Route steps: {len(pipeline.route_steps)}")
        print("   Both statistics and visualization data are available.")
        return 0

if __name__ == "__main__":
    exit(main())
