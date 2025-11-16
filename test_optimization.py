"""
Test script to verify optimized greedy implementation.

This script compares V1 (original) vs V2 (optimized) performance and quality.

Usage:
    python test_optimization.py path/to/your/file.kml
"""

import sys
import time
from collections import defaultdict

# Import both versions
try:
    from parallel_processing_addon_greedy import (
        parallel_cluster_routing as greedy_v1,
        _greedy_route_cluster as greedy_cluster_v1
    )
    V1_AVAILABLE = True
except ImportError:
    print("⚠️ V1 not available")
    V1_AVAILABLE = False

try:
    from parallel_processing_addon_greedy_v2 import (
        parallel_cluster_routing as greedy_v2,
        optimized_greedy_route_cluster as greedy_cluster_v2,
        precompute_distance_matrix
    )
    V2_AVAILABLE = True
except ImportError:
    print("⚠️ V2 not available")
    V2_AVAILABLE = False

# Import core functionality
from route_planner_complete import (
    parse_kml,
    build_graph,
    cluster_segments,
    order_clusters,
    two_opt_order,
    centroid_of_cluster
)


def test_single_cluster(graph, required_edges, seg_idxs, start_node):
    """Test a single cluster with both algorithms"""

    print(f"\n{'='*60}")
    print(f"TESTING CLUSTER WITH {len(seg_idxs)} SEGMENTS")
    print(f"{'='*60}")

    results = {}

    # Test V1
    if V1_AVAILABLE:
        print("\n[V1] Running original greedy...")
        start_time = time.time()
        path_v1, dist_v1, unreachable_v1 = greedy_cluster_v1(
            graph, required_edges, seg_idxs, start_node
        )
        time_v1 = time.time() - start_time

        results['v1'] = {
            'time': time_v1,
            'distance': dist_v1,
            'path_length': len(path_v1),
            'unreachable': len(unreachable_v1)
        }

        print(f"  ✓ Completed in {time_v1:.2f}s")
        print(f"  Distance: {dist_v1/1000:.2f} km")
        print(f"  Path points: {len(path_v1)}")
        print(f"  Unreachable: {len(unreachable_v1)}")

    # Test V2
    if V2_AVAILABLE:
        print("\n[V2] Running optimized greedy...")

        # Preprocessing
        print("  Preprocessing...")
        prep_start = time.time()
        distance_matrix, endpoints = precompute_distance_matrix(
            graph, required_edges, seg_idxs, start_node
        )
        prep_time = time.time() - prep_start
        print(f"    ✓ Precomputed {len(distance_matrix)} distances in {prep_time:.2f}s")
        print(f"    Endpoints: {len(endpoints)}")

        # Greedy
        print("  Greedy routing...")
        greedy_start = time.time()
        path_v2, dist_v2, unreachable_v2 = greedy_cluster_v2(
            graph, required_edges, seg_idxs, start_node,
            distance_matrix=distance_matrix,
            endpoints=endpoints
        )
        greedy_time = time.time() - greedy_start
        time_v2 = prep_time + greedy_time

        results['v2'] = {
            'time': time_v2,
            'preprocessing_time': prep_time,
            'greedy_time': greedy_time,
            'distance': dist_v2,
            'path_length': len(path_v2),
            'unreachable': len(unreachable_v2),
            'matrix_size': len(distance_matrix),
            'num_endpoints': len(endpoints)
        }

        print(f"  ✓ Completed in {time_v2:.2f}s (prep: {prep_time:.2f}s, greedy: {greedy_time:.2f}s)")
        print(f"  Distance: {dist_v2/1000:.2f} km")
        print(f"  Path points: {len(path_v2)}")
        print(f"  Unreachable: {len(unreachable_v2)}")

    # Comparison
    if V1_AVAILABLE and V2_AVAILABLE:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        speedup = results['v1']['time'] / results['v2']['time']
        quality = results['v2']['distance'] / results['v1']['distance']

        print(f"\n⚡ PERFORMANCE:")
        print(f"  V1 time: {results['v1']['time']:.2f}s")
        print(f"  V2 time: {results['v2']['time']:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")

        if speedup > 1.0:
            print(f"  🎉 V2 is {speedup:.1f}x FASTER!")
        else:
            print(f"  ⚠️ V2 is slower (preprocessing overhead on small cluster)")

        print(f"\n📏 QUALITY:")
        print(f"  V1 distance: {results['v1']['distance']/1000:.2f} km")
        print(f"  V2 distance: {results['v2']['distance']/1000:.2f} km")
        print(f"  Quality ratio: {quality:.3f}")

        if 0.95 <= quality <= 1.05:
            print(f"  ✅ Routes are similar quality")
        elif quality < 0.95:
            print(f"  🎉 V2 found a better route!")
        else:
            print(f"  ⚠️ V2 route is slightly worse")

        print(f"\n🔍 DETAILS:")
        print(f"  Matrix size: {results['v2']['matrix_size']}")
        print(f"  Endpoints: {results['v2']['num_endpoints']}")
        print(f"  Expected matrix size: {results['v2']['num_endpoints']**2}")
        print(f"  Coverage: {100*results['v2']['matrix_size']/(results['v2']['num_endpoints']**2):.1f}%")

    return results


def test_full_pipeline(kml_path):
    """Test full pipeline with both algorithms"""

    print(f"\n{'='*60}")
    print("FULL PIPELINE TEST")
    print(f"{'='*60}")

    # Parse KML
    print("\n[1] Parsing KML...")
    segments = parse_kml(kml_path)
    print(f"  ✓ Loaded {len(segments)} segments")

    # Build graph
    print("\n[2] Building graph...")
    graph, required_edges = build_graph(segments, treat_unspecified_as_two_way=True)
    print(f"  ✓ Graph has {len(graph.id_to_node):,} nodes")

    # Cluster
    print("\n[3] Clustering...")
    clusters = cluster_segments(segments, method='auto', k_clusters=10)
    print(f"  ✓ Created {len(clusters)} clusters")

    # Show cluster sizes
    cluster_sizes = [len(seg_idxs) for seg_idxs in clusters.values()]
    print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")

    # Order clusters
    print("\n[4] Ordering clusters...")
    order = order_clusters(clusters, segments, use_ortools=False)
    centroids = {cid: centroid_of_cluster(clusters[cid], segments) for cid in order}
    improved_order = two_opt_order(order, centroids) if len(order) > 2 else order

    # Test both algorithms
    results = {}

    # V1
    if V1_AVAILABLE:
        print("\n[5] Running V1 (original greedy)...")
        start_time = time.time()
        results_v1 = greedy_v1(
            graph=graph,
            required_edges=required_edges,
            clusters=clusters,
            cluster_order=improved_order,
            allow_return=True,
            num_workers=1  # Use 1 worker for fair comparison
        )
        time_v1 = time.time() - start_time

        total_dist_v1 = sum(r[1] for r in results_v1)
        results['v1'] = {
            'time': time_v1,
            'distance': total_dist_v1,
            'clusters': len(results_v1)
        }

        print(f"  ✓ Completed in {time_v1:.2f}s")
        print(f"  Total distance: {total_dist_v1/1000:.2f} km")

    # V2
    if V2_AVAILABLE:
        print("\n[6] Running V2 (optimized greedy)...")
        start_time = time.time()
        results_v2 = greedy_v2(
            graph=graph,
            required_edges=required_edges,
            clusters=clusters,
            cluster_order=improved_order,
            allow_return=True,
            num_workers=1  # Use 1 worker for fair comparison
        )
        time_v2 = time.time() - start_time

        total_dist_v2 = sum(r[1] for r in results_v2)
        results['v2'] = {
            'time': time_v2,
            'distance': total_dist_v2,
            'clusters': len(results_v2)
        }

        print(f"  ✓ Completed in {time_v2:.2f}s")
        print(f"  Total distance: {total_dist_v2/1000:.2f} km")

    # Final comparison
    if V1_AVAILABLE and V2_AVAILABLE:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")

        speedup = results['v1']['time'] / results['v2']['time']
        quality = results['v2']['distance'] / results['v1']['distance']

        print(f"\n⚡ PERFORMANCE:")
        print(f"  V1: {results['v1']['time']:.2f}s")
        print(f"  V2: {results['v2']['time']:.2f}s")
        print(f"  Speedup: {speedup:.1f}x")

        print(f"\n📏 QUALITY:")
        print(f"  V1: {results['v1']['distance']/1000:.2f} km")
        print(f"  V2: {results['v2']['distance']/1000:.2f} km")
        print(f"  Ratio: {quality:.3f}")

        print(f"\n🎯 VERDICT:")
        if speedup > 5.0 and 0.95 <= quality <= 1.05:
            print(f"  ✅ EXCELLENT! V2 is {speedup:.1f}x faster with similar quality")
        elif speedup > 2.0 and 0.90 <= quality <= 1.10:
            print(f"  ✅ GOOD! V2 is {speedup:.1f}x faster")
        elif speedup > 1.0:
            print(f"  ⚠️ MODEST GAINS. V2 is {speedup:.1f}x faster")
            print(f"     (Preprocessing overhead is significant on small dataset)")
        else:
            print(f"  ⚠️ V2 is slower on this dataset")
            print(f"     (Use V2 only for large clusters with 20+ segments)")

    return results


def main():
    """Main test entry point"""

    if len(sys.argv) < 2:
        print("Usage: python test_optimization.py path/to/file.kml")
        sys.exit(1)

    kml_path = sys.argv[1]

    print("="*60)
    print("OPTIMIZATION TEST SUITE")
    print("="*60)
    print(f"KML file: {kml_path}")
    print(f"V1 available: {V1_AVAILABLE}")
    print(f"V2 available: {V2_AVAILABLE}")
    print("="*60)

    if not V1_AVAILABLE and not V2_AVAILABLE:
        print("❌ Neither V1 nor V2 is available!")
        sys.exit(1)

    # Run full pipeline test
    results = test_full_pipeline(kml_path)

    print("\n✅ Test completed successfully!")


if __name__ == '__main__':
    main()
