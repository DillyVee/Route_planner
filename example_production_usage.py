"""
Production-Ready DRPP Solver - Example Usage

This example demonstrates how to use the V4 production-ready DRPP solver
with all best practices:
- Proper logging configuration
- Type hints
- Error handling
- Progress tracking
- Performance profiling
"""

from pathlib import Path
import logging
from typing import List, Dict, Tuple

# Import production-ready modules
from drpp_core import (
    parallel_cluster_routing,
    cluster_segments,
    ClusteringMethod,
    estimate_optimal_workers,
    PathResult,
    ClusterResult
)
from drpp_core.logging_config import setup_logging, LogTimer
from drpp_core.profiling import BenchmarkRunner, track_memory


def main():
    """Main example function."""

    # ===================================================================
    # STEP 1: Configure Logging
    # ===================================================================
    logger = setup_logging(
        level=logging.INFO,
        log_file=Path("logs/drpp_example.log"),
        console=True,
        detailed=False
    )

    logger.info("=" * 60)
    logger.info("DRPP Production Solver V4.0 - Example")
    logger.info("=" * 60)

    # ===================================================================
    # STEP 2: Load Data
    # ===================================================================
    logger.info("Loading graph and segments...")

    # TODO: Replace with your actual data loading
    # graph = load_your_graph()
    # segments = load_your_segments()
    # required_edges = extract_required_edges(segments)

    # For this example, we'll show the structure
    logger.info("NOTE: This is a template. Replace with your actual data loading.")
    logger.info("Expected structures:")
    logger.info("  - graph: Object with dijkstra(), node_to_id, id_to_node")
    logger.info("  - segments: List[Dict] with 'start', 'end', 'coords'")
    logger.info("  - required_edges: List[Tuple] of (start, end, coords, ...)")

    # Example data structure (replace with real data)
    """
    segments = [
        {
            'start': (40.7128, -74.0060),  # NYC
            'end': (40.7580, -73.9855),    # Times Square
            'coords': [(40.7128, -74.0060), (40.7580, -73.9855)],
            'speed_limit': 50,  # km/h (optional)
        },
        # ... more segments
    ]

    required_edges = [
        (segment['start'], segment['end'], segment['coords'])
        for segment in segments
    ]
    """

    # ===================================================================
    # STEP 3: Cluster Segments
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING")
    logger.info("=" * 60)

    # For demonstration, create dummy segments
    # In production, use your actual segments
    dummy_segments = create_dummy_segments()

    with LogTimer(logger, "Clustering"):
        with track_memory("Clustering"):
            result: ClusterResult = cluster_segments(
                dummy_segments,
                method=ClusteringMethod.GRID,  # Use GRID for demo (no sklearn needed)
                grid_x=5,
                grid_y=5
            )

    logger.info(f"Created {len(result.clusters)} clusters")
    logger.info(f"Noise points: {result.noise_count}")
    logger.info(f"Method used: {result.method_used}")

    # Show cluster sizes
    for cluster_id, seg_indices in result.clusters.items():
        logger.info(f"  Cluster {cluster_id}: {len(seg_indices)} segments")

    # ===================================================================
    # STEP 4: Route Through Clusters
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ROUTING")
    logger.info("=" * 60)

    # Determine cluster order (customize as needed)
    cluster_order = list(result.clusters.keys())

    # Estimate optimal workers
    num_workers = estimate_optimal_workers(
        num_clusters=len(result.clusters),
        num_segments=len(dummy_segments)
    )
    logger.info(f"Using {num_workers} worker processes")

    # For demonstration, we can't actually route without a real graph
    logger.info("\nNOTE: Actual routing requires a real graph object.")
    logger.info("Example routing code:")
    logger.info("""
    results: List[PathResult] = parallel_cluster_routing(
        graph=graph,
        required_edges=required_edges,
        clusters=result.clusters,
        cluster_order=cluster_order,
        start_node=dummy_segments[0]['start'],
        num_workers=num_workers,
        progress_callback=lambda done, total:
            logger.info(f"Progress: {done}/{total} ({100*done/total:.1f}%)")
    )

    # Analyze results
    for result in results:
        logger.info(f"Cluster {result.cluster_id}:")
        logger.info(f"  Distance: {result.distance / 1000:.1f} km")
        logger.info(f"  Covered: {result.segments_covered} segments")
        logger.info(f"  Unreachable: {result.segments_unreachable} segments")
        logger.info(f"  Time: {result.computation_time:.2f}s")

    total_distance = sum(r.distance for r in results)
    total_time = sum(r.computation_time for r in results)
    logger.info(f"\\nTotal distance: {total_distance / 1000:.1f} km")
    logger.info(f"Total computation time: {total_time:.2f}s")
    """)

    # ===================================================================
    # STEP 5: Performance Analysis
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE PROFILING")
    logger.info("=" * 60)

    logger.info("To profile your code, use:")
    logger.info("""
    from drpp_core.profiling import ProfilerContext

    with ProfilerContext("routing", top_n=20):
        results = parallel_cluster_routing(...)
    """)

    # ===================================================================
    # STEP 6: Benchmarking (optional)
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARKING")
    logger.info("=" * 60)

    logger.info("To benchmark different approaches:")
    logger.info("""
    bench = BenchmarkRunner()
    bench.add_scenario("greedy", lambda: parallel_cluster_routing(...))
    bench.add_scenario("optimized", lambda: optimized_routing(...))
    bench.run(iterations=5, warmup=2)
    bench.print_results()
    """)

    logger.info("\n" + "=" * 60)
    logger.info("Example complete!")
    logger.info("=" * 60)


def create_dummy_segments() -> List[Dict]:
    """Create dummy segments for demonstration.

    In production, replace this with your actual segment loading.
    """
    import random

    segments = []
    for i in range(50):
        # Random points in NYC area
        lat = 40.7 + random.random() * 0.2
        lon = -74.0 + random.random() * 0.2
        lat2 = lat + random.random() * 0.01
        lon2 = lon + random.random() * 0.01

        segments.append({
            'start': (lat, lon),
            'end': (lat2, lon2),
            'coords': [(lat, lon), (lat2, lon2)],
            'speed_limit': random.choice([30, 40, 50, 60]),
        })

    return segments


if __name__ == '__main__':
    main()
