#!/usr/bin/env python3
"""
Simple script to run the complete DRPP pipeline.

Usage:
    python run_drpp_pipeline.py your_file.kml

Features:
- Parses KML with segment IDs and directionality
- Builds directed graph
- Solves DRPP with multiple algorithm options
- Generates rich visualizations with:
  * Color-coded segments by requirement type
  * Segment ID labels
  * Route step numbering
  * Multiple output formats (HTML, GeoJSON, SVG)

Algorithms:
- 'industry': Industry-standard DRPP (Eulerian augmentation + Hierholzer) [RECOMMENDED]
- 'v4': Production V4 greedy (FAST, on-demand Dijkstra for large datasets)
- 'rfcs': Route-First Cluster-Second (HIGH QUALITY)
- 'greedy': Legacy greedy
- 'hungarian': Hungarian assignment
"""

import sys
from pathlib import Path

from drpp_pipeline import DRPPPipeline


def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python run_drpp_pipeline.py <kml_file> [algorithm]")
        print()
        print("Algorithms:")
        print("  industry  - Industry-standard DRPP (Eulerian + Hierholzer) [RECOMMENDED]")
        print("  v4        - Production V4 greedy (FAST, for large datasets)")
        print("  rfcs      - Route-First Cluster-Second (HIGH QUALITY)")
        print("  greedy    - Legacy greedy")
        print("  hungarian - Hungarian assignment")
        print()
        print("Example:")
        print("  python run_drpp_pipeline.py my_segments.kml industry")
        sys.exit(1)

    kml_file = Path(sys.argv[1])
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "industry"

    if not kml_file.exists():
        print(f"Error: KML file not found: {kml_file}")
        sys.exit(1)

    # Create pipeline
    print("🚀 Starting DRPP Pipeline")
    print(f"   KML: {kml_file}")
    print(f"   Algorithm: {algorithm.upper()}")
    print()

    pipeline = DRPPPipeline()

    # Run pipeline
    try:
        results = pipeline.run(
            kml_file=kml_file,
            algorithm=algorithm,
            output_dir=Path("./output"),
            output_formats=["html", "geojson", "svg"],
        )

        print()
        print("=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("📊 Results:")
        print(f"   Total distance: {results['total_distance'] / 1000:.1f} km")
        print(f"   Coverage: {results['coverage']:.1f}%")
        print()
        print("📁 Output files:")
        for fmt, path in results["output_files"].items():
            print(f"   {fmt.upper():10s}: {path}")
        print()
        print("🌐 To view the interactive map, open the HTML file in your browser:")
        print(f"   {results['output_files'].get('html', 'N/A')}")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
