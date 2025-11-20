#!/usr/bin/env python3
"""
Run Industry-Standard DRPP solver on KML input.

This script demonstrates the industry-standard Eulerian augmentation approach
to solving DRPP, following methodology used by DOTs, Esri, HERE, TomTom, etc.

Usage:
    python run_industry_drpp.py input.kml [--output-dir OUTPUT_DIR] [--start-lat LAT] [--start-lon LON]
"""

import argparse
import logging
import sys
from pathlib import Path

from drpp_pipeline import DRPPPipeline
from drpp_core import (
    solve_drpp_industry_standard,
    export_drpp_to_kml,
    generate_turn_by_turn
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Industry-Standard DRPP Solver with Eulerian Augmentation'
    )
    parser.add_argument('input_kml', help='Input KML file with road segments')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--start-lat', type=float,
                       help='Starting latitude (optional)')
    parser.add_argument('--start-lon', type=float,
                       help='Starting longitude (optional)')
    parser.add_argument('--snap-tolerance', type=float, default=2.0,
                       help='Endpoint snapping tolerance in meters (default: 2.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(args.input_kml)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("INDUSTRY-STANDARD DRPP SOLVER")
    logger.info("=" * 80)
    logger.info(f"Input KML: {input_path}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Parse KML using existing pipeline
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: PARSING KML")
    logger.info("=" * 80)

    pipeline = DRPPPipeline()
    segments = pipeline._parse_kml(str(input_path))

    logger.info(f"✓ Parsed {len(segments)} segments from KML")
    logger.info(f"  - Forward required: {sum(1 for s in segments if s.forward_required)}")
    logger.info(f"  - Backward required: {sum(1 for s in segments if s.backward_required)}")
    logger.info(f"  - One-way segments: {sum(1 for s in segments if s.one_way)}")

    # Step 2: Solve using industry-standard DRPP
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SOLVING DRPP (INDUSTRY-STANDARD EULERIAN APPROACH)")
    logger.info("=" * 80)

    start_coordinate = None
    if args.start_lat is not None and args.start_lon is not None:
        start_coordinate = (args.start_lat, args.start_lon)
        logger.info(f"Starting coordinate: {start_coordinate}")

    solution = solve_drpp_industry_standard(
        segments=segments,
        start_coordinate=start_coordinate,
        snap_tolerance_meters=args.snap_tolerance
    )

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("SOLUTION SUMMARY")
    logger.info("=" * 80)

    for message in solution.messages:
        logger.info(f"  {message}")

    logger.info("")
    logger.info(f"Valid solution: {solution.is_valid}")
    logger.info(f"Total distance: {solution.total_distance_km:.2f} km")
    logger.info(f"Required distance: {solution.required_distance_km:.2f} km")
    logger.info(f"Deadhead distance: {solution.deadhead_distance_km:.2f} km ({solution.deadhead_percentage:.1f}%)")
    logger.info(f"Number of route steps: {len(solution.tour.edges)}")

    # Step 3: Export results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EXPORTING RESULTS")
    logger.info("=" * 80)

    # Export KML
    output_kml = output_dir / "route_industry_standard.kml"
    export_drpp_to_kml(solution, str(output_kml))
    logger.info(f"✓ Exported KML: {output_kml}")

    # Export turn-by-turn directions
    output_csv = output_dir / "directions.csv"
    output_json = output_dir / "directions.json"
    output_text = output_dir / "directions.txt"

    generate_turn_by_turn(
        solution,
        output_csv=str(output_csv),
        output_json=str(output_json),
        output_text=str(output_text)
    )

    logger.info(f"✓ Exported turn-by-turn CSV: {output_csv}")
    logger.info(f"✓ Exported turn-by-turn JSON: {output_json}")
    logger.info(f"✓ Exported turn-by-turn text: {output_text}")

    # Display first few directions
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE DIRECTIONS (First 5 steps)")
    logger.info("=" * 80)

    instructions = generate_turn_by_turn(solution)
    for inst in instructions[:5]:
        logger.info(f"{inst.step_number}. {inst.instruction}")
        logger.info(f"   Distance: {inst.distance_m / 1000:.2f} km (Total: {inst.cumulative_distance_m / 1000:.2f} km)")

    if len(instructions) > 5:
        logger.info(f"... ({len(instructions) - 5} more steps)")

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
