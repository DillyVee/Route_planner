#!/usr/bin/env python3
"""
Quick script to check Dir field values in a KML file.
Usage: python check_dir_fields.py your_file.kml
"""

import sys
from collections import Counter
from drpp_pipeline import DRPPPipeline

if len(sys.argv) < 2:
    print("Usage: python check_dir_fields.py <kml_file>")
    sys.exit(1)

kml_file = sys.argv[1]

print(f"\nAnalyzing Dir fields in: {kml_file}\n")

pipeline = DRPPPipeline()
segments = pipeline._parse_kml(kml_file)

# Count Dir field values
dir_counter = Counter()
one_way_count = 0
two_way_count = 0

for seg in segments:
    dir_val = seg.metadata.get('direction_code', 'NONE')
    dir_counter[dir_val] += 1

    if seg.one_way:
        one_way_count += 1
    else:
        two_way_count += 1

print("=" * 60)
print("DIR FIELD DISTRIBUTION")
print("=" * 60)
for dir_val, count in dir_counter.most_common():
    pct = (count / len(segments)) * 100
    print(f"  {dir_val:10s}: {count:6d} segments ({pct:5.1f}%)")

print("\n" + "=" * 60)
print("ROUTING INTERPRETATION")
print("=" * 60)
print(f"  One-way segments:  {one_way_count:6d} ({one_way_count/len(segments)*100:5.1f}%)")
print(f"  Two-way segments:  {two_way_count:6d} ({two_way_count/len(segments)*100:5.1f}%)")
print(f"  Total segments:    {len(segments):6d}")

print("\n" + "=" * 60)
print("DIRECTION REQUIREMENTS")
print("=" * 60)
forward_count = sum(1 for s in segments if s.forward_required)
backward_count = sum(1 for s in segments if s.backward_required)
both_count = sum(1 for s in segments if s.forward_required and s.backward_required)

print(f"  Forward required:  {forward_count:6d}")
print(f"  Backward required: {backward_count:6d}")
print(f"  Both directions:   {both_count:6d}")

print("\n" + "=" * 60)
print("LEGEND")
print("=" * 60)
print("  I = Increasing (one-way along line)")
print("  D = Decreasing (one-way opposite to line)")
print("  B = Both directions (two-way)")
print("  T = Two-way (bidirectional)")
print("=" * 60)
