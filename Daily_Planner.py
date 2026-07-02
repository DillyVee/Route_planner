#!/usr/bin/env python3
"""
Daily Survey Route Planner
==========================

Builds a balanced multi-day collection plan from a hotel: several complete
daily routes that each start and end at the hotel, meet a minimum daily
mileage, and stay consistent in effort from day to day - the goal is a
sustainable field schedule, not one great day followed by garbage days of
scattered leftovers.

How the balance works
---------------------
1. Segments are grouped into continuous runs (chains) and swept by compass
   bearing around the hotel into one wedge per day, each wedge holding about
   the same survey mileage. Wedges keep every day's work in one compact
   area, so collecting a day never strands isolated segments inside another
   day's territory - the lookahead is structural: whatever remains after a
   day is still clustered for the following days.
2. Every day is then solved as a real road route (same engine as
   Route_Planner: OSM road network, direction-aware, transfers always on
   roads) from the hotel, through its wedge, and back to the hotel.
3. Days are rebalanced on their true driven totals: boundary runs shift
   from the heaviest day into the angularly adjacent lighter day - a run
   may leave its locally optimal day if that keeps tomorrow's mileage
   consistent - until no single day is dramatically bigger than another.

Usage
-----
    python Daily_Planner.py segments.mpz --hotel "8051 Peach St, Erie PA" --min-miles 120
    python Daily_Planner.py segments.mpz --hotel 42.05,-80.09 --min-miles 120 --days 4
    python Daily_Planner.py segments.mpz --hotel 42.05,-80.09 --min-miles 120 --lock 1

Outputs (in --out-dir, default daily_plan/):
    day_N.mpz / day_N.gpx   one complete hotel-to-hotel route per day
    plan_overview.html      all days on one map, one color per day
    plan_segments.mpz       every segment tagged with its planned day
    remaining_segments.mpz  (with --lock N) day N marked Scheduled=Yes;
                            feed this file back in tomorrow to replan the rest

Workflow: preview the plan, lock in the day you will drive (--lock N or the
interactive prompt), collect it, then re-run on remaining_segments.mpz (or
on a fresh Map Plus export with Collected=Yes set) for the next day.
"""

import argparse
import os
import sys
from math import atan2, cos, pi, radians, sin

from Route_Planner import (
    DEFAULT_SPEED_KMH, OVERPASS_CACHE_DIR, apply_osm_speeds, build_graph,
    build_visits, chain_sections, fetch_osm_ways, haversine, merge_endpoints,
    parse_input, snap_coord, solve_route, write_gpx, write_mpz,
    write_segments_mpz,
)

MILES_PER_M = 1 / 1609.344
DAY_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#008080", "#9a6324"]


# ============================================================================
# Hotel position
# ============================================================================

def resolve_hotel(text, log=print):
    """Resolve --hotel input: either 'LAT,LON' or a street address."""
    parts = text.split(",")
    if len(parts) == 2:
        try:
            return snap_coord(float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    import requests
    log(f"  Geocoding hotel: {text}")
    resp = requests.get("https://nominatim.openstreetmap.org/search",
                        params={"q": text, "format": "json", "limit": 1},
                        headers={"User-Agent": "SurveyRoutePlanner/1.0"},
                        timeout=30)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Address not found: {text!r} - "
                         f"try coordinates as LAT,LON instead")
    hit = results[0]
    log(f"  Found: {hit.get('display_name', text)}")
    return snap_coord(float(hit["lat"]), float(hit["lon"]))


def link_hotel(graph, hotel, log=print):
    """Insert the hotel into the routing graph at the nearest road node."""
    best, best_d = None, float("inf")
    for node in graph.nodes:
        d = haversine(hotel, node)
        if d < best_d:
            best, best_d = node, d
    if best is None:
        raise ValueError("Routing graph is empty")
    if best != hotel:
        mps = DEFAULT_SPEED_KMH * 1000.0 / 3600.0
        graph.add_edge(hotel, best, best_d / mps, best_d)
        graph.add_edge(best, hotel, best_d / mps, best_d)
    if best_d > 200:
        log(f"  Note: hotel is {best_d:.0f} m from the nearest mapped road")
    return hotel


# ============================================================================
# Balanced partition into days
# ============================================================================

def _bearing(a, b):
    """Initial bearing from a to b in radians, 0..2*pi."""
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    y = sin(lon2 - lon1) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
    return atan2(y, x) % (2 * pi)


def _centroid(chain):
    coords = chain["coords"]
    return (sum(lat for lat, _ in coords) / len(coords),
            sum(lon for _, lon in coords) / len(coords))


def sweep_partition(chains, hotel, days):
    """Split chains into `days` compass wedges of roughly equal survey work.

    Returns a list of `days` lists of chain indices, in sweep order, so
    adjacent groups are also adjacent on the ground (which is what lets the
    rebalancing step shift work between neighboring days sensibly).
    """
    order = sorted(range(len(chains)),
                   key=lambda ci: _bearing(hotel, _centroid(chains[ci])))
    if days <= 1:
        return [order]

    # Start the sweep at the largest angular gap so one natural cluster is
    # never split across the seam into two different days.
    angles = [_bearing(hotel, _centroid(chains[ci])) for ci in order]
    gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * pi)
            for i in range(len(angles))]
    seam = (gaps.index(max(gaps)) + 1) % len(order)
    order = order[seam:] + order[:seam]

    total = sum(chains[ci]["length_m"] for ci in order)
    target = total / days
    groups = [[] for _ in range(days)]
    cum = 0.0
    for ci in order:
        mid = cum + chains[ci]["length_m"] / 2
        day = min(int(mid / target), days - 1)
        groups[day].append(ci)
        cum += chains[ci]["length_m"]

    # A huge single run can leave a wedge empty; steal from a neighbor.
    for day in range(days):
        if groups[day]:
            continue
        for nb in (day - 1, day + 1):
            if 0 <= nb < days and len(groups[nb]) > 1:
                groups[day].append(groups[nb].pop(0 if nb > day else -1))
                break
    return [g for g in groups if g]


# ============================================================================
# Per-day routing (hotel -> wedge -> hotel)
# ============================================================================

def solve_day(graph, chains, group, hotel):
    """Route one day: from the hotel through its runs and back."""
    day_chains = [chains[ci] for ci in group]
    route, legs, total = solve_route(graph, day_chains, start=hotel,
                                     log=lambda *a: None)

    hotel_id = graph.node_id[hotel]
    back = graph.nearest_target(route[-1], {hotel_id})
    wrong_way_return = False
    if back is None:
        back = graph.nearest_target(route[-1], {hotel_id}, undirected=True)
        wrong_way_return = back is not None
    if back is not None:
        _, ret_s, ret_m, ret_path = back
    else:  # disconnected data (only possible without OSM)
        ret_m = haversine(route[-1], hotel)
        ret_s = ret_m / (DEFAULT_SPEED_KMH * 1000.0 / 3600.0)
        ret_path = [route[-1], hotel]
        total["jumps"] += 1

    full_route = route + ret_path[1:]
    return {
        "group": group,
        "chains": day_chains,
        "legs": legs,
        "route": full_route,
        "total": total,
        "return_path": ret_path,
        "return_m": ret_m,
        "return_s": ret_s,
        "wrong_way_return": wrong_way_return,
    }


def day_survey_mi(day):
    return day["total"]["survey_m"] * MILES_PER_M


def day_deadhead_mi(day):
    return (day["total"]["deadhead_m"] + day["return_m"]) * MILES_PER_M


def day_total_mi(day):
    return day_survey_mi(day) + day_deadhead_mi(day)


def day_hours(day):
    t = day["total"]
    return (t["survey_s"] + t["deadhead_s"] + day["return_s"]) / 3600


def rebalance(graph, chains, groups, solved, hotel, log, max_moves=10):
    """Even out the days on their true driven mileage.

    Shifts a boundary run from the heaviest day into the angularly adjacent
    lighter day and re-solves both; a move is kept only when it shrinks the
    gap between the heaviest and lightest day. This deliberately accepts a
    slightly worse single day when it keeps the whole plan consistent.
    """
    for _ in range(max_moves):
        miles = [day_total_mi(d) for d in solved]
        avg = sum(miles) / len(miles)
        spread = max(miles) - min(miles)
        if spread <= 0.15 * avg:
            break
        hi = miles.index(max(miles))

        candidates = []
        if hi > 0 and len(groups[hi]) > 1 and miles[hi - 1] < miles[hi]:
            candidates.append((hi - 1, 0))   # give first run to previous day
        if (hi < len(groups) - 1 and len(groups[hi]) > 1
                and miles[hi + 1] < miles[hi]):
            candidates.append((hi + 1, -1))  # give last run to next day
        if not candidates:
            break

        improved = False
        for dst, pos in candidates:
            ci = groups[hi][pos]
            new_hi = [c for c in groups[hi] if c != ci]
            new_dst = ([ci] + groups[dst]) if pos == -1 else (groups[dst] + [ci])
            trial_hi = solve_day(graph, chains, new_hi, hotel)
            trial_dst = solve_day(graph, chains, new_dst, hotel)
            trial_miles = list(miles)
            trial_miles[hi] = day_total_mi(trial_hi)
            trial_miles[dst] = day_total_mi(trial_dst)
            if max(trial_miles) - min(trial_miles) < spread:
                groups[hi], groups[dst] = new_hi, new_dst
                solved[hi], solved[dst] = trial_hi, trial_dst
                improved = True
                log(f"  Rebalance: moved a run from day {hi + 1} "
                    f"to day {dst + 1} "
                    f"({', '.join(f'{m:.0f}' for m in trial_miles)} mi)")
                break
        if not improved:
            break
    return groups, solved


# ============================================================================
# Reporting + outputs
# ============================================================================

def print_plan(solved, min_miles, log=print):
    log("\n" + "=" * 64)
    log("MULTI-DAY PLAN")
    log("=" * 64)
    for n, day in enumerate(solved, start=1):
        total_mi = day_total_mi(day)
        survey_mi = day_survey_mi(day)
        dh_mi = day_deadhead_mi(day)
        eff = survey_mi / total_mi * 100 if total_mi else 0
        visits = build_visits(day["chains"], day["legs"])
        ok = "OK" if total_mi >= min_miles else "UNDER MINIMUM"
        log(f"\n--- Route {n} -------------------------------------------------")
        log(f"  {total_mi:6.1f} mi total  =  {survey_mi:.1f} mi collection "
            f"+ {dh_mi:.1f} mi deadhead (incl. return to hotel)")
        log(f"  {len(visits)} segments | ~{day_hours(day):.1f} h driving | "
            f"efficiency {eff:.0f}% | minimum {min_miles:.0f} mi: {ok}")
        if day["wrong_way_return"]:
            log("  Note: return leg ignores one-way restrictions")
        log("  Collection order:")
        line = []
        for v in visits:
            section = v["section"]
            against = v["reversed"] != section["digitized_reversed"]
            collid = section["collid"] or section["name"]
            line.append(f"{collid}{'D' if against else 'I'}")
            if len(line) == 10:
                log("    " + "  ".join(line))
                line = []
        if line:
            log("    " + "  ".join(line))

    miles = [day_total_mi(d) for d in solved]
    log("\n" + "-" * 64)
    log(f"  Plan: {len(solved)} days | "
        f"{sum(miles):.0f} mi total | "
        f"days range {min(miles):.0f}-{max(miles):.0f} mi | "
        f"spread {(max(miles) - min(miles)) / (sum(miles) / len(miles)) * 100:.0f}% of average")
    log("  (I = blue, drive WITH the arrows; D = pink, drive AGAINST them)")


def write_plan_html(solved, hotel, collected, filename):
    """One Leaflet map with every day's loop in its own color."""
    import json as _json
    days = []
    for n, day in enumerate(solved, start=1):
        days.append({
            "label": f"Route {n}",
            "color": DAY_COLORS[(n - 1) % len(DAY_COLORS)],
            "route": [[lat, lon] for lat, lon in day["route"]],
            "miles": round(day_total_mi(day), 1),
            "segments": sum(len(c["parts"]) for c in day["chains"]),
            "hours": round(day_hours(day), 1),
        })
    data = {"days": days, "hotel": list(hotel),
            "collected": [[[lat, lon] for lat, lon in s["digitized_coords"]]
                          for s in collected]}

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Day Survey Plan</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
 html, body, #map { height: 100%; margin: 0; font-family: system-ui, sans-serif; }
 .info { position: absolute; top: 10px; right: 10px; z-index: 1000; background: #fff;
         padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.3);
         font-size: 14px; line-height: 1.7; }
 .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
       margin-right: 6px; vertical-align: -1px; }
</style></head><body>
<div id="map"></div>
<div class="info" id="legend"><b>Multi-Day Plan</b><br></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = __DATA__;
const map = L.map('map');
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(map);
data.collected.forEach(c => L.polyline(c,
  {color: '#888', weight: 2, opacity: .4, dashArray: '4 6'}).addTo(map));
let bounds = null;
const legend = document.getElementById('legend');
data.days.forEach(d => {
  const line = L.polyline(d.route, {color: d.color, weight: 3, opacity: .85})
    .bindPopup('<b>' + d.label + '</b><br>' + d.miles + ' mi, ' +
               d.segments + ' segments, ~' + d.hours + ' h').addTo(map);
  bounds = bounds ? bounds.extend(line.getBounds()) : line.getBounds();
  legend.innerHTML += '<span class="sw" style="background:' + d.color + '"></span>'
    + d.label + ': ' + d.miles + ' mi, ' + d.segments + ' seg, ~' + d.hours + ' h<br>';
});
L.marker(data.hotel).bindPopup('<b>HOTEL</b> - every route starts and ends here')
  .addTo(map);
if (bounds) map.fitBounds(bounds.pad(0.05));
</script></body></html>"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA__", _json.dumps(data)))
    return filename


# ============================================================================
# Pipeline
# ============================================================================

def plan_days(input_path, hotel_text, min_miles, days=None, lock=None,
              out_dir="daily_plan", use_osm=True,
              cache_dir=OVERPASS_CACHE_DIR, log=print):
    hotel = resolve_hotel(hotel_text, log=log)
    log(f"  Hotel position: {hotel[0]:.5f}, {hotel[1]:.5f}")

    log("\n[1/5] Parsing segments...")
    sections = parse_input(input_path, log=log)
    merge_endpoints(sections, log=log)
    unavailable = [s for s in sections if s["collected"] or s["scheduled"]]
    todo = [s for s in sections if not (s["collected"] or s["scheduled"])]
    if unavailable:
        n_col = sum(1 for s in unavailable if s["collected"])
        n_sch = len(unavailable) - n_col
        parts = []
        if n_col:
            parts.append(f"{n_col} collected")
        if n_sch:
            parts.append(f"{n_sch} locked into earlier days")
        log(f"  Skipping {' + '.join(parts)}; {len(todo)} segments to plan")
    if not todo:
        raise ValueError("Every segment is already collected or scheduled - "
                         "nothing left to plan")

    log("\n[2/5] Fetching OSM road network..." if use_osm
        else "\n[2/5] Skipping OSM (offline mode)")
    osm_ways = []
    if use_osm:
        points = [pt for s in sections for pt in s["coords"]] + [hotel]
        osm_ways = fetch_osm_ways(points, cache_dir=cache_dir, log=log)
        if not osm_ways:
            raise RuntimeError(
                "No OSM road data found for the survey area; daily loops "
                "could not be kept on roads. Re-run to retry, or pass "
                "--no-osm to accept straight-line transfers.")
        apply_osm_speeds(sections, osm_ways, log=log)

    log("\n[3/5] Building road graph...")
    chains = chain_sections(todo, log=log)
    graph = build_graph(chains, osm_ways, collected_sections=unavailable,
                        log=log)
    link_hotel(graph, hotel, log=log)

    total_survey_mi = sum(s["length_m"] for s in todo) * MILES_PER_M
    if days is None:
        days = max(1, min(5, round(total_survey_mi / min_miles)))
        while days > 1 and total_survey_mi / days < min_miles * 0.6:
            days -= 1
    log(f"\n[4/5] Planning {days} day(s) for {total_survey_mi:.0f} mi of "
        f"collection (min {min_miles:.0f} mi/day)...")

    groups = sweep_partition(chains, hotel, days)
    solved = [solve_day(graph, chains, g, hotel) for g in groups]
    groups, solved = rebalance(graph, chains, groups, solved, hotel, log)

    # If a day still can't reach the minimum, use fewer days.
    while (len(solved) > 1
           and min(day_total_mi(d) for d in solved) < min_miles):
        days = len(solved) - 1
        log(f"  A day fell under the {min_miles:.0f} mi minimum - "
            f"replanning with {days} day(s)")
        groups = sweep_partition(chains, hotel, days)
        solved = [solve_day(graph, chains, g, hotel) for g in groups]
        groups, solved = rebalance(graph, chains, groups, solved, hotel, log)

    jumps = sum(d["total"]["jumps"] for d in solved)
    if jumps:
        log(f"  WARNING: {jumps} transfers had no road connection at all; "
            f"straight-line jumps were used (run with OSM enabled).")

    print_plan(solved, min_miles, log=log)

    log("\n[5/5] Writing outputs...")
    os.makedirs(out_dir, exist_ok=True)
    day_of = {}
    for n, day in enumerate(solved, start=1):
        for chain in day["chains"]:
            for section, _ in chain["parts"]:
                day_of[id(section)] = n
        visits = build_visits(day["chains"], day["legs"])
        total = dict(day["total"])
        total["deadhead_m"] += day["return_m"]
        total["deadhead_s"] += day["return_s"]
        mpz = os.path.join(out_dir, f"day_{n}.mpz")
        gpx = os.path.join(out_dir, f"day_{n}.gpx")
        write_mpz(visits, total, mpz, title=f"Day {n} route",
                  return_path=day["return_path"])
        write_gpx(day["route"], day["legs"], total, gpx)
        log(f"  Route {n}: {mpz} + {gpx}")

    overview = os.path.join(out_dir, "plan_overview.html")
    write_plan_html(solved, hotel, unavailable, overview)
    log(f"  Map: {overview}")

    plan_file = os.path.join(out_dir, "plan_segments.mpz")
    write_segments_mpz(sections, plan_file, "Multi-day plan", day_of=day_of)
    log(f"  Plan state: {plan_file} (every segment tagged with its day)")

    # --- Lock in a day ------------------------------------------------------
    if lock is None and sys.stdin.isatty():
        try:
            answer = input(f"\nLock in a route to drive (1-{len(solved)}), "
                           f"or Enter to skip: ").strip()
            lock = int(answer) if answer else None
        except (ValueError, EOFError):
            lock = None
    if lock is not None:
        if not 1 <= lock <= len(solved):
            raise ValueError(f"--lock must be between 1 and {len(solved)}")
        for chain in solved[lock - 1]["chains"]:
            for section, _ in chain["parts"]:
                section["scheduled"] = True
        remaining_of = {sid: d for sid, d in day_of.items()}
        remaining = os.path.join(out_dir, "remaining_segments.mpz")
        write_segments_mpz(sections, remaining, "Remaining segments",
                           day_of=remaining_of)
        log(f"\n  Route {lock} locked in. Drive {os.path.join(out_dir, f'day_{lock}.mpz')}")
        log(f"  Tomorrow: python Daily_Planner.py {remaining} "
            f"--hotel ... --min-miles {min_miles:.0f}")

    return solved


def main():
    parser = argparse.ArgumentParser(
        description="Plan balanced daily collection loops from a hotel over "
                    "the remaining segments of a KML or Map Plus (.mpz) file.")
    parser.add_argument("segments", help="KML or .mpz file with the segments")
    parser.add_argument("--hotel", required=True, metavar="ADDR_OR_LATLON",
                        help='hotel: street address or "LAT,LON"')
    parser.add_argument("--min-miles", type=float, required=True,
                        help="minimum total miles each daily route must reach")
    parser.add_argument("--days", type=int,
                        help="number of routes to build (default: sized from "
                             "--min-miles, capped at 5)")
    parser.add_argument("--lock", type=int, metavar="N",
                        help="lock in route N non-interactively and write "
                             "remaining_segments.mpz")
    parser.add_argument("--out-dir", default="daily_plan",
                        help="output directory (default: daily_plan)")
    parser.add_argument("--no-osm", action="store_true",
                        help="skip OpenStreetMap fetch (transfers may leave "
                             "roads; not recommended)")
    args = parser.parse_args()

    plan_days(args.segments, args.hotel, args.min_miles, days=args.days,
              lock=args.lock, out_dir=args.out_dir, use_osm=not args.no_osm)


if __name__ == "__main__":
    main()
