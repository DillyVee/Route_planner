#!/usr/bin/env python3
"""
Daily Survey Route Planner
==========================

Builds a balanced multi-day collection plan from a hotel: several complete
daily routes that each start and end at the hotel, fit inside a driving-time
shift, and stay consistent in effort from day to day - the goal is a
sustainable field schedule, not one great day followed by garbage days of
scattered leftovers.

Time is the budget: --hours caps each day's estimated driving (default 6 h,
which leaves wiggle room for gas and breaks inside an 8-hour shift). Every
route ends back at the hotel within that cap. When there is more work than
the planned days can hold, runs close to the hotel are left unplanned as
safe end-of-shift filler - far-away segments always stay inside a planned,
time-boxed day so you are never stranded far out when the shift ends.

Hours are computed from observed field pace, not map speed limits (rural
roads mostly have no posted limit in OSM and would look far slower than
reality): 35 mph while collecting (rural roads) and 45 mph on deadhead.
At that pace a 6 h day holds roughly 200-250 mi of driving. No tuning
needed; --collect-mph / --transfer-mph exist only for unusual regions.

How the balance works
---------------------
1. Segments are grouped into continuous runs (chains) and swept by compass
   bearing around the hotel into one wedge per day, each wedge holding about
   the same collection time. Wedges keep every day's work in one compact
   area, so collecting a day never strands isolated segments inside another
   day's territory - the lookahead is structural: whatever remains after a
   day is still clustered for the following days.
2. Every day is then solved as a real road route (same engine as
   Route_Planner: OSM road network, direction-aware, transfers always on
   roads) from the hotel, through its wedge, and back to the hotel.
3. Days are rebalanced on their true driving time: boundary runs shift
   from the heaviest day into the angularly adjacent lighter day - a run
   may leave its locally optimal day if that keeps tomorrow consistent -
   until no single day is dramatically bigger than another.
4. Days still over the shift cap shed their nearest-to-hotel runs into the
   unplanned filler pool until they fit.

Usage
-----
    python Daily_Planner.py segments.mpz --hotel "8051 Peach St, Erie PA"
    python Daily_Planner.py segments.mpz --hotel 42.05,-80.09 --hours 6 --days 4
    python Daily_Planner.py segments.mpz --hotel 42.05,-80.09 --lock 1

Outputs (in --out-dir, default daily_plan/):
    day_N.mpz / day_N.gpx   one complete hotel-to-hotel route per day
    plan_overview.html      all days on one map, one color per day
    plan_segments.mpz       every segment tagged with its planned day
    remaining_segments.mpz  (with --lock N) day N marked Scheduled=Yes;
                            feed this file back in tomorrow to replan the rest

Workflow: preview the plan, lock in the day you will drive (--lock N or the
interactive prompt), collect it, then re-run on remaining_segments.mpz (or
on a fresh Map Plus export with Collected=Yes set) for the next day. If a
day finishes early, grab unplanned filler segments near the hotel and mark
them collected - the next re-run accounts for everything automatically.
"""

import argparse
import os
import sys
from math import atan2, ceil, cos, pi, radians, sin

from Route_Planner import (
    DEFAULT_SPEED_KMH, OVERPASS_CACHE_DIR, apply_osm_speeds, build_graph,
    build_visits, chain_sections, fetch_osm_ways, haversine, merge_endpoints,
    parse_input, snap_coord, solve_route, write_gpx, write_mpz,
    write_segments_mpz,
)

MILES_PER_M = 1 / 1609.344
DAY_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
              "#008080", "#9a6324"]
# Rough share of a day expected to be actual collection (the rest is
# transfers + the drive out and back); only used to size the initial day
# count - the real cap is enforced on solved routes.
COLLECTION_SHARE = 0.6
# Real-world pace used to turn route miles into shift hours. Posted speed
# limits are a bad predictor of a field day (most rural OSM roads carry no
# limit at all and fall back to slow class defaults), so day length is
# estimated from these observed averages instead. Rural collection roads
# run 35 mph; no tuning needed, but --collect-mph / --transfer-mph can
# override if a region drives differently.
COLLECT_MPH = 35.0    # average while collecting segments (rural roads)
TRANSFER_MPH = 45.0   # average on deadhead between segments / to the hotel


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
    """Split chains into `days` compass wedges of roughly equal collection time.

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

    total = sum(chains[ci]["time_s"] for ci in order)
    target = total / days
    groups = [[] for _ in range(days)]
    cum = 0.0
    for ci in order:
        mid = cum + chains[ci]["time_s"] / 2
        day = min(int(mid / target), days - 1)
        groups[day].append(ci)
        cum += chains[ci]["time_s"]

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

def solve_day(graph, chains, group, hotel, pace):
    """Route one day: from the hotel through its runs and back.

    `pace` is (collect_mph, transfer_mph); the day's shift hours come from
    those observed averages, not from map speed limits.
    """
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

    survey_mi = total["survey_m"] * MILES_PER_M
    deadhead_mi = (total["deadhead_m"] + ret_m) * MILES_PER_M
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
        "hours": survey_mi / pace[0] + deadhead_mi / pace[1],
        "wrong_way_return": wrong_way_return,
    }


def day_survey_mi(day):
    return day["total"]["survey_m"] * MILES_PER_M


def day_deadhead_mi(day):
    return (day["total"]["deadhead_m"] + day["return_m"]) * MILES_PER_M


def day_total_mi(day):
    return day_survey_mi(day) + day_deadhead_mi(day)


def day_hours(day):
    return day["hours"]


def rebalance(graph, chains, groups, solved, hotel, pace, log, max_moves=10):
    """Even out the days on their true driving time.

    Shifts a boundary run from the heaviest day into the angularly adjacent
    lighter day and re-solves both; a move is kept only when it shrinks the
    gap between the heaviest and lightest day. This deliberately accepts a
    slightly worse single day when it keeps the whole plan consistent.
    """
    if len(solved) < 2:
        return groups, solved
    for _ in range(max_moves):
        hours = [day_hours(d) for d in solved]
        avg = sum(hours) / len(hours)
        spread = max(hours) - min(hours)
        if spread <= 0.15 * avg:
            break
        hi = hours.index(max(hours))
        lo = hours.index(min(hours))

        # Candidate moves (src day, dst day, boundary position in src):
        # the heaviest day gives a boundary run to a lighter neighbor, or
        # the lightest day takes one from a heavier neighbor.
        candidates = []
        if hi > 0 and hours[hi - 1] < hours[hi]:
            candidates.append((hi, hi - 1, 0))
        if hi < len(groups) - 1 and hours[hi + 1] < hours[hi]:
            candidates.append((hi, hi + 1, -1))
        if lo > 0 and hours[lo - 1] > hours[lo]:
            candidates.append((lo - 1, lo, -1))
        if lo < len(groups) - 1 and hours[lo + 1] > hours[lo]:
            candidates.append((lo + 1, lo, 0))

        best = None
        for src, dst, pos in candidates:
            if len(groups[src]) <= 1:
                continue
            ci = groups[src][pos]
            new_src = [c for c in groups[src] if c != ci]
            new_dst = ([ci] + groups[dst]) if pos == -1 else (groups[dst] + [ci])
            trial_src = solve_day(graph, chains, new_src, hotel, pace)
            trial_dst = solve_day(graph, chains, new_dst, hotel, pace)
            trial_hours = list(hours)
            trial_hours[src] = day_hours(trial_src)
            trial_hours[dst] = day_hours(trial_dst)
            trial_spread = max(trial_hours) - min(trial_hours)
            if trial_spread < spread and (best is None or trial_spread < best[0]):
                best = (trial_spread, src, dst, new_src, new_dst,
                        trial_src, trial_dst, trial_hours)

        if best is None:
            break
        _, src, dst, new_src, new_dst, trial_src, trial_dst, trial_hours = best
        groups[src], groups[dst] = new_src, new_dst
        solved[src], solved[dst] = trial_src, trial_dst
        log(f"  Rebalance: moved a run from day {src + 1} to day {dst + 1} "
            f"({', '.join(f'{h:.1f}' for h in trial_hours)} h)")
    return groups, solved


def trim_to_shift(graph, chains, groups, solved, hotel, pace, max_hours, log):
    """Shed work from days that exceed the shift until they fit.

    Runs closest to the hotel are dropped first: they are the safe ones to
    leave unplanned (easy to grab with spare time at the end of any shift),
    while far-away runs must stay inside a planned, time-boxed day so the
    shift never ends with a long drive home. Returns the dropped chain
    indices (the filler pool).
    """
    filler = []
    for di in range(len(groups)):
        while day_hours(solved[di]) > max_hours and len(groups[di]) > 1:
            ci = min(groups[di],
                     key=lambda c: haversine(hotel, _centroid(chains[c])))
            groups[di].remove(ci)
            filler.append(ci)
            solved[di] = solve_day(graph, chains, groups[di], hotel, pace)
        if day_hours(solved[di]) > max_hours:
            log(f"  WARNING: day {di + 1} is a single run that alone takes "
                f"{day_hours(solved[di]):.1f} h - it cannot fit the "
                f"{max_hours:.1f} h shift")
    if filler:
        mi = sum(chains[ci]["length_m"] for ci in filler) * MILES_PER_M
        log(f"  {len(filler)} runs ({mi:.0f} mi) near the hotel left "
            f"unplanned as end-of-shift filler - far segments stay inside "
            f"time-boxed days")
    return filler


# ============================================================================
# Reporting + outputs
# ============================================================================

def print_plan(solved, max_hours, min_miles, filler_info, pace, log=print):
    log("\n" + "=" * 64)
    log("MULTI-DAY PLAN")
    log(f"(hours assume {pace[0]:.0f} mph collecting / {pace[1]:.0f} mph "
        f"transfers - tune with --collect-mph / --transfer-mph)")
    log("=" * 64)
    for n, day in enumerate(solved, start=1):
        total_mi = day_total_mi(day)
        survey_mi = day_survey_mi(day)
        dh_mi = day_deadhead_mi(day)
        hours = day_hours(day)
        eff = survey_mi / total_mi * 100 if total_mi else 0
        visits = build_visits(day["chains"], day["legs"])
        fit = "fits" if hours <= max_hours else "OVER"
        log(f"\n--- Route {n} -------------------------------------------------")
        log(f"  ~{hours:.1f} h driving ({fit} the {max_hours:.1f} h shift) | "
            f"{total_mi:.1f} mi total = {survey_mi:.1f} mi collection "
            f"+ {dh_mi:.1f} mi deadhead (incl. return)")
        log(f"  {len(visits)} segments | efficiency {eff:.0f}%"
            + (f" | minimum {min_miles:.0f} mi: "
               f"{'OK' if total_mi >= min_miles else 'UNDER'}"
               if min_miles else ""))
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

    hours = [day_hours(d) for d in solved]
    miles = [day_total_mi(d) for d in solved]
    log("\n" + "-" * 64)
    spread = ((max(hours) - min(hours)) / (sum(hours) / len(hours)) * 100
              if len(hours) > 1 else 0)
    log(f"  Plan: {len(solved)} days | {sum(miles):.0f} mi | "
        f"days {min(hours):.1f}-{max(hours):.1f} h | "
        f"day-to-day spread {spread:.0f}% of average")
    if filler_info:
        log(f"  Filler pool: {filler_info} - grab these near-hotel segments "
            f"whenever a shift ends early, then re-run")
    log("  (I = blue, drive WITH the arrows; D = pink, drive AGAINST them)")


def write_plan_html(solved, hotel, collected, filler_chains, filename):
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
                          for s in collected],
            "filler": [[[lat, lon] for lat, lon in c["coords"]]
                       for c in filler_chains]}

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
data.filler.forEach(c => L.polyline(c,
  {color: '#222', weight: 3, opacity: .6, dashArray: '2 8'})
  .bindPopup('Unplanned filler - near the hotel, grab with spare time').addTo(map));
let bounds = null;
const legend = document.getElementById('legend');
data.days.forEach(d => {
  const line = L.polyline(d.route, {color: d.color, weight: 3, opacity: .85})
    .bindPopup('<b>' + d.label + '</b><br>' + d.miles + ' mi, ' +
               d.segments + ' segments, ~' + d.hours + ' h').addTo(map);
  bounds = bounds ? bounds.extend(line.getBounds()) : line.getBounds();
  legend.innerHTML += '<span class="sw" style="background:' + d.color + '"></span>'
    + d.label + ': ~' + d.hours + ' h, ' + d.miles + ' mi<br>';
});
if (data.filler.length) legend.innerHTML +=
  '<span class="sw" style="background:#222"></span>filler (unplanned)<br>';
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

def plan_days(input_path, hotel_text, max_hours=6.0, min_miles=None,
              days=None, out_dir="daily_plan", use_osm=True,
              collect_mph=COLLECT_MPH, transfer_mph=TRANSFER_MPH,
              cache_dir=OVERPASS_CACHE_DIR, log=print):
    pace = (collect_mph, transfer_mph)
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

    # Size the plan in shift-hours at the observed collection pace;
    # transfers are assumed to take the rest of each day (~40%); the true
    # cap is enforced on the solved routes below.
    total_survey_mi = sum(s["length_m"] for s in todo) * MILES_PER_M
    total_survey_h = total_survey_mi / collect_mph
    max_days = days or 5
    if days is None:
        days = max(1, min(max_days,
                          ceil(total_survey_h / (max_hours * COLLECTION_SHARE))))
    log(f"\n[4/5] Planning {days} day(s): {total_survey_mi:.0f} mi "
        f"(~{total_survey_h:.1f} h) of collection, "
        f"{max_hours:.1f} h driving per shift "
        f"(pace: {collect_mph:.0f} mph collecting, "
        f"{transfer_mph:.0f} mph transfers)...")

    groups = sweep_partition(chains, hotel, days)
    solved = [solve_day(graph, chains, g, hotel, pace) for g in groups]
    groups, solved = rebalance(graph, chains, groups, solved, hotel, pace, log)

    # If the days run over the shift and we may add another day, do that
    # first (keeps everything planned); only then trim to the cap.
    while (days < max_days
           and max(day_hours(d) for d in solved) > max_hours):
        days += 1
        log(f"  Days exceed the {max_hours:.1f} h shift - "
            f"replanning with {days} day(s)")
        groups = sweep_partition(chains, hotel, days)
        solved = [solve_day(graph, chains, g, hotel, pace) for g in groups]
        groups, solved = rebalance(graph, chains, groups, solved, hotel,
                                   pace, log)

    filler = trim_to_shift(graph, chains, groups, solved, hotel, pace,
                           max_hours, log)
    if filler:
        groups, solved = rebalance(graph, chains, groups, solved, hotel,
                                   pace, log)

    jumps = sum(d["total"]["jumps"] for d in solved)
    if jumps:
        log(f"  WARNING: {jumps} transfers had no road connection at all; "
            f"straight-line jumps were used (run with OSM enabled).")

    filler_chains = [chains[ci] for ci in filler]
    filler_info = ""
    if filler:
        filler_mi = sum(c["length_m"] for c in filler_chains) * MILES_PER_M
        n_seg = sum(len(c["parts"]) for c in filler_chains)
        filler_info = f"{n_seg} segments, {filler_mi:.0f} mi near the hotel"
    print_plan(solved, max_hours, min_miles, filler_info, pace, log=log)

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
        # Report times at the observed pace, not map speed limits.
        total["survey_s"] = day_survey_mi(day) / collect_mph * 3600
        total["deadhead_s"] = day_deadhead_mi(day) / transfer_mph * 3600
        mpz = os.path.join(out_dir, f"day_{n}.mpz")
        gpx = os.path.join(out_dir, f"day_{n}.gpx")
        write_mpz(visits, total, mpz, title=f"Day {n} route",
                  return_path=day["return_path"])
        write_gpx(day["route"], day["legs"], total, gpx)
        log(f"  Route {n}: {mpz} + {gpx}")

    overview = os.path.join(out_dir, "plan_overview.html")
    write_plan_html(solved, hotel, unavailable, filler_chains, overview)
    log(f"  Map: {overview}")

    plan_file = os.path.join(out_dir, "plan_segments.mpz")
    write_segments_mpz(sections, plan_file, "Multi-day plan", day_of=day_of)
    log(f"  Plan state: {plan_file} (every segment tagged with its day)")

    return {
        "solved": solved,
        "sections": sections,
        "day_of": day_of,
        "out_dir": out_dir,
        "overview": overview,
        "plan_file": plan_file,
        "max_hours": max_hours,
        "filler_info": filler_info,
    }


def lock_route(plan, n, log=print):
    """Lock in route `n` of a plan: write remaining_segments.mpz.

    The locked route's segments are marked Scheduled=Yes so a re-run of the
    planner on the remaining file plans only what is left.
    """
    solved = plan["solved"]
    if not 1 <= n <= len(solved):
        raise ValueError(f"Route number must be between 1 and {len(solved)}")
    for chain in solved[n - 1]["chains"]:
        for section, _ in chain["parts"]:
            section["scheduled"] = True
    remaining = os.path.join(plan["out_dir"], "remaining_segments.mpz")
    write_segments_mpz(plan["sections"], remaining, "Remaining segments",
                       day_of=plan["day_of"])
    log(f"\n  Route {n} locked in. Drive "
        f"{os.path.join(plan['out_dir'], f'day_{n}.mpz')}")
    log(f"  Tomorrow: python Daily_Planner.py {remaining} "
        f"--hotel ... --hours {plan['max_hours']:g}")
    return remaining


def main():
    parser = argparse.ArgumentParser(
        description="Plan balanced daily collection loops from a hotel over "
                    "the remaining segments of a KML or Map Plus (.mpz) "
                    "file. Days are sized and capped by driving time so "
                    "every route is back at the hotel before the shift ends.")
    parser.add_argument("segments", help="KML or .mpz file with the segments")
    parser.add_argument("--hotel", required=True, metavar="ADDR_OR_LATLON",
                        help='hotel: street address or "LAT,LON"')
    parser.add_argument("--hours", type=float, default=6.0,
                        help="max driving hours per day (default 6.0 - "
                             "leaves wiggle room in an 8 h shift)")
    parser.add_argument("--collect-mph", type=float, default=COLLECT_MPH,
                        help=f"your average speed while collecting segments "
                             f"(default {COLLECT_MPH:.0f})")
    parser.add_argument("--transfer-mph", type=float, default=TRANSFER_MPH,
                        help=f"your average speed on transfers between "
                             f"segments and to/from the hotel "
                             f"(default {TRANSFER_MPH:.0f})")
    parser.add_argument("--min-miles", type=float, default=None,
                        help="optional: flag any route under this many total "
                             "miles (balance itself is driven by --hours)")
    parser.add_argument("--days", type=int,
                        help="number of routes to build (default: sized from "
                             "--hours, capped at 5)")
    parser.add_argument("--lock", type=int, metavar="N",
                        help="lock in route N non-interactively and write "
                             "remaining_segments.mpz")
    parser.add_argument("--out-dir", default="daily_plan",
                        help="output directory (default: daily_plan)")
    parser.add_argument("--no-osm", action="store_true",
                        help="skip OpenStreetMap fetch (transfers may leave "
                             "roads; not recommended)")
    args = parser.parse_args()

    plan = plan_days(args.segments, args.hotel, max_hours=args.hours,
                     min_miles=args.min_miles, days=args.days,
                     out_dir=args.out_dir, use_osm=not args.no_osm,
                     collect_mph=args.collect_mph,
                     transfer_mph=args.transfer_mph)

    lock = args.lock
    if lock is None and sys.stdin.isatty():
        try:
            answer = input(f"\nLock in a route to drive "
                           f"(1-{len(plan['solved'])}), or Enter to skip: ").strip()
            lock = int(answer) if answer else None
        except (ValueError, EOFError):
            lock = None
    if lock is not None:
        lock_route(plan, lock)


if __name__ == "__main__":
    main()
