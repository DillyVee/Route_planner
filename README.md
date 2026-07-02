# Survey Route Planner

Plans an efficient driving route that covers every road section in a KML file
(MapPlus/Duweis roadway-survey exports or generic KML) **or a Map Plus
project (.mpz)**, then writes a mobile-friendly **GPX** track, an interactive
**HTML map** preview, and a native **Map Plus project (.mpz)** built for the
field workflow. Segments already marked `Collected = Yes` in the input are
left out of the route and carried through faded, so re-uploading a partially
collected project plans only what's left.

Two programs:

- **`Route_Planner.py`** — one continuous route over everything remaining.
- **`Daily_Planner.py`** — a balanced multi-day plan of hotel-to-hotel loops
  (see [Daily planner](#daily-planner-hotel-loops) below).

Everything lives in one file: `Route_Planner.py`.

## Install

```bash
pip install -r requirements.txt   # requests (OSM data; needed unless --no-osm) + PyQt6 (GUI)
```

## Usage

```bash
python Route_Planner.py                     # GUI (both tools)
python Route_Planner.py sections.kml       # headless CLI
python Route_Planner.py region.mpz         # Map Plus project input
python Route_Planner.py sections.kml --no-osm --gpx out.gpx --mpz out.mpz --start 40.44,-79.99
```

The GUI covers both tools in one dark, modern window: a **Single route**
tab and a **Daily plan from hotel** tab with hotel address input (street
address or LAT,LON), shift hours, day count, drag-and-drop file loading, a
live log, and a one-click **Lock in route** button that writes
`remaining_segments.mpz` for tomorrow.

Try it: `python Route_Planner.py sample.kml --no-osm`

## How it works

1. **Parse KML or MPZ** — reads sections with direction (`Dir`
   NB/SB/EB/WB/I/D/B), `Collected` status, and speed-limit metadata; KML
   parsing repairs malformed XML; already-collected sections are set aside.
2. **Merge endpoints** — endpoints within ~15 m snap together, so sections the
   state chopped apart reconnect.
3. **Chain sections** — sections that run end-to-start are joined into single
   continuous runs (driven in one pass, and far fewer pieces to order).
4. **OSM network (cached)** — the survey area is fetched from Overpass in
   small tiles (large regions can't be fetched in one query), with retries
   across several Overpass servers. Tiles are cached on disk in
   `overpass_cache/`, so only the first run downloads. If the road network
   can't be fetched the planner stops with an error rather than producing a
   route with off-road jumps. Chain endpoints are linked to the nearest OSM
   nodes so the two networks connect, and missing speed limits are filled in.
5. **Solve** — greedy nearest-section ordering. Each step is a single Dijkstra
   search that stops the moment it reaches the closest remaining section, so
   back-to-back sections cost nothing to connect. Two-way runs can be entered
   from either end; one-way runs are driven in their required direction.
   Transfers always stay on roads: if one-way restrictions leave no legal
   path, the search retries ignoring one-way (still on real roads) before
   ever considering anything else. Already-collected sections stay routable
   as transfer roads.
6. **Output** — GPX with a named waypoint (distance/ETA) per run, plus a
   Leaflet HTML preview showing the route, waypoints, already-collected
   segments (faded gray), and survey/deadhead totals.

With OSM data on (the default) every transfer follows real roads. Only when
OSM is disabled *and* the survey sections themselves are physically
disconnected is a straight-line jump used, and it is loudly reported.

## Map Plus output (.mpz)

`survey_route.mpz` imports straight into Map Plus and matches the structure
**and color language** of the state-issued region files: one line feature per
original section, in a feature collection sorted by driving order, keeping
every field (`CollId`, `RouteName`, `Dir`, `Collected`, ...) and labeled
`[CollId]` on the map.

Direction colors (identical to the reference files):

- **Blue** (`Dir = I`) — drive **with** the digitized arrows.
- **Pink** (`Dir = D`) — drive **against** the digitized arrows.
- Segments keep their original digitized geometry, so the arrows point the
  same way as in the source map and the blob color tells you which way to
  drive. The description also spells it out.

Field workflow:

- Features are named `0001 · <CollId>`, `0002 · <CollId>`, ... so the feature
  list is the day's ordered task list, and tapping any segment shows its
  CollId, run number, length, ETA, and the CollId of the **next** segment
  (also stored in the `RouteOrder` / `NextCollId` properties).
- Transfers between runs appear as thin gray `→ 0042` features drawn along
  the actual connecting roads, telling you how to reach the next segment and
  which CollId comes next.
- Set `Collected` to `Yes` as you finish each segment and it fades out on the
  map (same conditional-style trick as the reference files), so the bright
  segments are always what's left to do.
- **Re-upload a partially collected project** (export it from Map Plus and
  run it through the planner again): segments already marked `Collected =
  Yes` keep their faded look as `✓ <CollId>` features, and a fresh route is
  planned over only the remaining segments — their roads are still used for
  transfers where helpful.

## Daily planner (hotel loops)

`Daily_Planner.py` turns the remaining segments into **balanced daily
routes** that each start and end at your hotel and fit inside a driving
shift (`--hours`, default 6 — wiggle room for gas and breaks inside an
8-hour day):

```bash
python Daily_Planner.py segments.mpz --hotel "8051 Peach St, Erie PA"
python Daily_Planner.py segments.mpz --hotel 42.05,-80.08 --hours 6 --days 4
python Daily_Planner.py segments.mpz --hotel 42.05,-80.08 --lock 1
```

How it balances the days:

1. Continuous runs are swept by compass bearing around the hotel into one
   wedge per day, each holding about the same collection time. Wedges
   keep every day's work in one compact area, so finishing a day never
   strands isolated segments in another day's territory — what remains is
   still clustered for tomorrow.
2. Each day is solved as a real road route (same engine as
   `Route_Planner.py`: OSM network, blue/pink directions, transfers on
   roads) from the hotel through its wedge and back.
3. Days are rebalanced on true driving time: boundary runs shift between
   neighboring days until no day is dramatically bigger than another — a
   run may leave its locally optimal day if that keeps the whole week
   consistent.
4. The shift cap is hard. If the days run over, more days are added
   (up to `--days`, default 5); if it still doesn't fit, days shed their
   **nearest-to-hotel** runs into an unplanned **filler pool** — far
   segments always stay inside a planned, time-boxed day so a shift never
   ends with a long drive home, and the filler near the hotel is what you
   grab when a day finishes early.

Every route is printed with estimated hours against the shift, total miles,
collection vs deadhead split, and the segment collection order (`123I` =
CollId 123 blue / with the arrows, `456D` = pink / against them).
`--min-miles` is optional and only flags routes under the threshold —
balance is driven by time.

Shift hours come from **real field pace**, not map speed limits (rural
roads mostly carry no posted limit in OSM and would look far slower than
reality): 35 mph while collecting on rural roads, 45 mph on deadhead. At
that pace a 6 h day holds roughly 200–250 mi of driving. No tuning needed —
`--collect-mph` / `--transfer-mph` exist only for unusual regions.

Outputs (in `--out-dir`, default `daily_plan/`):

- `day_N.mpz` / `day_N.gpx` — one complete hotel-to-hotel route per day.
- `plan_overview.html` — all days on one map, one color per day, hotel
  marked.
- `plan_segments.mpz` — every segment tagged with its planned day
  (`ScheduledDay`).
- `remaining_segments.mpz` (with `--lock N` or the interactive prompt) —
  route N's segments marked `Scheduled = Yes`.

Daily workflow: preview the plan, lock in the route you'll drive
(`--lock 1`), collect it, then re-run tomorrow on
`remaining_segments.mpz` — or on a fresh Map Plus export with
`Collected = Yes` set — and get a fresh balanced plan over what's left.
Finished early? Grab filler segments near the hotel, mark them collected,
and the next re-run accounts for everything automatically.

## Input format

- **Map Plus project (.mpz)**: state-issued region files or projects exported
  from Map Plus (including routes generated by this tool — their deadhead
  transfer features are ignored on re-import). Reads every line feature with
  its `CollId`, `RouteName`, `Dir`, `Collected`, and speed properties.
- **MapPlus/Duweis KML**: `CollId`, `RouteName`, `Dir` (`I`/`D`/`B`/`T`/`NB`/`SB`/`EB`/`WB`),
  `Collected`, speed fields in `ExtendedData`.
- **Generic KML**: any `LineString` placemarks; optional `oneway` and
  `maxspeed` extended data or values in the description/name.

## License

MIT — see [LICENSE](LICENSE).
