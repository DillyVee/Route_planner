# Survey Route Planner

Plans an efficient driving route that covers every road section in a KML file
(MapPlus/Duweis roadway-survey exports or generic KML), then writes a
mobile-friendly **GPX** track, an interactive **HTML map** preview, and a
native **Map Plus project (.mpz)** built for the field workflow.

Everything lives in one file: `Route_Planner.py`.

## Install

```bash
pip install -r requirements.txt   # requests (OSM data) + PyQt6 (GUI) — both optional
```

## Usage

```bash
python Route_Planner.py                     # GUI
python Route_Planner.py sections.kml       # headless CLI
python Route_Planner.py sections.kml --no-osm --gpx out.gpx --mpz out.mpz --start 40.44,-79.99
```

Try it: `python Route_Planner.py sample.kml --no-osm`

## How it works

1. **Parse KML** — reads sections with direction (`Dir` NB/SB/EB/WB/I/D/B) and
   speed-limit metadata; repairs malformed XML.
2. **Merge endpoints** — endpoints within ~15 m snap together, so sections the
   state chopped apart reconnect.
3. **Chain sections** — sections that run end-to-start are joined into single
   continuous runs (driven in one pass, and far fewer pieces to order).
4. **OSM network (optional, cached)** — one Overpass query supplies connecting
   roads for deadhead routing and fills in missing speed limits. Chain
   endpoints are linked to the nearest OSM nodes so the two networks connect.
5. **Solve** — greedy nearest-section ordering. Each step is a single Dijkstra
   search that stops the moment it reaches the closest remaining section, so
   back-to-back sections cost nothing to connect. Two-way runs can be entered
   from either end; one-way runs are driven in their required direction.
6. **Output** — GPX with a named waypoint (distance/ETA) per run, plus a
   Leaflet HTML preview showing the route, waypoints, and survey/deadhead
   totals.

Sections with no road connection at all are joined by straight-line jumps and
reported with a warning.

## Map Plus output (.mpz)

`survey_route.mpz` imports straight into Map Plus and matches the structure of
the state-issued region files: one line feature per original KML section, in a
feature collection sorted by driving order, keeping every field (`CollId`,
`RouteName`, `Dir`, `Collected`, ...) and labeled `[CollId]` on the map.

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
- Segments that must be driven opposite to their digitized direction say so
  in their description.

## Input format

- **MapPlus/Duweis KML**: `CollId`, `RouteName`, `Dir` (`I`/`D`/`B`/`T`/`NB`/`SB`/`EB`/`WB`),
  speed fields in `ExtendedData`.
- **Generic KML**: any `LineString` placemarks; optional `oneway` and
  `maxspeed` extended data or values in the description/name.

## License

MIT — see [LICENSE](LICENSE).
