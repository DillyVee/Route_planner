# Survey Route Planner

Plans an efficient driving route that covers every road section in a KML file
(MapPlus/Duweis roadway-survey exports or generic KML), then writes a
mobile-friendly **GPX** track and an interactive **HTML map** preview.

Everything lives in one file: `Route_Planner.py`.

## Install

```bash
pip install -r requirements.txt   # requests (OSM data) + PyQt6 (GUI) — both optional
```

## Usage

```bash
python Route_Planner.py                     # GUI
python Route_Planner.py sections.kml       # headless CLI
python Route_Planner.py sections.kml --no-osm --gpx out.gpx --start 40.44,-79.99
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

## Input format

- **MapPlus/Duweis KML**: `CollId`, `RouteName`, `Dir` (`I`/`D`/`B`/`T`/`NB`/`SB`/`EB`/`WB`),
  speed fields in `ExtendedData`.
- **Generic KML**: any `LineString` placemarks; optional `oneway` and
  `maxspeed` extended data or values in the description/name.

## License

MIT — see [LICENSE](LICENSE).
