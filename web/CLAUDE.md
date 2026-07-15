# CLAUDE.md — web

## What this is
Dashboard + API layer of Nimbus. Stores the observations and LoRa node
telemetry observatory POSTs, serves them — plus cached external provider
data — to any HTTP client, and hosts the widget dashboard. No ML here —
that's `academy/`. No camera or inference here — that's `observatory/`.

## Where it sits in the pipeline
```
academy  (train → export → ONNX → optimize)
    ↓
observatory  (load → infer → postprocess → observations)
    ↓  POST /api/v1/observations
web  (store → REST API → dashboard)   ← this sub-project
    ↓
browsers / mobile / e-paper / any program  (JSON API)
```

## Core principles
- **API-first.** The SPA is just another API client. Anything a widget renders
  must exist as a JSON endpoint first, so e-paper displays and scripts are
  first-class consumers.
- **`config.yaml` is the single source of truth** — server, storage, stations,
  providers, dashboard layout. Loaded into typed dataclasses by
  `server/config.py`, validated at load time. Relative paths resolve against
  the config file's directory.
- **Provider boundary.** Routes only import `server/providers/registry.py` and
  the `Provider` ABC — never a concrete provider. Same rule as academy's
  ModelAdapter: adding a data source must not touch the routes.
- **Cache-first.** External APIs are only reached through `provider_cache`
  (SQLite-backed, survives reboots). A failed fetch serves the stale payload
  flagged `"stale": true` instead of breaking the widget.
- **A provider doesn't have to be remote.** `astronomy` computes sun/moon
  ephemeris locally with `astral` (no network); the TTL cache just spares
  recomputation. Map tiles (OSM, RainViewer) are the one exception to
  cache-first: the server caches only the frame *index*, tiles load
  browser-side from the CDNs.
- **Some widgets need no server at all.** The `orrery` views (terminator
  map, Earth–Moon, orbit) run on `frontend/src/lib/astro.js` — low-precision
  Meeus formulas (~0.1–0.5°, fine for schematics; moon phase timing ~±2 h)
  plus bundled Natural Earth coastlines (`src/assets/land-rings.json`,
  public domain). The day/night map is equal-area Mollweide, rendered **per
  pixel** (inverse projection → land-mask lookup → twilight shading from
  the real sun vector, same brightness formula as the globe) and **centred
  on the station's longitude** from config.yaml, not on Greenwich; the
  Earth–Moon globe is a draggable, per-pixel rendered 3D view (default
  camera: ecliptic north, so the 23.4° axial tilt is visible). Earth and
  Moon share one rasterizer (`Orrery.svelte`'s `rasterSphere`); the Moon's
  orbital position comes from `moonPlane`/`moonDir` — the real ecliptic
  plane expressed in the same Earth-fixed frame as the sun vector,
  projected through the *same* camera basis as the globe. There is no
  separate "Sun" object drawn — only the per-pixel terminator shading,
  driven by the real sun vector, so there is nothing that can visually
  contradict it. The camera is stored in the inertial frame (dec + RA) and
  converted to Earth-fixed coordinates via sidereal time each frame —
  that's what spins the Earth on its own axis under a fixed camera instead
  of dragging the whole scene when the time slider moves. The orbit view
  carries the year's sky events from `lib/events.js` — meteor showers are
  pinned at their **solar-longitude** peaks (IMO values, so dates adapt per
  year; that's why they belong on the orbit), equinoxes/solstices at λ☉ =
  0/90/180/270, apsides, plus next full/new moon — with an upcoming-events
  countdown panel on the right (click an event to time-travel the orbit to
  it) and month ticks (Earth's true position on the 1st of each month).
  Event instants are solved by Newton iteration on solar longitude /
  lunar elongation (`nextSolarLongitude` / `nextMoonElongation`). Each
  view has a time slider (±24 h; ±12 months on the orbit). Precise
  rise/set times stay in the astral-backed provider; don't duplicate them
  client-side.
  # ponytail: rasterSphere composites each sphere via an offscreen scratch
  # canvas + drawImage, never ctx.putImageData() straight onto the shared
  # canvas — putImageData replaces pixels outright (no alpha blending), so
  # two overlapping sphere bounding boxes would punch a transparent hole
  # through whatever was drawn first. Keep this pattern for any future
  # canvas widget that layers multiple raster sprites.

## Stack
- Python 3 + FastAPI + SQLite (WAL). One uvicorn worker, one process serving
  both `/api/v1` and the built SPA — SQLite is the single writer and the Pi 5
  shares its RAM with observatory.
- Svelte 5 (runes) + Vite + uPlot + Leaflet (~120 KB gzip total, includes the
  bundled coastline data). Node runs at build time on the dev PC only, never
  on the Pi.
- Tests are plain `unittest` (repo convention), backed by FastAPI's TestClient.

## Commands
```bash
# From web/. Install (uv):
uv venv && uv pip install -r requirements.txt

# Tests (no network, temp DBs):
.venv/bin/python -m unittest discover -s tests -v

# Dev: API on :8080, SPA with HMR on :5173 (vite proxies /api and /images)
.venv/bin/uvicorn server.main:app --reload --port 8080
cd frontend && npm install && npm run dev

# Build the SPA (then FastAPI serves frontend/dist/ itself):
cd frontend && npm run build

# Background server (PID-file managed, logs to data/uvicorn.log):
scripts/restart.sh [port]           # stop-if-running, then start
scripts/stop.sh
scripts/build-and-restart.sh [port] # npm run build, then restart.sh
```

## Conventions
- JSON timestamps are ISO 8601 UTC (`...Z`); the DB stores integer epoch
  seconds (`ts_utc`). Naive timestamps are rejected at the API edge.
- Cloud metrics are **[0, 1] fractions** everywhere in nimbus; the UI formats
  percentages. Provider payloads keep their source's native units
  (Open-Meteo: percent).
- Images never travel through the API: observatory writes files into
  `storage.images_dir` and POSTs filenames; responses expose `/images/...`
  URLs.
- Ingest is an upsert on `(station_id, timestamp)` — re-sends after flaky
  comms are idempotent by design. Same rule for node telemetry on
  `(node_id, timestamp)`.
- Unknown `station_id` on ingest → 404: stations are declared in config.yaml,
  fail loudly. **Nodes are the deliberate exception**: LoRa sensors come and
  go, so they auto-register on their first `POST /api/v1/nodes/telemetry`.
- The dashboard is organized in **sections** (observatory-style pages) under
  `dashboard.sections`, each with its own widget grid; a bare
  `dashboard.layout` still parses as one implicit section. Navigation is
  hash-routed in the SPA — no client router library.
- Widgets are enabled/disabled per layout entry (`enabled: false` keeps the
  entry and its props but hides the widget); providers have their own
  `enabled`. The ambient effects layer is toggled by `dashboard.ambient`.
- `/observations/export.csv` streams unbounded ranges through a cursor —
  don't replace it with a fetchall. `/health` probes /proc and /sys directly
  (no psutil); probes return null where the host lacks them (e.g. no thermal
  zone under WSL).
- `TODO(design):` marks deliberately deferred decisions.
- **Gitignore trap (already bitten once):** the root `.gitignore`'s Python
  template has a bare `lib/` rule that matched `frontend/src/lib/` and kept
  those sources out of git until they were lost and had to be reconstructed
  from the dist bundle. The re-include `!web/frontend/src/lib/` now guards
  it — check `git status` shows new files under `src/lib/` before assuming
  they're tracked.

## Adding a widget
1. Only if it needs a new external source: `server/providers/<name>.py`
   implementing `Provider`, one line in `PROVIDERS` (`registry.py`), one block
   under `providers:` in config.yaml.
2. `frontend/src/widgets/<Name>.svelte`, one line in `WIDGETS`
   (`widgets/registry.js`).
3. One entry under `dashboard.layout` in config.yaml.

Widgets showing nimbus's own observations skip step 1 entirely.

## Open / not decided
- Auth: none — LAN deployment assumed. Revisit before exposing beyond it.
- Live updates: widget polling now. A future `GET /api/v1/stream` (SSE) hooks
  into the single write path (the ingest route) without restructuring.
- Server-rendered PNG endpoint for dumb e-paper clients.
- Background provider refresh task (read-through + stale fallback behaves the
  same at household scale).
- Multipart image ingest for remote stations (LoRa can't carry images anyway).
- Rollup/pre-aggregation table if multi-year series queries ever get slow.
- Docker, CI, TypeScript.
- How observatory's C++ side actually POSTs — its `communication/` module is
  itself `TODO(design)`; the v1 observation and node-telemetry schemas in
  `server/models.py` are the contract proposals.
- Node telemetry retention/pruning (rows are tiny; revisit if it ever grows).
- Dark tiles for the rain map (OSM tiles are light-only; a CSS filter would
  also recolor the radar overlay).
