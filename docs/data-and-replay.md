# Data sources, fixtures and offline replay

## Sources

| Data | Source | Licence | How it is used | Network at runtime? |
|---|---|---|---|---|
| Hiking routes, springs, huts, shelters, campsites, peaks, lakes, settlements | OpenStreetMap via the Overpass API | ODbL 1.0, © OpenStreetMap contributors | built once into `data/routes.json` | no |
| Heights of route points | SRTM 30 m (NASA) via the OpenTopoData public API | public domain | built once into `data/routes.json` | no |
| Weather forecast | Open-Meteo forecast API | CC BY 4.0 | `assess_weather_risk` | only in `live` mode (and when recording fixtures) |
| Plast norms and coefficients | 51 public sources (Plast documents, state norms, outdoor guidance), see [research/plast-norms-research.md](../research/plast-norms-research.md) | n/a | `data/rules.yaml` (DRAFT, every value with its source) | no |

## The route dataset

`uv run python scripts/build_dataset.py` (2–5 minutes the first time; then offline from `data/raw/`):

1. Overpass: all `route=hiking` relations in the bbox 47.95–48.75 N, 23.9–24.9 E (`out geom`), points of interest, settlements. Raw responses are cached in `data/raw/` (git-ignored).
2. Each relation's ways are chained into one line (ways come unordered and reversed). Routes are kept only if the longest chain covers ≥ 85 % of the member length and the length is 3–90 km.
3. The line is resampled every 250 m. Heights come from OpenTopoData: 100 points per request, 1 request per second, cached after every batch.
4. Ascent and descent are summed with 10 m hysteresis. Points of interest within 300 m (lakes 500 m) become waypoints `w0..wN` at their km position. A "lake" is `natural=water` with `water=lake` or no subtype; ponds (`water=pond`, names with «став…»), fish/zoo enclosures, rivers and reservoirs are excluded.
5. Orientation: "A – B" starts at A when the route name matches the end labels; otherwise the route starts from the lower end. Each route gets a region by the nearest region centre.

Snapshot in the committed file: OSM `2026-11-14T19:08:36Z`, 121 routes (skipped: 25 out of length range, 10 unnamed, 4 broken chains), 3039 points of interest.

## Weather fixtures and replay

`assess_weather_risk` has three modes, chosen with `WEATHER_MODE` when starting the server:

| Mode | Network | Writes | Use |
|---|---|---|---|
| `replay` (default) | none | nothing | demo, tests, offline |
| `live` | Open-Meteo | nothing | real planning |
| recording (`scripts/record_weather.py`, not the server) | Open-Meteo | `fixtures/open-meteo/<route>_<first-day>.json` | refreshing fixtures; the server refuses `WEATHER_MODE=record`, so the tool itself never writes |

- **What a fixture is.** One genuine Open-Meteo response per route, saved untouched together with the exact request: `{"recorded_at", "request": {"url", "params"}, "response": <body>}`. The request asks for 16 days of daily values at the route's highest point.
- **Fidelity.** Replay loads `response` and passes it through the **same** `parse_forecast()` (with the same completeness checks) and the same risk scoring as a live call. There is no shortcut that returns a prepared answer. If no recorded file covers the requested dates, the tool returns `WEATHER_FIXTURE_MISSING` with the recorded ranges.
- **Coverage now.** 121 fixtures recorded on 2026-11-14 cover 14–29 Nov 2026. Replay deliberately does not compare dates with today: a recorded window can be replayed later (for the demo); live mode only accepts today..today+15.
- **Refresh before the demo** (about 3 minutes, ~121 calls):
  ```bash
  uv run python scripts/record_weather.py                       # all routes
  uv run python scripts/record_weather.py --region chornohora   # or one region
  ```
  Then move the start date in the demo request note into the new window.

## Rate limits respected

| Service | Published limit | What the project does |
|---|---|---|
| Overpass API | shared public server; asks clients to back off | three queries in total, cached; exponential backoff on 502/429; mirror fallback |
| OpenTopoData public API | 100 locations per request, 1 request/s, 1000 requests/day | batches of 100, 1.1 s pause, cache saved after every batch (~54 requests for the whole dataset) |
| Open-Meteo (free, non-commercial) | 600 calls/min, 5000/hour, 10 000/day | ≥ 1 s between calls, 30-minute in-memory cache per route, up to 3 tries with backoff; a full recording is ~121 calls |
| OpenRouter free models | 20 requests/min, 50/day without credits | 3–6 model calls per agent run (one schema retry per call at most); the count is printed at the end of every run |

## Secrets and local-only files

| Path | Why it is git-ignored |
|---|---|
| `.env` | OpenRouter and Obsidian keys (template: `.env.example`) |
| `certs/` | the plugin's certificate, specific to one installation |
| `demo-vault/.obsidian/` | the plugin's `data.json` holds the API key and the TLS private key |
| `demo-vault/Плани/*.md`, `out/` | agent output |
| `data/raw/` | raw downloads, re-creatable |
