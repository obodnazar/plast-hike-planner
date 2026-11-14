"""Build data/routes.json: marked hiking routes of the Ukrainian Carpathians.

Sources (both public, no API key):
  * OpenStreetMap via the Overpass API: hiking route relations + points of
    interest (springs, shelters, huts, campsites, peaks, lakes). Licence ODbL,
    "© OpenStreetMap contributors".
  * OpenTopoData public API, SRTM 30 m dataset (public domain): height of every
    profile point. Limits: 100 points per request, 1 request per second.

Run once (needs internet; about 2-5 minutes):
    uv run python scripts/build_dataset.py

Raw downloads are cached in data/raw/ (git-ignored), so a re-run is offline and
deterministic. The server itself never touches the network for route data.

Steps per route:
  1. chain the relation's ways into one line (ways come unordered and reversed);
  2. resample the line every 250 m and look up elevation for each point;
  3. compute length, ascent/descent (10 m hysteresis against DEM noise), min/max height;
  4. attach POIs lying within 300 m of the line as waypoints, at their km position;
  5. assign a mountain region by the nearest region centre.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "routes.json"

BBOX = (47.95, 23.9, 48.75, 24.9)  # south, west, north, east
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
ELEVATION_URL = "https://api.opentopodata.org/v1/srtm30m"
USER_AGENT = "plast-hike-planner-student-project/0.1"

STEP_KM = 0.25          # profile resolution
JOIN_TOLERANCE_KM = 0.05  # two way ends closer than this count as connected
MIN_COVERAGE = 0.85     # longest chain must cover this share of member length
POI_RADIUS_KM = 0.30
LAKE_RADIUS_KM = 0.50
ASCENT_HYSTERESIS_M = 10
MIN_LEN_KM, MAX_LEN_KM = 3.0, 90.0

# Rough region centres (lat, lon). Assignment is "nearest centre": approximate by design.
REGIONS = {
    "chornohora": (48.12, 24.55),
    "svydovets": (48.25, 24.20),
    "gorgany": (48.50, 24.20),
    "marmarosy": (47.98, 24.25),
    "yaremche-vorokhta": (48.40, 24.55),
}


# ---------------------------------------------------------------- downloads
def overpass(query: str, cache_name: str) -> dict:
    cache = RAW / cache_name
    if cache.exists():
        return json.loads(cache.read_text())
    delay = 10
    for attempt in range(6):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = httpx.post(url, data={"data": query}, timeout=180,
                           headers={"User-Agent": USER_AGENT})
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                cache.write_text(r.text)
                return r.json()
            print(f"  overpass {url} -> HTTP {r.status_code}, retry in {delay}s")
        except httpx.HTTPError as exc:
            print(f"  overpass {url} -> {type(exc).__name__}, retry in {delay}s")
        time.sleep(delay)
        delay *= 2
    raise SystemExit("Overpass is unavailable; try again later.")


def _key(p: tuple[float, float]) -> str:
    return f"{p[0]:.4f},{p[1]:.4f}"  # rounded to about 11 m


def fetch_elevations(points: list[tuple[float, float]]) -> dict[str, float]:
    """Heights for all points: 100 per request, 1 request per second (public API limit).

    The cache is saved after every batch, so an interrupted run resumes where it stopped.
    """
    cache_file = RAW / "elevation_cache.json"
    cache: dict[str, float] = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    missing = sorted({_key(p) for p in points} - cache.keys())
    print(f"elevation: {len(missing)} points to fetch ({len(cache)} cached)")
    for i in range(0, len(missing), 100):
        batch = missing[i : i + 100]
        delay = 2
        while True:
            r = httpx.get(ELEVATION_URL, params={"locations": "|".join(batch)},
                          timeout=60, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200 and r.json().get("status") == "OK":
                break
            print(f"  elevation HTTP {r.status_code}, retry in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        for k, res in zip(batch, r.json()["results"], strict=True):
            if res.get("elevation") is not None:  # SRTM has rare voids; skip them
                cache[k] = res["elevation"]
        cache_file.write_text(json.dumps(cache))
        print(f"  elevation {min(i + 100, len(missing))}/{len(missing)}")
        time.sleep(1.1)
    return cache


def heights_for(points: list[tuple[float, float]], cache: dict[str, float]) -> list[float]:
    """Look up each point; fill a rare void with the previous known height."""
    out, last = [], None
    for p in points:
        h = cache.get(_key(p), last)
        out.append(h)
        last = h if h is not None else last
    first = next((h for h in out if h is not None), 0.0)
    return [first if h is None else h for h in out]


# ---------------------------------------------------------------- geometry
def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def line_length(line: list[tuple[float, float]]) -> float:
    return sum(haversine_km(line[i], line[i + 1]) for i in range(len(line) - 1))


def chain_ways(ways: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Greedy chaining: grow a line from each start way, keep the longest result."""
    best: list[tuple[float, float]] = []
    for start in range(len(ways)):
        line = list(ways[start])
        unused = set(range(len(ways))) - {start}
        grew = True
        while grew and unused:
            grew = False
            for i in sorted(unused):
                w = ways[i]
                if haversine_km(line[-1], w[0]) <= JOIN_TOLERANCE_KM:
                    line += w[1:]
                elif haversine_km(line[-1], w[-1]) <= JOIN_TOLERANCE_KM:
                    line += w[::-1][1:]
                elif haversine_km(line[0], w[-1]) <= JOIN_TOLERANCE_KM:
                    line = w[:-1] + line
                elif haversine_km(line[0], w[0]) <= JOIN_TOLERANCE_KM:
                    line = w[::-1][:-1] + line
                else:
                    continue
                unused.discard(i)
                grew = True
        if line_length(line) > line_length(best):
            best = line
    return best


def resample(line: list[tuple[float, float]], step: float) -> list[tuple[float, float, float]]:
    """Points (km, lat, lon) every `step` km along the line, plus the last point."""
    out = [(0.0, *line[0])]
    done, next_mark = 0.0, step
    for a, b in zip(line, line[1:]):
        seg = haversine_km(a, b)
        while seg > 0 and done + seg >= next_mark:
            t = (next_mark - done) / seg
            out.append((next_mark, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            next_mark += step
        done += seg
    if done - out[-1][0] > 0.01:
        out.append((done, *line[-1]))
    return out


def ascent_descent(heights: list[float]) -> tuple[int, int]:
    """Sum climbs/drops, ignoring wiggles smaller than the hysteresis (DEM noise)."""
    up = down = 0.0
    ref = heights[0]
    for h in heights[1:]:
        if h - ref >= ASCENT_HYSTERESIS_M:
            up += h - ref
            ref = h
        elif ref - h >= ASCENT_HYSTERESIS_M:
            down += ref - h
            ref = h
    return round(up), round(down)


# ---------------------------------------------------------------- POIs
def poi_kind(tags: dict) -> str | None:
    if tags.get("natural") == "spring" or tags.get("amenity") == "drinking_water":
        return "water"
    if tags.get("tourism") in ("alpine_hut", "wilderness_hut"):
        return "hut"
    if tags.get("amenity") == "shelter":
        return "shelter"
    if tags.get("tourism") == "camp_site":
        return "campsite"
    if tags.get("natural") == "peak":
        return "peak"
    if tags.get("natural") == "water":
        return "lake"
    return None


def load_pois() -> list[dict]:
    s, w, n, e = BBOX
    b = f"({s},{w},{n},{e})"
    q = f"""[out:json][timeout:180];
(
  node["natural"="spring"]{b};
  node["amenity"="drinking_water"]{b};
  node["amenity"="shelter"]{b};
  nwr["tourism"~"^(alpine_hut|wilderness_hut|camp_site)$"]{b};
  node["natural"="peak"]["name"]{b};
  way["natural"="water"]["name"]{b};
  relation["natural"="water"]["name"]{b};
);
out center tags;"""
    data = overpass(q, "pois.json")
    pois = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        kind = poi_kind(tags)
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if kind is None or lat is None:
            continue
        ele = tags.get("ele", "")
        pois.append({
            "kind": kind,
            "name": tags.get("name:uk") or tags.get("name") or "",
            "lat": lat,
            "lon": lon,
            "osm": f"{el['type']}/{el['id']}",
            "osm_ele": float(ele) if ele.replace(".", "", 1).isdigit() else None,
        })
    return pois


def nearest_on_profile(profile, lat, lon):
    best = min(profile, key=lambda p: haversine_km((p[1], p[2]), (lat, lon)))
    return best, haversine_km((best[1], best[2]), (lat, lon))


def load_places() -> list[dict]:
    """Named settlements and localities, used to label route ends and orient routes by name."""
    s, w, n, e = BBOX
    q = (f'[out:json][timeout:120];node["place"~"^(town|village|hamlet|isolated_dwelling|locality)$"]["name"]'
         f'({s},{w},{n},{e});out body;')
    data = overpass(q, "places.json")
    return [{"name": el["tags"].get("name:uk") or el["tags"]["name"], "place": el["tags"]["place"],
             "lat": el["lat"], "lon": el["lon"]} for el in data["elements"]]


PREFIXES = ("с.", "м.", "г.", "смт", "пол.", "пер.", "прис.", "хр.", "нсб", "оз.", "вдсп.", "кпп")


def name_parts(route_name: str) -> list[str]:
    """'с. Лазещина - г. Петрос - с. Ясіня' -> ['лазещина', 'петрос', 'ясіня']."""
    import re
    parts = []
    for raw in re.split(r"\s[-–―—]\s|[-–―—]", route_name):
        t = raw.strip().lower()
        for pre in PREFIXES:
            if t.startswith(pre):
                t = t[len(pre):].strip()
        if t:
            parts.append(t)
    return parts


def end_label(lat: float, lon: float, places: list[dict], pois: list[dict]) -> str:
    """Best human label for a route end, in order of preference:
    a named peak/hut/shelter/lake within 600 m, a settlement within 3 km, a locality within 1.5 km."""
    def nearest(items, radius_km):
        best = min(items, key=lambda x: haversine_km((lat, lon), (x["lat"], x["lon"])), default=None)
        if best and haversine_km((lat, lon), (best["lat"], best["lon"])) <= radius_km:
            return best["name"]
        return ""

    named_pois = [p for p in pois if p["name"] and p["kind"] in ("peak", "hut", "shelter", "lake")]
    settlements = [p for p in places if p["place"] in ("town", "village", "hamlet")]
    localities = [p for p in places if p["place"] in ("locality", "isolated_dwelling")]
    return nearest(named_pois, 0.6) or nearest(settlements, 3.0) or nearest(localities, 1.5)


def matches(part: str, label: str) -> bool:
    return bool(part) and bool(label) and part[:5] in label.lower()


def should_reverse(route_name: str, start_label: str, end_label_: str,
                   start_h: float, end_h: float, loop: bool) -> tuple[bool, str]:
    """Orient 'A - B' to start at A; otherwise start from the lower end (approach from the valley)."""
    parts = name_parts(route_name)
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        if matches(first, end_label_) and not matches(first, start_label):
            return True, "name"
        if matches(last, start_label) and not matches(last, end_label_):
            return True, "name"
        if matches(first, start_label) or matches(last, end_label_):
            return False, "name"
    if loop:
        return False, "loop"
    return (start_h > end_h + 50), "lower end first"


# ---------------------------------------------------------------- main
def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    s, w, n, e = BBOX
    routes_raw = overpass(
        f'[out:json][timeout:180];relation["route"="hiking"]({s},{w},{n},{e});out geom;',
        "routes_geom.json",
    )
    osm_time = routes_raw.get("osm3s", {}).get("timestamp_osm_base", "")
    pois = load_pois()
    places = load_places()
    print(f"relations: {len(routes_raw['elements'])}, pois: {len(pois)}, places: {len(places)}")

    skipped: dict[str, int] = {}
    candidates = []  # (relation, name, tags, resampled points, length)
    for rel in routes_raw["elements"]:
        tags = rel.get("tags", {})
        name = tags.get("name:uk") or tags.get("name")
        if not name:
            skipped["no name"] = skipped.get("no name", 0) + 1
            continue
        ways = [
            [(p["lat"], p["lon"]) for p in m["geometry"]]
            for m in rel.get("members", [])
            if m.get("type") == "way" and m.get("geometry")
            and m.get("role", "") in ("", "main", "forward", "backward")
        ]
        if not ways:
            skipped["no geometry"] = skipped.get("no geometry", 0) + 1
            continue
        line = chain_ways(ways)
        total = sum(line_length(wy) for wy in ways)
        length = line_length(line)
        if total == 0 or length / total < MIN_COVERAGE:
            skipped["broken chain"] = skipped.get("broken chain", 0) + 1
            continue
        if not (MIN_LEN_KM <= length <= MAX_LEN_KM):
            skipped["length out of range"] = skipped.get("length out of range", 0) + 1
            continue
        candidates.append((rel, name, tags, resample(line, STEP_KM), length))

    cache = fetch_elevations([(p[1], p[2]) for c in candidates for p in c[3]])

    built = []
    for rel, name, tags, pts, length in candidates:
        heights = heights_for([(p[1], p[2]) for p in pts], cache)
        loop = haversine_km((pts[0][1], pts[0][2]), (pts[-1][1], pts[-1][2])) < 0.5
        a_label = end_label(pts[0][1], pts[0][2], places, pois)
        b_label = end_label(pts[-1][1], pts[-1][2], places, pois)
        reverse, orientation = should_reverse(name, a_label, b_label, heights[0], heights[-1], loop)
        if reverse:
            total_km = pts[-1][0]
            pts = [(total_km - p[0], p[1], p[2]) for p in reversed(pts)]
            heights = list(reversed(heights))
            a_label, b_label = b_label, a_label
        profile = [(round(p[0], 2), round(p[1], 5), round(p[2], 5), round(h)) for p, h in zip(pts, heights)]
        up, down = ascent_descent([p[3] for p in profile])

        lats = [p[1] for p in profile]
        lons = [p[2] for p in profile]
        box = (min(lats) - 0.01, min(lons) - 0.01, max(lats) + 0.01, max(lons) + 0.01)
        found = []
        for poi in pois:
            if not (box[0] <= poi["lat"] <= box[2] and box[1] <= poi["lon"] <= box[3]):
                continue
            near, dist = nearest_on_profile(profile, poi["lat"], poi["lon"])
            radius = LAKE_RADIUS_KM if poi["kind"] == "lake" else POI_RADIUS_KM
            if dist <= radius:
                found.append({
                    "kind": poi["kind"],
                    "name": poi["name"],
                    "km": near[0],
                    "elevation_m": round(poi["osm_ele"]) if poi["osm_ele"] else near[3],
                    "off_route_m": round(dist * 1000),
                    "lat": round(poi["lat"], 5),
                    "lon": round(poi["lon"], 5),
                    "osm": poi["osm"],
                })
        start = {"kind": "start", "name": a_label or "початок маршруту",
                 "km": 0.0, "elevation_m": profile[0][3], "off_route_m": 0,
                 "lat": profile[0][1], "lon": profile[0][2], "osm": ""}
        finish = {"kind": "finish", "name": b_label or "кінець маршруту", "km": profile[-1][0],
                  "elevation_m": profile[-1][3], "off_route_m": 0,
                  "lat": profile[-1][1], "lon": profile[-1][2], "osm": ""}
        waypoints = [start] + sorted(found, key=lambda x: x["km"]) + [finish]
        for i, wp in enumerate(waypoints):
            wp["id"] = f"w{i}"

        clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)
        region = min(REGIONS, key=lambda r: haversine_km((clat, clon), REGIONS[r]))

        built.append({
            "id": f"r{rel['id']}",
            "osm_relation": rel["id"],
            "name": name,
            "region": region,
            "marking": tags.get("osmc:symbol", ""),
            "loop": loop,
            "orientation_rule": orientation,
            "length_km": round(length, 1),
            "ascent_m": up,
            "descent_m": down,
            "min_elevation_m": min(p[3] for p in profile),
            "max_elevation_m": max(p[3] for p in profile),
            "waypoints": waypoints,
            "profile": profile,
        })

    built.sort(key=lambda r: (r["region"], r["name"]))
    meta = {
        "description": "Marked hiking routes of the Ukrainian Carpathians with elevation profiles and waypoints",
        "sources": {
            "routes_and_pois": "© OpenStreetMap contributors, ODbL 1.0, via Overpass API",
            "elevation": "SRTM 30 m (NASA, public domain) via OpenTopoData public API",
        },
        "osm_snapshot": osm_time,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox_south_west_north_east": BBOX,
        "profile_step_km": STEP_KM,
        "profile_columns": ["km", "lat", "lon", "elevation_m"],
        "ascent_hysteresis_m": ASCENT_HYSTERESIS_M,
        "poi_radius_m": int(POI_RADIUS_KM * 1000),
        "route_count": len(built),
        "skipped": skipped,
    }
    OUT.write_text(json.dumps({"meta": meta, "routes": built}, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(built)} routes; skipped {skipped}")


if __name__ == "__main__":
    main()
