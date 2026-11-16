"""A picture of the planned hike for the Obsidian plan note (SVG: plain text, so vault_write can store it).

Top left: a sketch of the route from its coordinates, each day in its own colour, with the start, the finish,
the night spots and the main peaks. Top right: one card per day with the climb, the drop, the walking time,
the steep km and whether the day is within the rules. Bottom: the elevation profile, coloured by steepness.
Every number comes from plast-trail (get_route_profile and validate_hike_plan); nothing is computed here
except the drawing itself. No plugin and no internet needed: Obsidian shows an SVG file as an image.
"""

from __future__ import annotations

import math
import re
from html import escape
from typing import Any

W, H = 980, 700
DAY_COLOURS = ["#1f6feb", "#d97706", "#7c3aed", "#059669", "#db2777", "#0891b2", "#65a30d"]
INK, MUTED, GRID, PANEL = "#1f2328", "#57606a", "#eaeef2", "#f6f8fa"
GENTLE, STEEP, VERY_STEEP = "#16a34a", "#ea580c", "#dc2626"
FONT = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"


class _Svg:
    def __init__(self) -> None:
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
                      f'font-family="{FONT}">', f'<rect width="{W}" height="{H}" rx="14" fill="#ffffff"/>']

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x: float, y: float, s: str, size: int = 13, colour: str = INK, anchor: str = "start",
             weight: str = "normal") -> None:
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{colour}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{escape(s)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, colour: str, width: float = 3, extra: str = "") -> None:
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" '
                 f'stroke-width="{width}" stroke-linecap="round" {extra}/>')

    def done(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def _grade_colour(a: list[float], b: list[float], steep_grade: float) -> str:
    grade = abs(b[3] - a[3]) / max(1.0, (b[0] - a[0]) * 1000)
    return GENTLE if grade < 0.08 else STEEP if grade < steep_grade else VERY_STEEP


def _main_peaks(landmarks: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    peaks = [x for x in landmarks if x["kind"] == "peak" and x["name"] and x["elevation_m"]]
    return sorted(sorted(peaks, key=lambda x: -x["elevation_m"])[:n], key=lambda x: x["km"])


def _day_verdict(day: int, validation: dict[str, Any]) -> tuple[str, str]:
    """From the rules check of that day: errors / warnings / within the rules."""
    found = [x for x in validation.get("violations", []) if x.get("day") == day]
    errors = sum(1 for x in found if x["severity"] == "error")
    if errors:
        return f"✗ порушень норм: {errors}", VERY_STEEP
    if found:
        return f"⚠ попереджень: {len(found)} (див. нижче)", STEEP
    return "✓ у межах норм" + (" (чернетка)" if "DRAFT" in (validation.get("rules_status") or "") else ""), GENTLE


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def route_svg(profile: dict[str, Any], validation: dict[str, Any], start_date: str | None = None,
              wished: list[dict[str, Any]] | None = None) -> str:
    """The whole picture as SVG text. `profile` = get_route_profile result; `validation` = validate_hike_plan
    result; `wished` = the places the leader named ({name, lat, lon, elevation_m}, from resolve_places), drawn
    as stars, with the distance from the route when the route does not pass them."""
    wished = wished or []
    s = _Svg()
    days = profile["days"]
    points = [p for d in days for p in d["points"]]
    total = profile["length_km"]
    title = f"{profile['route_name']} · {total:g} км · днів: {len(days)}" + (f" · старт {start_date}" if start_date else "")
    s.text(24, 36, title[:90], 20, weight="600")
    s.text(24, 58, "Схема маршруту (північ угорі) і профіль висот: колір лінії профілю — крутизна", 13, MUTED)
    peaks = _main_peaks(profile["landmarks"])

    # ---------------------------------------------------------------- sketch map
    mx, my, mw, mh = 24, 74, 430, 360
    s.add(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" fill="{PANEL}" stroke="#d0d7de"/>')
    lat0 = sum(p[1] for p in points) / len(points)
    scale_x = math.cos(math.radians(lat0))
    xs = [p[2] * scale_x for p in points] + [w["lon"] * scale_x for w in wished]
    ys = [-p[1] for p in points] + [-w["lat"] for w in wished]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-4)
    k = (min(mw, mh) - 70) / span
    ox = mx + (mw - (max(xs) - min(xs)) * k) / 2 - min(xs) * k
    oy = my + (mh - (max(ys) - min(ys)) * k) / 2 - min(ys) * k

    def xy(lat: float, lon: float) -> tuple[float, float]:
        return ox + lon * scale_x * k, oy - lat * k

    for i, d in enumerate(days):
        pts = d["points"]
        for a, b in zip(pts, pts[1:]):
            s.line(*xy(a[1], a[2]), *xy(b[1], b[2]), DAY_COLOURS[i % len(DAY_COLOURS)], 4)
    boxes: list[tuple[float, float, float, float]] = []

    def label(x: float, y: float, text: str, size: int = 11, weight: str = "normal", colour: str = INK) -> None:
        """Right of the point, else left, below or above; never over another label or outside the panel."""
        w, h = len(text) * size * 0.56, size + 2
        for x0, y0 in ((x + 10, y - h / 2), (x - 10 - w, y - h / 2), (x - w / 2, y + 8), (x - w / 2, y - 8 - h)):
            box = (x0, y0, x0 + w, y0 + h)
            inside = mx + 4 <= box[0] and box[2] <= mx + mw - 4 and my + 4 <= box[1] and box[3] <= my + mh - 4
            if inside and not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                boxes.append(box)
                s.text(x0, y0 + h - 3, text, size, colour, "start", weight)
                return

    for p in peaks:
        x, y = xy(p["lat"], p["lon"])
        s.add(f'<path d="M{x:.1f},{y - 8:.1f} l7,12 h-14 z" fill="{MUTED}"/>')
        label(x, y, f"{p['name']} {p['elevation_m']}")
    for lake in [x for x in profile["landmarks"] if x["kind"] == "lake"][:3]:
        x, y = xy(lake["lat"], lake["lon"])
        s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#0ea5e9"/>')
    for n in [x for x in profile["landmarks"] if x["night"]]:
        x, y = xy(n["lat"], n["lon"])
        s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#ffffff" stroke="{INK}" stroke-width="2.5"/>')
        label(x, y, f"ночівля {n['night']}", 11, "600")
    ends = {x["kind"]: x for x in profile["landmarks"] if x["kind"] in ("start", "finish")}
    for kind, colour, word in (("start", GENTLE, "старт"), ("finish", VERY_STEEP, "фініш")):
        if kind in ends:
            e = ends[kind]
            x, y = xy(e["lat"], e["lon"])
            s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colour}"/>')
            label(x, y, f"{word}: {e['name']}" if e["name"] else word, 12, "600")
    for wp in wished:   # the places the leader named: a star, and a dashed line when the route misses them
        x, y = xy(wp["lat"], wp["lon"])
        near = min(points, key=lambda p: _distance_km(wp["lat"], wp["lon"], p[1], p[2]))
        gap = _distance_km(wp["lat"], wp["lon"], near[1], near[2])
        if gap > 0.3:
            s.line(*xy(near[1], near[2]), x, y, "#b45309", 2, 'stroke-dasharray="5 4"')
        star = " ".join(f"{x + r * math.sin(a * math.pi / 5):.1f},{y - r * math.cos(a * math.pi / 5):.1f}"
                        for a, r in ((i, 9 if i % 2 == 0 else 4) for i in range(10)))
        s.add(f'<polygon points="{star}" fill="#f59e0b" stroke="#b45309" stroke-width="1"/>')
        height = f" {wp['elevation_m']}" if wp.get("elevation_m") else ""
        label(x, y, f"{wp['name']}{height}" + (f" · {gap:.1f} км від маршруту" if gap > 0.3 else ""), 11, "600", "#b45309")
    bar_km = 1 if span * k / scale_x < 300 else 2
    bar = bar_km / 111.0 * k
    s.line(mx + 16, my + mh - 16, mx + 16 + bar, my + mh - 16, INK, 3)
    s.text(mx + 16, my + mh - 22, f"{bar_km} км", 11)
    s.add(f'<path d="M{mx + mw - 24},{my + 16} l7,16 l-7,-5 l-7,5 z" fill="{INK}"/>')
    s.text(mx + mw - 24, my + 46, "Пн", 11, anchor="middle")

    # ---------------------------------------------------------------- day cards
    vdays = {d["day"]: d for d in validation.get("days", [])}
    cx, cy, cw = 476, 74, 480
    compact = len(days) > 3
    card_h = 50 if compact else 106
    for i, d in enumerate(days):
        y = cy + i * (card_h + (6 if compact else 12))
        colour = DAY_COLOURS[i % len(DAY_COLOURS)]
        vd = vdays.get(d["day"], {})
        verdict, verdict_colour = _day_verdict(d["day"], validation)
        hours = f"≈ {vd['walking_hours']} год ходу" if vd.get("walking_hours") is not None else ""
        times = f", прихід {vd['est_arrival']} (захід {vd['sunset']})" if vd.get("est_arrival") and vd.get("sunset") else ""
        s.add(f'<rect x="{cx}" y="{y}" width="{cw}" height="{card_h}" rx="10" fill="#ffffff" stroke="{colour}" stroke-width="2"/>')
        s.add(f'<rect x="{cx}" y="{y}" width="8" height="{card_h}" rx="4" fill="{colour}"/>')
        if compact:
            s.text(cx + 20, y + 21, f"День {d['day']}: {d['km']:g} км · ↗ +{d['ascent_m']} м · ↘ −{d['descent_m']} м · {hours}",
                   13, colour, weight="700")
            s.text(cx + 20, y + 40, f"дуже круто: {d['steep_km']:g} км · {verdict}", 12, verdict_colour)
        else:
            s.text(cx + 22, y + 28, f"День {d['day']}: {d['km']:g} км", 17, colour, weight="700")
            s.text(cx + 22, y + 52, f"↗ +{d['ascent_m']} м   ↘ −{d['descent_m']} м   найвища точка {d['max_elevation_m']} м", 14)
            s.text(cx + 22, y + 74, f"{hours}{times}", 13)
            s.text(cx + 22, y + 96, f"дуже круто (> {profile['steep_grade']:.0%}): {d['steep_km']:g} км   ·   {verdict}",
                   13, verdict_colour)
    notes_y = cy + len(days) * (card_h + (6 if compact else 12)) + 14
    if notes_y < my + mh:
        s.text(cx, notes_y, "Числа — з plast-trail: профіль SRTM 30 м, час — правило Ленгмюра.", 11, MUTED)

    # ---------------------------------------------------------------- elevation profile
    px, py, pw, ph = 70, 476, 880, 190
    kmax = max(p[0] for p in points) or total
    emin = min(p[3] for p in points) // 100 * 100 - 100
    emax = (max(p[3] for p in points) // 100 + 1) * 100 + 100

    def X(km: float) -> float:
        return px + km / kmax * pw

    def Y(e: float) -> float:
        return py + ph - (e - emin) / (emax - emin) * ph

    step = 200 if emax - emin <= 1600 else 400
    for e in range(int(emin), int(emax) + 1, step):
        s.line(px, Y(e), px + pw, Y(e), GRID, 1)
        s.text(px - 8, Y(e) + 4, f"{e}", 11, MUTED, "end")
    km_step = 5 if kmax <= 50 else 10
    for km in range(0, int(kmax) + 1, km_step):
        s.text(X(km), py + ph + 18, f"{km} км", 11, MUTED, "middle")
    for i, d in enumerate(days):
        pts = d["points"]
        colour = DAY_COLOURS[i % len(DAY_COLOURS)]
        poly = " ".join(f"{X(p[0]):.1f},{Y(p[3]):.1f}" for p in pts)
        s.add(f'<polygon points="{X(pts[0][0]):.1f},{py + ph} {poly} {X(pts[-1][0]):.1f},{py + ph}" '
              f'fill="{colour}" fill-opacity="0.10"/>')
        s.text((X(d["start_km"]) + X(d["end_km"])) / 2, py + ph - 8, f"день {d['day']}", 12, colour, "middle", "700")
        for a, b in zip(pts, pts[1:]):
            if b[0] > a[0]:
                s.line(X(a[0]), Y(a[3]), X(b[0]), Y(b[3]), _grade_colour(a, b, profile["steep_grade"]), 3)
    for n in [x for x in profile["landmarks"] if x["night"]]:
        x = X(n["km"])
        s.line(x, py, x, py + ph, INK, 1, 'stroke-dasharray="4 4"')
        s.text(x, py - 6, f"ночівля {n['night']}" + (f", {n['elevation_m']} м" if n["elevation_m"] else ""),
               12, anchor="middle", weight="600")
    last_x = -100.0
    for p in peaks:
        x = X(p["km"])
        if x - last_x < 70:
            continue
        last_x = x
        s.add(f'<circle cx="{x:.1f}" cy="{Y(p["elevation_m"]):.1f}" r="3.5" fill="{INK}"/>')
        s.text(x + 6, Y(p["elevation_m"]) - 6, f"{p['name']} {p['elevation_m']}", 11)
    lx, ly = px + pw - 380, py - 26
    for i, (colour, words) in enumerate(((GENTLE, "до 8 %"), (STEEP, f"8–{profile['steep_grade']:.0%}"),
                                         (VERY_STEEP, f"понад {profile['steep_grade']:.0%}: дуже круто"))):
        s.add(f'<rect x="{lx + i * 100}" y="{ly - 8}" width="14" height="5" fill="{colour}"/>')
        s.text(lx + i * 100 + 18, ly - 3, words, 11, MUTED)
    return s.done()


# ------------------------------------------------------------------ real maps: Mapy.com links and a GPX file
MAPY_ROUTE = "https://mapy.com/fnc/v1/route"
MAX_VIA = 13   # Mapy.com takes at most 15 waypoints; 13 between start and end keep its path on ours


def _dedupe(points: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for p in points:
        if not out or (round(p[1], 5), round(p[2], 5)) != (round(out[-1][1], 5), round(out[-1][2], 5)):
            out.append(p)
    return out


def day_track(profile: dict[str, Any], day: dict[str, Any]) -> list[list[float]]:
    """The day exactly: from its start point (the start or a night spot) to its end, without the neighbouring
    points the drawing adds so the coloured lines join."""
    ends = {round(x["km"], 3): x for x in profile["landmarks"] if x["kind"] in ("start", "finish") or x["night"]}
    inside = [p for p in day["points"] if day["start_km"] - 1e-6 <= p[0] <= day["end_km"] + 1e-6]
    ele = inside[0][3] if inside else 0

    def at(km: float) -> list[list[float]]:
        x = ends.get(round(km, 3))
        return [[x["km"], x["lat"], x["lon"], x["elevation_m"] if x["elevation_m"] is not None else ele]] if x else []

    return _dedupe(at(day["start_km"]) + inside + at(day["end_km"]))


def route_track(profile: dict[str, Any]) -> list[list[float]]:
    return _dedupe([p for d in profile["days"] for p in day_track(profile, d)])


def mapy_link(points: list[list[float]]) -> str:
    """A hiking route on Mapy.com's outdoor map through our points ([km, lat, lon, ele]): it opens with the
    topographic map, the route and an interactive elevation profile (climb and drop), in a browser or the app.
    Mapy.com draws its own path between the points, so a few intermediate points keep it on our trail."""
    pts = _dedupe(points)
    if len(pts) < 2:
        return ""
    inner = pts[1:-1]
    step = max(1, math.ceil(len(inner) / MAX_VIA))
    via = inner[step // 2::step][:MAX_VIA]
    fmt = lambda p: f"{p[2]:.5f},{p[1]:.5f}"   # noqa: E731  Mapy.com wants lon,lat
    url = f"{MAPY_ROUTE}?mapset=outdoor&routeType=foot_hiking&start={fmt(pts[0])}&end={fmt(pts[-1])}"
    return url + (f"&waypoints={';'.join(fmt(p) for p in via)}" if via else "")


def side_trip_link(points: list[list[float]], place: dict[str, Any]) -> tuple[str, float]:
    """From the route's point nearest to a wished place to the place itself, on foot along trails."""
    near = min(points, key=lambda p: _distance_km(place["lat"], place["lon"], p[1], p[2]))
    gap = _distance_km(place["lat"], place["lon"], near[1], near[2])
    return (f"{MAPY_ROUTE}?mapset=outdoor&routeType=foot_hiking&start={near[2]:.5f},{near[1]:.5f}"
            f"&end={place['lon']:.5f},{place['lat']:.5f}"), gap


def route_gpx(profile: dict[str, Any], wished: list[dict[str, Any]] | None = None, title: str = "") -> str:
    """GPX 1.1: one track per day with heights, waypoints for the start, night spots, finish, main peaks and the
    wished places. Opens in Mapy, OsmAnd, Organic Maps, gpx.studio, and in Obsidian map plugins."""
    def wpt(lat: float, lon: float, ele: Any, name: str, sym: str) -> str:
        e = f"<ele>{ele}</ele>" if ele is not None else ""
        return f'  <wpt lat="{lat:.6f}" lon="{lon:.6f}">{e}<name>{escape(name)}</name><sym>{sym}</sym></wpt>'

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="plast-hike-planner" xmlns="http://www.topografix.com/GPX/1/1">',
             f"  <metadata><name>{escape(title or profile['route_name'])}</name>"
             f"<desc>{escape(profile['source'])}</desc></metadata>"]
    for lm in profile["landmarks"]:
        if lm["kind"] in ("start", "finish") or lm["night"]:
            name = f"Ночівля {lm['night']}" if lm["night"] else ("Старт" if lm["kind"] == "start" else "Фініш")
            lines.append(wpt(lm["lat"], lm["lon"], lm["elevation_m"], f"{name}: {lm['name']}".rstrip(": "),
                             "Campground" if lm["night"] else "Flag"))
    for p in _main_peaks(profile["landmarks"]):
        lines.append(wpt(p["lat"], p["lon"], p["elevation_m"], p["name"], "Summit"))
    for w in wished or []:
        lines.append(wpt(w["lat"], w["lon"], w.get("elevation_m"), f"Бажане місце: {w['name']}", "Scenic Area"))
    for d in profile["days"]:
        lines.append(f"  <trk><name>День {d['day']}: {d['km']:g} км, +{d['ascent_m']} / −{d['descent_m']} м</name><trkseg>")
        lines += [f'    <trkpt lat="{p[1]:.6f}" lon="{p[2]:.6f}"><ele>{p[3]:g}</ele></trkpt>' for p in day_track(profile, d)]
        lines.append("  </trkseg></trk>")
    lines.append("</gpx>")
    return "\n".join(lines)


# ------------------------------------------------------------------ an interactive map inside Obsidian
# Base maps that load without a key (checked 15 Nov 2026). The plugin's own default (CARTO) now shows only
# "API KEY REQUIRED", so it is switched off (osmLayer: false). OpenTopoMap has the subdomains a, b, c only
# (the plugin's default a-d left every 4th tile empty).
TILE_LAYERS = ["https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png|Топокарта",
               "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}|Супутник"]
TILE_SUBDOMAINS = "a,b,c"
MAP_CREDITS = ("Карта: © OpenStreetMap contributors, SRTM; стиль © OpenTopoMap (CC-BY-SA). "
               "Супутник: © Esri, Maxar, Earthstar Geographics.")


def _plain(text: str) -> str:
    """Marker text for the leaflet block. The plugin parses the block as YAML, then splits a marker at commas:
    a colon («Старт: Бистрець») broke the whole map (YAML error, then a fallback that cut https: off the tile
    URL), a # starts a comment, a comma would shift the fields."""
    return re.sub(r"\s+", " ", re.sub(r"[,:#\n\r\[\]|\"'{}]", " ", text)).strip()


def fit_zoom(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
             width_px: int = 850, height_px: int = 460) -> int:
    """The largest web-map zoom at which the box fits a map of about this size (Obsidian's note column, the
    block is 500 px high), with some margin. Zoom z shows 256 * 2^z pixels for 360 degrees of longitude."""
    def mercator_y(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    lon_span = max(lon_max - lon_min, 1e-4) / 360
    lat_span = max(mercator_y(lat_max) - mercator_y(lat_min), 1e-6) / (2 * math.pi)
    z = min(math.log2(width_px / 256 / lon_span), math.log2(height_px / 256 / lat_span)) - 0.3
    return max(9, min(15, math.floor(z)))


def leaflet_block(profile: dict[str, Any], gpx_path: str, wished: list[dict[str, Any]] | None = None,
                  map_id: str = "plast-plan") -> str:
    """A `leaflet` code block for the Obsidian Leaflet plugin: the plan's GPX on a topographic map, with
    the start, the night spots, the finish and the wished places as markers. Without the plugin Obsidian
    would show this as plain text, so the agent adds it only when the plugin is installed."""
    pts = route_track(profile)
    lats = [p[1] for p in pts] + [w["lat"] for w in wished or []]
    lons = [p[2] for p in pts] + [w["lon"] for w in wished or []]
    zoom = fit_zoom(min(lats), max(lats), min(lons), max(lons))
    lines = ["```leaflet", f"id: {map_id}", "height: 500px", f"lat: {(min(lats) + max(lats)) / 2:.5f}",
             f"long: {(min(lons) + max(lons)) / 2:.5f}", f"defaultZoom: {zoom}", "minZoom: 7", "maxZoom: 17",
             # no zoomFeatures: it fitted the view before the GPX was loaded and left a grey map ("0 mi")
             "unit: km", "osmLayer: false", f'tileSubdomains: "{TILE_SUBDOMAINS}"', "tileServer:",
             *[f'  - "{layer}"' for layer in TILE_LAYERS], f"gpx: [[{gpx_path}]]",
             'gpxColor: "#ff1f5a"', "marker:"]   # a YAML list: repeated `marker:` keys are invalid YAML
    for x in profile["landmarks"]:
        if x["kind"] in ("start", "finish") or x["night"]:
            label = f"Ночівля {x['night']}" if x["night"] else ("Старт" if x["kind"] == "start" else "Фініш")
            name = f"{label} — {x['name']}" if x["name"] else label
            lines.append(f"  - default,{x['lat']:.5f},{x['lon']:.5f},,{_plain(name)}")
    for w in wished or []:
        lines.append(f"  - default,{w['lat']:.5f},{w['lon']:.5f},,{_plain('Бажане місце — ' + w['name'])}")
    lines.append("```")
    return "\n".join(lines)
