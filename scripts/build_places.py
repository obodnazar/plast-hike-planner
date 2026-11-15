"""Build data/places.json: the gazetteer of named places for the resolve_places tool.

Reads the OpenStreetMap downloads that scripts/build_dataset.py already cached in data/raw/
(pois.json: peaks, lakes, springs, huts, shelters, campsites; places.json: villages, towns, polonyny,
urochyshcha), keeps the named ones, and gives each a mountain region by the same nearest-centre rule
as the routes. No network.

    uv run python scripts/build_places.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import RAW, REGIONS, haversine_km  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "places.json"
CYRILLIC = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
GENERIC = {"джерело", "криниця", "колиба", "притулок", "хатка", "хата", "кемпінг", "стоянка", "табір", "озеро",
           "вершина", "полонина", "колодязь", "бювет", "навіс", "альтанка", "стоянка туристів"}
ALT_KEYS = ("name", "name:uk", "name:ru", "alt_name", "alt_name:uk", "old_name", "official_name", "loc_name")


def kind_of(tags: dict) -> str | None:
    if tags.get("natural") == "peak":
        return "peak"
    if tags.get("natural") == "water":
        name = (tags.get("name:uk") or tags.get("name") or "").lower()
        return None if "став" in name else "lake"
    if tags.get("natural") == "spring" or tags.get("amenity") == "drinking_water":
        return "spring"
    if tags.get("tourism") in ("alpine_hut", "wilderness_hut"):
        return "hut"
    if tags.get("amenity") == "shelter":
        return "shelter"
    if tags.get("tourism") == "camp_site":
        return "campsite"
    if tags.get("place") in ("village", "hamlet", "isolated_dwelling"):
        return "village"
    if tags.get("place") == "town":
        return "town"
    if tags.get("place") == "locality":
        return "locality"   # polonyny, urochyshcha
    return None


def elevation(tags: dict) -> int | None:
    try:
        return round(float(str(tags.get("ele", "")).replace(",", ".").split()[0]))
    except (ValueError, IndexError):
        return None


def main() -> None:
    elements = []
    for name in ("pois.json", "places.json"):
        path = RAW / name
        if not path.exists():
            sys.exit(f"{path} is missing: run scripts/build_dataset.py first (it downloads and caches OSM data).")
        elements += json.loads(path.read_text())["elements"]

    places: list[dict] = []
    for el in elements:
        tags = el.get("tags") or {}
        main_name = tags.get("name:uk") or tags.get("name") or ""
        kind = kind_of(tags)
        if not kind or not CYRILLIC.search(main_name) or main_name.strip().lower() in GENERIC:
            continue
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        alt = sorted({tags[k].strip() for k in ALT_KEYS if tags.get(k) and CYRILLIC.search(tags[k])} - {main_name.strip()})
        region = min(REGIONS, key=lambda r: haversine_km((lat, lon), REGIONS[r]))
        entry = {"id": f"{el['type'][0]}{el['id']}", "name": main_name.strip(), "alt_names": alt, "kind": kind,
                 "lat": round(lat, 5), "lon": round(lon, 5), "elevation_m": elevation(tags), "region": region}
        twin = next((p for p in places if p["name"] == entry["name"] and p["kind"] == kind
                     and haversine_km((p["lat"], p["lon"]), (lat, lon)) < 1.0), None)
        if twin is None:   # the same name and kind within 1 km is one place (a node and a way of one lake, say)
            places.append(entry)

    places.sort(key=lambda p: (p["name"].lower(), p["kind"]))
    OUT.write_text(json.dumps({
        "description": "Named places of the Ukrainian Carpathians for resolve_places: peaks, lakes, springs, huts, "
                       "shelters, campsites, villages, polonyny",
        "source": "© OpenStreetMap contributors, ODbL 1.0, via Overpass API (the raw cache of build_dataset.py)",
        "region_rule": "nearest region centre, the same rule as data/routes.json (approximate by design)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(places),
        "places": places,
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    kinds: dict[str, int] = {}
    for p in places:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"{OUT.relative_to(ROOT)}: {len(places)} named places {kinds}")


if __name__ == "__main__":
    main()
