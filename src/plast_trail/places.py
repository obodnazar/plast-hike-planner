"""The gazetteer behind resolve_places: a name as people write it -> a named place of the dataset.

data/places.json (built by scripts/build_places.py from OpenStreetMap) holds ~1500 named peaks, lakes,
villages, polonyny, huts. Matching, from strict to loose:
  1. the find_routes rule: every word stem of the written name occurs in the place's name
     («Говерлу» -> Говерла, «озеро Несамовите» -> Несамовите озеро);
  2. otherwise a misspelling: similar letters (difflib), also against the start of a long name
     («піпівана» -> Піп Іван Чорногірський, «бребенеско» -> Бребенескул).
Ties go to the requested region, then to peaks and lakes, places some route passes, the higher peak.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from functools import lru_cache
from typing import Any

from plast_trail.data import DATA_DIR, load_routes
from plast_trail.errors import DATA_UNAVAILABLE, DomainError
from plast_trail.models import PlaceCandidate, PlaceResolution, ResolvePlacesResult
from plast_trail.routes import PLACE_PREFIXES, matched_places, place_stems

PLACES_FILE = DATA_DIR / "places.json"
FUZZY_MIN = 0.8        # below this a written name is "not found" rather than a misspelling
TIE = 0.02             # two misspelling matches closer than this are "ambiguous"
KIND_RANK = {"peak": 0, "lake": 0, "locality": 1, "village": 2, "town": 2, "hut": 3, "shelter": 3,
             "campsite": 4, "spring": 4}


@lru_cache(maxsize=1)
def load_places() -> tuple[dict[str, Any], ...]:
    try:
        data = json.loads(PLACES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DomainError(DATA_UNAVAILABLE, f"Gazetteer not found at {PLACES_FILE}",
                          "Run: uv run python scripts/build_places.py") from exc
    return tuple(data["places"])


def _norm(text: str) -> str:
    t = text.lower().replace("ґ", "г").replace("ё", "е")
    return re.sub(r"[’ʼ`´]", "'", t).strip()


def _bare(text: str) -> str:
    """Without a prefix like «оз.» / «гора», case or spaces: «Піп Іван» -> «піпіван»."""
    t = _norm(text)
    for pre in PLACE_PREFIXES:
        if t.startswith(pre):
            t = t[len(pre):].strip()
    return re.sub(r"[\s\-'\".«»]", "", t)


def _first_word(text: str) -> str:
    t = _norm(text)
    for pre in PLACE_PREFIXES:
        if t.startswith(pre):
            t = t[len(pre):].strip()
    return _bare(re.split(r"[\s\-]", t)[0]) if t else ""


@lru_cache(maxsize=4096)
def routes_through(name: str) -> tuple[str, ...]:
    """Route ids whose names and waypoints contain this place (find_routes' own matching)."""
    return tuple(rid for rid, r in load_routes().items() if matched_places(r, [name]))


def _similarity(query: str, name: str) -> tuple[bool, float]:
    """(stem match as find_routes would see it, letter similarity 0-1)."""
    stems = place_stems(query)
    exact = bool(stems) and all(s in _norm(name) for s in stems)
    q, full = _bare(query), _bare(name)
    if exact:   # compare with the first word, so «Сивуля» is equally close to «Сивуля Велика» and «Сивуля Мала»
        return True, max(difflib.SequenceMatcher(None, q, _first_word(name)).ratio(),
                         difflib.SequenceMatcher(None, q, full).ratio())
    ratios = [difflib.SequenceMatcher(None, q, full).ratio()]
    if len(q) >= 5:   # a glued or shortened long name: «піпівана» vs «піпіванчорногірський»
        ratios += [difflib.SequenceMatcher(None, q, full[:n]).ratio() for n in (len(q), len(q) + 2)]
    return False, max(ratios)


def _candidate(place: dict[str, Any], score: float) -> PlaceCandidate:
    ids = routes_through(place["name"])
    return PlaceCandidate(name=place["name"], kind=place["kind"], region=place["region"],
                          elevation_m=place["elevation_m"], lat=place["lat"], lon=place["lon"], on_routes=len(ids),
                          route_ids=list(ids[:5]), score=round(score, 3))


def resolve_one(query: str, region: str | None = None) -> PlaceResolution:
    scored = []
    for place in load_places():
        best = max((_similarity(query, n) for n in [place["name"], *place["alt_names"]]), key=lambda s: (s[0], s[1]))
        if best[0] or best[1] >= FUZZY_MIN:
            scored.append((best, place))
    if not scored:
        return PlaceResolution(query=query, status="not_found", best=None, alternatives=[])

    def rank(item):
        (exact, score), place = item
        return (exact, round(score, 2), place["region"] == region, -KIND_RANK.get(place["kind"], 5),
                bool(routes_through(place["name"])), place["elevation_m"] or 0)

    scored.sort(key=rank, reverse=True)
    (exact, score), top = scored[0]
    others = [(s, p) for s, p in scored[1:] if p["name"] != top["name"]][:3]
    if exact:
        status = "found"
    elif others and not others[0][0][0] and score - others[0][0][1] < TIE:
        status = "ambiguous"
    else:
        status = "corrected"
    return PlaceResolution(query=query, status=status, best=_candidate(top, score),
                           alternatives=[_candidate(p, s[1]) for s, p in others])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def locate(name: str, region: str | None = None) -> PlaceCandidate | None:
    """Where a written place name is (the gazetteer's best match), or None if unknown."""
    r = resolve_one(name, region)
    return r.best if r.status != "not_found" else None


def distance_to_route_km(place: PlaceCandidate, route: dict[str, Any]) -> float:
    """How close the route comes to the place: the nearest profile point (every 250 m), straight line."""
    return min(haversine_km(place.lat, place.lon, p[1], p[2]) for p in route["profile"])


def resolve_places(names: list[str], region: str | None = None) -> ResolvePlacesResult:
    places = load_places()
    return ResolvePlacesResult(results=[resolve_one(n, region) for n in names], gazetteer_size=len(places),
                               source="© OpenStreetMap contributors, ODbL 1.0")
