"""find_routes: pick routes that fit a group's section, number of days and wishes.

Filtering rules (each excluded route is counted under its reason):
  * region does not match (if a region was asked for);
  * a wished feature (lake, peak, ...) is missing;
  * too long: length > days x daily hard limit of the section;
  * too short for the days: length < 3 km per day (a planner heuristic, not a Plast norm);
  * not enough night spots: fewer allowed overnight options than nights.
Ranking: closeness of km/day to the section's target (warning level, or 75 % of the hard limit).
"""

from __future__ import annotations

from typing import Any

from plast_trail.data import load_routes, rules_status, section_rule, water_distance_km
from plast_trail.models import FindRoutesResult, OvernightOption, RouteSummary

OVERNIGHT_PRIORITY = {"hut": 0, "campsite": 1, "shelter": 2, "water": 3, "lake": 4}
MIN_KM_PER_DAY = 3.0
MIN_DAY_KM = 1.0  # night spots must be at least this far from both ends


def daily_targets(section: str) -> tuple[float, float | None]:
    """(target km/day for ranking, hard km/day limit or None)."""
    hard = section_rule(section, "km_per_day_max")["value"]
    warn = section_rule(section, "km_per_day_warn")["value"]
    if warn:
        return float(warn), (float(hard) if hard else None)
    if hard:
        return round(0.75 * hard, 1), float(hard)
    return 15.0, None


def overnight_options(route: dict[str, Any], section: str, limit: int = 12) -> list[OvernightOption]:
    """Allowed night spots, one per 0.5 km stretch (best kind wins), with a water flag."""
    allowed = set(section_rule(section, "overnight_allowed_kinds")["value"]) - {"start", "finish", "peak"}
    best: dict[int, dict[str, Any]] = {}
    total = route["waypoints"][-1]["km"]
    for w in route["waypoints"]:
        if w["kind"] not in allowed or w["kind"] not in OVERNIGHT_PRIORITY:
            continue
        if not MIN_DAY_KM <= w["km"] <= total - MIN_DAY_KM:  # would make a ~0 km day
            continue
        bucket = int(w["km"] // 0.5)
        cur = best.get(bucket)
        if cur is None or OVERNIGHT_PRIORITY[w["kind"]] < OVERNIGHT_PRIORITY[cur["kind"]]:
            best[bucket] = w
    picked = sorted(best.values(), key=lambda w: w["km"])
    if len(picked) > limit:  # keep an even spread along the route
        step = len(picked) / limit
        picked = [picked[int(i * step)] for i in range(limit)]
    return [
        OvernightOption(
            id=w["id"], kind=w["kind"], name=w["name"], km=w["km"], elevation_m=w["elevation_m"],
            water_nearby=water_distance_km(route, w["km"]) is not None,
        )
        for w in picked
    ]


NEAR_KM = 8.0   # a named place farther than this from a route is not "near" it

PLACE_PREFIXES = ("г.", "гора ", "оз.", "озеро ", "с.", "село ", "смт ", "пол.", "полонина ", "пер.")


def place_stems(place: str) -> list[str]:
    """'Говерлу' -> ['говер']; 'Піп Іван' -> ['піп', 'іва']: word stems that survive Ukrainian case endings."""
    t = place.lower().replace("ґ", "г").strip()
    for pre in PLACE_PREFIXES:
        if t.startswith(pre):
            t = t[len(pre):].strip()
    words = [w for w in t.replace("-", " ").split() if len(w) >= 3]
    # drop the case ending: 2 letters from long words ('говерлу' -> 'говер'), 1 from short ones
    # ('петрос' -> 'петро', so it does not match 'Петраш'); never shorter than 4 letters
    return [w if len(w) <= 4 else w[: max(4, len(w) - (2 if len(w) >= 7 else 1))] for w in words]


def route_text(r: dict[str, Any]) -> str:
    return " ".join([r["name"]] + [w["name"] for w in r["waypoints"]]).lower().replace("ґ", "г")


def matched_places(r: dict[str, Any], places: list[str]) -> list[str]:
    text = route_text(r)
    return [p for p in places if place_stems(p) and all(stem in text for stem in place_stems(p))]


def find_routes(section: str, days: int, region: str | None, must_include: list[str],
                max_results: int, group_max_km_per_day: float | None = None,
                places: list[str] | None = None) -> FindRoutesResult:
    from plast_trail.places import distance_to_route_km, locate   # here: places.py imports this module

    routes = load_routes()
    places = [p for p in (places or []) if place_stems(p)]
    note = ""
    # A named place that no route passes (Бребенескул is on the main ridge, which only the long trails follow):
    # do not drop the wish, rank the routes by how close they come to it instead.
    off_route = [p for p in places if not any(matched_places(r, [p]) for r in routes.values())]
    near: dict[str, Any] = {}
    off_notes = []
    for p in off_route:
        where = locate(p, region)
        if where is None:
            off_notes.append(f"'{p}' is not a known place name (check the spelling); it was ignored.")
        else:
            near[where.name] = where
            off_notes.append(f"No route in the dataset passes {where.name} ({where.kind}, {where.elevation_m or '?'} m); "
                             f"routes are ranked by how close they come to it (distance_to_places_km). Reaching it is "
                             "a side trip: tell the user.")
    places = [p for p in places if p not in off_route]
    if places:  # named places: all of them if any route allows it, otherwise at least one
        full = [r for r in routes.values() if len(matched_places(r, places)) == len(places)]
        if not full:
            partial = {k: r for k, r in routes.items() if matched_places(r, places)}
            if partial:
                note = (f"No route in the dataset passes all of: {', '.join(places)}. Showing routes that pass "
                        "at least one of them; tell the user.")
                routes = partial
            else:  # cannot happen any more (off-route names were taken out above); kept for safety
                places = []
        else:
            note = ""
            routes = {r["id"]: r for r in full}
    target, hard = daily_targets(section)
    if group_max_km_per_day:  # the group's own limit only makes things stricter
        hard = min(hard, group_max_km_per_day) if hard else group_max_km_per_day
        target = min(target, group_max_km_per_day)
    excluded: dict[str, int] = {}
    notes: list[str] = ([note] if note else []) + off_notes

    max_days = section_rule(section, "max_days")
    if max_days["value"] and days > max_days["value"]:
        notes.append(
            f"{days} days exceeds the {section} limit of {max_days['value']} days "
            f"({max_days['severity']}, {max_days['source']}). validate_hike_plan will flag it."
        )

    def exclude(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    found: list[RouteSummary] = []
    for r in routes.values():
        if region and r["region"] != region:
            exclude("other region")
            continue
        kinds = {w["kind"] for w in r["waypoints"]}
        missing = [f for f in must_include if f not in kinds]
        if missing:
            exclude(f"missing {', '.join(missing)}")
            continue
        if hard and r["length_km"] > days * hard:
            exclude("too long for the days")
            continue
        if r["length_km"] < days * MIN_KM_PER_DAY:
            exclude("too short for the days")
            continue
        options = overnight_options(r, section)
        if len(options) < days - 1:
            exclude("not enough allowed night spots")
            continue

        avg = r["length_km"] / days
        score = max(0.0, 1 - abs(avg - target) / target)
        hit = matched_places(r, places) if places else []
        if places:
            score = round(score * (0.5 + 0.5 * len(hit) / len(places)), 2)  # more of the named places = better
        reasons = [f"{avg:.1f} km/day vs target {target:g} km/day"]
        if hard:
            reasons.append(f"hard limit {hard:g} km/day leaves {hard - avg:.1f} km/day margin")
        if must_include:
            reasons.append("has " + ", ".join(must_include))
        if places:
            reasons.append(f"passes {', '.join(hit)}" if hit else "passes none of the named places")
        distances = {n: round(distance_to_route_km(pl, r), 1) for n, pl in near.items()}
        if distances:   # the leader named the place: closeness counts more than the km/day fit
            closest = min(distances.values())
            score = round(0.3 * score + 0.7 * max(0.0, 1 - closest / NEAR_KM), 2)
            reasons.append("comes within " + ", ".join(f"{d:g} km of {n}" for n, d in distances.items()))
        if days > 1:
            reasons.append(f"{len(options)} allowed night spots for {days - 1} night(s)")

        found.append(RouteSummary(
            id=r["id"], name=r["name"], region=r["region"], length_km=r["length_km"],
            ascent_m=r["ascent_m"], descent_m=r["descent_m"], max_elevation_m=r["max_elevation_m"],
            loop=r["loop"], start=r["waypoints"][0]["name"], finish=r["waypoints"][-1]["name"],
            avg_km_per_day=round(avg, 1), features=sorted(kinds - {"start", "finish"}),
            overnight_options=options if days > 1 else [],
            fit_score=round(score, 2), fit_reasons=reasons, places_matched=hit, distance_to_places_km=distances,
        ))

    if near and any(min(s.distance_to_places_km.values()) < NEAR_KM for s in found):
        far = [s for s in found if min(s.distance_to_places_km.values()) >= NEAR_KM]
        for _ in far:
            exclude(f"{NEAR_KM:g} km or more from {', '.join(near)}")
        found = [s for s in found if s not in far]
    found.sort(key=lambda s: (-s.fit_score, s.ascent_m))
    if not found:
        notes.append("No route matched. Try fewer wishes, another region, or a different number of days.")
    return FindRoutesResult(
        rules_status=rules_status(), daily_km_target=target, daily_km_limit=hard,
        total_candidates=len(routes), routes=found[:max_results], excluded=excluded, notes=notes,
    )
