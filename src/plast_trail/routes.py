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

from plast_trail.data import load_routes, rules_status, section_rule
from plast_trail.models import FindRoutesResult, OvernightOption, RouteSummary

OVERNIGHT_PRIORITY = {"hut": 0, "campsite": 1, "shelter": 2, "water": 3, "lake": 4}
WATER_RADIUS_KM = 0.3
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
    water_kms = [w["km"] for w in route["waypoints"] if w["kind"] == "water"]
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
            water_nearby=w["kind"] == "water" or any(abs(k - w["km"]) <= WATER_RADIUS_KM for k in water_kms),
        )
        for w in picked
    ]


def find_routes(section: str, days: int, region: str | None, must_include: list[str],
                max_results: int) -> FindRoutesResult:
    routes = load_routes()
    target, hard = daily_targets(section)
    excluded: dict[str, int] = {}
    notes: list[str] = []

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
        reasons = [f"{avg:.1f} km/day vs target {target:g} km/day"]
        if hard:
            reasons.append(f"hard limit {hard:g} km/day leaves {hard - avg:.1f} km/day margin")
        if must_include:
            reasons.append("has " + ", ".join(must_include))
        if days > 1:
            reasons.append(f"{len(options)} allowed night spots for {days - 1} night(s)")

        found.append(RouteSummary(
            id=r["id"], name=r["name"], region=r["region"], length_km=r["length_km"],
            ascent_m=r["ascent_m"], descent_m=r["descent_m"], max_elevation_m=r["max_elevation_m"],
            loop=r["loop"], start=r["waypoints"][0]["name"], finish=r["waypoints"][-1]["name"],
            avg_km_per_day=round(avg, 1), features=sorted(kinds - {"start", "finish"}),
            overnight_options=options if days > 1 else [],
            fit_score=round(score, 2), fit_reasons=reasons,
        ))

    found.sort(key=lambda s: (-s.fit_score, s.ascent_m))
    if not found:
        notes.append("No route matched. Try fewer wishes, another region, or a different number of days.")
    return FindRoutesResult(
        rules_status=rules_status(), daily_km_target=target, daily_km_limit=hard,
        total_candidates=len(routes), routes=found[:max_results], excluded=excluded, notes=notes,
    )
