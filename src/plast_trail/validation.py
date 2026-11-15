"""validate_hike_plan: split a route into days at the chosen night spots and check every rule.

A plan is: route + the waypoint ids where the group sleeps (in route order).
N night spots -> N+1 days. All numbers (km, ascent, hours) are computed here from
the route's elevation profile; the model never has to invent them.
"""

from __future__ import annotations

import itertools
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from plast_trail.data import (
    ascent_descent,
    common_rule,
    get_route,
    water_distance_km,
    load_rules,
    profile_between,
    rules_status,
    section_rule,
)
from plast_trail.errors import INVALID_INPUT, WAYPOINT_NOT_FOUND, WAYPOINT_ORDER, DomainError
from plast_trail.models import (
    DayPlan,
    GroupLimits,
    OvernightCheck,
    Severity,
    ValidationResult,
    Violation,
    Waypoint,
)
from plast_trail.routes import overnight_options

KYIV = ZoneInfo("Europe/Kyiv")
MIN_DAY_KM = 1.0  # a night spot closer than this to the start or finish would make a ~0 km day


# ------------------------------------------------------------------ time helpers
# A route that ends this high and near its top ends on a summit or ridge: the descent is not planned.
ENDS_HIGH_M = 1300
ENDS_NEAR_TOP_M = 60


def parse_hhmm(value: str) -> int:
    """'08:30' -> minutes after midnight."""
    try:
        h, m = value.split(":")
        minutes = int(h) * 60 + int(m)
    except ValueError as exc:
        raise DomainError(INVALID_INPUT, f"start_time '{value}' is not HH:MM", "Use e.g. '08:00'.") from exc
    if not 0 <= minutes < 24 * 60:
        raise DomainError(INVALID_INPUT, f"start_time '{value}' is outside 00:00-23:59")
    return minutes


def fmt_hhmm(minutes: float) -> str:
    minutes = int(round(minutes)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def sunset_local_minutes(day: date, lat: float, lon: float) -> float:
    """Local (Europe/Kyiv) sunset in minutes after midnight. NOAA solar-position approximation."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    phi = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(decl))
              - math.tan(phi) * math.tan(decl))
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    utc_minutes = 720 - 4 * (lon - ha) - eqtime
    midnight_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    local = (midnight_utc + timedelta(minutes=utc_minutes)).astimezone(KYIV)
    return local.hour * 60 + local.minute + local.second / 60


# ------------------------------------------------------------------ plan -> days
def resolve_night_spots(route: dict[str, Any], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {w["id"]: w for w in route["waypoints"]}
    last_id = route["waypoints"][-1]["id"]
    spots = []
    for wid in ids:
        if wid not in by_id:
            raise DomainError(
                WAYPOINT_NOT_FOUND,
                f"Route {route['id']} has no waypoint '{wid}'.",
                f"Use night spots from find_routes overnight_options for {route['id']}, e.g. "
                f"{', '.join(night_spot_examples(route))}. Never invent ids.",
            )
        if wid in ("w0", last_id):
            raise DomainError(WAYPOINT_ORDER, f"'{wid}' is the start or finish, not a night spot.",
                              "Night spots must lie strictly inside the route.")
        w = by_id[wid]
        total = route["waypoints"][-1]["km"]
        if not MIN_DAY_KM <= w["km"] <= total - MIN_DAY_KM:
            raise DomainError(WAYPOINT_ORDER, f"'{wid}' is at km {w['km']}, too close to the start or finish "
                              f"(route length {total} km).", f"Night spots must be at least {MIN_DAY_KM:g} km "
                              "from both ends; take them from find_routes overnight_options.")
        spots.append(w)
    for a, b in zip(spots, spots[1:]):
        if a["id"] == b["id"]:
            raise DomainError(WAYPOINT_ORDER, f"Night spot {a['id']} is listed twice.", "Use each night spot once.")
        if b["km"] <= a["km"]:
            raise DomainError(WAYPOINT_ORDER,
                              f"Night spots must go in route order: {a['id']} (km {a['km']}) "
                              f"comes before {b['id']} (km {b['km']}).",
                              "Sort the ids by their km value.")
    return spots


def night_spot_examples(route: dict[str, Any], n: int = 6) -> list[str]:
    """A few real night-spot ids of this route, spread along it (for error hints)."""
    total = route["waypoints"][-1]["km"]
    ids = [w["id"] for w in route["waypoints"]
           if w["kind"] in ("hut", "campsite", "shelter", "water", "lake") and MIN_DAY_KM <= w["km"] <= total - MIN_DAY_KM]
    step = max(1, len(ids) // n)
    return ids[::step][:n] or ["(this route has no night spots: use another route)"]


def wp_model(w: dict[str, Any]) -> Waypoint:
    return Waypoint(id=w["id"], kind=w["kind"], name=w["name"], km=w["km"], elevation_m=w["elevation_m"])


def day_stats(route: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pts = profile_between(route, a["km"], b["km"])
    heights = [p[3] for p in pts]
    up, down = ascent_descent(heights)
    top = max(pts, key=lambda p: p[3])
    pace = load_rules()["pace"]
    km = round(b["km"] - a["km"], 1)
    walking = km / pace["flat_km_per_hour"]["value"] + up / pace["ascent_m_per_hour"]["value"]
    return {
        "km": km, "ascent": up, "descent": down, "min": min(heights), "max": max(heights),
        "walking": round(walking, 1),
        "elapsed": round(walking * (1 + common_rule("break_share")["value"]), 1),
        "top": {"km": top[0], "lat": top[1], "lon": top[2], "elevation_m": top[3]},
        "end_lat": pts[-1][1], "end_lon": pts[-1][2],
    }


def ridge_risk(route: dict[str, Any], km: float, elevation: int | None = None) -> bool:
    """A night spot on a crest or top: near the highest point within ±1 km AND the ground drops on both sides.

    Heights come from the route profile (not the OSM `ele` tag) so all points are compared on one scale.
    A spot in the middle of a long climb, or on a plateau, is not a ridge.
    """
    margin = common_rule("ridge_overnight")["value"]  # metres
    window = profile_between(route, km - 1.0, km + 1.0)
    if not window:
        return False
    here = min(window, key=lambda p: abs(p[0] - km))[3]
    left = [p[3] for p in window if p[0] < km]
    right = [p[3] for p in window if p[0] > km]
    near_top = here >= max(p[3] for p in window) - margin
    drops_both_sides = bool(left and right) and min(left) <= here - margin and min(right) <= here - margin
    return near_top and drops_both_sides


# ------------------------------------------------------------------ main entry
def validate_hike_plan(route_id: str, section: str, participants: int, adults: int,
                       overnight_waypoint_ids: list[str], start_date: date | None,
                       start_time: str | None, group_limits: GroupLimits | None) -> ValidationResult:
    route = get_route(route_id)
    spots = resolve_night_spots(route, overnight_waypoint_ids)
    group = group_limits or GroupLimits()
    start_min = parse_hhmm(start_time or common_rule("default_start")["value"])
    stops = [route["waypoints"][0], *spots, route["waypoints"][-1]]
    violations: list[Violation] = []
    applied: dict[str, str] = {}

    def add(code, severity, day, message, limit, actual, source, suggestion=""):
        violations.append(Violation(code=code, severity=Severity(severity), day=day, message=message,
                                    limit=limit, actual=actual, source=source, suggestion=suggestion))

    def limit_for(rule_name: str, group_value: float | None, label: str) -> tuple[float | None, str, str]:
        """Strictest of rules value and group value -> (limit, severity, source)."""
        rule = section_rule(section, rule_name)
        rv = rule["value"]
        if group_value is not None and (rv is None or group_value < rv):
            applied[label] = f"{group_value:g} (group note, stricter than rules: {rv})"
            return group_value, "error", "group note (Obsidian request)"
        if rv is not None:
            applied[label] = f"{rv:g} (rules.yaml, {rule['severity']}, approved={rule['approved']})"
            return rv, rule["severity"], rule["source"]
        applied[label] = "none (reported only)"
        return None, "warning", rule["source"]

    km_max, km_sev, km_src = limit_for("km_per_day_max", group.max_km_per_day, "km_per_day_max")
    up_max, up_sev, up_src = limit_for("ascent_per_day_max", group.max_ascent_per_day_m, "ascent_per_day_max")
    h_max, h_sev, h_src = limit_for("walking_hours_per_day_max", group.max_walking_hours_per_day,
                                    "walking_hours_per_day_max")
    km_warn = section_rule(section, "km_per_day_warn")
    allowed_kinds = set(section_rule(section, "overnight_allowed_kinds")["value"])
    # The same night spots find_routes shows, so every suggested id is one the caller has seen.
    options = overnight_options(route, section)
    total_km = route["waypoints"][-1]["km"]
    n_days = len(stops) - 1
    nights = n_days - 1

    def even_split(max_km: float) -> list | None:
        """Night spots (from the shown options) that keep every day <= max_km, as even as possible."""
        best, best_key = None, None
        for combo in itertools.combinations(options, nights):
            cuts = [0.0] + [o.km for o in combo] + [total_km]
            legs = [round(b - a, 1) for a, b in zip(cuts, cuts[1:])]
            if max(legs) > max_km:
                continue
            key = (max(legs), max(legs) - min(legs))
            if best_key is None or key < best_key:
                best, best_key = (combo, legs), key
        return best

    def better_stop(day_start_km: float, max_km: float, last_day: bool) -> str:
        """A fix that keeps the plan's number of days, or says honestly that it cannot."""
        max_km = round(max_km, 1)
        need = math.ceil(total_km / max_km) if max_km > 0 else n_days + 1
        if need > n_days:
            return (f"At {max_km:g} km/day this {total_km:g} km route needs at least {need} days, but the plan has "
                    f"{n_days}: choose a shorter route, or ask the leader for more days.")
        found = even_split(max_km) if nights else None
        if found:
            combo, legs = found
            ids = ", ".join(f"{o.id} (km {o.km:g})" for o in combo)
            return (f"Keep {nights} night(s) but split the route more evenly: night spots {ids} "
                    f"give days of {' / '.join(f'{x:g}' for x in legs)} km.")
        return (f"No combination of this route's overnight options keeps every day within {max_km:g} km: "
                "choose another route.")

    days: list[DayPlan] = []
    for i, (a, b) in enumerate(zip(stops, stops[1:]), start=1):
        s = day_stats(route, a, b)
        last = i == len(stops) - 1
        sunset = None
        if start_date:
            sunset_min = sunset_local_minutes(start_date + timedelta(days=i - 1), s["end_lat"], s["end_lon"])
            sunset = fmt_hhmm(sunset_min)
            arrive = start_min + s["elapsed"] * 60
            margin_h = common_rule("arrive_hours_before_sunset")["value"]
            if arrive > sunset_min - margin_h * 60:
                latest_start = sunset_min - margin_h * 60 - s["elapsed"] * 60
                earliest = parse_hhmm(common_rule("earliest_start")["value"])
                if latest_start >= earliest:
                    hint = f"Start by {fmt_hhmm(latest_start)} (earliest allowed {fmt_hhmm(earliest)})."
                else:
                    budget_h = (sunset_min - margin_h * 60 - earliest) / 60
                    hint = (f"Even a {fmt_hhmm(earliest)} start is too late; shorten the day to about "
                            f"{budget_h:.1f} h. " + better_stop(a["km"], round(s["km"] * budget_h / s["elapsed"], 1), last))
                add("ARRIVAL_AFTER_SAFE_TIME", "warning", i,
                    f"Arrival {fmt_hhmm(arrive)} is less than {margin_h} h before sunset {sunset}.",
                    fmt_hhmm(sunset_min - margin_h * 60), fmt_hhmm(arrive),
                    common_rule("arrive_hours_before_sunset")["source"], hint)
        if km_max is not None and s["km"] > km_max:
            add("KM_PER_DAY_MAX", km_sev, i, f"Day {i} is {s['km']} km, limit {km_max:g} km.",
                km_max, s["km"], km_src, better_stop(a["km"], km_max, last))
        elif km_warn["value"] and s["km"] > km_warn["value"]:
            add("KM_PER_DAY_WARN", "warning", i, f"Day {i} is {s['km']} km, above the comfortable {km_warn['value']} km.",
                km_warn["value"], s["km"], km_warn["source"], better_stop(a["km"], km_warn["value"], last))
        if up_max is not None and s["ascent"] > up_max:
            add("ASCENT_PER_DAY_MAX", up_sev, i, f"Day {i} climbs {s['ascent']} m, limit {up_max:g} m.",
                up_max, s["ascent"], up_src, "Split the climb over two days or pick a route with less ascent.")
        if h_max is not None and s["walking"] > h_max:
            add("WALKING_HOURS_MAX", h_sev, i, f"Day {i} needs {s['walking']} h of walking, limit {h_max:g} h.",
                h_max, s["walking"], h_src,
                better_stop(a["km"], s["km"] * h_max / max(s["walking"], 0.1), last))

        overnight = None
        if i < len(stops) - 1:
            w = b
            water = water_distance_km(route, w["km"])
            ridge = ridge_risk(route, w["km"], w["elevation_m"])
            overnight = OvernightCheck(waypoint=wp_model(w),
                                       water_within_m=None if water is None else round(water * 1000),
                                       ridge_risk=ridge)
            if w["kind"] not in allowed_kinds:
                rule = section_rule(section, "overnight_allowed_kinds")
                alt = [o for o in options if abs(o.km - w["km"]) <= 3]
                add("OVERNIGHT_KIND_NOT_ALLOWED", rule["severity"], i,
                    f"Night {i} at a {w['kind']} ({w['name']!r}) is not allowed for {section}.",
                    ", ".join(sorted(allowed_kinds - {'start', 'finish'})), w["kind"], rule["source"],
                    (f"Use {alt[0].id} ({alt[0].kind}, km {alt[0].km})." if alt else "No allowed spot within 3 km."))
            if group.max_overnight_elevation_m and w["elevation_m"] > group.max_overnight_elevation_m:
                add("OVERNIGHT_TOO_HIGH", "error", i,
                    f"Night {i} is at {w['elevation_m']} m, group limit {group.max_overnight_elevation_m} m.",
                    group.max_overnight_elevation_m, w["elevation_m"], "group note (Obsidian request)",
                    "Pick a lower night spot from find_routes overnight_options.")
            if ridge:
                lower = [o for o in options if abs(o.km - w["km"]) <= 2 and o.elevation_m < w["elevation_m"] - 50]
                add("RIDGE_OVERNIGHT", "warning", i, f"Night {i} ({w['id']}) looks like a ridge or top.",
                    "not within 30 m of the local high point", w["elevation_m"], common_rule("ridge_overnight")["source"],
                    (f"Lower spot nearby: {lower[0].id} ({lower[0].kind}, {lower[0].elevation_m} m)." if lower
                     else "No lower allowed spot within 2 km."))
            if water is None:
                wet = [o for o in options if o.water_nearby and abs(o.km - w["km"]) <= 3]
                add("OVERNIGHT_NO_WATER", "warning", i, f"No water within 300 m of night {i} ({w['id']}).",
                    "water within 300 m", "none", common_rule("overnight_water_radius_km")["source"],
                    (f"Move the night to {wet[0].id} (km {wet[0].km}), or carry water (calculate_supplies dry_nights)."
                     if wet else "Carry water for the evening and morning (pass dry_nights to calculate_supplies)."))

        days.append(DayPlan(
            day=i, start=wp_model(a), end=wp_model(b), km=s["km"], ascent_m=s["ascent"],
            descent_m=s["descent"], min_elevation_m=s["min"], max_elevation_m=s["max"],
            walking_hours=s["walking"], elapsed_hours=s["elapsed"], start_time=fmt_hhmm(start_min),
            est_arrival=fmt_hhmm(start_min + s["elapsed"] * 60), sunset=sunset,
            highest_point=s["top"], overnight=overnight,
        ))

    # ---- whole-group checks
    n_days = len(days)
    rule = section_rule(section, "max_days")
    if rule["value"] and n_days > rule["value"]:
        add("TOO_MANY_DAYS", rule["severity"], None, f"{n_days} days, limit {rule['value']} for {section}.",
            rule["value"], n_days, rule["source"], "Shorten the hike or split it into separate outings.")
    rule = section_rule(section, "min_adults")
    people_adults = adults + (participants if section == "senior" else 0)
    if rule["value"] and people_adults < rule["value"]:
        add("TOO_FEW_ADULTS", rule["severity"], None, f"{people_adults} adult(s), at least {rule['value']} needed.",
            rule["value"], people_adults, rule["source"], "Add adult leaders.")
    rule = section_rule(section, "children_per_adult_max")
    if rule["value"] and section != "senior":
        need = math.ceil(participants / rule["value"])
        if adults < need:
            add("ADULT_RATIO", rule["severity"], None,
                f"{participants} participants need at least {need} adults (1 per {rule['value']}), have {adults}.",
                need, adults, rule["source"], f"Add {need - adults} adult(s) or reduce the group.")
    rule = section_rule(section, "group_size_max")
    if rule["value"] and participants + adults > rule["value"]:
        add("GROUP_TOO_LARGE", rule["severity"], None,
            f"Group of {participants + adults}, recommended at most {rule['value']}.",
            rule["value"], participants + adults, rule["source"], "Split into two groups, each with its own adults.")
    if start_min < parse_hhmm(common_rule("earliest_start")["value"]):
        add("START_TOO_EARLY", "warning", None, f"Start at {fmt_hhmm(start_min)} is before the earliest allowed.",
            common_rule("earliest_start")["value"], fmt_hhmm(start_min), common_rule("earliest_start")["source"])

    # ---- the end of the route: routes are planned one way, so some end on a summit or a ridge
    finish = route["waypoints"][-1]
    last = days[-1]
    top_at_end = finish["elevation_m"] if finish["elevation_m"] is not None else last.max_elevation_m
    if top_at_end >= ENDS_HIGH_M and top_at_end >= route["max_elevation_m"] - ENDS_NEAR_TOP_M:
        back = stops[-2]
        pace = load_rules()["pace"]["flat_km_per_hour"]["value"]
        back_h = round(last.km / pace * (1 + common_rule("break_share")["value"]), 1)
        back_at = parse_hhmm(last.est_arrival) + back_h * 60
        late = ""
        if last.sunset:
            margin = common_rule("arrive_hours_before_sunset")["value"] * 60
            if back_at > parse_hhmm(last.sunset) - margin:
                late = f" Back at about {fmt_hhmm(back_at)}, later than {margin / 60:g} h before sunset {last.sunset}."
        add("ENDS_ON_TOP", "warning", len(days),
            f"The route ends high, at {finish['name'] or 'its end'} ({top_at_end} m); the way down is not in the plan.",
            ENDS_HIGH_M, top_at_end, "plast-trail: the dataset plans every route one way",
            f"Plan the descent: back the same way to {back['name'] or back['id']} is {last.km:g} km, about "
            f"{back_h:g} h with rests.{late} Or choose a route that ends in a village.")

    docs = list(load_rules()["documents"]["all"])
    if section != "senior":
        docs += load_rules()["documents"]["minors"]

    return ValidationResult(
        valid=not any(v.severity == Severity.error for v in violations),
        rules_status=rules_status(), route_id=route["id"], route_name=route["name"], section=section,
        days=days, total_km=round(sum(d.km for d in days), 1), total_ascent_m=sum(d.ascent_m for d in days),
        violations=violations, applied_limits=applied, documents=docs,
        dry_nights=sum(1 for v in violations if v.code == "OVERNIGHT_NO_WATER"),
    )
