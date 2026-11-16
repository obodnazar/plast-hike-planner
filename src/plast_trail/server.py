"""plast-trail: MCP server for planning Plast hikes in the Ukrainian Carpathians.

Start it on its own, in a separate terminal:

    uv run plast-trail                 # streamable HTTP on 127.0.0.1:8770/mcp
    uv run plast-trail --port 8771     # another port
    WEATHER_MODE=live uv run plast-trail

Tools (all return structured JSON with an output schema):
    find_routes          - routes that fit the group (OSM dataset)
    validate_hike_plan   - per-day numbers + rule check with fix suggestions
    assess_weather_risk  - Open-Meteo forecast -> risk per day (live / record / replay)
    calculate_supplies   - food, water, gas, tents, pack weight
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
import time
from datetime import date
from typing import Annotated, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from plast_trail import places as gazetteer
from plast_trail import profile, routes, supplies, validation, weather
from plast_trail.data import PROJECT_ROOT, dataset_meta, load_routes, rules_status
from plast_trail.errors import DomainError
from plast_trail.models import (
    DayLoad,
    Feature,
    FindRoutesResult,
    GroupLimits,
    Region,
    ResolvePlacesResult,
    RouteProfileResult,
    Section,
    SuppliesResult,
    ValidationResult,
    WeatherResult,
)

log = logging.getLogger("plast-trail")

INSTRUCTIONS = """\
Tools for planning a Plast (Ukrainian scouting) hike on marked trails of the Ukrainian Carpathians.
Typical flow:
0. If the request names peaks, lakes or villages: resolve_places(names) -> use the corrected names,
   ask the user about ambiguous ones, and the region of the places if the request gives none.
1. find_routes(section, days, ...) -> pick a route and night spots from its overnight_options.
2. validate_hike_plan(route_id, overnight_waypoint_ids, ...) -> if valid=false, apply the
   suggestions (they name better waypoint ids) and validate again.
3. assess_weather_risk(same route and night spots, start_date) -> if a day is high risk,
   change dates, night spots or route, then validate again.
4. calculate_supplies(day_loads from step 2 days[], dry_nights from step 2, night_min_c = coldest_night_c from step 3).
5. Optional, to show the plan: get_route_profile(same route and night spots) -> per-day profile and landmarks.
Never invent km, ascent or ids: take them from tool results. Rules are a DRAFT until a
Plast leader approves them; always mention rules_status to the user.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def logged(fn):
    """Log every call on the server side (visible in the server terminal during the demo)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        shown = {k: getattr(v, "value", v) for k, v in kwargs.items() if v not in (None, [], "")}
        try:
            result = fn(*args, **kwargs)
        except DomainError as exc:
            log.warning("%s(%s) -> ERROR %s (%.0f ms)", fn.__name__, shown, exc.code,
                        (time.monotonic() - started) * 1000)
            raise exc.to_tool_error() from exc
        except Exception as exc:  # a bug must still come back as a structured error, not a stack trace
            log.exception("%s(%s) -> INTERNAL ERROR", fn.__name__, shown)
            raise DomainError("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}",
                              "This is a server bug; see the server log.").to_tool_error() from exc
        log.info("%s(%s) -> ok (%.0f ms)", fn.__name__, shown, (time.monotonic() - started) * 1000)
        return result
    return wrapper


mcp = FastMCP("plast-trail", instructions=INSTRUCTIONS)


@mcp.tool(
    title="Find routes for a Plast group",
    description=(
        "Find marked hiking routes in the Ukrainian Carpathians (OpenStreetMap) that suit a Plast group: "
        "age section, number of days, optional region and wished features (lake, peak, ...). "
        "Returns routes ranked by how well km/day fits the section's target, each with allowed overnight "
        "spots (waypoint ids for validate_hike_plan). Use it first. An empty `routes` list means nothing "
        "matched (see `excluded` for why); it is not an error. "
        "Not a free-text search; to check or correct a place name first, use resolve_places. Only the bundled "
        "Carpathian dataset. Route ids are "
        "'r' + the OpenStreetMap relation id; waypoint ids ('w12') are valid only within their own route."
    ),
    annotations=READ_ONLY,
)
@logged
def find_routes(
    section: Annotated[Section, Field(description="Plast age section: novatstvo (6-11), yunatstvo (11-18), senior (18+)")],
    days: Annotated[int, Field(ge=1, le=7, description="Number of hiking days (nights = days - 1)")],
    region: Annotated[Region | None, Field(description="Mountain area; omit to search all")] = None,
    must_include: Annotated[list[Feature], Field(max_length=4, description="Features the route must pass, e.g. ['lake']")] = [],  # noqa: B006
    max_results: Annotated[int, Field(ge=1, le=10, description="How many routes to return")] = 5,
    max_km_per_day: Annotated[float | None, Field(ge=1, le=40, description="The group's own stricter daily limit (from the request); routes longer than days × this are excluded and the ranking aims at it")] = None,
    places: Annotated[list[Annotated[str, Field(min_length=3, max_length=40)]], Field(max_length=3, description="Named peaks, lakes or villages to pass, e.g. ['Говерла', 'Петрос'] (any case form). Routes passing all of them first; if none does, routes passing at least one, with a note")] = [],  # noqa: B006
) -> FindRoutesResult:
    return routes.find_routes(section.value, days, region.value if region else None,
                              [f.value for f in must_include], max_results, max_km_per_day, places)


@mcp.tool(
    title="Resolve place names",
    description=(
        "Look up peaks, lakes, villages, polonyny or huts by the names people write, including misspellings and "
        "case forms ('піпівана' -> 'Піп Іван Чорногірський', 'Говерлу' -> 'Говерла'), in a gazetteer of ~1500 "
        "named places from OpenStreetMap. For each name: status found / corrected / ambiguous / not_found, the "
        "best place with its kind, region and height, how many routes of the dataset pass it (on_routes = 0: "
        "find_routes cannot use it), and alternatives. Use it before find_routes when a request names places: "
        "pass best.name on to find_routes(places=...), ask the user when the status is ambiguous, and take the "
        "region from the places if the request gives none. not_found is a normal result, not an error. "
        "Not a route search and not a geocoder outside the Ukrainian Carpathians."
    ),
    annotations=READ_ONLY,
)
@logged
def resolve_places(
    names: Annotated[list[Annotated[str, Field(min_length=2, max_length=60)]], Field(
        min_length=1, max_length=5, description="Place names as written by the user, e.g. ['піпівана', 'Говерлу']")],
    region: Annotated[Region | None, Field(description="Prefer places in this area when names repeat (Петрос is in "
                                                       "three areas); omit if unknown")] = None,
) -> ResolvePlacesResult:
    return gazetteer.resolve_places(names, region.value if region else None)


@mcp.tool(
    title="Validate a hike plan",
    description=(
        "Check a concrete hike plan against Plast safety rules and the group's own limits. "
        "A plan is route_id + the waypoint ids where the group sleeps, in route order (N ids = N+1 days). "
        "Computes per-day km, ascent, walking time and arrival vs sunset, checks night spots (allowed kind, "
        "water, ridge), number of adults and group size. Returns valid=false with violations; each violation "
        "has a concrete suggestion, often a better waypoint id. Re-validate after every change until valid=true. "
        "Use only a route_id from find_routes and night spots from its overnight_options; never invent ids. "
        "Warnings do not make valid=false but must be reported to the user together with rules_status. "
        "The number of night spots fixes the number of days: do not add nights without the user's consent."
    ),
    annotations=READ_ONLY,
)
@logged
def validate_hike_plan(
    route_id: Annotated[str, Field(pattern=r"^r\d+$", description="Route id from find_routes, e.g. 'r8613349'")],
    section: Annotated[Section, Field(description="Plast age section of the participants")],
    participants: Annotated[int, Field(ge=1, le=60, description="Participants of the section (children/youth; for senior: members)")],
    adults: Annotated[int, Field(ge=0, le=30, description="Adult leaders accompanying the group")],
    overnight_waypoint_ids: Annotated[list[Annotated[str, Field(pattern=r"^w\d+$")]], Field(
        max_length=6, description="Night spots in route order, e.g. ['w30', 'w56']; [] for a one-day hike")] = [],  # noqa: B006
    start_date: Annotated[date | None, Field(description="First day, YYYY-MM-DD; enables the sunset check")] = None,
    start_time: Annotated[str | None, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="Daily start time HH:MM (24 h), default from rules (08:00)")] = None,
    group_limits: Annotated[GroupLimits | None, Field(description="Stricter limits from the leader's request note")] = None,
) -> ValidationResult:
    return validation.validate_hike_plan(route_id, section.value, participants, adults, overnight_waypoint_ids,
                                         start_date, start_time, group_limits)


@mcp.tool(
    title="Assess weather risk for a plan",
    description=(
        "Weather risk for each day of a plan (same route_id and overnight_waypoint_ids as validate_hike_plan) "
        "starting on start_date: thunderstorm, snow, wind and rain at the route's highest point, and night cold "
        "at each night spot. Returns per-day risk low/medium/high with triggers and recommendations. "
        "If any day is high, change dates or route and validate again. "
        "Call it only after validate_hike_plan returned valid=true, with the same start_date. "
        "Not a general weather lookup. Data: Open-Meteo forecast (16 days ahead); in replay mode, recorded "
        "genuine responses, so only recorded dates work."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
)
@logged
def assess_weather_risk(
    route_id: Annotated[str, Field(pattern=r"^r\d+$", description="Route id from find_routes")],
    start_date: Annotated[date, Field(description="First hiking day, YYYY-MM-DD, inside the 16-day forecast window")],
    overnight_waypoint_ids: Annotated[list[Annotated[str, Field(pattern=r"^w\d+$")]], Field(
        max_length=6, description="Same night spots as in validate_hike_plan")] = [],  # noqa: B006
) -> WeatherResult:
    return weather.assess_weather_risk(route_id, start_date, overnight_waypoint_ids)


@mcp.tool(
    title="Calculate supplies and pack weight",
    description=(
        "Compute food, water, cooking gas, tents, stoves, pots, first-aid kits and pack weights for a "
        "validated plan. Call it last. Pass day_loads = one object per day copied from validate_hike_plan "
        "days[] (km, ascent_m, walking_hours), dry_nights = validate_hike_plan dry_nights, night_min_c = "
        "assess_weather_risk coldest_night_c. Every coefficient used is listed in `assumptions` with its source."
    ),
    annotations=READ_ONLY,
)
@logged
def calculate_supplies(
    section: Annotated[Section, Field(description="Plast age section of the participants")],
    participants: Annotated[int, Field(ge=1, le=60, description="Participants of the section (same as in validate_hike_plan)")],
    adults: Annotated[int, Field(ge=0, le=30, description="Adult leaders")],
    day_loads: Annotated[list[DayLoad], Field(min_length=1, max_length=7, description="Array with one object per hiking day (NOT a number): km, ascent_m, walking_hours copied from validate_hike_plan days[]")],
    dry_nights: Annotated[int, Field(ge=0, le=6, description="validate_hike_plan dry_nights: night spots without water nearby")] = 0,
    night_min_c: Annotated[float | None, Field(ge=-40, le=30, description="assess_weather_risk coldest_night_c, °C; omit if the weather was not assessed")] = None,
    tent_capacity: Annotated[Literal[2, 3, 4], Field(description="People per tent")] = 3,
    lightest_body_weight_kg: Annotated[float | None, Field(ge=15, le=150, description="Body weight of the lightest participant, for the % body-weight pack rule")] = None,
    participant_base_gear_kg: Annotated[float | None, Field(ge=1, le=25, description="Measured weight of a participant's packed rucksack without food and water")] = None,
    adult_base_gear_kg: Annotated[float | None, Field(ge=1, le=30, description="Same for an adult leader")] = None,
) -> SuppliesResult:
    if dry_nights > len(day_loads) - 1:
        raise DomainError("INVALID_INPUT", f"dry_nights={dry_nights} but a {len(day_loads)}-day plan has only "
                          f"{len(day_loads) - 1} night(s).", "Pass validate_hike_plan dry_nights.").to_tool_error()
    return supplies.calculate_supplies(section.value, participants, adults, day_loads, dry_nights, night_min_c,
                                       tent_capacity, lightest_body_weight_kg, participant_base_gear_kg,
                                       adult_base_gear_kg)


@mcp.tool(
    title="Route profile for a plan",
    description=(
        "The shape of a planned hike, for drawing it: for the same route_id and overnight_waypoint_ids as "
        "validate_hike_plan, the elevation profile of each day every 250 m with coordinates ([km, lat, lon, "
        "elevation_m]), the day's climb, drop, highest point and how many km are steeper than 15 %, plus "
        "landmarks (start, finish, night spots, named peaks, lakes, huts). Use it after the plan is valid, to "
        "show the user how hard each day is (a map sketch and an elevation chart). It does not check rules and "
        "adds no new decisions: for rules use validate_hike_plan. Ids must come from find_routes."
    ),
    annotations=READ_ONLY,
)
@logged
def get_route_profile(
    route_id: Annotated[str, Field(pattern=r"^r\d+$", description="Route id from find_routes, e.g. 'r8613349'")],
    overnight_waypoint_ids: Annotated[list[Annotated[str, Field(pattern=r"^w\d+$")]], Field(
        max_length=6, description="Night spots in route order, as validated; [] for a one-day hike")] = [],  # noqa: B006
) -> RouteProfileResult:
    return profile.route_profile(route_id, overnight_waypoint_ids)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="plast-trail MCP server")
    parser.add_argument("--host", default=os.environ.get("PLAST_TRAIL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PLAST_TRAIL_PORT", "8770")))
    parser.add_argument("--stdio", action="store_true", help="serve over stdio instead of HTTP")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):   # Ukrainian place names in the log, whatever the console encoding
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr, force=True)
    try:
        n_routes = len(load_routes())
        meta = dataset_meta()
        status = rules_status()
        wmode = weather.mode()
        if wmode == "record":
            raise DomainError("INVALID_INPUT", "WEATHER_MODE=record is not served by the tool",
                              "Record fixtures with: uv run python scripts/record_weather.py")
    except DomainError as exc:
        sys.exit(f"Cannot start: {exc.code}: {exc.message}. {exc.hint}")
    log.info("plast-trail: %d routes (OSM snapshot %s), rules: %s, weather mode: %s",
             n_routes, meta.get("osm_snapshot"), status, wmode)

    if args.stdio:
        mcp.run(transport="stdio")
        return
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    log.info("serving MCP over streamable HTTP at http://%s:%d%s", args.host, args.port,
             mcp.settings.streamable_http_path)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
