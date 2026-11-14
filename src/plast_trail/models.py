"""Typed contracts of the plast-trail tools.

Every tool returns one of the *Result models below. FastMCP turns them into the
tool's outputSchema, and the field descriptions are what the model reads.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WaypointKind = Literal["start", "finish", "water", "hut", "shelter", "campsite", "peak", "lake"]
ViolationCode = Literal[
    "KM_PER_DAY_MAX", "KM_PER_DAY_WARN", "ASCENT_PER_DAY_MAX", "WALKING_HOURS_MAX", "ARRIVAL_AFTER_SAFE_TIME",
    "OVERNIGHT_KIND_NOT_ALLOWED", "OVERNIGHT_TOO_HIGH", "RIDGE_OVERNIGHT", "OVERNIGHT_NO_WATER",
    "TOO_MANY_DAYS", "TOO_FEW_ADULTS", "ADULT_RATIO", "GROUP_TOO_LARGE", "START_TOO_EARLY",
]
TriggerCode = Literal["THUNDERSTORM", "SNOW", "WIND", "RAIN", "NIGHT_COLD"]
TriggerUnit = Literal["wmo_code", "m/s", "mm", "°C"]


# ------------------------------------------------------------------ enums
class Section(str, Enum):
    """Plast age section of the group."""

    novatstvo = "novatstvo"  # новацтво, 6-11
    yunatstvo = "yunatstvo"  # юнацтво, 11-18
    senior = "senior"        # старші пластуни / сеніори, 18+


class Region(str, Enum):
    """Mountain area of the Ukrainian Carpathians (assigned to each route by its centre)."""

    chornohora = "chornohora"
    svydovets = "svydovets"
    gorgany = "gorgany"
    marmarosy = "marmarosy"
    yaremche_vorokhta = "yaremche-vorokhta"


class Feature(str, Enum):
    """Something the group wants on the route."""

    lake = "lake"
    peak = "peak"
    water = "water"
    hut = "hut"
    shelter = "shelter"
    campsite = "campsite"


class Severity(str, Enum):
    """error = the plan breaks a rule and is not valid; warning = shown to the leader, does not block."""

    error = "error"
    warning = "warning"


class RiskLevel(str, Enum):
    """Weather risk of a day or of the whole plan."""

    low = "low"
    medium = "medium"
    high = "high"


# ------------------------------------------------------------------ shared pieces
class Waypoint(BaseModel):
    """A point on a route: start, finish, or a point of interest within 300 m of the trail."""

    id: str = Field(description="Waypoint id within its route, e.g. 'w12'. Pass these ids to validate_hike_plan.")
    kind: WaypointKind = Field(description="What the point is; start/finish are the route ends")
    name: str = Field(description="Name from OpenStreetMap, may be empty")
    km: float = Field(description="Distance from the route start, km")
    elevation_m: int = Field(description="Height above sea level, m (OSM tag or SRTM profile)")


class GeoPoint(BaseModel):
    """A point on the map."""

    lat: float = Field(description="Latitude, WGS84")
    lon: float = Field(description="Longitude, WGS84")
    elevation_m: int = Field(description="Height above sea level, m")


class RoutePoint(GeoPoint):
    """A point of a route's elevation profile."""

    km: float = Field(description="Distance from the route start, km")


class OvernightOption(Waypoint):
    """A waypoint where this section may stop for the night."""

    water_nearby: bool = Field(description="A spring or drinking water within 300 m along the route")


class GroupLimits(BaseModel):
    """Extra limits from the leader's request note. They can only make rules stricter."""

    model_config = ConfigDict(extra="forbid")

    max_km_per_day: float | None = Field(default=None, ge=1, le=40, description="Group's own km/day limit")
    max_ascent_per_day_m: int | None = Field(default=None, ge=100, le=3000, description="Group's own ascent limit per day, m")
    max_walking_hours_per_day: float | None = Field(default=None, ge=1, le=12, description="Group's own walking-time limit per day, h")
    max_overnight_elevation_m: int | None = Field(default=None, ge=300, le=2100, description="Highest allowed night spot, m")


class DayLoad(BaseModel):
    """One day of a validated plan: copy km, ascent_m and walking_hours from a validate_hike_plan days[] entry."""

    model_config = ConfigDict(extra="ignore")  # extra fields of days[] (start, end, ...) may be passed and are ignored

    km: float = Field(ge=0, le=40, description="Distance of the day, km")
    ascent_m: int = Field(ge=0, le=3000, description="Total climb of the day, m")
    walking_hours: float = Field(ge=0, le=16, description="Pure walking time of the day, h")


# ------------------------------------------------------------------ find_routes
class RouteSummary(BaseModel):
    """A candidate route with its numbers and the night spots this section may use."""

    id: str = Field(description="Route id; pass it to the other tools")
    name: str = Field(description="Route name from OpenStreetMap")
    region: Region = Field(description="Mountain area")
    length_km: float = Field(description="Length along the trail, km")
    ascent_m: int = Field(description="Total climb from start to finish, m")
    descent_m: int = Field(description="Total descent from start to finish, m")
    max_elevation_m: int = Field(description="Highest point of the route, m")
    loop: bool = Field(description="True if the route ends where it starts")
    start: str = Field(description="Name of the start point")
    finish: str = Field(description="Name of the finish point")
    avg_km_per_day: float = Field(description="length_km / days")
    features: list[Feature] = Field(description="Kinds of points of interest on the route")
    overnight_options: list[OvernightOption] = Field(
        description="Up to 12 night spots allowed for this section, spread along the route (at least 1 km from "
                    "both ends). validate_hike_plan suggestions only name ids from this list. Empty for a one-day hike")
    fit_score: float = Field(ge=0, le=1, description="How well the length suits the daily distance, 1 = ideal")
    fit_reasons: list[str] = Field(description="Plain-language reasons behind the score")
    places_matched: list[str] = Field(default_factory=list, description="Which of the requested named places this route passes")


class FindRoutesResult(BaseModel):
    rules_status: str = Field(description="DRAFT or approved status of the rules used; tell it to the user")
    daily_km_target: float = Field(description="Daily distance the ranking aims for: the rules, or the group limit if stricter")
    daily_km_limit: float | None = Field(description="Hard daily limit: the section rule, or the group limit if stricter")
    total_candidates: int = Field(description="Routes in the dataset before filtering")
    routes: list[RouteSummary] = Field(description="Matching routes, best first. Empty list = nothing matched (not an error)")
    excluded: dict[str, int] = Field(description="Why other routes were filtered out: reason -> count")
    notes: list[str] = Field(description="Advice, e.g. what to relax when nothing matched")


# ------------------------------------------------------------------ validate_hike_plan
class Violation(BaseModel):
    """One broken rule or warning."""

    code: ViolationCode = Field(description="Stable rule code")
    severity: Severity = Field(description="error blocks the plan, warning does not")
    day: int | None = Field(description="Day number (1-based) or null for the whole plan")
    message: str = Field(description="What is wrong, with the numbers")
    limit: float | str | None = Field(description="The limit that applies, in the unit of the rule (km, m, h, HH:MM, people, days)")
    actual: float | str | None = Field(description="The plan's value, in the same unit as limit")
    source: str = Field(description="Where the limit comes from: rules.yaml source text or the group note")
    suggestion: str = Field(description="Concrete way to fix the plan; may name waypoint ids")


class OvernightCheck(BaseModel):
    """Checks of the night spot at the end of a day."""

    waypoint: Waypoint = Field(description="The night spot")
    water_within_m: int | None = Field(description="Distance along the route to the nearest water point, null if none within 300 m")
    ridge_risk: bool = Field(description="Night spot on a crest or top: near the highest point within ±1 km and the ground drops ≥30 m on both sides")


class DayPlan(BaseModel):
    """Numbers of one day, all computed from the route's elevation profile."""

    day: int = Field(description="Day number, 1-based")
    start: Waypoint = Field(description="Where the day starts")
    end: Waypoint = Field(description="Where the day ends (night spot or finish)")
    km: float = Field(description="Distance, km")
    ascent_m: int = Field(description="Total climb, m")
    descent_m: int = Field(description="Total descent, m")
    min_elevation_m: int = Field(description="Lowest point of the day, m")
    max_elevation_m: int = Field(description="Highest point of the day, m")
    walking_hours: float = Field(description="Pure walking time: flat pace + ascent time (coefficients in rules.yaml `pace`)")
    elapsed_hours: float = Field(description="Walking time plus rests")
    start_time: str = Field(description="Daily start, HH:MM")
    est_arrival: str = Field(description="Estimated arrival, HH:MM local time")
    sunset: str | None = Field(description="Local sunset at the day's end point, if a start date was given")
    highest_point: RoutePoint = Field(description="The day's highest point")
    overnight: OvernightCheck | None = Field(description="Check of the night spot after this day, null on the last day")


class ValidationResult(BaseModel):
    valid: bool = Field(description="True when there are no error-severity violations")
    rules_status: str = Field(description="DRAFT or approved status of the rules used; tell it to the user")
    route_id: str = Field(description="The checked route")
    route_name: str = Field(description="Route name from OpenStreetMap")
    section: Section = Field(description="Section the rules were applied for")
    days: list[DayPlan] = Field(description="One entry per hiking day")
    total_km: float = Field(description="Sum of the days, km")
    total_ascent_m: int = Field(description="Sum of the days' climbs, m")
    violations: list[Violation] = Field(description="Errors and warnings; empty when everything passes")
    dry_nights: int = Field(description="Night spots without water within the rules' radius; pass to calculate_supplies")
    applied_limits: dict[str, str] = Field(description="Limit name -> value and where it came from")
    documents: list[str] = Field(description="Paperwork checklist for this group")


# ------------------------------------------------------------------ assess_weather_risk
class Trigger(BaseModel):
    """A forecast value that crossed a threshold."""

    code: TriggerCode = Field(description="Which threshold was crossed")
    value: float = Field(description="Forecast value, in `unit`")
    threshold: float = Field(description="Threshold from rules.yaml `weather`, in `unit`")
    unit: TriggerUnit = Field(description="wmo_code for THUNDERSTORM/SNOW, m/s for WIND, mm for RAIN, °C for NIGHT_COLD")
    risk: RiskLevel = Field(description="Risk this trigger causes")
    recommendation: str = Field(description="What to change in the plan (Ukrainian)")
    source: str = Field(description="Source of the threshold")


class DayWeather(BaseModel):
    """Forecast and risk for one hiking day."""

    day: int = Field(description="Day number, 1-based")
    date: str = Field(description="YYYY-MM-DD")
    exposed: bool = Field(description="Day reaches above the tree line (rules: weather.treeline_m)")
    max_elevation_m: int = Field(description="Highest point of the day, m")
    weather_code: int = Field(description="WMO weather code from Open-Meteo")
    weather: str = Field(description="Plain-language WMO weather description")
    temp_min_c: float = Field(description="Daily minimum at the forecast point, °C")
    temp_max_c: float = Field(description="Daily maximum at the forecast point, °C")
    precipitation_mm: float = Field(description="Daily precipitation sum, mm")
    precipitation_probability_pct: float | None = Field(description="Max precipitation probability, %")
    wind_max_ms: float = Field(description="Max sustained wind at 10 m, m/s")
    gusts_max_ms: float | None = Field(description="Max gusts, m/s")
    night_spot_min_c: float | None = Field(description="Night minimum at the overnight elevation (lapse-rate adjusted), null on the last day")
    risk: RiskLevel = Field(description="Highest risk among the triggers; low if none")
    triggers: list[Trigger] = Field(description="Thresholds crossed on this day")


class WeatherResult(BaseModel):
    route_id: str = Field(description="The assessed route")
    start_date: str = Field(description="First hiking day, YYYY-MM-DD")
    mode: Literal["live", "replay"] = Field(description="live = fresh Open-Meteo call; replay = recorded genuine response from fixtures/")
    forecast_point: GeoPoint = Field(description="The route's highest point, where the forecast is taken")
    fetched_at: str = Field(description="When the forecast was downloaded (UTC)")
    days: list[DayWeather] = Field(description="One entry per hiking day")
    overall_risk: RiskLevel = Field(description="Highest daily risk; high means re-plan")
    coldest_night_c: float | None = Field(description="Lowest night_spot_min_c; pass to calculate_supplies as night_min_c")
    rules_status: str = Field(description="DRAFT or approved status of the thresholds")
    source: str = Field(description="Data source and licence")


# ------------------------------------------------------------------ calculate_supplies
class Assumption(BaseModel):
    """A coefficient used in the calculation."""

    name: str = Field(description="Key in rules.yaml `supplies`")
    value: float | str = Field(description="Value used")
    source: str = Field(description="Where the value comes from")


class PackEstimate(BaseModel):
    """Weight of one rucksack at the start of the hike."""

    personal_gear_kg: float | None = Field(description="Measured base gear without food and water; null if not given")
    food_kg: float = Field(description="Food for the whole hike, carried at the start")
    water_kg: float = Field(description="Water carried on the heaviest day: the flask, plus dry-night water if any night has no source")
    total_kg: float = Field(description="Peak weight: all food + carried water (+ personal gear if given)")
    limit_kg: float | None = Field(description="Strictest applicable limit: kg rule and % of body weight")
    limit_source: str = Field(description="Which rule the limit comes from")
    remaining_for_gear_kg: float | None = Field(description="Limit minus food and water: what is left for all other gear")
    within_limit: bool | None = Field(description="Null when the limit or the gear weight is unknown")


class SuppliesResult(BaseModel):
    people: int = Field(description="Participants + adults")
    num_days: int = Field(description="Number of hiking days")
    kcal_per_person_day: int = Field(description="Energy target for a participant of this section")
    food_kg_per_person: float = Field(description="Food per participant for the whole hike, kg")
    food_kg_group: float = Field(description="Food for the whole group, kg")
    water_l_per_person_by_day: list[float] = Field(description="Water need per person for each day, l")
    water_to_carry_for_dry_nights_l: float = Field(description="Extra water for nights without a source, whole group, l")
    gas_g_group: float = Field(description="Cooking gas for the whole hike, g")
    stoves: int = Field(description="Stoves (one per kitchen group)")
    pots: int = Field(description="Pots of at least 3 l")
    tents: int = Field(description="Tents of the given capacity")
    first_aid_kits: int = Field(description="Group first-aid kits")
    pack_participant: PackEstimate = Field(description="Start-of-hike pack of a participant")
    pack_adult: PackEstimate = Field(description="Start-of-hike pack of an adult leader")
    gear_additions: list[str] = Field(description="Extra gear for this plan (Ukrainian): cold nights, dry nights, long days")
    warnings: list[str] = Field(description="Problems to solve before the hike (Ukrainian, for the leader)")
    assumptions: list[Assumption] = Field(description="Every coefficient used, with its source")
