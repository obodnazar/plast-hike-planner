"""assess_weather_risk: Open-Meteo forecast at the route's highest point -> risk per day.

Modes (env WEATHER_MODE):
  live    - call Open-Meteo (no API key; free tier 600 calls/min, 10 000/day);
  record  - same as live, and save the genuine response to fixtures/open-meteo/;
  replay  - no network: read a recorded response from fixtures/open-meteo/.

Replay goes through exactly the same parse_forecast() and scoring as live,
so a fixture is only a different *source* of the same bytes.

Rate limiting: at least 1 s between calls, and a 30-minute in-memory cache
per route, so a re-planning loop does not hit the API again.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from plast_trail.data import PROJECT_ROOT, get_route, load_rules, rules_status
from plast_trail.errors import (
    DATE_OUT_OF_RANGE,
    INVALID_INPUT,
    WEATHER_FIXTURE_MISSING,
    WEATHER_UNAVAILABLE,
    DomainError,
)
from plast_trail.models import DayWeather, RiskLevel, Trigger, WeatherResult
from plast_trail.validation import day_stats, resolve_night_spots

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FIXTURES = Path(os.environ.get("PLAST_TRAIL_FIXTURES", PROJECT_ROOT / "fixtures" / "open-meteo"))
FORECAST_DAYS = 16
DAILY_VARS = [
    "weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max",
]
LAPSE_C_PER_KM = 6.5  # standard atmosphere: temperature drops ~6.5 °C per 1000 m
KYIV = ZoneInfo("Europe/Kyiv")
USER_AGENT = "plast-hike-planner-student-project/0.1"

WMO = {
    0: "ясно", 1: "переважно ясно", 2: "мінлива хмарність", 3: "хмарно", 45: "туман", 48: "туман з інеєм",
    51: "слабка мряка", 53: "мряка", 55: "сильна мряка", 56: "крижана мряка", 57: "сильна крижана мряка",
    61: "слабкий дощ", 63: "дощ", 65: "сильний дощ", 66: "крижаний дощ", 67: "сильний крижаний дощ",
    71: "слабкий сніг", 73: "сніг", 75: "сильний сніг", 77: "снігова крупа",
    80: "слабкі зливи", 81: "зливи", 82: "сильні зливи", 85: "снігопад", 86: "сильний снігопад",
    95: "гроза", 96: "гроза з градом", 99: "гроза з сильним градом",
}

_last_call = 0.0
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_SECONDS = 30 * 60


def mode() -> str:
    m = os.environ.get("WEATHER_MODE", "replay").strip().lower()
    if m not in ("live", "record", "replay"):
        raise DomainError(INVALID_INPUT, f"WEATHER_MODE '{m}' is not live, record or replay")
    return m


def forecast_point(route: dict[str, Any]) -> dict[str, float]:
    top = max(route["profile"], key=lambda p: p[3])
    return {"lat": top[1], "lon": top[2], "elevation_m": top[3]}


def request_params(point: dict[str, float]) -> dict[str, Any]:
    return {
        "latitude": point["lat"], "longitude": point["lon"], "elevation": point["elevation_m"],
        "daily": ",".join(DAILY_VARS), "wind_speed_unit": "ms", "timezone": "Europe/Kyiv",
        "forecast_days": FORECAST_DAYS,
    }


# ------------------------------------------------------------------ sources
def fetch_live(route_id: str, point: dict[str, float]) -> tuple[dict[str, Any], str]:
    """Call Open-Meteo (with throttle, cache and two retries). Returns (body, fetched_at)."""
    global _last_call
    cached = _cache.get(route_id)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1], datetime.fromtimestamp(cached[0], timezone.utc).isoformat(timespec="seconds")
    params = request_params(point)
    last_error = ""
    for attempt in range(3):
        wait = 1.0 - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = httpx.get(FORECAST_URL, params=params, timeout=15, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code == 200:
                body = r.json()
                fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _cache[route_id] = (time.time(), body)
                if mode() == "record":
                    save_fixture(route_id, params, body, fetched)
                return body, fetched
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        time.sleep(2 ** attempt)
    raise DomainError(WEATHER_UNAVAILABLE, f"Open-Meteo did not answer: {last_error}",
                      "Retry later, or run the server with WEATHER_MODE=replay to use recorded forecasts.")


def save_fixture(route_id: str, params: dict[str, Any], body: dict[str, Any], fetched: str) -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    first_day = body["daily"]["time"][0]
    path = FIXTURES / f"{route_id}_{first_day}.json"
    path.write_text(json.dumps({"recorded_at": fetched, "request": {"url": FORECAST_URL, "params": params},
                                "response": body}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_fixture(route_id: str, needed: list[str]) -> tuple[dict[str, Any], str]:
    """Newest recorded response for this route whose days cover all needed dates."""
    files = sorted(FIXTURES.glob(f"{route_id}_*.json"), reverse=True)
    ranges = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        days = rec["response"]["daily"]["time"]
        ranges.append(f"{days[0]}..{days[-1]}")
        if all(d in days for d in needed):
            return rec["response"], rec["recorded_at"]
    raise DomainError(
        WEATHER_FIXTURE_MISSING,
        f"No recorded forecast for {route_id} covering {needed[0]}..{needed[-1]}.",
        ("Recorded ranges for this route: " + ", ".join(ranges)) if ranges
        else "No fixtures for this route. Record one: WEATHER_MODE=record, or use scripts/record_weather.py.",
    )


# ------------------------------------------------------------------ scoring
def parse_forecast(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Open-Meteo daily arrays -> {date: {variable: value}}."""
    daily = body["daily"]
    return {d: {v: daily[v][i] for v in DAILY_VARS} for i, d in enumerate(daily["time"])}


def worst(levels: list[RiskLevel]) -> RiskLevel:
    order = [RiskLevel.low, RiskLevel.medium, RiskLevel.high]
    return max(levels, key=order.index, default=RiskLevel.low)


def assess_weather_risk(route_id: str, start_date: date, overnight_waypoint_ids: list[str]) -> WeatherResult:
    route = get_route(route_id)
    spots = resolve_night_spots(route, overnight_waypoint_ids)
    stops = [route["waypoints"][0], *spots, route["waypoints"][-1]]
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(len(stops))]  # +1 day for the last night
    needed = dates[: len(stops) - 1]
    point = forecast_point(route)
    m = mode()

    if m == "replay":
        body, fetched = load_fixture(route_id, needed)
    else:
        today = datetime.now(KYIV).date()
        last_ok = today + timedelta(days=FORECAST_DAYS - 1)
        if start_date < today or date.fromisoformat(needed[-1]) > last_ok:
            raise DomainError(DATE_OUT_OF_RANGE,
                              f"Forecast covers {today}..{last_ok}; plan needs {needed[0]}..{needed[-1]}.",
                              "Pick dates inside the forecast window.")
        body, fetched = fetch_live(route_id, point)

    fc = parse_forecast(body)
    w = {k: v["value"] for k, v in load_rules()["weather"].items()}
    src = {k: v["source"] for k, v in load_rules()["weather"].items()}

    days: list[DayWeather] = []
    for i, (a, b) in enumerate(zip(stops, stops[1:]), start=1):
        d = dates[i - 1]
        f = fc[d]
        s = day_stats(route, a, b)
        exposed = s["max"] >= w["treeline_m"]
        triggers: list[Trigger] = []

        if f["weather_code"] in w["thunderstorm_codes"]:
            lvl = RiskLevel.high if exposed else RiskLevel.medium
            triggers.append(Trigger(code="THUNDERSTORM", value=f["weather_code"], threshold=95, risk=lvl,
                                    recommendation=("Не виходити на хребет і відкриті вершини в час грози; "
                                                    "перенести хребтову ділянку або обрати лісовий варіант."),
                                    source=src["thunderstorm_codes"] + "; " + src["wind_medium_ms"].split(";")[0]))
        if f["weather_code"] in w["snow_codes"]:
            triggers.append(Trigger(code="SNOW", value=f["weather_code"], threshold=71,
                                    risk=RiskLevel.high if exposed else RiskLevel.medium,
                                    recommendation=("Сніг на маршруті: зимове спорядження й досвід зимових походів; "
                                                    "вище межі лісу — не виходити без зимової підготовки."),
                                    source=src["snow_codes"]))
        wind = f["wind_speed_10m_max"] or 0.0
        if wind >= w["wind_high_ms"]:
            triggers.append(Trigger(code="WIND", value=wind, threshold=w["wind_high_ms"],
                                    risk=RiskLevel.high if exposed else RiskLevel.medium,
                                    recommendation="Не виходити вище межі лісу; перенести високогірний день.",
                                    source=src["wind_high_ms"]))
        elif wind >= w["wind_medium_ms"] and exposed:
            triggers.append(Trigger(code="WIND", value=wind, threshold=w["wind_medium_ms"], risk=RiskLevel.medium,
                                    recommendation="Пройти ділянку нижче межі лісу або скоротити день на хребті.",
                                    source=src["wind_medium_ms"]))
        rain = f["precipitation_sum"] or 0.0
        if rain >= w["rain_high_mm"]:
            triggers.append(Trigger(code="RAIN", value=rain, threshold=w["rain_high_mm"], risk=RiskLevel.high,
                                    recommendation="Перенести або скасувати цей день.", source=src["rain_high_mm"]))
        elif rain >= w["rain_medium_mm"]:
            triggers.append(Trigger(code="RAIN", value=rain, threshold=w["rain_medium_mm"], risk=RiskLevel.medium,
                                    recommendation="Без бродів; не ставати на ночівлю біля потоків і в руслах.",
                                    source=src["rain_medium_mm"]))

        night_min = None
        if i < len(stops) - 1:
            next_day = fc.get(dates[i]) or f  # night after day i ~ minimum of the next morning
            night_min = round(next_day["temperature_2m_min"]
                              + LAPSE_C_PER_KM * (point["elevation_m"] - b["elevation_m"]) / 1000, 1)
            if night_min <= w["night_cold_c"]:
                triggers.append(Trigger(code="NIGHT_COLD", value=night_min, threshold=w["night_cold_c"],
                                        risk=RiskLevel.medium,
                                        recommendation="Зимові спальники й каремати, досвід холодних ночівель; або нижча ночівля.",
                                        source=src["night_cold_c"]))

        days.append(DayWeather(
            day=i, date=d, exposed=exposed, max_elevation_m=s["max"], weather_code=f["weather_code"],
            weather=WMO.get(f["weather_code"], f"код {f['weather_code']}"),
            temp_min_c=f["temperature_2m_min"], temp_max_c=f["temperature_2m_max"], precipitation_mm=rain,
            precipitation_probability_pct=f.get("precipitation_probability_max"),
            wind_max_ms=wind, gusts_max_ms=f.get("wind_gusts_10m_max"), night_spot_min_c=night_min,
            risk=worst([t.risk for t in triggers]), triggers=triggers,
        ))

    return WeatherResult(
        route_id=route_id, start_date=start_date.isoformat(), mode=m, forecast_point=point, fetched_at=fetched,
        days=days, overall_risk=worst([d.risk for d in days]), rules_status=rules_status(),
        source="Open-Meteo forecast API (CC BY 4.0), daily values at the route's highest point",
    )
