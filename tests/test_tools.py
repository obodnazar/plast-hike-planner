"""Tests for the plast-trail domain logic (no network: weather uses fixtures or synthetic data).

Run:  uv run pytest -q
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from plast_trail import weather
from plast_trail.errors import DomainError
from plast_trail.models import GroupLimits, RiskLevel
from plast_trail.routes import find_routes
from plast_trail.validation import fmt_hhmm, sunset_local_minutes, validate_hike_plan

ROUTE = "r8613349"  # с.Лазещина - г.Петрос - с.Ясіня, 33.3 km


# ------------------------------------------------------------------ find_routes
def test_find_routes_ranks_and_explains_exclusions():
    r = find_routes("yunatstvo", 2, "chornohora", ["lake"], 3)
    assert r.routes, "expected at least one route"
    assert [x.fit_score for x in r.routes] == sorted((x.fit_score for x in r.routes), reverse=True)
    assert all("lake" in x.features for x in r.routes)
    assert r.excluded.get("other region", 0) > 0


def test_find_routes_empty_is_a_result_not_an_error():
    r = find_routes("novatstvo", 7, "marmarosy", ["lake"], 5)
    assert r.routes == []
    assert r.excluded
    assert any("exceeds" in n for n in r.notes)  # 7 days > novatstvo max 2


def test_novatstvo_night_spots_are_only_huts_or_campsites():
    r = find_routes("novatstvo", 2, None, [], 10)
    kinds = {o.kind for x in r.routes for o in x.overnight_options}
    assert kinds <= {"hut", "campsite"}


# ------------------------------------------------------------------ validate_hike_plan
def test_too_long_day_suggests_an_even_split_with_shown_night_spots():
    v = validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w23", "w47"], None, None, GroupLimits(max_km_per_day=12))
    assert not v.valid
    km = [x for x in v.violations if x.code == "KM_PER_DAY_MAX"]
    assert km and km[0].limit == 12 and km[0].source.startswith("group note")
    assert "w30" in km[0].suggestion and "w56" in km[0].suggestion


def test_too_few_days_is_said_honestly():
    v = validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w47"], None, None, GroupLimits(max_km_per_day=12))
    km = [x for x in v.violations if x.code == "KM_PER_DAY_MAX"]
    assert "needs at least 3 days" in km[0].suggestion


def test_night_spot_at_the_very_start_is_refused():
    with pytest.raises(DomainError) as e:
        validate_hike_plan("r21241643", "yunatstvo", 10, 2, ["w1"], None, None, None)
    assert e.value.code == "WAYPOINT_ORDER"


def test_following_the_suggestions_gives_a_valid_plan():
    v = validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w30", "w56"], None, None, GroupLimits(max_km_per_day=12))
    assert v.valid
    assert len(v.days) == 3
    assert v.total_km == pytest.approx(33.3, abs=0.2)
    assert all(d.km <= 12 for d in v.days)


def test_adult_rules():
    v = validate_hike_plan(ROUTE, "yunatstvo", 10, 1, ["w30", "w56"], None, None, None)
    codes = {x.code for x in v.violations}
    assert {"TOO_FEW_ADULTS", "ADULT_RATIO"} <= codes


def test_waypoint_errors():
    with pytest.raises(DomainError) as e:
        validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w56", "w30"], None, None, None)
    assert e.value.code == "WAYPOINT_ORDER"
    with pytest.raises(DomainError) as e:
        validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w999"], None, None, None)
    assert e.value.code == "WAYPOINT_NOT_FOUND"
    with pytest.raises(DomainError) as e:
        validate_hike_plan("r1", "yunatstvo", 10, 2, [], None, None, None)
    assert e.value.code == "ROUTE_NOT_FOUND"


def test_sunset_is_close_to_reality():
    # Kyiv, summer solstice: real sunset is about 21:12 local time.
    m = sunset_local_minutes(date(2026, 6, 21), 50.45, 30.52)
    assert abs(m - (21 * 60 + 12)) <= 5, fmt_hhmm(m)


def test_error_payload_is_json_with_code():
    err = DomainError("ROUTE_NOT_FOUND", "No route", "hint").to_tool_error()
    assert json.loads(str(err))["error"]["code"] == "ROUTE_NOT_FOUND"


# ------------------------------------------------------------------ assess_weather_risk
def synthetic_body(start: str, n: int, **overrides):
    """An Open-Meteo-shaped daily block; same parser as live and replay."""
    days = [(date.fromisoformat(start) + timedelta(days=i)).isoformat() for i in range(n)]
    base = {"weather_code": 1, "temperature_2m_max": 15.0, "temperature_2m_min": 5.0, "precipitation_sum": 0.0,
            "precipitation_probability_max": 10, "wind_speed_10m_max": 5.0, "wind_gusts_10m_max": 9.0}
    base.update(overrides)
    return {"daily": {"time": days, **{k: [v] * n for k, v in base.items()}}}


@pytest.fixture
def replay_with(monkeypatch):
    def install(body):
        monkeypatch.setenv("WEATHER_MODE", "replay")
        monkeypatch.setattr(weather, "load_fixture", lambda route_id, needed: (body, "2026-11-14T00:00:00+00:00"))
    return install


def test_thunderstorm_above_treeline_is_high(replay_with):
    replay_with(synthetic_body("2026-07-10", 4, weather_code=95))
    r = weather.assess_weather_risk(ROUTE, date(2026, 7, 10), ["w30", "w56"])
    exposed_days = [d for d in r.days if d.exposed]
    assert exposed_days and all(d.risk == RiskLevel.high for d in exposed_days)
    assert r.overall_risk == RiskLevel.high


def test_calm_summer_is_low(replay_with):
    replay_with(synthetic_body("2026-07-10", 4))
    r = weather.assess_weather_risk(ROUTE, date(2026, 7, 10), ["w30", "w56"])
    assert r.overall_risk == RiskLevel.low


def test_night_temperature_uses_lapse_rate(replay_with):
    replay_with(synthetic_body("2026-07-10", 4, temperature_2m_min=-1.0))
    r = weather.assess_weather_risk(ROUTE, date(2026, 7, 10), ["w30", "w56"])
    first_night = r.days[0].night_spot_min_c
    # w30 is ~1100 m below the forecast point, so the night there is ~7 °C warmer than at the top.
    assert first_night > 5


def test_replay_of_recorded_fixture_matches_structure(monkeypatch):
    files = sorted(weather.FIXTURES.glob(f"{ROUTE}_*.json"))
    if not files:
        pytest.skip("no recorded fixture")
    rec = json.loads(files[-1].read_text(encoding="utf-8"))
    first = rec["response"]["daily"]["time"][1]
    monkeypatch.setenv("WEATHER_MODE", "replay")
    r = weather.assess_weather_risk(ROUTE, date.fromisoformat(first), ["w30", "w56"])
    assert r.mode == "replay" and len(r.days) == 3
    assert r.coldest_night_c == min(d.night_spot_min_c for d in r.days if d.night_spot_min_c is not None)


# ------------------------------------------------------------------ calculate_supplies
from plast_trail.models import DayLoad  # noqa: E402
from plast_trail.supplies import calculate_supplies  # noqa: E402


def _days():
    return [DayLoad(km=11.3, ascent_m=201, walking_hours=3.3), DayLoad(km=11.0, ascent_m=1087, walking_hours=5.2),
            DayLoad(km=11.0, ascent_m=31, walking_hours=2.8)]


def test_supplies_numbers_trace_to_coefficients():
    s = calculate_supplies("yunatstvo", 10, 2, _days(), 0, None, 3, None, None, None)
    assert s.people == 12 and s.num_days == 3
    assert s.food_kg_group == pytest.approx(3 * 12 * 0.9)          # 0.9 kg/person/day
    assert s.gas_g_group == 12 * 3 * 25                            # 25 g/person/day, no frost
    assert s.stoves == 2 and s.pots == 6 and s.tents == 4 and s.first_aid_kits == 2
    assert all(w >= 3.0 for w in s.water_l_per_person_by_day)     # 3 l/day floor
    names = {a.name for a in s.assumptions}
    assert "food_kg_per_person_day.yunatstvo" in names and "gas_g_per_person_day" in names


def test_cold_night_raises_gas_and_adds_gear():
    s = calculate_supplies("yunatstvo", 10, 2, _days(), 1, -3.0, 3, None, None, None)
    assert s.gas_g_group == round(12 * 3 * 25 * 1.5)
    assert any("зимові спальники" in g for g in s.gear_additions)
    assert s.water_to_carry_for_dry_nights_l == pytest.approx(12 * 1.5)


def test_water_for_every_dry_night_is_in_the_pack():
    """Codex P2: two dry nights in a row, but the pack weight counted water for one."""
    one = calculate_supplies("yunatstvo", 10, 2, _days(), 1, None, 3, None, None, None)
    two = calculate_supplies("yunatstvo", 10, 2, _days(), 2, None, 3, None, None, None)
    assert one.pack_participant.water_kg == pytest.approx(1.0 + 1.5)
    assert two.pack_participant.water_kg == pytest.approx(1.0 + 2 * 1.5)
    assert two.water_to_carry_for_dry_nights_l == pytest.approx(2 * 12 * 1.5)
    assert any("всі ці ночівлі" in w for w in two.warnings)


def test_pack_check_uses_measured_gear_only():
    no_gear = calculate_supplies("yunatstvo", 10, 2, _days(), 0, None, 3, None, None, None)
    assert no_gear.pack_participant.within_limit is None           # no invented base weight
    assert no_gear.pack_participant.remaining_for_gear_kg == pytest.approx(9 - (2.7 + 1.0))
    heavy = calculate_supplies("yunatstvo", 10, 2, _days(), 0, None, 3, None, 6.0, None)
    assert heavy.pack_participant.within_limit is False            # 2.7 + 1.0 + 6.0 > 9


# ------------------------------------------------------------------ P3 coverage
from plast_trail.data import load_routes  # noqa: E402
from plast_trail.validation import ridge_risk  # noqa: E402


def test_ridge_warning_needs_a_real_top():
    r = load_routes()[ROUTE]
    w23 = next(w for w in r["waypoints"] if w["id"] == "w23")      # campsite in a valley on a long climb
    assert not ridge_risk(r, w23["km"])
    flagged = sum(ridge_risk(x, w["km"]) for x in load_routes().values() for w in x["waypoints"]
                  if w["kind"] in ("hut", "campsite", "shelter", "water", "lake"))
    total = sum(1 for x in load_routes().values() for w in x["waypoints"]
                if w["kind"] in ("hut", "campsite", "shelter", "water", "lake"))
    assert flagged / total < 0.15                                  # was ~30 % before the fix


def test_novatstvo_rules_apply():
    r = find_routes("novatstvo", 2, None, [], 10)
    assert r.daily_km_limit == 10
    route = next(x for x in r.routes if x.overnight_options)
    v = validate_hike_plan(route.id, "novatstvo", 8, 2, [route.overnight_options[0].id], None, None, None)
    assert {"ADULT_RATIO", "TOO_FEW_ADULTS"}.isdisjoint({x.code for x in v.violations})   # 8 children, 2 adults ok
    v = validate_hike_plan(route.id, "novatstvo", 11, 2, [route.overnight_options[0].id], None, None, None)
    assert "ADULT_RATIO" in {x.code for x in v.violations}                                # 11 > 2 × 5


def test_senior_has_no_adult_ratio_and_one_day_hike_works():
    v = validate_hike_plan(ROUTE, "senior", 6, 0, [], None, None, None)
    assert len(v.days) == 1 and v.days[0].km == pytest.approx(33.3, abs=0.2)
    codes = {x.code for x in v.violations}
    assert "ADULT_RATIO" not in codes and "TOO_FEW_ADULTS" not in codes


def test_sunset_follows_the_dst_change():
    # Kyiv leaves summer time on 25 Oct 2026: the sunset clock time jumps back by about an hour.
    before = sunset_local_minutes(date(2026, 10, 24), 50.45, 30.52)
    after = sunset_local_minutes(date(2026, 10, 26), 50.45, 30.52)
    assert 50 <= before - after <= 70


def test_missing_forecast_value_is_an_error_not_a_zero(replay_with):
    body = synthetic_body("2026-07-10", 4)
    body["daily"]["precipitation_sum"][1] = None
    replay_with(body)
    with pytest.raises(DomainError) as e:
        weather.assess_weather_risk(ROUTE, date(2026, 7, 10), ["w30", "w56"])
    assert e.value.code == "WEATHER_UNAVAILABLE" and "precipitation_sum" in e.value.message


def test_malformed_forecast_body_is_an_error(replay_with):
    replay_with({"hourly": {}})
    with pytest.raises(DomainError) as e:
        weather.assess_weather_risk(ROUTE, date(2026, 7, 10), ["w30", "w56"])
    assert e.value.code == "WEATHER_UNAVAILABLE"


def test_pack_uses_body_weight_and_dry_night_water():
    s = calculate_supplies("yunatstvo", 10, 2, _days(), 0, None, 3, 40, 5.0, None)
    assert s.pack_participant.limit_kg == 8.0                       # min(9 kg, 20 % of 40 kg)
    assert s.pack_participant.within_limit is False                 # 2.7 food + 1 water + 5 gear
    dry = calculate_supplies("yunatstvo", 10, 2, _days(), 1, None, 3, None, None, None)
    assert dry.pack_participant.water_kg == pytest.approx(1.0 + 1.5)


def test_group_km_limit_narrows_the_search():
    loose = find_routes("yunatstvo", 2, "chornohora", [], 10)
    tight = find_routes("yunatstvo", 2, "chornohora", [], 10, group_max_km_per_day=8)
    assert tight.daily_km_limit == 8
    assert all(x.length_km <= 16 for x in tight.routes)                       # 2 days × 8 km
    assert tight.excluded.get("too long for the days", 0) > loose.excluded.get("too long for the days", 0)



def test_unknown_place_names_are_ignored_not_fatal():
    r = find_routes("yunatstvo", 2, "chornohora", [], 5, None, ["mountain lake"])
    assert r.routes, "an unknown name must not wipe out all routes"
    assert any("not a known place name" in n for n in r.notes)


def test_a_place_on_no_route_ranks_routes_by_distance():
    # the leader's test of 15 Nov: «бребенеско» is on no route, so the plan went to Кукул far away
    r = find_routes("yunatstvo", 2, "chornohora", [], 4, None, ["Бребенеско"])
    assert r.routes and all("Бребенескул" in x.distance_to_places_km for x in r.routes)
    assert all(x.distance_to_places_km["Бребенескул"] < 8 for x in r.routes)
    assert any("No route in the dataset passes Бребенескул" in n for n in r.notes)


# ------------------------------------------------------------------ resolve_places (gazetteer)
from plast_trail.places import resolve_places  # noqa: E402


def _one(name: str, region: str | None = None):
    return resolve_places([name], region).results[0]


def test_misspelt_names_are_corrected():
    r = _one("піпівана")                              # the leader's real request, 15 Nov
    assert r.status == "corrected" and r.best.name == "Піп Іван Чорногірський"
    assert r.best.region.value == "chornohora" and r.best.on_routes >= 1
    assert _one("Піпіван").best.name == "Піп Іван Чорногірський"


def test_case_forms_and_prefixes_are_found_as_written():
    assert _one("Говерлу").status == "found" and _one("Говерлу").best.name == "Говерла"
    assert _one("озеро Несамовите").best.name == "Несамовите озеро"
    assert _one("петроса").best.name == "Петрос"                       # Chornohora's, the highest of three


def test_region_decides_between_namesakes():
    assert _one("Петрос", "marmarosy").best.name == "Петрос Мармароський"
    assert {a.name for a in _one("Петрос").alternatives} >= {"Петрос Мармароський", "Петрос Ґорґанський"}


def test_known_place_on_no_route_is_said_and_unknown_is_not_an_error():
    r = _one("бребенеско")
    assert r.best.name == "Бребенескул" and r.best.on_routes == 0   # the data gap the leader hit: on no route
    assert _one("mountain lake").status == "not_found"


def test_a_plan_that_ends_on_a_summit_is_warned():
    # the leader's run of 16 Nov: «Лазещина-Говерла» ends high, the descent was nowhere in the plan
    v = validate_hike_plan("r8613461", "senior", 3, 0, ["w13"], date(2026, 11, 28), None, None)
    top = [x for x in v.violations if x.code == "ENDS_ON_TOP"]
    assert top and top[0].severity.value == "warning" and top[0].day == 2
    assert "about" in top[0].suggestion and "h with rests" in top[0].suggestion
    assert v.valid                                   # a warning, the plan stays valid
    v = validate_hike_plan(ROUTE, "yunatstvo", 10, 2, ["w30", "w56"], None, None, None)
    assert "ENDS_ON_TOP" not in [x.code for x in v.violations]   # ends in Ясіня, a village


def test_common_rule_severity_comes_from_the_rules_file(monkeypatch):
    """Codex P3: changing arrive_hours_before_sunset to severity: error in rules.yaml had no effect."""
    import copy
    from plast_trail import data, validation
    rules = copy.deepcopy(data.load_rules())
    rules["common"]["arrive_hours_before_sunset"]["severity"] = "error"
    monkeypatch.setattr(data, "load_rules", lambda: rules)
    monkeypatch.setattr(validation, "load_rules", lambda: rules, raising=False)
    from datetime import date
    v = validation.validate_hike_plan("r8613349", "yunatstvo", 10, 2, ["w30", "w56"], date(2026, 11, 23), None, None)
    late = [x for x in v.violations if x.code == "ARRIVAL_AFTER_SAFE_TIME"]
    assert late and all(x.severity == "error" for x in late) and not v.valid


def test_gusts_count_as_wind(monkeypatch):
    """The leader's decision (16.11.2026): 40 m/s gusts with 5 m/s steady wind are not «low»."""
    from datetime import date
    from plast_trail import weather
    real = weather.day_values

    def gusty(fc, d):
        v = dict(real(fc, d))
        v.update(weather_code=1, precipitation_sum=0.0, wind_speed_10m_max=5.0, wind_gusts_10m_max=40.0,
                 temperature_2m_min=5.0)
        return v
    monkeypatch.setattr(weather, "day_values", gusty)
    r = weather.assess_weather_risk("r8613349", date(2026, 11, 23), ["w30", "w56"])
    wind = [t for d in r.days for t in d.triggers if t.code == "WIND"]
    assert wind and wind[0].value == 40.0 and "Пориви до 40" in wind[0].recommendation
    assert getattr(r.overall_risk, "value", r.overall_risk) == "high"   # the route goes above the tree line
