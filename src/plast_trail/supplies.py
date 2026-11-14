"""calculate_supplies: food, water, fuel, group gear and pack weight for a validated plan.

Inputs are the per-day loads from validate_hike_plan and, optionally, the coldest
night from assess_weather_risk. Every coefficient comes from rules.yaml `supplies`
(research §7.3) and is echoed back in `assumptions`, so each number can be traced.

There is no default base-gear weight (no source exists). If the leader gives a
measured one, the pack is checked against the limit; if not, the result says how
much of the limit is left for gear after food and water.
"""

from __future__ import annotations

import math

from plast_trail.data import load_rules, section_rule
from plast_trail.errors import DATA_UNAVAILABLE, DomainError
from plast_trail.models import Assumption, DayLoad, PackEstimate, SuppliesResult


def _entry(name: str) -> dict:
    block = load_rules().get("supplies") or {}
    if name not in block or block[name].get("value") is None:
        raise DomainError(DATA_UNAVAILABLE, f"Supply coefficient '{name}' is missing in rules.yaml.",
                          "Add it under `supplies:` with a value and a source.")
    return block[name]


def calculate_supplies(section: str, participants: int, adults: int, days: list[DayLoad], dry_nights: int,
                       night_min_c: float | None, tent_capacity: int, lightest_body_weight_kg: float | None,
                       participant_base_gear_kg: float | None, adult_base_gear_kg: float | None) -> SuppliesResult:
    n_days = len(days)
    people = participants + adults
    used: dict[str, Assumption] = {}

    def coef(name: str, key: str | None = None) -> float:
        e = _entry(name)
        v = e["value"][key] if key else e["value"]
        label = f"{name}.{key}" if key else name
        used[label] = Assumption(name=label, value=v, source=e["source"])
        return float(v)

    adult_key = "senior"
    food_p = coef("food_kg_per_person_day", section)
    food_a = coef("food_kg_per_person_day", adult_key)
    food_group = round(n_days * (participants * food_p + adults * food_a), 1)
    kcal = int(coef("kcal_per_person_day", section))

    per_hour = coef("water_l_per_walking_hour")
    camp = coef("water_l_camp_per_day")
    floor = coef("water_l_min_per_day")
    water_by_day = [round(max(floor, camp + per_hour * d.walking_hours), 1) for d in days]
    per_dry_night = coef("water_l_per_person_dry_night")
    water_dry = round(dry_nights * people * per_dry_night, 1)
    start_water = coef("water_l_carried_at_start")

    cold = night_min_c is not None and night_min_c <= load_rules()["weather"]["night_cold_c"]["value"]
    gas_factor = coef("gas_cold_factor") if cold else 1.0
    gas = round(people * n_days * coef("gas_g_per_person_day") * gas_factor)
    stoves = math.ceil(people / coef("people_per_stove"))
    pots = stoves * int(coef("pots_per_stove"))
    kits = math.ceil(people / coef("people_per_first_aid_kit"))
    tents = math.ceil(people / tent_capacity)

    # Heaviest moment: all the food at the start, the flask, and (if any night is dry) the extra water
    # filled at the last source before that night.
    carried_water = round(start_water + (per_dry_night if dry_nights else 0.0), 1)

    def pack(rule_section: str, food_per_day: float, base: float | None, body_kg: float | None) -> PackEstimate:
        food = round(n_days * food_per_day, 1)
        consumables = round(food + carried_water, 1)
        kg = section_rule(rule_section, "pack_weight_max_kg")
        pct = section_rule(rule_section, "pack_weight_max_pct_body")
        limits = []
        if kg["value"]:
            limits.append((float(kg["value"]), f"{kg['value']} kg: {kg['source']}"))
        if pct["value"] and body_kg:
            limits.append((round(pct["value"] / 100 * body_kg, 1),
                           f"{pct['value']}% of {body_kg:g} kg body weight: {pct['source']}"))
        limit, source = min(limits, default=(None, "no limit known (give body weight for the % rule)"))
        total = round(consumables + (base or 0.0), 1)
        return PackEstimate(
            personal_gear_kg=base, food_kg=food, water_kg=carried_water, total_kg=total, limit_kg=limit,
            limit_source=source,
            remaining_for_gear_kg=None if limit is None else round(limit - consumables, 1),
            within_limit=None if (limit is None or base is None) else total <= limit,
        )

    member = pack(section, food_p, participant_base_gear_kg, lightest_body_weight_kg)
    adult = pack(adult_key, food_a, adult_base_gear_kg, None)

    warnings: list[str] = []
    if member.remaining_for_gear_kg is not None and member.remaining_for_gear_kg <= 0:
        warnings.append(f"Самі їжа й вода ({member.food_kg + member.water_kg:.1f} кг) вже досягають ліміту учасника "
                        f"{member.limit_kg} кг. Частину їжі мають нести дорослі, потрібне поповнення запасів "
                        "або коротша мандрівка.")
    elif member.within_limit is False:
        warnings.append(f"Наплічник учасника {member.total_kg} кг перевищує ліміт {member.limit_kg} кг "
                        f"({member.limit_source}). Перекладіть частину їжі чи групового спорядження дорослим.")
    if participant_base_gear_kg is None:
        warnings.append("Вагу спорядження не вказано (джерела для типового значення немає): зважте зібраний "
                        "наплічник без їжі й води і передайте participant_base_gear_kg.")
    if dry_nights:
        warnings.append(f"Ночівель без води поруч: {dry_nights}. Групі треба нести додатково {water_dry} л "
                        f"({per_dry_night:g} л на людину на ночівлю); це враховано у вазі наплічника.")

    additions = [f"{pots} казанки ≥3 л, групова ємність для води 5–6 л [S2][S13]"]
    if cold:
        additions += [f"Ніч до {night_min_c} °C: зимові спальники (комфорт нижче {night_min_c} °C) й утеплені каремати",
                      "Запасні рукавиці, шапка, термобілизна",
                      f"Газу ×{gas_factor:g} (холод); нижче −12 °C лише пропан-бутанова суміш або балон у теплі"]
    if dry_nights:
        additions.append(f"Ємності для води на {round(per_dry_night + start_water, 1)} л на людину")
    if any(d.walking_hours > 6 for d in days):
        additions.append("Налобні ліхтарі з запасними батареями: довгий ходовий день")

    return SuppliesResult(
        people=people, num_days=n_days, kcal_per_person_day=kcal, food_kg_per_person=round(n_days * food_p, 1),
        food_kg_group=food_group, water_l_per_person_by_day=water_by_day, water_to_carry_for_dry_nights_l=water_dry,
        gas_g_group=gas, stoves=stoves, pots=pots, tents=tents, first_aid_kits=kits,
        pack_participant=member, pack_adult=adult, gear_additions=additions, warnings=warnings,
        assumptions=list(used.values()),
    )
