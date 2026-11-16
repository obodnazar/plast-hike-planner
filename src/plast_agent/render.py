"""Turn the final workflow state into a human-readable Obsidian note (Markdown).

Every number in the note is copied from a tool result; nothing here computes domain values.
"""

from __future__ import annotations

import re
from typing import Any

from plast_agent.understanding import REGION_UA
from plast_agent.picture import MAP_CREDITS, day_track, leaflet_block, mapy_link, route_track, side_trip_link

STATUS = {
    "ok": "✅ План готовий",
    "attention": "⚠️ План валідний, але є попередження: перегляньте їх перед виходом",
    "risky": "⚠️ План складено, але НЕ рекомендовано без змін (погода)",
    "group": "✅ План готовий · за можливості візьміть ще дорослого на супровід",
    "failed": "❌ Валідного плану скласти не вдалося",
}
SECTION_UA = {"novatstvo": "новацтво", "yunatstvo": "юнацтво", "senior": "старші / сеніори"}
FEATURE_UA = {"lake": "озеро", "peak": "вершина", "water": "джерело", "hut": "хатина", "shelter": "притулок",
              "campsite": "табірне місце"}

# Tool messages are English (for the model); the note for the leader is Ukrainian, built from the stable codes.
VIOLATION_UA = {
    "KM_PER_DAY_MAX": "День {day}: {actual} км, ліміт {limit} км.",
    "KM_PER_DAY_WARN": "День {day}: {actual} км, більше за комфортні {limit} км.",
    "ASCENT_PER_DAY_MAX": "День {day}: набір висоти {actual} м, ліміт {limit} м.",
    "WALKING_HOURS_MAX": "День {day}: {actual} год ходу, ліміт {limit} год.",
    "ARRIVAL_AFTER_SAFE_TIME": "День {day}: прихід о {actual}, а безпечно — до {limit} (за 3 год до заходу сонця).",
    "ENDS_ON_TOP": "День {day}: маршрут закінчується на висоті {actual} м, а спуску в плані немає — заплануйте дорогу вниз.",
    "OVERNIGHT_KIND_NOT_ALLOWED": "Ночівля після дня {day}: тип «{actual}» для цієї групи не дозволено (можна: {limit}).",
    "OVERNIGHT_TOO_HIGH": "Ночівля після дня {day} на {actual} м, а ліміт групи {limit} м.",
    "RIDGE_OVERNIGHT": "Ночівля після дня {day} ({actual} м) схожа на гребінь або вершину.",
    "OVERNIGHT_NO_WATER": "Біля ночівлі після дня {day} немає води в межах 300 м.",
    "TOO_MANY_DAYS": "{actual} дн. мандрівки, ліміт для групи {limit} дн.",
    "TOO_FEW_ADULTS": "Дорослих {actual}, а потрібно щонайменше {limit}.",
    "ADULT_RATIO": "Замало дорослих на таку кількість учасників: є {actual}, потрібно {limit}.",
    "GROUP_TOO_LARGE": "Група {actual} осіб, рекомендовано не більше {limit}.",
    "START_TOO_EARLY": "Старт о {actual}, раніше дозволеного {limit}.",
}


def num(v):
    return f"{v:g}" if isinstance(v, (int, float)) else v


def violation_ua(x: dict[str, Any]) -> str:
    template = VIOLATION_UA.get(x["code"])
    if not template:
        return x["message"]
    return template.format(day=x["day"], actual=num(x["actual"]), limit=num(x["limit"]))


def clean(text: str | None, limit: int = 600) -> str:
    """Model text going into the vault: no links, embeds, HTML or table breaks, bounded length."""
    text = re.sub(r"[\[\]!<>`|]", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def request_lines(r: dict[str, Any], notes: list[str]) -> list[str]:
    """What the model understood from the leader's note, so the leader can catch a misreading."""
    limits = {"max_km_per_day": "км/день", "max_ascent_per_day_m": "м набору/день",
              "max_walking_hours_per_day": "год ходу/день", "max_overnight_elevation_m": "м — найвища ночівля"}
    lim = ", ".join(f"{num(r[k])} {u}" for k, u in limits.items() if r.get(k)) or "немає"

    def v(key):
        return "не вказано" if r.get(key) is None else r.get(key)

    people = (f"вас {v('participants')}" if r.get("section") == "senior" and not r.get("adults")
              else f"учасників {v('participants')}, дорослих {v('adults')}")
    out = [
        f"- Група: {SECTION_UA.get(r.get('section'), 'не вказано')}, {people}",
        f"- Дні: {v('days')}, старт {v('start_date')} о {r.get('start_time') or '08:00'}"
        + (f", можна зсунути до {r['date_flexibility_days']} дн. пізніше" if r.get("date_flexibility_days") else ""),
        f"- Район: {REGION_UA.get(r.get('region'), r.get('region')) or 'будь-який'}; побажання: "
        + (", ".join(FEATURE_UA.get(f, f) for f in r.get("must_include") or []) or "немає")
        + (f"; місця: {', '.join(r['places'])}" if r.get("places") else ""),
        f"- Обмеження групи: {lim}",
    ]
    if r.get("lightest_body_weight_kg") or r.get("participant_base_gear_kg"):
        out.append(f"- Вага: найлегший учасник {num(r.get('lightest_body_weight_kg'))} кг; спорядження учасника "
                   f"{num(r.get('participant_base_gear_kg'))} кг, дорослого {num(r.get('adult_base_gear_kg'))} кг")
    out += [f"- ⚠️ {n}" for n in notes]
    return out


KIND_UA = {"hut": "хатина", "campsite": "стоянка", "shelter": "притулок", "water": "джерело",
           "lake": "озеро", "peak": "вершина", "start": "старт", "finish": "фініш"}


def wp(w: dict[str, Any]) -> str:
    name = w.get("name") or KIND_UA.get(w["kind"], w["kind"])
    return f"{name} ({w['id']}, {w['elevation_m']} м)"


def render_plan(state: dict[str, Any], request_path: str, model: str, generated_at: str) -> str:
    r = state.get("request", {})
    v = state.get("validation") or {}
    w = state.get("weather") or {}
    s = state.get("supplies") or {}
    status = state.get("status", "failed")
    L: list[str] = []
    add = L.append

    add("---")
    add(f"status: {status}")
    add(f"route: {v.get('route_id', '')}")
    add(f"generated: {generated_at}")
    add("source: plast-hike-planner agent")
    add("---")
    add(f"# План мандрівки: {request_path.split('/')[-1].removesuffix('.md')}\n")
    add(f"**{STATUS.get(status, status)}** · запит: [[{request_path.removesuffix('.md')}]]")
    if v.get("rules_status"):
        add(f"\n> [!warning] Норми: {v['rules_status']}. Перевірте й затвердьте `data/rules.yaml`.")
    add(f"\n## Висновок\n{clean(state.get('conclusion'), 1500)}\n")
    if r:
        add("## Що агент зрозумів із запиту")
        L.extend(request_lines(r, state.get("search_notes", [])))
        add("")

    if v.get("days"):
        rid = v["route_id"]
        add(f"## Маршрут\n**{v['route_name']}** — {v['total_km']} км, набір {v['total_ascent_m']} м. "
            f"[OpenStreetMap](https://www.openstreetmap.org/relation/{rid.removeprefix('r')}) · id `{rid}`\n")
        if state.get("picture_path"):
            add(f"![[{state['picture_path']}]]\n")
        prof = state.get("route_profile") or {}
        if prof.get("days"):
            add("**На справжній карті** (Mapy.com: туристична карта, профіль висот, підйом і спуск; браузер або застосунок):")
            add(f"- 🗺 [Увесь маршрут]({mapy_link(route_track(prof))})"
                + "".join(f" · [день {d['day']}]({mapy_link(day_track(prof, d))})" for d in prof["days"] if len(prof["days"]) > 1))
            for place in (state.get("request") or {}).get("place_points") or []:
                link, gap = side_trip_link(route_track(prof), place)
                if gap > 0.3:
                    add(f"- ⭐ [Радіальний вихід: {place['name']}]({link}) — {gap:.1f} км по прямій від маршруту")
            if state.get("gpx_path"):
                add(f"- 📍 Трек для телефона (Mapy, OsmAnd, Organic Maps): [[{state['gpx_path']}]]")
            if state.get("leaflet") and state.get("gpx_path"):   # an interactive map right here (Leaflet plugin)
                add("\n**Інтерактивна карта** (колесо миші — масштаб; кнопка шарів праворуч угорі — топокарта або "
                    "супутник; потрібен інтернет):\n")
                add(leaflet_block(prof, state["gpx_path"], (state.get("request") or {}).get("place_points"),
                                  state.get("leaflet_id", "plast-plan")))
                add(f"_{MAP_CREDITS}_\n")
            add("_Mapy.com сам прокладає стежку через наші точки, тож може трохи відрізнятися від маршруту в даних._\n")
        add("| День | Звідки → куди | км | набір, м | год ходу | старт → прихід | захід сонця | ночівля |")
        add("|---|---|---|---|---|---|---|---|")
        for d in v["days"]:
            night = "—"
            if d.get("overnight"):
                o = d["overnight"]
                water = f"вода {o['water_within_m']} м" if o["water_within_m"] is not None else "без води"
                night = f"{KIND_UA.get(o['waypoint']['kind'], o['waypoint']['kind'])}, {water}" + \
                        (", гребінь!" if o["ridge_risk"] else "")
            add(f"| {d['day']} | {wp(d['start'])} → {wp(d['end'])} | {d['km']} | {d['ascent_m']} | "
                f"{d['walking_hours']} | {d['start_time']} → {d['est_arrival']} | {d.get('sunset') or '—'} | {night} |")
        errors = [x for x in v.get("violations", []) if x["severity"] == "error"]
        warns = [x for x in v.get("violations", []) if x["severity"] == "warning"]
        if errors and state.get("status") == "group":
            add("\n**Супровід (порада):**")
            for x in errors:
                add(f"- {violation_ua(x)} За можливості візьміть ще когось на супровід (норми — чернетка).")
        elif errors:
            add("\n**Порушення (план не валідний):**")
            for x in errors:
                add(f"- ✗ {violation_ua(x)} _Порада інструмента: {clean(x['suggestion'], 300)}_")
        if warns:
            add("\n**Попередження:**")
            for x in warns:
                add(f"- {violation_ua(x)} _Порада інструмента: {clean(x['suggestion'], 300)}_")

    others = [x for x in (state.get("search") or {}).get("routes", []) if x["id"] != v.get("route_id")]
    if v.get("days") and others:
        add("\n## Інші варіанти маршруту")
        add("| Маршрут | км | набір, м | найвища точка, м | названі місця |")
        add("|---|---|---|---|---|")
        for x in others:
            near = [f"{n}: {d:g} км від маршруту" for n, d in (x.get("distance_to_places_km") or {}).items()]
            passes = [f"через {n}" for n in x.get("places_matched") or []]
            add(f"| [{clean(x['name'], 80)}](https://www.openstreetmap.org/relation/{x['id'].removeprefix('r')}) | "
                f"{x['length_km']} | {x['ascent_m']} | {x['max_elevation_m']} | {'; '.join(passes + near) or '—'} |")
        add("_Щоб спланувати інший варіант, допишіть у запит його назву й запустіть агента знову._")

    if w.get("days"):
        add(f"\n## Погода (загальний ризик: **{w['overall_risk']}**)")
        add(f"Open-Meteo на найвищій точці маршруту ({w['forecast_point']['elevation_m']} м), "
            f"режим `{w['mode']}`, прогноз від {w['fetched_at']}.\n")
        add("| Дата | Погода | °C | Опади, мм | Вітер, м/с | Ніч на ночівлі, °C | Ризик | Що робити |")
        add("|---|---|---|---|---|---|---|---|")
        for d in w["days"]:
            todo = "; ".join(t["recommendation"] for t in d["triggers"]) or "—"
            add(f"| {d['date']} | {d['weather']} | {d['temp_min_c']}…{d['temp_max_c']} | {d['precipitation_mm']} | "
                f"{d['wind_max_ms']} | {d.get('night_spot_min_c') if d.get('night_spot_min_c') is not None else '—'} | "
                f"{d['risk']} | {todo} |")
    elif w.get("error"):
        add(f"\n## Погода\nПрогноз недоступний: {w['error']}")

    if s.get("error"):
        add(f"\n## Запаси й спорядження\nНе пораховано: {s['error']}")
    elif s:
        pk, pa = s["pack_participant"], s["pack_adult"]
        add("\n## Запаси й спорядження")
        add(f"- Їжа: **{s['food_kg_group']} кг** на групу ({s['food_kg_per_person']} кг на учасника, "
            f"~{s['kcal_per_person_day']} ккал/день)")
        add(f"- Вода: {', '.join(str(x) for x in s['water_l_per_person_by_day'])} л на людину по днях"
            + (f"; додатково нести {s['water_to_carry_for_dry_nights_l']} л на ночівлі без води"
               if s["water_to_carry_for_dry_nights_l"] else ""))
        add(f"- Газ: **{s['gas_g_group']:.0f} г**; пальники: {s['stoves']}; казанки: {s['pots']}; "
            f"намети: {s['tents']}; аптечки: {s['first_aid_kits']}")
        if pk["limit_kg"] is None:
            add(f"- Наплічник учасника на старті: {pk['total_kg']} кг (ліміт невідомий: {pk['limit_source']})")
        else:
            add(f"- Наплічник учасника на старті: {pk['total_kg']} кг (ліміт {pk['limit_kg']} кг, "
                f"на спорядження лишається {pk['remaining_for_gear_kg']} кг)")
        add(f"- Наплічник виховника: {pa['total_kg']} кг")
        for g in s["gear_additions"]:
            add(f"- {g}")
        for x in s["warnings"]:
            add(f"- ⚠️ {x}")

    if v.get("documents"):
        add("\n## Документи")
        for doc in v["documents"]:
            add(f"- [ ] {doc}")

    add("\n## Як агент дійшов до цього плану")
    for a in state.get("attempts", []):
        p = a["proposal"]
        add(f"{a['n']}. `{p['route_id']}`, ночівлі {', '.join(p['overnight_waypoint_ids']) or '—'}, "
            f"старт {p['start_date']} → **{a['outcome']}**. _{clean(p.get('reasoning'), 400)}_")
        for f in a["feedback"][:4]:
            add(f"    - {clean(f, 400)}")
    for n in state.get("search_notes", []):
        add(f"- Пошук: {n}")

    add("\n## Звідки дані")
    add("- Маршрути й точки: © OpenStreetMap contributors (ODbL); висоти: SRTM 30 м через OpenTopoData")
    add("- Погода: Open-Meteo (CC BY 4.0); норми: `data/rules.yaml` (джерела в research/plast-norms-research.md)")
    add(f"- Модель: `{model}`; згенеровано {generated_at}")
    for sw in state.get("model_switches", []):
        add(f"- ↪ {clean(sw, 300)}")
    return "\n".join(L) + "\n"
