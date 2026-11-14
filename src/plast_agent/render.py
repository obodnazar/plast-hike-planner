"""Turn the final workflow state into a human-readable Obsidian note (Markdown).

Every number in the note is copied from a tool result; nothing here computes domain values.
"""

from __future__ import annotations

from typing import Any

STATUS = {
    "ok": "✅ План готовий",
    "attention": "⚠️ План валідний, але є попередження: перегляньте їх перед виходом",
    "risky": "⚠️ План складено, але НЕ рекомендовано без змін (погода)",
    "failed": "❌ Валідного плану скласти не вдалося",
}
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
    add(f"\n## Висновок\n{state.get('conclusion', '')}\n")

    if v.get("days"):
        rid = v["route_id"]
        add(f"## Маршрут\n**{v['route_name']}** — {v['total_km']} км, набір {v['total_ascent_m']} м. "
            f"[OpenStreetMap](https://www.openstreetmap.org/relation/{rid.removeprefix('r')}) · id `{rid}`\n")
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
        if errors:
            add("\n**Порушення (план не валідний):**")
            for x in errors:
                add(f"- ✗ {x['message']} {x['suggestion']}")
        if warns:
            add("\n**Попередження:**")
            for x in warns:
                add(f"- {x['message']} {x['suggestion']}")

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
            f"старт {p['start_date']} → **{a['outcome']}**. _{p.get('reasoning', '')}_")
        for f in a["feedback"][:4]:
            add(f"    - {f}")
    for n in state.get("search_notes", []):
        add(f"- Пошук: {n}")

    add("\n## Звідки дані")
    add("- Маршрути й точки: © OpenStreetMap contributors (ODbL); висоти: SRTM 30 м через OpenTopoData")
    add("- Погода: Open-Meteo (CC BY 4.0); норми: `data/rules.yaml` (джерела в research/plast-norms-research.md)")
    add(f"- Модель: `{model}`; згенеровано {generated_at}")
    return "\n".join(L) + "\n"
