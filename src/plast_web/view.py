"""The one boundary between the agent's data and the page: present(record) -> PlanView.

The agent's record holds tool results as they came: English messages for the model, ids like r8613349 / w30,
codes like ENDS_ON_TOP, source tags like [S13], region keys like chornohora. The page must read like a person
wrote it, so everything for the leader is built here, in Ukrainian, from the stable fields (codes, numbers,
names). Model text is used only if it has none of that technical text (looks_technical); otherwise the
conclusion is written by code from the tool results.

Only three parts of a PlanView are not for reading as text: `map` (coordinates and place names for drawing),
the `*_url` links and `tech` (the MCP calls, shown only behind «технічні деталі»). The tests check the rest.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

from plast_agent.picture import day_track, mapy_link, route_track, side_trip_link
from plast_agent.render import FEATURE_UA, KIND_UA, SECTION_UA, VIOLATION_UA, rules_note, violation_ua
from plast_agent.understanding import REGION_UA
from plast_agent.workflow import ADULT_CODES, SAFETY_ORDER, fallback_conclusion, group_conclusion

MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
          "листопада", "грудня"]
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]
RISK_UA = {"low": ("низький", "ok"), "medium": ("середній", "warn"), "high": ("високий", "danger"),
           "unknown": ("невідомий", "muted")}
STATUS_UA = {
    "ok": ("ok", "Готово"),
    "attention": ("warn", "Готово, є що врахувати"),
    "group": ("warn", "Готово · порада щодо супроводу"),
    "risky": ("danger", "Не радимо без змін: погода"),
    "failed": ("danger", "Плану скласти не вдалося"),
}
ADVICE_UA = {   # what to do, by violation code (the tool's own suggestion is English, for the model)
    "KM_PER_DAY_MAX": "Оберіть коротший маршрут або додайте день.",
    "KM_PER_DAY_WARN": "Закладіть більше перерв або оберіть ближчу ночівлю.",
    "ASCENT_PER_DAY_MAX": "Розподіліть підйом на два дні або оберіть інший маршрут.",
    "WALKING_HOURS_MAX": "Скоротіть день або додайте ще один.",
    "ARRIVAL_AFTER_SAFE_TIME": "Вийдіть раніше або оберіть ближчу ночівлю.",
    "OVERNIGHT_KIND_NOT_ALLOWED": "Оберіть іншу ночівлю.",
    "OVERNIGHT_TOO_HIGH": "Оберіть нижчу ночівлю.",
    "RIDGE_OVERNIGHT": "Оберіть ночівлю нижче, під захистом лісу.",
    "OVERNIGHT_NO_WATER": "Несіть воду на ночівлю з собою (її вагу враховано в запасах).",
    "TOO_MANY_DAYS": "Скоротіть мандрівку.",
    "TOO_FEW_ADULTS": "За можливості візьміть ще дорослого на супровід.",
    "ADULT_RATIO": "За можливості візьміть ще дорослого на супровід.",
    "GROUP_TOO_LARGE": "Розділіть групу на дві, кожну зі своїми дорослими.",
    "START_TOO_EARLY": "Перенесіть старт пізніше.",
}
OUTCOME_UA = [   # the agent's attempt outcomes (English, for the log) -> a line for the leader
    ("rejected by the agent check", "агент відхилив ще до перевірки: {why}"),
    ("rejected by the tool", "інструмент не прийняв план"),
    ("invalid", "не пройшла норми безпеки"),
    ("valid but HIGH weather", "маршрут пройшов норми, але погода ризикована: шукаю інший варіант"),
    ("valid, weather unknown", "маршрут пройшов норми; прогнозу на ці дати ще немає"),
    ("valid, weather risk", "маршрут пройшов норми, погодний ризик {risk}"),
    ("valid", "маршрут пройшов норми безпеки"),
]
REJECT_WHY = [   # the agent's own check (feedback is English, for the model) -> the reason in Ukrainian
    ("night spot(s) are needed", "не та кількість ночівель"),
    ("outside the allowed window", "дата поза дозволеним вікном"),
    ("not one of the candidates", "маршрут не зі списку знайдених"),
    ("stays HIGH whatever the night spots", "на цьому маршруті в ці дні погода ризикована за будь-яких ночівель"),
    ("repeats attempt", "вона повторює вже відхилену спробу"),
    ("not a night spot", "ночівля не з цього маршруту"),
]
DAY_COLORS = 3
# Words that may stay in Latin letters on the page: names of services, formats and licences.
LATIN_OK = {"mapy", "com", "gpx", "openstreetmap", "opentopomap", "open", "meteo", "obsidian", "srtm", "cc", "by",
            "sa", "odbl", "osmand", "organic", "maps", "nasa", "opentopodata", "osm", "contributors"}
# Boundaries are «not a Latin letter or digit», not \b: «Маршрутr8613349» has no word boundary before the r.
TECH = [re.compile(p) for p in (
    r"(?<![A-Za-z0-9])[rw]\d+(?![A-Za-z0-9])",          # route and waypoint ids
    r"(?<![A-Za-z0-9])[A-Z]{2,}(?:_[A-Z]+)+(?![A-Za-z0-9])",   # codes like ENDS_ON_TOP
    r"\[S\d+\]",                     # source tags of the norms research
    r"\b(?:DRAFT|PROPOSAL|HTTP|null|None|True|False|replay|live)\b",
    r"\b(?:chornohora|svydovets|gorgany|marmarosy|borzhava|yunatstvo|novatstvo|senior)\b",
    r"\b(?:low|medium|high|unknown|valid|invalid|attempt|error|warning|km|kg)\b",
)]


def num(v: Any) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def temp(c: float | int) -> str:
    """−7.9 °C with a real minus sign."""
    return f"{num(c)} °C".replace("-", "−")


def strip_sources(text: str) -> str:
    """«Медичні довідки [S2][S3]» -> «Медичні довідки»; English asides in brackets go too."""
    text = re.sub(r"\s*(\[S\d+\])+", "", text or "")
    text = re.sub(r"\s*\([^()]*[A-Za-z]{3,}[^()]*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_technical(text: str, names: set[str] = frozenset()) -> bool:
    """Ids, codes, source tags or English words that are not a place name from the data."""
    if any(p.search(text) for p in TECH):
        return True
    latin = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", text)}
    allowed = LATIN_OK | {w.lower() for n in names for w in re.findall(r"[A-Za-z]{2,}", n)}
    return bool(latin - allowed)


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "дата не вказана"
    d = date.fromisoformat(iso[:10])
    return f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]}"


def fmt_moment(text: str | None) -> str:
    """«2026-11-14T20:37:22+00:00» or «2026-11-16 14:05» -> «16 листопада, 14:05»."""
    try:
        t = datetime.fromisoformat((text or "").replace(" ", "T"))
    except ValueError:
        return text or ""
    if t.tzinfo is not None:
        from zoneinfo import ZoneInfo
        t = t.astimezone(ZoneInfo("Europe/Kyiv"))
    return f"{t.day} {MONTHS[t.month - 1]}, {t:%H:%M}"


def minutes(hhmm: str | None) -> int | None:
    if not hhmm or not re.match(r"^\d{1,2}:\d{2}$", hhmm):
        return None
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def point_name(w: dict[str, Any]) -> str:
    """A named point by its name; a nameless one by what it is and where («стоянка на 9.1 км, 866 м»)."""
    if w.get("name"):
        return w["name"]
    kind = {"start": "старт", "finish": "фініш"}.get(w.get("kind"), KIND_UA.get(w.get("kind"), "точка"))
    km = w.get("km")
    return f"{kind} на {km:.1f} км, {w.get('elevation_m')} м" if isinstance(km, (int, float)) else kind


def violation_text(x: dict[str, Any]) -> str:
    """render.violation_ua, with night-spot kinds in Ukrainian («shelter» -> «притулок»)."""
    if x["code"] not in VIOLATION_UA:
        return "Є зауваження до плану: перегляньте його перед виходом."

    def ua(v: Any) -> Any:
        if isinstance(v, str):   # «campsite, hut» comes as one string
            v = [k.strip() for k in v.split(",")] if "," in v else v
        if isinstance(v, list):
            return ", ".join(KIND_UA.get(k, k) for k in v)
        return KIND_UA.get(v, v) if isinstance(v, str) else v
    return violation_ua({**x, "actual": ua(x.get("actual")), "limit": ua(x.get("limit"))})


def advice(x: dict[str, Any]) -> str:
    """What to do about one violation, in Ukrainian; numbers taken from the tool's suggestion where it has them."""
    s = x.get("suggestion") or ""
    if x["code"] == "ENDS_ON_TOP":
        m = re.search(r"back the same way to (.+?) is ([\d.]+) km, about ([\d.]+) h", s)
        text = "Заплануйте спуск"
        if m:
            back = "попередньої ночівлі" if re.fullmatch(r"w\d+", m.group(1)) else f"«{m.group(1)}»"
            text += f": назад тією ж стежкою до {back} — {m.group(2)} км, близько {m.group(3)} год з перервами"
        late = re.search(r"Back at about (\d{1,2}:\d{2}), later than ([\d.]+) h before sunset (\d{1,2}:\d{2})", s)
        if late:
            text += (f". Повернення близько {late.group(1)}, а безпечно — за {late.group(2)} год до заходу сонця "
                     f"({late.group(3)})")
        return text + ". Або оберіть маршрут, що закінчується в селі."
    if x["code"] == "ARRIVAL_AFTER_SAFE_TIME":
        m = re.search(r"Start by (\d{1,2}:\d{2})", s)
        if m:
            return f"Вийдіть не пізніше {m.group(1)} або оберіть ближчу ночівлю."
    return ADVICE_UA.get(x["code"], "")


TOOL_TEXT_UA = [   # Ukrainian tool messages that still name a tool parameter (written for the model)
    (re.compile(r"\s*і передайте participant_base_gear_kg"), " і впишіть його вагу в запит"),
]


def human_items(items: list[str], names: set[str]) -> list[str]:
    """Tool texts for the page: sources stripped, parameter names reworded; one still technical is left out."""
    out = []
    for text in items:
        text = strip_sources(text)
        for pattern, ua in TOOL_TEXT_UA:
            text = pattern.sub(ua, text)
        text = re.sub(r"(?<![\w−-])-(?=\d)", "−", text)
        if text and not looks_technical(text, names):
            out.append(text)
    return out


def clean_note(text: str, names: set[str]) -> str | None:
    """A search note of the agent (Ukrainian, but may carry a region key or an English reason)."""
    for key, ua in REGION_UA.items():
        text = re.sub(rf"\b{key}\b", ua, text)
    text = strip_sources(text)
    return None if looks_technical(text, names) else text


def place_names(record: dict[str, Any]) -> set[str]:
    """Names that come from OpenStreetMap and may be in Latin letters («Blue path (modrá značka)»)."""
    names: set[str] = set()
    for x in (record.get("search") or {}).get("routes", []):
        names.add(x.get("name") or "")
        names.update(o.get("name") or "" for o in x.get("overnight_options") or [])
        names.update(x.get("places_matched") or [])
        names.update((x.get("distance_to_places_km") or {}).keys())
    v = record.get("validation") or {}
    names.add(v.get("route_name") or "")
    for d in v.get("days") or []:
        names.update({d["start"].get("name") or "", d["end"].get("name") or ""})
    for m in (record.get("route_profile") or {}).get("landmarks") or []:
        names.add(m.get("name") or "")
    for p in (record.get("request") or {}).get("place_points") or []:
        names.add(p.get("name") or "")
    names.update((record.get("request") or {}).get("places") or [])
    return {n for n in names if n}


# --------------------------------------------------------------------------------------------- parts of the page
def understood(r: dict[str, Any]) -> list[dict[str, str]]:
    if not r:
        return []
    section = r.get("section")
    people = (f"{r.get('participants')} осіб" if section == "senior" and not r.get("adults")
              else f"{r.get('participants')} учасників, дорослих {r.get('adults')}")
    rows = [
        ("Група", f"{SECTION_UA.get(section, 'не вказано')} · {people}"),
        ("Тривалість", f"{r.get('days')} дн." if r.get("days") else "не вказано"),
        ("Старт", f"{fmt_date(r.get('start_date'))}, {r.get('start_time') or '08:00'}"
         + (f" (можна зсунути на {r['date_flexibility_days']} дн.)" if r.get("date_flexibility_days") else "")),
        ("Район", REGION_UA.get(r.get("region"), "будь-який")),
    ]
    if r.get("places"):
        rows.append(("Місця", ", ".join(r["places"])))
    if r.get("must_include"):
        rows.append(("Побажання", ", ".join(FEATURE_UA.get(f, f) for f in r["must_include"])))
    limits = [(r.get("max_km_per_day"), "км на день"), (r.get("max_ascent_per_day_m"), "м підйому на день"),
              (r.get("max_walking_hours_per_day"), "год ходу на день"),
              (r.get("max_overnight_elevation_m"), "м — найвища ночівля")]
    shown = [f"до {num(v)} {u}" for v, u in limits if v]
    if shown:
        rows.append(("Обмеження", "; ".join(shown)))
    if r.get("lightest_body_weight_kg"):
        rows.append(("Найлегший учасник", f"{num(r['lightest_body_weight_kg'])} кг"))
    return [{"label": k, "value": v} for k, v in rows]


def safety(record: dict[str, Any]) -> list[dict[str, str]]:
    v, w = record.get("validation") or {}, record.get("weather") or {}
    out = []
    for x in sorted(v.get("violations") or [], key=lambda x: (x["severity"] != "error", SAFETY_ORDER.get(x["code"], 9))):
        level = "warn" if x["severity"] == "warning" or x["code"] in ADULT_CODES else "danger"
        out.append({"level": level, "text": violation_text(x), "advice": advice(x)})
    for d in w.get("days") or []:
        for t in d.get("triggers") or []:
            if t.get("risk") == "high":
                out.append({"level": "danger", "text": f"{fmt_date(d['date'])}: високий погодний ризик.",
                            "advice": strip_sources(t.get("recommendation", ""))})
    if w.get("coldest_night_c") is not None and w["coldest_night_c"] <= 0:
        out.append({"level": "warn", "text": f"Ніч до {temp(w['coldest_night_c'])}.",
                    "advice": "Зимові спальники й утеплені каремати."})
    return out


def days(record: dict[str, Any]) -> list[dict[str, Any]]:
    v = record.get("validation") or {}
    start = (record.get("proposal") or {}).get("start_date")
    out = []
    for i, d in enumerate(v.get("days") or []):
        night = None
        if d.get("overnight"):
            o = d["overnight"]
            gap = o.get("water_within_m")
            water = "води поруч немає" if gap is None else "вода поруч" if gap < 50 else f"вода за {gap} м"
            wp = o["waypoint"]
            night = (f"{wp['name']} ({KIND_UA.get(wp['kind'], 'ночівля')})" if wp.get("name") else point_name(wp)) + \
                    f", {water}" + \
                    (" · схоже на гребінь" if o.get("ridge_risk") else "")
        sunset = minutes(d.get("sunset"))
        safe_by = sunset - 180 if sunset is not None else None
        arrival = minutes(d.get("est_arrival"))
        out.append({
            "n": d["day"], "color": i % DAY_COLORS,
            "title": f"{point_name(d['start'])} → {point_name(d['end'])}",
            "date": fmt_date((date.fromisoformat(start) + timedelta(days=i)).isoformat()) if start else "",
            "night": night,
            "numbers": [{"value": f"{num(d['km'])} км", "label": "відстань"},
                        {"value": f"+{d['ascent_m']} м", "label": "підйом"},
                        {"value": f"−{d['descent_m']} м" if d["descent_m"] else "0 м", "label": "спуск"},
                        {"value": f"{num(d['walking_hours'])} год", "label": "ходу"}],
            "start": d.get("start_time"), "arrival": d.get("est_arrival"), "sunset": d.get("sunset"),
            "start_min": minutes(d.get("start_time")), "arrival_min": arrival, "sunset_min": sunset,
            "late": bool(arrival is not None and safe_by is not None and arrival > safe_by),
        })
    return out


def weather(record: dict[str, Any]) -> dict[str, Any]:
    w = record.get("weather") or {}
    if w.get("days"):
        when = fmt_moment(w.get("fetched_at"))
        risk, kind = RISK_UA.get(w.get("overall_risk"), RISK_UA["unknown"])
        return {
            "state": "ok", "risk": {"label": f"Ризик {risk}", "kind": kind},
            "note": (f"Прогноз Open-Meteo на найвищій точці маршруту ({w['forecast_point']['elevation_m']} м), "
                     + (f"записаний {when}." if w.get("mode") == "replay" else f"отриманий {when}.")),
            "days": [{
                "date": fmt_date(d["date"]), "text": d.get("weather") or "",
                "temp": f"{round(d['temp_min_c'])}…{round(d['temp_max_c'])} °C".replace("-", "−"),
                "details": f"опади {num(d['precipitation_mm'])} мм · вітер {num(d['wind_max_ms'])} м/с"
                           + (f" · ніч на ночівлі {temp(d['night_spot_min_c'])}"
                              if d.get("night_spot_min_c") is not None else ""),
                "risk": {"label": f"ризик {RISK_UA.get(d['risk'], RISK_UA['unknown'])[0]}",
                         "kind": RISK_UA.get(d["risk"], RISK_UA["unknown"])[1]},
                "advice": [strip_sources(t["recommendation"]) for t in d.get("triggers") or []],
            } for d in w["days"]],
        }
    if record.get("validation", {}) and (record.get("validation") or {}).get("days"):
        return {"state": "missing",
                "note": "Прогнозу на ці дати ще немає: Open-Meteo дає прогноз лише на 16 днів уперед, а записані "
                        "прогнози в проєкті покривають 14–29 листопада 2026. Перевірте погоду ближче до виходу."}
    return {"state": "none", "note": ""}


def pack_text(p: dict[str, Any]) -> str:
    """«8.7 кг: спорядження 5 + їжа 2.7 + вода 1» or «3.7 кг їжі й води; спорядження ще не зважено»."""
    food_water = f"їжа {num(p['food_kg'])} + вода {num(p['water_kg'])}"
    if p.get("personal_gear_kg") is not None:
        return f"{num(p['total_kg'])} кг разом зі спорядженням (спорядження {num(p['personal_gear_kg'])} + {food_water})"
    return f"{num(p['total_kg'])} кг їжі й води ({food_water}); спорядження ще не зважено"


def supplies(record: dict[str, Any], names: set[str] = frozenset()) -> dict[str, Any] | None:
    s = record.get("supplies") or {}
    if not s:
        return None
    if s.get("error"):
        return {"tiles": [], "lines": ["Запаси не пораховано: інструмент не відповів."], "gear": [], "warnings": []}
    pk, pa = s["pack_participant"], s["pack_adult"]
    water = ", ".join(num(x) for x in s["water_l_per_person_by_day"])
    lines = [f"Їжа: {num(s['food_kg_per_person'])} кг на людину, близько {s['kcal_per_person_day']} ккал на день",
             f"Вода: {water} л на людину по днях"
             + (f"; ще {num(s['water_to_carry_for_dry_nights_l'])} л нести на ночівлі без води"
                if s.get("water_to_carry_for_dry_nights_l") else ""),
             f"Пальники: {s['stoves']} · казанки: {s['pots']} · аптечки: {s['first_aid_kits']}"]
    limit = (f"ліміт {num(pk['limit_kg'])} кг" if pk.get("limit_kg") is not None
             else "ліміту немає, бо в запиті не вказано вагу найлегшого учасника")
    lines.append(f"Наплічник учасника на старті: {pack_text(pk)}; {limit}"
                 + (f", на спорядження лишається {num(pk['remaining_for_gear_kg'])} кг"
                    if pk.get("remaining_for_gear_kg") is not None and pk.get("personal_gear_kg") is None else ""))
    lines.append(f"Наплічник дорослого на старті: {pack_text(pa)}")
    return {
        "tiles": [{"value": f"{num(s['food_kg_group'])} кг", "label": "їжі на групу"},
                  {"value": f"{round(s['gas_g_group'])} г", "label": "газу"},
                  {"value": f"{s['tents']}", "label": "намети"}],
        "lines": lines,
        "gear": human_items(s.get("gear_additions") or [], names),
        "warnings": human_items(s.get("warnings") or [], names),
    }


def alternatives(record: dict[str, Any]) -> list[dict[str, Any]]:
    chosen = (record.get("validation") or {}).get("route_id")
    out = []
    for x in (record.get("search") or {}).get("routes", []):
        if x["id"] == chosen:
            continue
        near = [f"{n}: {num(d)} км від маршруту" for n, d in (x.get("distance_to_places_km") or {}).items()]
        passes = [f"через {n}" for n in x.get("places_matched") or []]
        out.append({"name": x["name"], "facts": f"{num(x['length_km'])} км · +{x['ascent_m']} м · найвища точка "
                                                f"{x['max_elevation_m']} м",
                    "places": "; ".join(passes + near),
                    "osm_url": f"https://www.openstreetmap.org/relation/{x['id'].removeprefix('r')}"})
    return out


def how(record: dict[str, Any], names: set[str]) -> list[str]:
    """The agent's way to the plan, as a short story without ids and codes."""
    r, search = record.get("request") or {}, record.get("search") or {}
    routes = {x["id"]: x for x in search.get("routes", [])}
    out = []
    if r:
        out.append(f"Прочитав запит: {SECTION_UA.get(r.get('section'), 'група')}, {r.get('days')} дн., "
                   f"старт {fmt_date(r.get('start_date'))}, район — {REGION_UA.get(r.get('region'), 'будь-який')}.")
    if search.get("routes"):
        out.append(f"Знайшов маршрутів, що підходять: {len(search['routes'])}.")
    for note in record.get("search_notes") or []:
        cleaned = clean_note(note, names)
        if cleaned:
            out.append(cleaned)
    risk_names = {k: v[0] for k, v in RISK_UA.items()}
    tries: list[list] = []   # [first n, last n, text]: the same attempt repeated is one line («Спроби 2–4: …»)
    for a in record.get("attempts") or []:
        p = a.get("proposal") or {}
        route = routes.get(p.get("route_id"))
        nights_by_id = {o["id"]: o for o in (route or {}).get("overnight_options") or []}
        nights = [point_name(nights_by_id[i]) for i in p.get("overnight_waypoint_ids") or [] if i in nights_by_id]
        outcome = a.get("outcome") or ""
        said = next((ua for en, ua in OUTCOME_UA if outcome.startswith(en)), "")
        risk = re.search(r"weather risk (\w+)", outcome)
        feedback = " ".join(a.get("feedback") or [])
        why = "; ".join(ua for en, ua in REJECT_WHY if en in feedback) or "план не відповідав запиту"
        said = said.format(risk=risk_names.get(risk.group(1), "") if risk else "", why=why)
        text = (f"«{route['name'] if route else 'маршрут поза списком'}», старт {fmt_date(p.get('start_date'))}"
                + (f", ночівлі: {', '.join(nights)}" if nights else "") + (f" — {said}." if said else "."))
        if tries and tries[-1][2] == text:
            tries[-1][1] = a["n"]
        else:
            tries.append([a["n"], a["n"], text])
    out += [f"Спроба {f}: {text}" if f == last else f"Спроби {f}–{last}: {text}" for f, last, text in tries]
    if record.get("supplies") and not (record.get("supplies") or {}).get("error"):
        out.append("Порахував їжу, воду, газ і спорядження; підготував карту, профіль висот і трек.")
    return out


def tech(record: dict[str, Any]) -> dict[str, Any]:
    """For the defence and for debugging: the real calls, ids and the model's own words. Not human text."""
    return {
        "model": record.get("model") or "",
        "switches": list(record.get("model_switches") or []),
        "calls": [f"{c['server']}.{c['tool']}({c['args']}) → {c['result']}"
                  + (f", {c['ms']} ms" if c.get("ms") is not None else "") for c in record.get("trace") or []],
        "attempts": [f"{a['n']}. {a['proposal'].get('route_id')} {a['proposal'].get('overnight_waypoint_ids')} "
                     f"{a['proposal'].get('start_date')} → {a.get('outcome')}. {a['proposal'].get('reasoning') or ''}"
                     + "".join(f" | {f}" for f in a.get("feedback") or []) for a in record.get("attempts") or []],
    }


def map_data(record: dict[str, Any]) -> dict[str, Any] | None:
    prof = record.get("route_profile") or {}
    if not prof.get("days"):
        return None
    tracks = [[[p[1], p[2]] for p in day_track(prof, d)] for d in prof["days"]]
    profile = []
    for i, d in enumerate(prof["days"]):
        profile += [[p[0], p[1], p[2], p[3], i % DAY_COLORS] for p in day_track(prof, d)]
    marks = []
    for m in prof.get("landmarks") or []:
        if m["kind"] == "start":
            marks.append({"kind": "start", "label": f"Старт · {point_name(m)}", "lat": m["lat"], "lon": m["lon"],
                          "km": m["km"]})
        elif m["kind"] == "finish":
            marks.append({"kind": "finish", "label": f"Фініш · {point_name(m)}", "lat": m["lat"], "lon": m["lon"],
                          "km": m["km"]})
        elif m.get("night"):
            marks.append({"kind": "night", "label": f"Ночівля {m['night']} · {point_name(m)}", "lat": m["lat"],
                          "lon": m["lon"], "km": m["km"]})
    for p in (record.get("request") or {}).get("place_points") or []:
        if any(abs(p["lat"] - m["lat"]) < 0.0015 and abs(p["lon"] - m["lon"]) < 0.002 for m in marks):
            continue   # the wished place is the start, a night spot or the finish: one pin is enough
        marks.append({"kind": "place", "label": p["name"] + (f" · {p['elevation_m']} м" if p.get("elevation_m")
                                                             is not None else ""),
                      "lat": p["lat"], "lon": p["lon"], "km": None})
    return {"tracks": tracks, "profile": profile, "marks": marks}


def links(record: dict[str, Any]) -> dict[str, Any]:
    prof = record.get("route_profile") or {}
    pid = record["id"]
    out: dict[str, Any] = {"gpx_url": "", "picture_url": "", "mapy_url": "", "mapy_days": [], "side_trips": [],
                           "osm_url": "", "obsidian_url": ""}
    if prof.get("days"):
        out["gpx_url"] = f"/api/plans/{pid}/route.gpx"
        out["picture_url"] = f"/api/plans/{pid}/route.svg"
        out["mapy_url"] = mapy_link(route_track(prof))
        if len(prof["days"]) > 1:
            out["mapy_days"] = [{"label": f"День {d['day']}", "url": mapy_link(day_track(prof, d))}
                                for d in prof["days"]]
        for place in (record.get("request") or {}).get("place_points") or []:
            url, gap = side_trip_link(route_track(prof), place)
            if gap > 0.3:
                out["side_trips"].append({"label": f"Радіальний вихід до «{place['name']}»: {gap:.1f} км по прямій "
                                                   "від маршруту", "url": url})
    rid = (record.get("validation") or {}).get("route_id")
    if rid:
        out["osm_url"] = f"https://www.openstreetmap.org/relation/{rid.removeprefix('r')}"
    if record.get("obsidian_plan"):
        vault = os.environ.get("OBSIDIAN_VAULT_NAME", "demo-vault")
        out["obsidian_url"] = (f"obsidian://open?vault={quote(vault)}"
                               f"&file={quote(record['obsidian_plan'].removesuffix('.md'))}")
    return out


# --------------------------------------------------------------------------------------------- the boundary
def conclusion(record: dict[str, Any], names: set[str]) -> str:
    text = re.sub(r"[*_`#>]+", "", record.get("conclusion") or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or looks_technical(text, names):
        state = {k: record.get(k) for k in ("validation", "weather", "status", "proposal")}
        text = (group_conclusion(state["validation"] or {}, state["weather"]) if record.get("status") == "group"
                else fallback_conclusion(state))
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", lambda m: fmt_date(m.group(0)), text)
    return re.sub(r"(?<![\w−-])-(?=\d)", "−", text)   # −6.3 °C with a real minus


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """One line of the plan list."""
    v, p = record.get("validation") or {}, record.get("proposal") or {}
    kind, label = STATUS_UA.get(record.get("status"), STATUS_UA["failed"])
    title = v.get("route_name") or (record.get("request_path") or "").split("/")[-1].removesuffix(".md")
    facts = []
    if v.get("days"):
        facts = [f"{len(v['days'])} дн. з {fmt_date(p.get('start_date'))}", f"{num(v['total_km'])} км",
                 f"+{v['total_ascent_m']} м"]
    return {"id": record["id"], "title": title, "status": {"kind": kind, "label": label},
            "request": (record.get("request_path") or "").split("/")[-1].removesuffix(".md"),
            "created": fmt_moment(record.get("generated_at")), "facts": facts}


def present(record: dict[str, Any]) -> dict[str, Any]:
    """A saved plan -> everything the plan page shows, in Ukrainian, without ids, codes or English."""
    names = place_names(record)
    v, w, r = record.get("validation") or {}, record.get("weather") or {}, record.get("request") or {}
    p = record.get("proposal") or {}
    kind, label = STATUS_UA.get(record.get("status"), STATUS_UA["failed"])
    ds = days(record)
    meta = []
    if ds:
        last = (date.fromisoformat(p["start_date"]) + timedelta(days=len(ds) - 1)).isoformat()
        meta.append(fmt_date(p["start_date"]) + (f" – {fmt_date(last)}" if len(ds) > 1 else ""))
    if r:
        meta.append(understood(r)[0]["value"])
        if r.get("region"):
            meta.append(REGION_UA.get(r["region"], ""))
    stats = []
    if v.get("days"):
        stats = [{"value": f"{num(v['total_km'])} км", "label": "відстань"},
                 {"value": f"+{v['total_ascent_m']} м", "label": "набір висоти"},
                 {"value": f"{num(round(sum(d['walking_hours'] for d in v['days']), 1))} год", "label": "чистого ходу"}]
        if w.get("coldest_night_c") is not None:
            stats.append({"value": temp(w["coldest_night_c"]), "label": "найхолодніша ніч"})
        else:
            stats.append({"value": f"{max(d['max_elevation_m'] for d in v['days'])} м", "label": "найвища точка"})
    notes = []
    said = conclusion(record, names)
    if "DRAFT" in (v.get("rules_status") or "") and "чернетка" not in said:
        notes.append("Норми Пласту в системі поки чернетка: їх має затвердити провідник.")
    return {
        **summary(record),
        "status": {"kind": kind, "label": label},
        "meta": " · ".join(m for m in meta if m),
        "conclusion": said,
        "stats": stats,
        "safety": safety(record),
        "notes": notes,
        "understood": understood(r),
        "days": ds,
        "weather": weather(record),
        "supplies": supplies(record, names),
        "documents": human_items(v.get("documents") or [], names),
        "alternatives": alternatives(record),
        "how": how(record, names),
        "sources": ("Маршрути й точки © OpenStreetMap contributors (ODbL) · висоти SRTM · погода Open-Meteo "
                    "(CC BY 4.0) · карта OpenTopoMap (CC-BY-SA)"
                    + (" · " + rules_note(v["rules_status"]) if v.get("rules_status", "").startswith("approved") else "")),
        "map": map_data(record),
        "links": links(record),
        "tech": tech(record),
    }
