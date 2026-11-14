"""The agent's workflow as a LangGraph graph.

    read_request ─► parse_request ─► search_routes ─► propose ─► validate ─┬─► weather ─┬─► supplies ─► conclude ─► write_plan
                                                        ▲                  │            │
                                                        └── violations ────┘            │
                                                        └── high weather risk ──────────┘

Code fixes the order of steps; the model works inside two of them:
  parse_request - turns the leader's free-text Obsidian note into typed fields;
  propose       - picks route, night spots and start date, and on every loop gets
                  the tool feedback (violations with suggestions, weather triggers).
A loop ends when the plan is valid and not high-risk, or after MAX_ATTEMPTS.

The model never calls MCP tools itself: the code calls them and gives the model the results.
Every proposal is checked in code (route among the candidates, exact number of night spots
from the shown options, date inside the allowed window, no repeat of a rejected plan) before
any tool call. If attempts run out, the best valid plan found so far is kept.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from plast_agent.llm import LLM, ParsedRequest, Proposal
from plast_agent.mcp_links import McpCallError, McpLinks

MAX_ATTEMPTS = 4
BOLD, YELLOW, RESET = "\033[1m", "\033[33m", "\033[0m"


class AgentStop(RuntimeError):
    """The run cannot continue (e.g. the request note is missing). The message is for the person."""


class State(TypedDict, total=False):
    request_path: str
    request_text: str
    request: dict[str, Any]
    search: dict[str, Any]
    search_notes: list[str]
    attempts: list[dict[str, Any]]
    proposal: dict[str, Any]
    validation: dict[str, Any]
    weather: dict[str, Any]
    supplies: dict[str, Any]
    proposal_ok: bool
    best: dict[str, Any]  # best valid plan so far: {proposal, validation, weather}
    status: str           # ok | attention | risky | failed
    conclusion: str


def step(title: str) -> None:
    print(f"\n{BOLD}━━ {title}{RESET}")


# ------------------------------------------------------------------ prompts
PARSE_SYSTEM = (
    "You extract the facts of a Plast (Ukrainian scouting) hike request note into the schema. "
    "Today is {today}. Use JSON null for anything the note does not state; never guess numbers."
)

PROPOSE_SYSTEM = """You plan a Plast hike. Choose ONE route from the candidates, its night spots and the start date.
Hard requirements:
- exactly {nights} night spot id(s), taken only from that route's overnight options, in increasing km;
- start_date between {first_date} and {last_date};
- respect the group's limits: {limits};
- split the route so that each day's km (difference of night-spot km values) is as even as possible
  and below the km limit.
If feedback from earlier attempts is given, fix exactly what it says:
- for rule violations, apply the suggested waypoint ids;
- weather depends on the dates and the route, hardly on night spots. After a HIGH weather result,
  use the date probes in the feedback: move start_date to a date whose risk is not high, or pick another route.
Never repeat a rejected route + start_date pair: {rejected}.
Write `reasoning` in Ukrainian, 1-2 sentences."""


def candidates_text(search: dict[str, Any]) -> str:
    lines = []
    for r in search["routes"]:
        lines.append(f"ROUTE {r['id']}: {r['name']} | {r['length_km']} km, +{r['ascent_m']} m, max "
                     f"{r['max_elevation_m']} m | {r['start']} -> {r['finish']} | avg {r['avg_km_per_day']} km/day")
        for o in r["overnight_options"]:
            lines.append(f"   night option {o['id']}: {o['kind']} {o['name'] or ''} at km {o['km']}, "
                         f"{o['elevation_m']} m, water nearby: {o['water_nearby']}")
    return "\n".join(lines)


def feedback_text(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No earlier attempts."
    out = []
    for a in attempts:
        p = a["proposal"]
        out.append(f"Attempt {a['n']}: route {p['route_id']}, nights {p['overnight_waypoint_ids']}, "
                   f"start {p['start_date']} -> {a['outcome']}")
        out += [f"   - {line}" for line in a["feedback"]]
    return "\n".join(out)


RISK_ORDER = {"low": 0, "medium": 1, "unknown": 2, "high": 3}


def group_limits(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r[k] for k in ("max_km_per_day", "max_ascent_per_day_m", "max_walking_hours_per_day",
                              "max_overnight_elevation_m") if r.get(k)}


def normalise_time(value: str | None) -> str:
    """'8:00' -> '08:00'; anything unusable -> '08:00'."""
    try:
        h, m = (value or "08:00").strip().split(":")
        h, m = int(h), int(m)
        if 0 <= h < 24 and 0 <= m < 60:
            return f"{h:02d}:{m:02d}"
    except ValueError:
        pass
    return "08:00"


def check_proposal(prop: dict[str, Any], state: State) -> tuple[list[str], dict[str, Any]]:
    """Enforce in code what the prompt asks for. Returns (problems, normalised proposal)."""
    r = state["request"]
    routes = {x["id"]: x for x in state["search"]["routes"]}
    problems: list[str] = []
    start = prop["start_date"]
    prop = {**prop, "start_date": start.isoformat(), "start_time": normalise_time(r.get("start_time"))}
    route = routes.get(prop["route_id"])
    if route is None:
        return [f"Route {prop['route_id']} is not one of the candidates {sorted(routes)}."], prop
    options = {o["id"]: o for o in route["overnight_options"]}
    ids = list(dict.fromkeys(prop["overnight_waypoint_ids"]))  # drop duplicates, keep order
    unknown = [i for i in ids if i not in options]
    if unknown:
        problems.append(f"Night spots {unknown} are not overnight options of {route['id']}; "
                        f"choose from {list(options)}.")
    ids = sorted((i for i in ids if i in options), key=lambda i: options[i]["km"])
    nights = r["days"] - 1
    if len(ids) != nights:
        problems.append(f"The group asked for {r['days']} days, so exactly {nights} night spot(s) are needed; "
                        f"got {len(ids)}.")
    first = date.fromisoformat(r["start_date"])
    last = first + timedelta(days=r["date_flexibility_days"])
    if not first <= start <= last:
        problems.append(f"start_date {start} is outside the allowed window {first}..{last}.")
    norm = {**prop, "overnight_waypoint_ids": ids}
    for a in state["attempts"]:
        q = a["proposal"]
        same_plan = (q["route_id"], q["overnight_waypoint_ids"], q["start_date"]) == \
                    (norm["route_id"], ids, norm["start_date"])
        same_bad_weather = "HIGH weather" in a["outcome"] and \
            (q["route_id"], q["start_date"]) == (norm["route_id"], norm["start_date"])
        if same_plan or same_bad_weather:
            problems.append(f"This repeats attempt {a['n']} ({a['outcome']}); change the night spots, the date "
                            "or the route.")
            break
    return problems, norm


# ------------------------------------------------------------------ graph
def build_graph(links: McpLinks, llm: LLM):
    async def read_request(state: State) -> State:
        step(f"1. Читаю запит з Obsidian: {state['request_path']}")
        try:
            note = await links.call("obsidian", "vault_read", {"path": state["request_path"]})
        except McpCallError as exc:
            listing = await links.call("obsidian", "vault_list", {"path": "Запити"})
            files = listing.get("files", []) if isinstance(listing, dict) else []
            raise AgentStop(f"Запит не знайдено в Obsidian ({exc.message}). "
                            f"Наявні запити: {', '.join('Запити/' + f for f in files) or 'немає'}") from exc
        text = note["content"] if isinstance(note, dict) else str(note)
        print(f"    прочитано {len(text)} символів")
        return {"request_text": text, "attempts": [], "search_notes": []}

    async def parse_request(state: State) -> State:
        step("2. Розбираю запит (LLM → структуровані поля)")
        req: ParsedRequest = llm.structured(ParsedRequest, PARSE_SYSTEM.format(today=date.today()),
                                            state["request_text"])
        d = req.model_dump(mode="json")
        print("    " + json.dumps({k: v for k, v in d.items() if v not in (None, [])}, ensure_ascii=False))
        return {"request": d}

    async def search_routes(state: State) -> State:
        step("3. Шукаю маршрути (plast-trail.find_routes)")
        r = state["request"]
        args = {"section": r["section"], "days": r["days"], "region": r["region"],
                "must_include": r["must_include"], "max_results": 4}
        args = {k: v for k, v in args.items() if v not in (None, [])}
        notes = []
        result = await links.call("plast-trail", "find_routes", args)
        # An empty list is a valid answer: relax the wishes step by step and say so.
        for relax in ("must_include", "region"):
            if result["routes"] or relax not in args:
                break
            notes.append(f"Немає маршрутів з умовою {relax}={args.pop(relax)}; шукаю без неї.")
            print(f"    {YELLOW}{notes[-1]}{RESET}")
            result = await links.call("plast-trail", "find_routes", args)
        if not result["routes"]:
            raise AgentStop(f"Жоден маршрут не підходить. Причини: {result['excluded']}. {result['notes']}")
        for x in result["routes"]:
            print(f"    {x['id']} {x['name']} — {x['length_km']} км, ~{x['avg_km_per_day']} км/день, fit {x['fit_score']}")
        return {"search": result, "search_notes": notes}

    async def propose(state: State) -> State:
        n = len(state["attempts"]) + 1
        step(f"4.{n} Пропоную план (LLM), спроба {n}/{MAX_ATTEMPTS}")
        r = state["request"]
        first = date.fromisoformat(r["start_date"])
        limits = group_limits(r)
        rejected = sorted({f"{a['proposal']['route_id']} from {a['proposal']['start_date']}"
                           for a in state["attempts"] if "HIGH weather" in a["outcome"]})
        system = PROPOSE_SYSTEM.format(nights=r["days"] - 1, first_date=first,
                                       last_date=first + timedelta(days=r["date_flexibility_days"]),
                                       limits=limits or "none beyond the Plast rules",
                                       rejected=", ".join(rejected) or "none yet")
        user = (f"Group: {r['section']}, {r['participants']} participants + {r['adults']} adults, "
                f"{r['days']} days.\n\nCandidates:\n"
                f"{candidates_text(state['search'])}\n\nFeedback:\n{feedback_text(state['attempts'])}")
        p: Proposal = llm.structured(Proposal, system, user)
        problems, prop = check_proposal(p.model_dump(), state)
        print(f"    маршрут {prop['route_id']}, ночівлі {prop['overnight_waypoint_ids']}, старт "
              f"{prop['start_date']} {prop['start_time']}\n    «{prop['reasoning']}»")
        if problems:  # rejected by the code check: feedback for the model, no tool call spent
            for x in problems:
                print(f"    {YELLOW}✗ перевірка агента: {x}{RESET}")
            attempt = {"n": n, "proposal": prop, "outcome": "rejected by the agent check", "feedback": problems}
            return {"proposal": prop, "proposal_ok": False, "attempts": state["attempts"] + [attempt]}
        return {"proposal": prop, "proposal_ok": True}

    def after_propose(state: State) -> str:
        if state["proposal_ok"]:
            return "validate"
        return "propose" if len(state["attempts"]) < MAX_ATTEMPTS else "give_up"

    async def validate(state: State) -> State:
        step("5. Перевіряю план (plast-trail.validate_hike_plan)")
        r, p = state["request"], state["proposal"]
        limits = group_limits(r)
        args = {"route_id": p["route_id"], "section": r["section"], "participants": r["participants"],
                "adults": r["adults"], "overnight_waypoint_ids": p["overnight_waypoint_ids"],
                "start_date": p["start_date"], "start_time": p["start_time"]}
        if limits:
            args["group_limits"] = limits
        attempt = {"n": len(state["attempts"]) + 1, "proposal": p, "feedback": []}
        try:
            v = await links.call("plast-trail", "validate_hike_plan", args)
        except McpCallError as exc:  # a wrong id or order is feedback for the model, not a crash
            attempt.update(outcome=f"rejected by the tool: {exc.code}", feedback=[f"{exc.message} {exc.hint}"])
            return {"attempts": state["attempts"] + [attempt], "validation": {}, "weather": {}}
        for d in v["days"]:
            print(f"    день {d['day']}: {d['start']['id']}→{d['end']['id']} {d['km']} км, +{d['ascent_m']} м, "
                  f"{d['walking_hours']} год ходу, прихід {d['est_arrival']} (захід {d['sunset']})")
        errors = [x for x in v["violations"] if x["severity"] == "error"]
        for x in v["violations"]:
            mark = "✗" if x["severity"] == "error" else "!"
            print(f"    {mark} {x['code']} (день {x['day']}): {x['message']} → {x['suggestion']}")
        attempt["outcome"] = "valid" if v["valid"] else f"invalid ({len(errors)} errors)"
        attempt["feedback"] = [f"{x['code']} day {x['day']}: {x['message']} Suggestion: {x['suggestion']}"
                               for x in errors]
        print(f"    → {'план валідний' if v['valid'] else 'план НЕ валідний'}")
        # weather always belongs to the plan just validated: never carry it over from an earlier attempt
        return {"attempts": state["attempts"] + [attempt], "validation": v, "weather": {}}

    def after_validate(state: State) -> str:
        if state["validation"].get("valid"):
            return "weather"
        return "propose" if len(state["attempts"]) < MAX_ATTEMPTS else "give_up"

    async def weather(state: State) -> State:
        step("6. Оцінюю погоду (plast-trail.assess_weather_risk)")
        p = state["proposal"]
        try:
            w = await links.call("plast-trail", "assess_weather_risk", {
                "route_id": p["route_id"], "start_date": p["start_date"],
                "overnight_waypoint_ids": p["overnight_waypoint_ids"]})
        except McpCallError as exc:  # no forecast is not a reason to drop a valid plan, but it is reported
            last = {**state["attempts"][-1], "outcome": f"valid, weather unknown ({exc.code})"}
            print(f"    {YELLOW}погода невідома: {exc.message}{RESET}")
            w = {"overall_risk": "unknown", "error": exc.message, "days": []}
            return {"weather": w, "attempts": state["attempts"][:-1] + [last], "best": better(state, w)}
        for d in w["days"]:
            trig = ", ".join(f"{t['code']}={t['value']}" for t in d["triggers"]) or "—"
            print(f"    {d['date']}: {d['weather']}, {d['temp_min_c']}…{d['temp_max_c']} °C, "
                  f"опади {d['precipitation_mm']} мм, вітер {d['wind_max_ms']} м/с → {d['risk']} ({trig})")
        print(f"    → загальний ризик: {w['overall_risk']} (джерело: {w['mode']}, {w['fetched_at']})")
        last = dict(state["attempts"][-1])
        if w["overall_risk"] == "high":
            last["outcome"] = "valid but HIGH weather risk"
            last["feedback"] = [f"Weather {d['date']}: risk {d['risk']}: " + "; ".join(
                f"{t['code']} {t['value']} (threshold {t['threshold']}): {t['recommendation']}" for t in d["triggers"])
                for d in w["days"] if d["risk"] == "high"]
            # Probe the other allowed start dates for the same plan: facts for the next proposal.
            r = state["request"]
            first = date.fromisoformat(r["start_date"])
            current = date.fromisoformat(p["start_date"])
            for k in range(r["date_flexibility_days"] + 1):
                alt = first + timedelta(days=k)
                if alt == current:
                    continue
                try:
                    probe = await links.call("plast-trail", "assess_weather_risk", {
                        "route_id": p["route_id"], "start_date": alt.isoformat(),
                        "overnight_waypoint_ids": p["overnight_waypoint_ids"]})
                    risk = probe["overall_risk"]
                except McpCallError as exc:
                    risk = f"unknown ({exc.code})"
                print(f"    проба дати {alt}: ризик {risk}")
                last["feedback"].append(f"Date probe: same route and night spots starting {alt}: weather risk {risk}")
        else:
            last["outcome"] = f"valid, weather risk {w['overall_risk']}"
        return {"weather": w, "attempts": state["attempts"][:-1] + [last], "best": better(state, w)}

    def better(state: State, w: dict[str, Any]) -> dict[str, Any]:
        """Keep the valid plan with the lowest weather risk seen so far."""
        current = {"proposal": state["proposal"], "validation": state["validation"], "weather": w}
        best = state.get("best")
        if not best or RISK_ORDER[w["overall_risk"]] < RISK_ORDER[best["weather"]["overall_risk"]]:
            return current
        return best

    def after_weather(state: State) -> str:
        if state["weather"].get("overall_risk") != "high":
            return "supplies"
        return "propose" if len(state["attempts"]) < MAX_ATTEMPTS else "give_up"

    async def supplies(state: State) -> State:
        step("7. Рахую запаси (plast-trail.calculate_supplies)")
        r, v, w = state["request"], state["validation"], state["weather"]
        nights = [d["night_spot_min_c"] for d in w.get("days", []) if d.get("night_spot_min_c") is not None]
        args = {"section": r["section"], "participants": r["participants"], "adults": r["adults"],
                "days": [{"km": d["km"], "ascent_m": d["ascent_m"], "walking_hours": d["walking_hours"]}
                         for d in v["days"]],
                "dry_nights": sum(1 for x in v["violations"] if x["code"] == "OVERNIGHT_NO_WATER")}
        for k_in, k_out in (("lightest_body_weight_kg", "lightest_body_weight_kg"),
                            ("participant_base_gear_kg", "participant_base_gear_kg"),
                            ("adult_base_gear_kg", "adult_base_gear_kg")):
            if r.get(k_in):
                args[k_out] = r[k_in]
        if nights:
            args["night_min_c"] = min(nights)
        try:
            s = await links.call("plast-trail", "calculate_supplies", args)
        except McpCallError as exc:  # the plan stays valid; the note says supplies could not be computed
            s = {"error": f"{exc.code}: {exc.message}", "warnings": [f"Запаси не пораховано: {exc.message}"]}
        else:
            pk = s["pack_participant"]
            print(f"    їжа {s['food_kg_group']} кг на групу, газ {s['gas_g_group']} г, намети {s['tents']}, "
                  f"пальники {s['stoves']}; наплічник учасника {pk['total_kg']} кг (ліміт {pk['limit_kg']})")
        warnings = [x for x in v["violations"] if x["severity"] == "warning"] + s["warnings"]
        if w.get("overall_risk") in ("high", "unknown"):
            status = "risky"
        elif w.get("overall_risk") == "medium" or warnings:
            status = "attention"
        else:
            status = "ok"
        return {"supplies": s, "status": status}

    async def give_up(state: State) -> State:
        best = state.get("best")
        if best:  # an earlier attempt was valid (only the weather was bad): keep it, marked risky
            step("! Спроби вичерпано: беру найкращий валідний план")
            print(f"    {YELLOW}маршрут {best['proposal']['route_id']}, старт {best['proposal']['start_date']}, "
                  f"погода: {best['weather']['overall_risk']}{RESET}")
            return {"proposal": best["proposal"], "validation": best["validation"], "weather": best["weather"]}
        step("✗ Не вдалося скласти валідний план")
        print(f"    {YELLOW}{MAX_ATTEMPTS} спроби вичерпано; записую звіт про причини{RESET}")
        return {"status": "failed", "weather": {}}

    def after_give_up(state: State) -> str:
        return "supplies" if state.get("best") else "conclude"

    async def conclude(state: State) -> State:
        step("8. Висновок для провідника (LLM)")
        final = state.get("validation") or {}
        is_plan = state["status"] != "failed" and final.get("valid")
        facts = {
            "status": state["status"],
            "rules_status": final.get("rules_status"),
            "history": [{"attempt": a["n"], "outcome": a["outcome"]} for a in state["attempts"]],
            "final_plan": {
                "route": final.get("route_name"),
                "start_date": state["proposal"]["start_date"],
                "days": [{k: d[k] for k in ("day", "km", "ascent_m", "walking_hours", "est_arrival")}
                         for d in final.get("days", [])],
                "weather_overall": (state.get("weather") or {}).get("overall_risk"),
                "weather_triggers": [{"date": d["date"], "code": t["code"], "risk": t["risk"]}
                                     for d in (state.get("weather") or {}).get("days", []) for t in d["triggers"]],
                "warnings": [x["message"] for x in final.get("violations", []) if x["severity"] == "warning"][:6]
                            + (state.get("supplies") or {}).get("warnings", [])[:3],
            } if is_plan else None,
            "last_errors": [f for a in state["attempts"][-1:] for f in a["feedback"]][:4] if not is_plan else [],
        }
        try:
            text = llm.text("Ти помічник пластового провідника. Напиши висновок українською, 3-5 речень, "
                            "лише з наданих фактів, без нових чисел. Якщо правила DRAFT — скажи, що їх треба "
                            "затвердити. Статус attention — план валідний, але назви головні попередження й що з ними "
                            "робити. Якщо статус risky або failed — чітко скажи, що план не рекомендований без змін і чому. "
                            "history — це відхилені спроби, не описуй їх як чинний план. Не згадуй чисел, дат, назв чи "
                            "погодних явищ, яких немає в JSON.", json.dumps(facts, ensure_ascii=False))
        except Exception as exc:  # the plan note must still be written
            text = f"(Висновок моделі недоступний: {type(exc).__name__}. Дивіться таблиці нижче.)"
        print(f"    {text[:300]}")
        return {"conclusion": text.strip()}

    g = StateGraph(State)
    for name, fn in [("read_request", read_request), ("parse_request", parse_request),
                     ("search_routes", search_routes), ("propose", propose), ("validate", validate),
                     ("weather", weather), ("supplies", supplies), ("give_up", give_up), ("conclude", conclude)]:
        g.add_node(name, fn)
    g.add_edge(START, "read_request")
    g.add_edge("read_request", "parse_request")
    g.add_edge("parse_request", "search_routes")
    g.add_edge("search_routes", "propose")
    g.add_conditional_edges("propose", after_propose, {"validate": "validate", "propose": "propose",
                                                       "give_up": "give_up"})
    g.add_conditional_edges("validate", after_validate, {"weather": "weather", "propose": "propose",
                                                         "give_up": "give_up"})
    g.add_conditional_edges("weather", after_weather, {"propose": "propose", "supplies": "supplies"})
    g.add_edge("supplies", "conclude")
    g.add_conditional_edges("give_up", after_give_up, {"supplies": "supplies", "conclude": "conclude"})
    g.add_edge("conclude", END)
    return g.compile()


def now_kyiv() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
