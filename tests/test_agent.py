"""Tests of the agent workflow: the REAL plast-trail MCP server over an in-memory MCP session,
a fake Obsidian and a scripted fake LLM (no network, no model quota).

Run:  uv run pytest -q
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from plast_agent import cli
from plast_agent.llm import LLMError, ParsedRequest, Proposal
from plast_agent.mcp_links import McpLinks
from plast_agent.workflow import build_graph
from plast_trail.server import mcp as plast_trail_server

REQUEST = "Запити/Мандрівка гуртка Сокіл.md"
NOTE = (Path(__file__).resolve().parents[1] / "demo-vault" / REQUEST).read_text(encoding="utf-8")
PARSED = {"section": "yunatstvo", "participants": 10, "adults": 2, "days": 3, "start_date": "2026-11-21",
          "date_flexibility_days": 2, "region": "chornohora", "must_include": ["lake"], "start_time": "8:00",
          "max_km_per_day": 12, "max_ascent_per_day_m": None, "max_walking_hours_per_day": None,
          "max_overnight_elevation_m": 1600, "lightest_body_weight_kg": 40, "participant_base_gear_kg": 5,
          "adult_base_gear_kg": 7}
GOOD = {"route_id": "r8613349", "overnight_waypoint_ids": ["w56", "w30"], "start_date": "2026-11-23",
        "reasoning": "рівні дні"}


# The demo note with one adult: for the tests of the escort advice (the text must say what the reading says).
NOTE_ONE_ADULT = NOTE.replace("і 2 дорослі виховники", "і 1 дорослий виховник")


class FakeObsidian:
    """Answers vault_read / vault_list / vault_write / vault_append like the plugin does (JSON text)."""

    def __init__(self, note: str = NOTE) -> None:
        self.written: dict[str, str] = {}
        self.note = note

    async def call_tool(self, tool: str, args: dict) -> CallToolResult:
        if tool == "vault_read" and args["path"] in self.written:  # notes saved during this run
            body = {"content": self.written[args["path"]], "path": args["path"]}
        elif tool == "vault_read":
            if args["path"] != REQUEST:
                return CallToolResult(content=[TextContent(type="text", text=f"File not found: {args['path']}")],
                                      isError=True)
            body = {"content": self.note, "path": REQUEST}
        elif tool == "vault_list":
            body = {"files": [Path(REQUEST).name]}
        elif tool == "vault_append":  # like the plugin: add to the end, create if missing
            self.written[args["path"]] = self.written.get(args["path"], "") + args["content"]
            body = {"version": "2"}
        else:
            self.written[args["path"]] = args["content"]
            body = {"version": "1"}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(body, ensure_ascii=False))])


class FakeLLM:
    """Returns scripted answers in order; an Exception in the script is raised instead."""

    model_name = "fake-model"

    def __init__(self, parsed: dict, proposals: list) -> None:
        self.script = [parsed, *proposals]
        self.calls = 0

    async def structured(self, schema, system: str, user: str):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert schema in (ParsedRequest, Proposal)
        return schema(**item)

    async def text(self, system: str, user: str) -> str:
        self.calls += 1
        return "Висновок."


def run_agent(llm: FakeLLM, note: str = NOTE) -> tuple[dict, McpLinks]:
    settings = SimpleNamespace(obsidian_mcp_url="https://obsidian.test/mcp/", plast_trail_url="memory://plast-trail")

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(settings)
            links.sessions = {"plast-trail": session, "obsidian": FakeObsidian(note)}
            state = await build_graph(links, llm).ainvoke({"request_path": REQUEST}, {"recursion_limit": 80})
            return state, links

    return asyncio.run(go())


def test_wrong_night_count_is_rejected_in_code_before_any_tool_call():
    state, _ = run_agent(FakeLLM(PARSED, [{**GOOD, "overnight_waypoint_ids": ["w30"]}, GOOD]))
    first = state["attempts"][0]
    assert first["outcome"] == "rejected by the agent check"
    assert any("exactly 2 night spot" in f for f in first["feedback"])
    assert state["validation"]["valid"] and state["status"] == "attention"
    assert state["proposal"]["overnight_waypoint_ids"] == ["w30", "w56"]  # sorted by km in code
    assert state["proposal"]["start_time"] == "08:00"                     # normalised from "8:00"


def test_route_outside_candidates_and_date_outside_window_are_rejected():
    bad_route = {**GOOD, "route_id": "r14097891"}
    bad_date = {**GOOD, "start_date": "2026-11-15"}
    state, _ = run_agent(FakeLLM(PARSED, [bad_route, bad_date, GOOD]))
    fb = [f for a in state["attempts"][:2] for f in a["feedback"]]
    assert any("not one of the candidates" in f for f in fb)
    assert any("outside the allowed window" in f for f in fb)
    assert state["validation"]["valid"]


def test_repeat_of_a_rejected_plan_is_refused():
    high = {**GOOD, "start_date": "2026-11-21"}            # valid but HIGH weather in the fixtures
    state, _ = run_agent(FakeLLM(PARSED, [high, high, GOOD]))
    assert "HIGH weather" in state["attempts"][0]["outcome"]
    assert "repeats attempt 1" in " ".join(state["attempts"][1]["feedback"])
    assert state["weather"]["overall_risk"] == "medium"


def test_give_up_keeps_the_best_valid_plan_instead_of_failing():
    high = {**GOOD, "start_date": "2026-11-21"}
    bad = {**GOOD, "route_id": "r1"}
    state, _ = run_agent(FakeLLM(PARSED, [high, bad, bad, bad]))
    assert state["status"] == "risky"                       # not "failed": a valid plan existed
    assert state["validation"]["valid"] and state["proposal"]["start_date"] == "2026-11-21"
    assert state["weather"]["overall_risk"] == "high"       # the weather of THAT plan
    assert state["supplies"]["food_kg_group"] > 0


def test_server_suggestion_names_shown_night_spots():
    bad_split = {**GOOD, "overnight_waypoint_ids": ["w23", "w47"]}
    state, _ = run_agent(FakeLLM(PARSED, [bad_split, GOOD]))
    fb = " ".join(state["attempts"][0]["feedback"])
    assert "w30" in fb and "w56" in fb and "Add one more night" not in fb


def test_model_failure_is_reported_as_the_model(capsys):
    llm = FakeLLM(PARSED, [LLMError("OpenRouter: ліміт запитів вичерпано (HTTP 429)")])

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": FakeObsidian()}
            return await cli.plan_and_write(links, None, REQUEST, dry_run=True, llm=llm)

    code = asyncio.run(go())
    out = capsys.readouterr().out
    assert code == 1
    assert "Модель недоступна" in out and "429" in out and "✗ plast-trail" not in out


def test_missing_request_note_lists_existing_requests(capsys):
    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": FakeObsidian()}
            return await cli.plan_and_write(links, None, "Запити/Немає.md", dry_run=True, llm=FakeLLM(PARSED, []))

    assert asyncio.run(go()) == 1
    assert "Наявні запити: Запити/Мандрівка гуртка Сокіл.md" in capsys.readouterr().out


# ------------------------------------------------------------------ schema contracts over real MCP (P2)
def call(tool: str, args: dict):
    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
            return tools, await session.call_tool(tool, args)
    return asyncio.run(go())


PLAN = {"route_id": "r8613349", "section": "yunatstvo", "participants": 10, "adults": 2,
        "overnight_waypoint_ids": ["w30", "w56"], "start_date": "2026-11-23"}


def test_invalid_time_and_unknown_limit_are_rejected_by_the_schema():
    _, res = call("validate_hike_plan", {**PLAN, "start_time": "12:99"})
    assert res.isError
    _, res = call("validate_hike_plan", {**PLAN, "group_limits": {"max_km_per_dayy": 5}})
    assert res.isError and "extra" in res.content[0].text.lower()


def test_outputs_are_typed_and_chain_into_supplies():
    tools, v = call("validate_hike_plan", PLAN)
    schema = json.dumps(tools["validate_hike_plan"].outputSchema)
    assert '"OVERNIGHT_NO_WATER"' in schema and '"campsite"' in schema   # enums, not free strings
    v = v.structuredContent
    _, w = call("assess_weather_risk", {k: PLAN[k] for k in ("route_id", "start_date", "overnight_waypoint_ids")})
    w = w.structuredContent
    assert all(t["unit"] in ("wmo_code", "m/s", "mm", "°C") for d in w["days"] for t in d["triggers"])
    assert w["coldest_night_c"] == min(d["night_spot_min_c"] for d in w["days"] if d["night_spot_min_c"] is not None)
    loads = [{k: d[k] for k in ("km", "ascent_m", "walking_hours")} for d in v["days"]]
    _, s = call("calculate_supplies", {"section": "yunatstvo", "participants": 10, "adults": 2,
                                       "day_loads": loads, "dry_nights": v["dry_nights"],
                                       "night_min_c": w["coldest_night_c"]})
    assert not s.isError and s.structuredContent["num_days"] == 3        # the three numbers of each day chain in


def test_dry_nights_cannot_exceed_nights():
    _, res = call("calculate_supplies", {"section": "yunatstvo", "participants": 10, "adults": 2,
                                         "day_loads": [{"km": 5, "ascent_m": 100, "walking_hours": 2}], "dry_nights": 3})
    assert res.isError and "INVALID_INPUT" in res.content[0].text


def test_out_of_range_limit_from_the_note_is_dropped_and_reported():
    from plast_agent.workflow import sanitise_request
    d, notes = sanitise_request({**PARSED, "max_km_per_day": 50})
    assert d["max_km_per_day"] is None and notes and d["start_time"] == "08:00"


def test_incomplete_request_stops_and_says_what_is_missing():
    from plast_agent.mcp_links import _leaves
    from plast_agent.workflow import AgentStop
    with pytest.raises(Exception) as e:
        run_agent(FakeLLM({**PARSED, "adults": None, "section": None, "start_date": None}, []))
    stop = next(x for x in _leaves(e.value) if isinstance(x, AgentStop))   # unwrapped like cli.py does
    assert "кількість дорослих" in str(stop) and "вікова група" in str(stop)
    assert "дата старту" not in str(stop)        # the code read «21 листопада 2026» from the note itself


def test_note_shows_what_the_agent_understood_and_cleans_model_text():
    from plast_agent.render import render_plan
    sneaky = {**GOOD, "reasoning": "ok ![img](http://evil) [[Плани/x]] <b>x</b>"}
    state, _ = run_agent(FakeLLM({**PARSED, "date_flexibility_days": 10}, [sneaky]))
    note = render_plan(state, REQUEST, "fake-model", "2026-11-15 10:00")
    assert "## Що агент зрозумів із запиту" in note and "юнацтво, учасників 10, дорослих 2" in note
    assert "можна зсунути до 3 дн." in note                          # 10 days clamped to 3, and said so
    assert "![" not in note.split("## Як агент")[1] and "[[Плани/x]]" not in note



# ------------------------------------------------------------------ chat mode
def run_chat(llm: FakeLLM, replies: list[str]):
    obsidian = FakeObsidian()
    answers = iter(replies)
    asked: list[str] = []

    async def ask(prompt: str) -> str:
        asked.append(prompt)
        return next(answers)

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": obsidian}
            return await cli.plan_and_write(links, None, None, dry_run=False, llm=llm, chat_mode=True, ask=ask)

    return asyncio.run(go()), obsidian, asked


def test_chat_asks_for_what_is_missing_then_saves_and_plans():
    no_adults = {**PARSED, "adults": None}
    llm = FakeLLM(no_adults, [GOOD])   # the reading misses adults; the answer «2» is read by code
    code, obsidian, asked = run_chat(llm, [
        "Юнацтво, 10 людей, Чорногора, хочемо озеро, 3 дні з 21 листопада, можемо зсунути на 2 дні", "",
        "2",                                                                       # answer: adults
        "",                                                                        # confirm: yes
        "Похід Сокіл з чату",                                                      # note title
    ])
    assert code == 0
    assert any("дорослих" in q for q in asked)                    # the agent asked about the adults
    request = obsidian.written["Запити/Похід Сокіл з чату.md"]
    assert "Уточнення" in request and "дорослих: 2" in request.lower()
    assert "Плани/Похід Сокіл з чату — план.md" in obsidian.written    # planned and written back
    assert llm.calls == 3                                          # 1 reading + 1 proposal + conclusion


def test_chat_can_be_cancelled_without_writing_anything():
    code, obsidian, _ = run_chat(FakeLLM(PARSED, []), ["Юнацтво, 10 + 2, Чорногора, 3 дні з 21.11", "", "ні"])
    assert code == 0 and obsidian.written == {}


def test_leaders_brebeneskul_request_gets_a_nearby_route_and_a_picture(capsys):
    """The leader's test of 15 Nov 21:05: «7 хлопців … на бребенеско … 2 дні», 1 adult. It ended as `failed`
    on a route near Кукул with no picture. Now: the nearest routes to Бребенескул, status `group`, a picture."""
    from plast_trail.routes import find_routes
    near = find_routes("yunatstvo", 2, "chornohora", [], 4, None, ["Бребенескул"]).routes[0]
    options = near.overnight_options
    proposal = {"route_id": near.id, "overnight_waypoint_ids": [options[len(options) // 2].id],
                "start_date": "2026-11-23", "reasoning": "найближче до Бребенескула"}
    reading = {"section": "yunatstvo", "participants": 7, "adults": 1, "days": 2, "start_date": None,
               "date_flexibility_days": 0, "region": None, "must_include": [], "places": ["Бребенеско"],
               "start_time": None}
    code, obsidian, asked = run_chat(FakeLLM(reading, [proposal]), [
        "планую піти я із юнацтвом, 7 хлопців, думаємо піти на бребенеско, план на 2 дні", "",
        "думаєм 23 листопада", "", ""])
    out = capsys.readouterr().out
    plan = next(v for k, v in obsidian.written.items() if k.endswith("план.md"))
    assert code == 0 and "План готовий · за можливості візьміть ще дорослого" in plan
    assert "## Інші варіанти маршруту" in plan
    assert "місця: Бребенескул" in plan                              # the dataset's name, not «Бребенеско»
    assert "Найближче підходить" in plan and "Перевірте написання" not in plan
    assert any(k.endswith("маршрут.svg") for k in obsidian.written) and "маршрут.svg]]" in plan
    assert "Район: Чорногора" in out or "chornohora" in plan


# ------------------------------------------------------------------ model provider chain (OpenRouter -> Ollama)
import httpx  # noqa: E402
import openai  # noqa: E402

from plast_agent import llm as L  # noqa: E402


class FakeRunner:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    async def ainvoke(self, messages):
        b = self.behaviour.pop(0) if isinstance(self.behaviour, list) else self.behaviour
        if b == "slow":
            await asyncio.sleep(5)
        if isinstance(b, Exception):
            raise b
        return b


class FakeChat:
    def __init__(self, behaviour):
        self.behaviour, self.methods = behaviour, []

    def with_structured_output(self, schema, method, **kwargs):
        self.methods.append(method)
        return FakeRunner(self.behaviour)


def provider(name, behaviour, deadline=2):
    chat = FakeChat(behaviour)
    return L.Provider(name=name, model=f"{name}-model", chat=chat, quick=FakeRunner(behaviour),
                      method="json_schema" if name == "ollama" else "function_calling",
                      structured_deadline=deadline, text_deadline=deadline)


def rate_limited():
    return openai.RateLimitError("429", response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                                 body=None)


@pytest.fixture
def quota_file(tmp_path, monkeypatch):
    path = tmp_path / "quota.json"
    monkeypatch.setattr(L, "QUOTA_FILE", path)
    return path


def test_quota_error_switches_to_ollama_and_is_remembered(quota_file):
    ok = ParsedRequest(**PARSED)
    cloud, local = provider("openrouter", rate_limited()), provider("ollama", ok)
    model = L.LLM(providers=[cloud, local])
    result = asyncio.run(model.structured(ParsedRequest, "s", "u"))
    assert result.participants == 10
    assert model.provider.name == "ollama" and len(model.switches) == 1
    assert local.chat.methods == ["json_schema"]              # constrained decoding for Ollama
    assert model.calls_by == {"openrouter/openrouter-model": 1, "ollama/ollama-model": 1}
    assert quota_file.exists() and L.quota_exhausted_now()      # no knocking on OpenRouter until 02:00 Kyiv


def test_slow_cloud_model_falls_back_too(quota_file):
    model = L.LLM(providers=[provider("openrouter", "slow", deadline=0.1), provider("ollama", ParsedRequest(**PARSED))])
    assert asyncio.run(model.structured(ParsedRequest, "s", "u")).days == 3
    assert "не відповіла за" in model.switches[0] and not quota_file.exists()   # a timeout is not a quota


def test_both_providers_down_gives_one_clear_error(quota_file):
    model = L.LLM(providers=[provider("openrouter", rate_limited()),
                             provider("ollama", openai.APIConnectionError(request=httpx.Request("POST", "https://x")))])
    with pytest.raises(L.LLMError) as e:
        asyncio.run(model.structured(ParsedRequest, "s", "u"))
    assert "Ollama не відповідає" in str(e.value)


def test_start_with_fallback_skips_the_cloud(quota_file):
    cloud, local = provider("openrouter", rate_limited()), provider("ollama", ParsedRequest(**PARSED))
    model = L.LLM(providers=[cloud, local], start_with_fallback=True)
    asyncio.run(model.structured(ParsedRequest, "s", "u"))
    assert cloud.chat.methods == [] and model.model_name == "ollama/ollama-model"


def test_quota_reset_is_midnight_utc():
    from datetime import datetime, timezone
    assert L.next_quota_reset(datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)).isoformat() == "2026-11-16T00:00:00+00:00"


def test_chat_with_two_adults_and_named_peaks_like_the_real_session():
    """Replays the leader's real test: 'іду я із сестрою… на Говерлу 23-го, спуститися біля Петроса 24-го'."""
    first = {"section": None, "participants": 2, "adults": None, "days": 2, "start_date": "2026-11-23",
             "date_flexibility_days": 0, "region": "chornohora", "must_include": ["peak"],
             "places": ["Говерла", "Петрос"], "start_time": None}
    proposal = {"route_id": "r8613511", "overnight_waypoint_ids": ["w4"], "start_date": "2026-11-23",
                "reasoning": "через Говерлу"}
    code, obsidian, asked = run_chat(FakeLLM(first, [proposal]), [   # the age answer needs no 2nd reading
        "іду я із сестрою, хочемо на говерлу піднятися 23 листопада, а 24 спуститися біля петроса", "",
        "",               # an empty answer: the agent must ask again, not continue with None
        "20 років нам",   # the group from an age, no model call needed
        "",               # confirm
        "",               # automatic title
    ])
    assert code == 0
    section_questions = [q for q in asked if "Хто йде" in q]
    assert len(section_questions) == 2                                   # asked again after the empty answer
    assert not any("дорослих (виховників" in q for q in asked)          # adults are not asked for an adult group
    request = next(v for k, v in obsidian.written.items() if k.startswith("Запити/"))
    assert "Група: 20 років нам" in request and "Дорослих супровідників: 0" in request
    plan = next(v for k, v in obsidian.written.items() if k.startswith("Плани/") and k.endswith(".md"))
    assert any(k.endswith("маршрут.svg") for k in obsidian.written)     # the route picture next to the plan
    assert "старші / сеніори, вас 2" in plan
    assert "Жоден маршрут у даних не проходить через усі місця (Говерла, Петрос)" in plan


def test_guess_section_from_words_and_ages():
    from plast_agent.chat import guess_section
    assert guess_section("20 років нам") == "senior"
    assert guess_section("нам по 14") == "yunatstvo"
    assert guess_section("новачки, рій") == "novatstvo"
    assert guess_section("дорослі") == "senior"
    assert guess_section("не знаю") is None


def test_chat_reads_answers_by_code_like_the_real_session(capsys):
    """The leader's real dialogue of 15 Nov: earlier every answer cost a model request and the 5th hit the
    daily limit (429). Now the model reads the description once; the answers are read by code."""
    reading = {"section": "yunatstvo", "participants": 11, "adults": 0, "days": None, "start_date": None,
               "date_flexibility_days": 0, "region": "chornohora", "must_include": ["hut", "campsite"],
               "places": ["Піпівана", "Бребенеско"], "start_time": None}   # like gemma4:e4b's real reading
    llm = FakeLLM(reading, [])
    code, obsidian, asked = run_chat(llm, [
        "хочу повести своїх дітей, наш гурток 10 хлопців десь 11-13 років, думав почати із піпівана "
        "і закінчити бребенеско", "",
        "я думаю 1 буду",                                  # adults
        "не важливо, хочемо вийти 19 листопада",           # days: an assumption + the start date in passing
        "ні",                                              # cancel at the summary
    ])
    out = capsys.readouterr().out
    assert llm.calls == 1                                  # one model reading for the whole dialogue
    assert [q for q in asked if "Скільки дорослих" in q] and [q for q in asked if "Скільки днів" in q]
    assert not [q for q in asked if "Коли старт" in q]     # the date came with the days answer
    assert "учасників 10" in out and "моє припущення" in out and "Прибрано побажання" in out
    assert "«Піпівана» → Піп Іван Чорногірський" in out              # plast-trail.resolve_places
    assert "Бребенескул" in out and "жоден маршрут у даних" in out     # the data gap, said plainly
    assert obsidian.written == {}


def test_too_few_adults_gives_the_full_plan_and_asks_to_fix_the_group():
    """Only the group breaks the rules: the route is fine, so the plan is finished (weather, supplies, picture)
    with status `group`, after one attempt: another route cannot add an adult."""
    llm = FakeLLM({**PARSED, "adults": 1}, [GOOD])
    state, _ = run_agent(llm, NOTE_ONE_ADULT)
    codes = {x["code"] for x in state["validation"]["violations"] if x["severity"] == "error"}
    assert codes and codes <= {"TOO_FEW_ADULTS", "ADULT_RATIO"}
    assert len(state["attempts"]) == 1 and state["status"] == "group"
    assert state["weather"]["days"] and state["supplies"] and state["route_profile"]["days"]
    assert llm.calls == 2                                  # reading + one proposal; the conclusion is written by code
    assert "підходить" in state["conclusion"] and "За можливості візьміть ще когось" in state["conclusion"]


def test_plan_note_gets_a_route_picture():
    """get_route_profile -> an SVG the plan note embeds: one colour per day, night spots, peaks, steepness."""
    import xml.etree.ElementTree as ET

    from plast_agent.picture import route_svg
    from plast_agent.render import render_plan
    state, links = run_agent(FakeLLM(PARSED, [GOOD]))
    prof = state["route_profile"]
    assert [d["day"] for d in prof["days"]] == [1, 2, 3]
    assert sum(d["km"] for d in prof["days"]) == pytest.approx(sum(d["km"] for d in state["validation"]["days"]), abs=0.2)
    assert [d["ascent_m"] for d in prof["days"]] == [d["ascent_m"] for d in state["validation"]["days"]]
    svg = route_svg(prof, state["validation"], state["proposal"]["start_date"])
    ET.fromstring(svg)                                   # well-formed: Obsidian can show it
    assert "День 1" in svg and "День 3" in svg and "ночівля 2" in svg and "Петрос" in svg
    note = render_plan({**state, "picture_path": "Плани/x — маршрут.svg"}, REQUEST, "fake-model", "2026-11-15 10:00")
    assert "![[Плани/x — маршрут.svg]]" in note
    Path(__file__).resolve().parents[1].joinpath("out").mkdir(exist_ok=True)
    Path(__file__).resolve().parents[1].joinpath("out", "test-route-picture.svg").write_text(svg, encoding="utf-8")


def test_route_profile_rejects_unknown_night_spots():
    _, res = call("get_route_profile", {"route_id": "r8613349", "overnight_waypoint_ids": ["w999"]})
    assert res.isError and "WAYPOINT_NOT_FOUND" in res.content[0].text


def test_plan_note_links_to_a_real_map_and_a_gpx_track():
    import xml.etree.ElementTree as ET

    from plast_agent.picture import mapy_link, route_gpx
    from plast_agent.render import render_plan
    state, _ = run_agent(FakeLLM(PARSED, [GOOD]))
    prof = state["route_profile"]
    from plast_agent.picture import day_track, route_track
    link = mapy_link(route_track(prof))
    nights = [x for x in prof["landmarks"] if x["night"]]
    day2 = day_track(prof, prof["days"][1])
    assert (day2[0][1], day2[0][2]) == (nights[0]["lat"], nights[0]["lon"])   # day 2 starts exactly at night 1
    assert [p[0] for p in route_track(prof)] == sorted(p[0] for p in route_track(prof))   # no back-and-forth
    assert link.startswith("https://mapy.com/fnc/v1/route?mapset=outdoor&routeType=foot_hiking&start=")
    assert 1 <= link.split("waypoints=")[1].count(";") + 1 <= 13          # Mapy.com takes at most 15 points
    gpx = ET.fromstring(route_gpx(prof, [{"name": "Бребенескул", "lat": 48.1, "lon": 24.56, "elevation_m": 2038}]))
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    assert len(gpx.findall("g:trk", ns)) == 3 and len(gpx.findall(".//g:trkpt", ns)) > 100
    assert any("Бребенескул" in (w.findtext("g:name", "", ns)) for w in gpx.findall("g:wpt", ns))
    note = render_plan({**state, "gpx_path": "Плани/x — маршрут.gpx"}, REQUEST, "fake-model", "2026-11-15 10:00")
    assert "[Увесь маршрут](https://mapy.com/" in note and "[день 3](" in note and "[[Плани/x — маршрут.gpx]]" in note


def test_interactive_map_block_for_the_leaflet_plugin():
    from plast_agent.picture import leaflet_block
    from plast_agent.render import render_plan
    state, _ = run_agent(FakeLLM(PARSED, [GOOD]))
    wished = [{"name": "Бребенескул, озеро", "lat": 48.1, "lon": 24.56, "elevation_m": 2038}]
    block = leaflet_block(state["route_profile"], "Плани/x — маршрут.gpx", wished, "plast-test")
    import re

    import yaml
    lines = block.splitlines()
    assert lines[0] == "```leaflet" and lines[-1] == "```" and "gpx: [[Плани/x — маршрут.gpx]]" in lines
    # the plugin hides [[links]] and parses the rest as YAML; a YAML error broke the whole map (15 Nov)
    body = re.sub(r"\[\[[^\]]*\]\]", "LEAFLET_INTERNAL_LINK_0", "\n".join(lines[1:-1]))
    params = yaml.safe_load(body)
    assert params["gpxColor"] == "#ff1f5a" and params["unit"] == "km"
    # CARTO (the plugin's default) needs a key since 2026: off; OpenTopoMap first, on its real subdomains a-c
    assert params["osmLayer"] is False and params["tileSubdomains"] == "a,b,c"
    assert params["tileServer"][0].startswith("https://{s}.tile.opentopomap.org") and len(params["tileServer"]) == 2
    assert all(len(layer.split("|")) == 2 for layer in params["tileServer"])   # the plugin splits url|name
    assert "zoomFeatures" not in params                       # it left a grey map before the GPX loaded
    assert len(params["marker"]) == 2 + 2 + 1                 # start, finish, 2 nights, the wished place
    assert all(m.count(",") == 4 for m in params["marker"])   # a comma in a name would shift the fields
    assert any("Бребенескул" in m for m in params["marker"])
    plain = render_plan({**state, "gpx_path": "Плани/x — маршрут.gpx"}, REQUEST, "m", "2026-11-15 10:00")
    with_map = render_plan({**state, "gpx_path": "Плани/x — маршрут.gpx", "leaflet": True}, REQUEST, "m", "2026-11-15 10:00")
    assert "```leaflet" not in plain and "```leaflet" in with_map   # only when the plugin is there


def test_map_zoom_fits_the_route():
    from plast_agent.picture import fit_zoom
    # the plan «Перевірка карти»: Бистрець -> Ґаджина and Бребенескул, ~9 x 5 km: zoom 12 (11 left it tiny)
    assert fit_zoom(48.0976, 48.1400, 24.5552, 24.6804) == 12
    assert fit_zoom(48.0, 48.6, 23.9, 24.9) == 9               # a long trip: the whole range, not a crop
    assert fit_zoom(48.10, 48.105, 24.55, 24.555) == 15        # a short walk: not zoomed past 15


def test_a_named_place_with_no_fitting_route_is_relaxed_not_a_dead_end(capsys):
    """Offline test of 15 Nov: «Горгани, 3 дні, хочемо на Сивулю, ≤12 км/день» stopped: the only route
    through Сивуля is too short for 3 days. Now the place is relaxed like the other wishes, and said so."""
    from plast_trail.routes import find_routes
    from itertools import combinations

    from plast_trail.models import GroupLimits
    from plast_trail.validation import validate_hike_plan
    near = find_routes("yunatstvo", 3, "gorgany", [], 4, 12, []).routes[0]
    opts = near.overnight_options
    # the most even split into 3 days that the rules accept (the model's job in a real run)
    pairs = sorted(combinations(opts, 2), key=lambda ab: max(ab[0].km, ab[1].km - ab[0].km, near.length_km - ab[1].km))
    nights = next([a.id, b.id] for a, b in pairs if a.km < b.km and validate_hike_plan(
        near.id, "yunatstvo", 10, 2, [a.id, b.id], None, None, GroupLimits(max_km_per_day=12)).valid)
    proposal = {"route_id": near.id, "overnight_waypoint_ids": nights, "start_date": "2026-11-21",
                "reasoning": "Горгани без Сивулі"}
    reading = {"section": "yunatstvo", "participants": 10, "adults": 2, "days": 3, "start_date": "2026-11-21",
               "date_flexibility_days": 0, "region": "gorgany", "must_include": [], "places": ["Сивуля"],
               "start_time": None, "max_km_per_day": 12}
    code, obsidian, _ = run_chat(FakeLLM(reading, [proposal]), [
        "Гурток: 10 юнаків і 2 виховники. 3 дні з 21 листопада, Горгани, хочемо на Сивулю. Не більше 12 км на день.",
        "", "", ""])
    plan = next(v for k, v in obsidian.written.items() if k.endswith("план.md"))
    assert "Немає маршруту на 3 дн. з умовою «маршрут через Сивуля» (закороткий на стільки днів" in plan
    assert code == 0 and "status: failed" not in plan


def test_a_place_the_model_missed_is_found_in_the_text():
    from plast_agent.workflow import resolve_named_places

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session}
            d = {"region": None, "places": []}
            notes = await resolve_named_places(links, d, "на вихідних ідемо курінем у Свидовець, хочемо на "
                                                          "Близницю, ночувати біля джерела")
            return d, notes
    d, notes = asyncio.run(go())
    assert d["places"] == ["Близниця"] and d["region"] == "svydovets"
    assert any("Місце з тексту: Близниця" in n for n in notes)
    assert "джерела" not in str(d["places"]).lower()


def test_conclusion_gets_safety_first_and_no_gear_weight_noise():
    class Recording(FakeLLM):
        async def text(self, system: str, user: str) -> str:
            self.facts = json.loads(user)
            return await super().text(system, user)
    llm = Recording({**PARSED, "participant_base_gear_kg": None, "adult_base_gear_kg": None}, [GOOD])
    state, _ = run_agent(llm)
    warnings = llm.facts["final_plan"]["warnings"]
    assert not any(w.startswith("Вагу спорядження не вказано") for w in warnings)
    from plast_agent.workflow import PROPOSE_SYSTEM
    assert "Do not state what Plast rules allow" in PROPOSE_SYSTEM


def test_the_agents_own_status_lines_are_not_read_as_the_request():
    """A re-run read «**Агент (2026-11-16 01:30):** …» from the note: «01:30» passed as the leader's start time
    and a model's invented 20:00 stayed (arrival at 04:24, 16 Nov)."""
    obsidian = FakeObsidian()
    obsidian.written[REQUEST] = NOTE + "\n\n---\n**Агент (2026-11-16 01:30):** attention → [[Плани/x]]\n"
    llm = FakeLLM({**PARSED, "start_time": "20:00"}, [GOOD])

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": obsidian}
            return await build_graph(links, llm).ainvoke({"request_path": REQUEST}, {"recursion_limit": 80})
    state = asyncio.run(go())
    assert "Агент" not in state["request_text"] and state["request"]["start_time"] == "08:00"


def test_factual_conclusion_puts_safety_first():
    from plast_agent.workflow import fallback_conclusion
    state, _ = run_agent(FakeLLM(PARSED, [GOOD]))
    text = fallback_conclusion(state)
    assert text.startswith("Маршрут «") and "Погодний ризик" in text and "найхолодніша ніч" in text
    assert "Вагу спорядження" not in text


# ------------------------------------------------------------------ independent review (Codex, 16 Nov)
def test_high_weather_is_never_hidden_by_the_escort_advice():
    """Codex P1: 1 adult + a high-risk forecast came out as «Готово · порада щодо супроводу»."""
    from plast_agent.render import render_plan
    high, bad = {**GOOD, "start_date": "2026-11-21"}, {**GOOD, "route_id": "r1"}
    state, _ = run_agent(FakeLLM({**PARSED, "adults": 1}, [high, bad, bad, bad]), NOTE_ONE_ADULT)
    assert state["weather"]["overall_risk"] == "high" and state["status"] == "risky"
    note = render_plan(state, REQUEST, "fake", "2026-11-16 10:00")
    assert "НЕ рекомендовано" in note and "**Супровід (порада):**" in note    # both, the weather first


def test_status_rule_puts_safety_first():
    from plast_agent.workflow import plan_status
    adults = {"code": "TOO_FEW_ADULTS", "severity": "error"}
    days = {"code": "TOO_MANY_DAYS", "severity": "error"}
    v = lambda *xs: {"violations": list(xs)}   # noqa: E731
    assert plan_status(v(adults), {"overall_risk": "high"}, []) == "risky"
    assert plan_status(v(adults), {"overall_risk": "unknown"}, []) == "risky"
    assert plan_status(v(adults), {"overall_risk": "low"}, []) == "group"
    assert plan_status(v(days), {"overall_risk": "low"}, []) == "failed"      # too many days is not advice
    assert plan_status(v(adults, days), {"overall_risk": "low"}, []) == "failed"
    assert plan_status(v(), {"overall_risk": "medium"}, []) == "attention"
    assert plan_status(v(), {"overall_risk": "low"}, []) == "ok"


def test_a_conclusion_with_invented_numbers_is_replaced_by_the_facts():
    """Codex P2 (twice): «1 км … +25 °C» and then «+10.4 °C» for a −10.4 °C night went into the plan. Numbers
    now come from the tools only: a model paragraph with any digit is replaced by the factual one."""
    class Writer(FakeLLM):
        def __init__(self, paragraph):
            super().__init__(PARSED, [GOOD])
            self.paragraph = paragraph

        async def text(self, system, user):
            self.calls += 1
            return self.paragraph
    for invented in ("Безпечний маршрут, 1 км, повертайтеся о 10:00, прогноз +25 °C.", "Ніч до +10.4 °C, це тепло."):
        state, _ = run_agent(Writer(invented))
        assert state["conclusion"].startswith("Маршрут «") and "+10.4" not in state["conclusion"]
        assert "−10.4" in state["conclusion"] or "-10.4" in state["conclusion"]     # the real frost, from the tool
    words = "Другий день найважчий: вийдіть раніше й візьміть зимові спальники."
    state, _ = run_agent(Writer(words))
    assert state["conclusion"] == words                                               # words only: kept as written



def test_unknown_arguments_and_schema_failures_are_structured_errors():
    """Codex P2: find_routes(typo_limit=1) succeeded silently; a schema failure was plain Pydantic text."""
    tools, res = call("find_routes", {"section": "yunatstvo", "days": 1, "typo_limit": 1})
    assert all(t.inputSchema.get("additionalProperties") is False for t in tools.values())
    assert res.isError
    for args in ({"section": "yunatstvo", "days": 1, "typo_limit": 1}, {"section": "WRONG", "days": 1},
                 {"section": "yunatstvo", "days": 0}):
        _, res = call("find_routes", args)
        text = res.content[0].text
        err = json.loads(text[text.index("{"):])["error"]
        assert res.isError and err["code"] == "INVALID_INPUT" and err["hint"]
    _, nested = call("calculate_supplies", {"section": "yunatstvo", "participants": 10, "adults": 2,
                                            "day_loads": [{"km": 5, "ascent_m": 100, "walking_hours": 2, "kmm": 1}]})
    assert nested.isError and "INVALID_INPUT" in nested.content[0].text     # nested fields are strict too (round 2)
    assert all(t.outputSchema.get("additionalProperties") is False for t in tools.values())


def test_too_many_dry_nights_is_invalid_input_not_a_server_bug():
    """Found while fixing the second review: the tool raised an already wrapped error, so the caller saw
    INTERNAL_ERROR («server bug») for a plain input mistake."""
    _, res = call("calculate_supplies", {"section": "yunatstvo", "participants": 8, "adults": 2, "dry_nights": 1,
                                         "day_loads": [{"km": 5, "ascent_m": 100, "walking_hours": 2}]})
    text = res.content[0].text
    assert res.isError and json.loads(text[text.index("{"):])["error"]["code"] == "INVALID_INPUT"
