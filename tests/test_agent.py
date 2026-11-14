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


class FakeObsidian:
    """Answers vault_read / vault_list / vault_write / vault_append like the plugin does (JSON text)."""

    def __init__(self) -> None:
        self.written: dict[str, str] = {}

    async def call_tool(self, tool: str, args: dict) -> CallToolResult:
        if tool == "vault_read" and args["path"] in self.written:  # notes saved during this run
            body = {"content": self.written[args["path"]], "path": args["path"]}
        elif tool == "vault_read":
            if args["path"] != REQUEST:
                return CallToolResult(content=[TextContent(type="text", text=f"File not found: {args['path']}")],
                                      isError=True)
            body = {"content": NOTE, "path": REQUEST}
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


def run_agent(llm: FakeLLM) -> tuple[dict, McpLinks]:
    settings = SimpleNamespace(obsidian_mcp_url="https://obsidian.test/mcp/", plast_trail_url="memory://plast-trail")

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(settings)
            links.sessions = {"plast-trail": session, "obsidian": FakeObsidian()}
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
    _, s = call("calculate_supplies", {"section": "yunatstvo", "participants": 10, "adults": 2,
                                       "day_loads": v["days"], "dry_nights": v["dry_nights"],
                                       "night_min_c": w["coldest_night_c"]})
    assert not s.isError and s.structuredContent["num_days"] == 3        # days[] can be passed as-is


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
        run_agent(FakeLLM({**PARSED, "adults": None, "start_date": None}, []))
    stop = next(x for x in _leaves(e.value) if isinstance(x, AgentStop))   # unwrapped like cli.py does
    assert "кількість дорослих" in str(stop) and "дата старту" in str(stop)


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
    llm = FakeLLM(no_adults, [PARSED, GOOD])   # 1st reading misses adults; after the answer it is complete
    code, obsidian, asked = run_chat(llm, [
        "Юнацтво, 10 людей, Чорногора, хочемо озеро, 3 дні з 21 листопада", "",   # description + end
        "2",                                                                       # answer: adults
        "",                                                                        # confirm: yes
        "Похід Сокіл з чату",                                                      # note title
    ])
    assert code == 0
    assert any("дорослих" in q for q in asked)                    # the agent asked about the adults
    request = obsidian.written["Запити/Похід Сокіл з чату.md"]
    assert "Уточнення" in request and "дорослих: 2" in request.lower()
    assert "Плани/Похід Сокіл з чату — план.md" in obsidian.written    # planned and written back
    assert llm.calls == 4                                          # 2 readings + 1 proposal + conclusion


def test_chat_can_be_cancelled_without_writing_anything():
    code, obsidian, _ = run_chat(FakeLLM(PARSED, []), ["Юнацтво, 10 + 2, Чорногора, 3 дні з 21.11", "", "ні"])
    assert code == 0 and obsidian.written == {}


def test_model_deadline_gives_a_readable_error():
    from plast_agent import llm as L
    model = L.LLM.__new__(L.LLM)
    model.model_name = "slow-model"

    class Slow:
        async def ainvoke(self, messages):
            await asyncio.sleep(5)

    with pytest.raises(L.LLMError) as e:
        asyncio.run(model._deadline(Slow(), [], 0.1))
    assert "не відповіла за" in str(e.value)


def test_chat_with_two_adults_and_named_peaks_like_the_real_session():
    """Replays the leader's real test: 'іду я із сестрою… на Говерлу 23-го, спуститися біля Петроса 24-го'."""
    first = {"section": None, "participants": 2, "adults": None, "days": 2, "start_date": "2026-11-23",
             "date_flexibility_days": 0, "region": "chornohora", "must_include": ["peak"],
             "places": ["Говерла", "Петрос"], "start_time": None}
    second = {**first, "section": "senior", "adults": 0}
    proposal = {"route_id": "r8613511", "overnight_waypoint_ids": ["w4"], "start_date": "2026-11-23",
                "reasoning": "через Говерлу"}
    code, obsidian, asked = run_chat(FakeLLM(first, [second, proposal]), [
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
    plan = next(v for k, v in obsidian.written.items() if k.startswith("Плани/"))
    assert "старші / сеніори, вас 2" in plan
    assert "Жоден маршрут у даних не проходить через усі місця (Говерла, Петрос)" in plan


def test_guess_section_from_words_and_ages():
    from plast_agent.chat import guess_section
    assert guess_section("20 років нам") == "senior"
    assert guess_section("нам по 14") == "yunatstvo"
    assert guess_section("новачки, рій") == "novatstvo"
    assert guess_section("дорослі") == "senior"
    assert guess_section("не знаю") is None
