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
        if tool == "vault_read":
            if args["path"] != REQUEST:
                return CallToolResult(content=[TextContent(type="text", text=f"File not found: {args['path']}")],
                                      isError=True)
            body = {"content": NOTE, "path": REQUEST}
        elif tool == "vault_list":
            body = {"files": [Path(REQUEST).name]}
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

    def structured(self, schema, system: str, user: str):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert schema in (ParsedRequest, Proposal)
        return schema(**item)

    def text(self, system: str, user: str) -> str:
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
