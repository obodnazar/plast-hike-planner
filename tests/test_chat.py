"""The web chat: a whole conversation from description to plan card, through the REAL plast-trail server in
memory, a fake Obsidian and a scripted model (no network, no quota); and the WebSocket around it.

Run:  uv run pytest -q tests/test_chat.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from datetime import date
from types import SimpleNamespace

from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from plast_agent.mcp_links import McpLinks
from plast_agent.plan_store import list_plans
from plast_trail.server import mcp as plast_trail_server
from plast_web.app import create_app
from plast_web.conversation import Conversation
from plast_web.view import looks_technical
from test_agent import GOOD, PARSED, FakeLLM, FakeObsidian

TODAY = date(2026, 11, 16)
DESCRIPTION = ("Гурток юнацтва, 10 учасників і 2 дорослі, хочемо в Чорногору на 3 дні з 21 листопада, можна зсунути "
               "на 2 дні пізніше, не більше 12 км на день, бажано біля озера.")


class FakeRuntime:
    """Like plast_web.runtime.Runtime, with both connections already open."""

    def __init__(self, links):
        self.links, self.settings, self.plan_lock = links, SimpleNamespace(), asyncio.Lock()

    def ready(self):
        return True

    def status(self):
        return {"ready": True, "problem": "", "items": []}


def talk(script, llm) -> tuple[Conversation, FakeObsidian]:
    """Run the messages in `script` one by one, each after the previous work has finished."""
    obsidian = FakeObsidian()

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": obsidian}
            conv = Conversation(FakeRuntime(links), today=TODAY, llm=llm)
            for msg in script:
                conv.handle(msg if isinstance(msg, dict) else {"type": "say", "text": msg})
                if conv.task:
                    await conv.task
            return conv

    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(go()), obsidian


def of(conv, kind):
    return [e for e in conv.events if e["type"] == kind]


def test_whole_conversation_from_description_to_plan_card():
    conv, obsidian = talk([DESCRIPTION, {"type": "set_field", "key": "participants", "value": 9},
                           {"type": "confirm"}], FakeLLM(PARSED, [GOOD]))
    cards = of(conv, "card")
    assert cards[0]["fields"][0] == {**cards[0]["fields"][0], "key": "section", "value": "юнацтво"}
    assert {"key": "participants", "value": "9"}.items() <= next(f for f in cards[-2]["fields"]
                                                                 if f["key"] == "participants").items()
    assert cards[-1]["active"] is False                                   # the card is closed once planning starts
    progress = of(conv, "progress")
    assert progress[-1]["finished"] and all(s["state"] == "done" for s in progress[-1]["steps"])
    assert any("спроба 1" in s["note"] for p in progress for s in p["steps"])
    plan = of(conv, "plan")[-1]
    assert plan["status"]["label"] == "Готово, є що врахувати" and plan["links"]["page_url"] == f"/plan/{plan['plan_id']}"
    assert plan["spark"] and plan["top"]
    assert conv.state == "done"
    saved = list_plans()
    assert len(saved) == 1 and saved[0]["request"]["participants"] == 9   # the edit reached the plan
    written = list(obsidian.written)
    assert any(p.startswith("Запити/Запит 2026") for p in written) and any(p.startswith("Плани/") for p in written)
    request_note = next(v for p, v in obsidian.written.items() if p.startswith("Запити/"))
    assert "Виправлення: Учасників — 9" in request_note                # the note in Obsidian has the correction


def test_missing_facts_are_asked_and_read_by_code():
    parsed = {**PARSED, "participants": None, "adults": None}
    llm = FakeLLM(parsed, [GOOD])
    conv, _ = talk(["Хочемо в Чорногору на 3 дні з 21 листопада, юнацтво.", "10", "двоє"], llm)
    questions = [e["text"] for e in of(conv, "msg") if e["role"] == "agent"]
    assert any("Скільки учасників" in q for q in questions) and any("Скільки дорослих" in q for q in questions)
    assert llm.calls == 1                                                  # answers cost no model request
    assert conv.state == "confirm" and conv.request["participants"] == 10 and conv.request["adults"] == 2


def test_cancel_writes_nothing():
    conv, obsidian = talk([DESCRIPTION, {"type": "cancel"}], FakeLLM(PARSED, []))
    assert conv.state == "describe" and not obsidian.written and not list_plans()


def test_everything_the_leader_reads_is_human():
    conv, _ = talk([DESCRIPTION, {"type": "confirm"}], FakeLLM(PARSED, [GOOD]))
    texts = [e["text"] for e in of(conv, "msg") if e["role"] == "agent"]
    texts += [f["value"] for c in of(conv, "card") for f in c["fields"]] + [n for c in of(conv, "card") for n in c["notes"]]
    texts += [s["label"] + " " + s["note"] for p in of(conv, "progress") for s in p["steps"]]
    plan = of(conv, "plan")[-1]
    texts += [plan["title"], plan["meta"], plan["top"]["text"], plan["top"]["advice"]] + [s["value"] for s in plan["stats"]]
    leaks = [t for t in texts if looks_technical(t, {plan["title"]})]
    assert not leaks, leaks


def test_work_goes_on_without_the_tab_and_a_new_tab_catches_up():
    """A conversation keeps its events; a reloaded tab asks for what came after the last one it saw."""
    conv, _ = talk([DESCRIPTION], FakeLLM(PARSED, []))
    seen = conv.events[2]["seq"]
    assert [e["seq"] for e in conv.since(seen)] == list(range(seen + 1, len(conv.events) + 1))


# ---------------------------------------------------------------------------------------------- the socket
def test_socket_greets_and_resumes(tmp_path):
    app = create_app(8780, new_conversation=lambda: Conversation(FakeRuntime(None), today=TODAY))
    client = TestClient(app, base_url="http://127.0.0.1:8780")
    with client.websocket_connect("ws://127.0.0.1:8780/ws", headers={"origin": "http://127.0.0.1:8780"}) as ws:
        ws.send_json({"type": "hello"})
        hello = ws.receive_json()
        greeting = ws.receive_json()
        assert hello["type"] == "hello" and greeting["text"].startswith("Привіт")
        cid = hello["conversation_id"]
    with client.websocket_connect("ws://127.0.0.1:8780/ws", headers={"origin": "http://127.0.0.1:8780"}) as ws:
        ws.send_json({"type": "hello", "conversation_id": cid, "last_seq": 0})
        assert ws.receive_json()["conversation_id"] == cid                # the same conversation
        assert ws.receive_json()["text"].startswith("Привіт")              # replayed from the start


def test_socket_refuses_pages_from_elsewhere():
    client = TestClient(create_app(8780), base_url="http://127.0.0.1:8780")
    try:
        with client.websocket_connect("ws://127.0.0.1:8780/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.send_json({"type": "hello"})
            ws.receive_json()
        refused = False
    except Exception:
        refused = True
    assert refused


def test_without_the_agent_the_chat_says_so():
    client = TestClient(create_app(8780), base_url="http://127.0.0.1:8780")
    assert client.get("/api/status").json()["ready"] is False
    with client.websocket_connect("ws://127.0.0.1:8780/ws", headers={"origin": "http://127.0.0.1:8780"}) as ws:
        ws.send_json({"type": "hello"})
        ws.receive_json(), ws.receive_json(), ws.receive_json()           # hello, greeting, state
        ws.send_json({"type": "say", "text": "Хочемо в гори"})
        assert "лише збережені плани" in ws.receive_json()["text"]


def test_a_question_after_the_plan_changes_that_plan():
    """16 Nov, Nazar: «а на Говерлу ми зайдем?» after a plan started a new dialogue from «Хто йде?»."""
    again = {**PARSED, "section": None, "participants": None, "places": ["Говерла"], "must_include": []}
    conv, _ = talk([DESCRIPTION, {"type": "confirm"}, "а на Говерлу ми зайдем?"], FakeLLM(PARSED, [GOOD, again]))
    assert conv.state == "confirm"                                         # straight to the card, no questions
    card = of(conv, "card")[-1]
    fields = {f["key"]: f["value"] for f in card["fields"]}
    assert fields["places"] == "Говерла" and fields["section"] == "юнацтво" and fields["participants"] == "10"
    assert "Зміна після плану: а на Говерлу ми зайдем?" in conv.text


def test_a_range_of_people_is_read_by_code():
    parsed = {**PARSED, "participants": None}
    llm = FakeLLM(parsed, [])
    conv, _ = talk(["Хочемо в Чорногору на 3 дні з 21 листопада, юнацтво, 2 дорослі.", "6-7"], llm)
    assert llm.calls == 1 and conv.request["participants"] == 7            # no second model call, no second question
    assert any("більше число" in n for c in of(conv, "card") for n in c["notes"])


def test_a_new_date_after_the_plan_is_kept():
    """Codex P2: «зміни старт на 23 листопада» after a plan for 21 Nov came back as 21 Nov."""
    again = {**PARSED, "start_date": "2026-11-21"}            # the model keeps the old date
    conv, _ = talk([DESCRIPTION, {"type": "confirm"}, "Зміни старт на 23 листопада"], FakeLLM(PARSED, [GOOD, again]))
    fields = {f["key"]: f["value"] for f in of(conv, "card")[-1]["fields"]}
    assert fields["start_date"].startswith("пн, 23 листопада")


def test_a_wish_taken_back_after_the_plan_stays_out():
    """Codex P2 (round 2): the chat put the old «озеро» back when the new reading had none."""
    again = {**PARSED, "must_include": []}
    conv, _ = talk([DESCRIPTION, {"type": "confirm"}, "Більше не хочу озера"], FakeLLM(PARSED, [GOOD, again]))
    assert conv.request["must_include"] == [] and conv.request["participants"] == 10
