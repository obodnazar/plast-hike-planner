"""The plan pages (plast-web): saved plans, the PlanView boundary and the local web server.

Plans come from real agent runs: the REAL plast-trail server in memory, a fake Obsidian and a scripted
model (no network, no quota), one per plan status.

Run:  uv run pytest -q tests/test_web.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from plast_agent import cli
from plast_agent.mcp_links import McpLinks
from plast_agent.plan_store import list_plans, load_plan, new_id, plan_record, save_plan
from plast_trail.server import mcp as plast_trail_server
from plast_web.app import create_app
from plast_web.view import looks_technical, place_names, present
from test_agent import GOOD, PARSED, REQUEST, FakeLLM, FakeObsidian, run_agent

HIGH = {**GOOD, "start_date": "2026-11-21"}    # valid, but HIGH weather in the recorded forecasts
BAD = {**GOOD, "route_id": "r1"}               # not one of the candidates
SCRIPTS = {
    "attention": (PARSED, [{**GOOD, "overnight_waypoint_ids": ["w30"]}, GOOD]),
    "risky": (PARSED, [HIGH, BAD, BAD, BAD]),
    "failed": (PARSED, [BAD, BAD, BAD, BAD]),
    "group": ({**PARSED, "adults": 1}, [GOOD]),
}


def record_for(name: str) -> dict:
    parsed, proposals = SCRIPTS[name]
    with contextlib.redirect_stdout(io.StringIO()):
        state, links = run_agent(FakeLLM(parsed, proposals))
    record = plan_record(state, REQUEST, "fake-model", "2026-11-16 14:05", trace=links.trace,
                         model_switches=["openrouter: HTTP 429, switched to ollama"])
    if name == "attention":   # the model's own words, with an id and a code: must not reach the page
        record["conclusion"] = "Plan r8613349 is valid; ENDS_ON_TOP [S13] — **йдіть**."
    return record


@pytest.fixture(scope="module")
def records() -> dict[str, dict]:
    out = {name: record_for(name) for name in SCRIPTS}
    no_forecast = json.loads(json.dumps(out["attention"]))   # the state the agent keeps when there is no forecast
    no_forecast["weather"] = {"overall_risk": "unknown", "error": "No recorded forecast covers 2026-12-12.",
                              "days": []}
    no_forecast["conclusion"] = "Висновок."
    no_forecast["id"] = new_id()
    out["no_forecast"] = no_forecast
    return out


def human_strings(view, path=""):
    """Every text the page shows as text: not the map data, not links, not the technical details."""
    if isinstance(view, dict):
        for k, v in view.items():
            if k in ("map", "tech", "id", "kind", "level", "state", "color") or k.endswith("url") or k.endswith("_min"):
                continue
            yield from human_strings(v, f"{path}.{k}")
    elif isinstance(view, list):
        for i, v in enumerate(view):
            yield from human_strings(v, f"{path}[{i}]")
    elif isinstance(view, str):
        yield path, view


@pytest.mark.parametrize("name", ["attention", "risky", "failed", "group", "no_forecast"])
def test_page_text_has_no_ids_codes_or_english(records, name):
    record = records[name]
    view = present(record)
    names = place_names(record)
    leaks = [(p, t) for p, t in human_strings(view) if looks_technical(t, names)]
    assert not leaks, leaks
    text = " ".join(t for _, t in human_strings(view))
    for bad in ("r8613349", "w30", "ENDS_ON_TOP", "[S", "chornohora", "yunatstvo", "medium", "replay", "DRAFT",
                "valid", "HTTP"):
        assert bad not in text, bad


def test_detector_catches_what_the_agent_says_in_english():
    assert looks_technical("Route r8613349, night w30")
    assert looks_technical("ризик medium") and looks_technical("ENDS_ON_TOP") and looks_technical("довідки [S2]")
    assert looks_technical("The group asked for 3 days")
    assert not looks_technical("Маршрут через Говерлу, трек GPX для Mapy.com")
    assert not looks_technical("«Blue path (modrá značka)»", {"Blue path (modrá značka)"})   # an OSM name


def test_each_status_reads_right(records):
    views = {k: present(v) for k, v in records.items()}
    assert views["attention"]["status"] == {"kind": "warn", "label": "Готово, є що врахувати"}
    assert views["attention"]["conclusion"].startswith("Маршрут «")      # the model's English was replaced
    assert views["risky"]["status"]["kind"] == "danger"
    assert any(x["level"] == "danger" and "високий погодний ризик" in x["text"] for x in views["risky"]["safety"])
    assert views["failed"]["status"]["label"] == "Плану скласти не вдалося"
    assert views["failed"]["map"] is None and views["failed"]["days"] == []
    assert any("маршрут не зі списку знайдених" in t for t in views["failed"]["how"])
    group = views["group"]
    assert group["status"]["kind"] == "warn"                                # advice, not a verdict
    assert any("візьміть ще дорослого" in x["advice"] for x in group["safety"] if x["level"] == "warn")
    w = views["no_forecast"]["weather"]
    assert w["state"] == "missing" and "16 днів" in w["note"]               # «no forecast yet» is not «risky»
    assert views["no_forecast"]["status"]["kind"] == "warn"


def test_plan_page_has_what_the_obsidian_note_has(records):
    v = present(records["attention"])
    assert [d["n"] for d in v["days"]] == [1, 2, 3]
    assert v["days"][0]["title"] == "Лазещина → стоянка на 11.3 км, 920 м"   # a nameless spot by where it is
    assert v["days"][1]["late"] is True and "07:25" in " ".join(x["advice"] for x in v["safety"])
    assert v["weather"]["state"] == "ok" and len(v["weather"]["days"]) == 3
    assert "−" in v["stats"][3]["value"]                                       # a real minus sign
    assert v["supplies"]["tiles"][0]["label"] == "їжі на групу" and v["documents"]
    assert v["alternatives"] and v["understood"][0]["label"] == "Група"
    assert v["links"]["mapy_url"].startswith("https://mapy.com/") and len(v["links"]["mapy_days"]) == 3
    assert len(v["map"]["tracks"]) == 3 and {m["kind"] for m in v["map"]["marks"]} >= {"start", "finish", "night"}
    assert any("plast-trail.validate_hike_plan" in c for c in v["tech"]["calls"])   # for the defence
    assert v["how"][0].startswith("Прочитав запит") and any(t.startswith("Спроба 2") for t in v["how"])


def test_store_round_trip_and_bad_ids(tmp_path, records):
    path = save_plan(records["attention"], tmp_path)
    assert load_plan(path.stem, tmp_path)["status"] == "attention"
    assert [r["id"] for r in list_plans(tmp_path)] == [path.stem]
    for bad in ("../../etc/passwd", "x", "", path.stem + ".json"):
        assert load_plan(bad, tmp_path) is None
    (tmp_path / "20260101-000000-abcdef.json").write_text("{broken", encoding="utf-8")
    assert len(list_plans(tmp_path)) == 1                                       # a broken file is skipped


def test_the_agent_saves_its_plan_for_the_web(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAST_PLANS_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)                          # dry-run pictures go here too
    from types import SimpleNamespace

    async def go():
        async with create_connected_server_and_client_session(plast_trail_server) as session:
            links = McpLinks(SimpleNamespace(obsidian_mcp_url="x", plast_trail_url="y"))
            links.sessions = {"plast-trail": session, "obsidian": FakeObsidian()}
            return await cli.plan_and_write(links, None, REQUEST, dry_run=True, llm=FakeLLM(PARSED, [GOOD]))

    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert asyncio.run(go()) == 0
    saved = list_plans(tmp_path)
    assert len(saved) == 1 and saved[0]["status"] == "attention" and saved[0]["obsidian_plan"] == ""
    assert len(saved[0]["trace"]) >= 5 and f"/plan/{saved[0]['id']}" in out.getvalue()


# ---------------------------------------------------------------------------------------------- the server
@pytest.fixture
def client(tmp_path, monkeypatch, records):
    monkeypatch.setenv("PLAST_PLANS_DIR", str(tmp_path))
    for r in records.values():
        save_plan(r, tmp_path)
    return TestClient(create_app(8780), base_url="http://127.0.0.1:8780")


def test_server_lists_plans_and_serves_one(client, records):
    plans = client.get("/api/plans").json()["plans"]
    assert len(plans) == 5 and {p["status"]["kind"] for p in plans} == {"warn", "danger"}
    pid = records["attention"]["id"]
    view = client.get(f"/api/plans/{pid}").json()
    assert view["title"] == "с.Лазещина - г.Петрос - с.Ясіня"
    gpx = client.get(f"/api/plans/{pid}/route.gpx")
    assert gpx.status_code == 200 and gpx.headers["content-type"].startswith("application/gpx+xml")
    assert "<trk>" in gpx.text and "attachment" in gpx.headers["content-disposition"]
    svg = client.get(f"/api/plans/{pid}/route.svg")
    assert svg.status_code == 200 and svg.text.lstrip().startswith("<svg")
    assert client.get(f"/plan/{pid}").text.count('type="module"') == 1          # the page shell


def test_server_refuses_unknown_plans_and_other_hosts(client):
    assert client.get("/api/plans/20990101-000000-000000").status_code == 404
    assert client.get("/api/plans/..%2F..%2Fpyproject").status_code == 404
    assert client.get("/api/plans", headers={"host": "evil.example:8780"}).status_code == 403   # DNS rebinding
    assert client.get("/api/plans", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/plans", headers={"host": "localhost:8780"}).status_code == 200


def test_security_headers(client):
    r = client.get("/")
    csp = r.headers["content-security-policy"]
    assert "script-src 'self' https://cdnjs.cloudflare.com" in csp and "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp and r.headers["x-content-type-options"] == "nosniff"
    html = r.text
    assert "<script>" not in html and " style=" not in html                         # nothing inline (CSP)
    assert html.count('integrity="sha512-') == 2                                    # Leaflet is pinned


def test_frontend_never_inserts_html():
    static = Path(__file__).resolve().parents[1] / "src" / "plast_web" / "static"
    js = (static / "app.js").read_text(encoding="utf-8")
    code = re.sub(r"//.*", "", js)   # the rule is stated in a comment; check the code only
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "setAttribute(\"style\""):
        assert sink not in code, sink
