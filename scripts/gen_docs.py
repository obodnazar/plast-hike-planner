"""Generate docs/tool-contracts.md and docs/obsidian-contract.md from the LIVE servers.

Names, model-facing descriptions and schemas are read from tools/list, and every
example output is a real call, so the docs cannot drift from what the model sees.
Purpose, errors and side effects are written by hand below (TOOL_NOTES).

    uv run plast-trail                        # terminal 1 (WEATHER_MODE=replay)
    uv run python scripts/gen_docs.py         # terminal 2, Obsidian open
"""

from __future__ import annotations

import asyncio
import json
import ssl
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plast_agent.config import load_settings  # noqa: E402

DOCS = ROOT / "docs"
PLAN_NOTE = "Плани/Мандрівка гуртка Сокіл — план.md"
REQUEST_NOTE = "Запити/Мандрівка гуртка Сокіл.md"

# ------------------------------------------------------------------ hand-written parts
TOOL_NOTES: dict[str, dict[str, Any]] = {
    "find_routes": {
        "purpose": "First step of planning. Narrows 121 marked Carpathian routes (OpenStreetMap) to the few that fit "
                   "the group: section, number of days, region, wished features. Ranks them by how well km/day fits "
                   "the section's target and lists, for each route, the night spots this section may use (waypoint ids). "
                   "The agent calls it once per run (again with relaxed wishes if nothing matched).",
        "processing": "Filters by region and features; drops routes longer than days × daily hard limit or shorter "
                      "than 3 km/day; requires at least days-1 allowed night spots (новацтво: only huts/campsites); "
                      "deduplicates night spots per 0.5 km and flags water within 300 m; scores fit = 1 - |km/day - target| / target.",
        "errors": [
            ("Arguments break the schema (days=0, unknown section/region/feature, more than 4 features)",
             "`isError: true`, text = pydantic validation message from FastMCP; the tool body does not run"),
            ("Dataset file missing or corrupt", "`isError: true`, `{\"error\": {\"code\": \"DATA_UNAVAILABLE\", ...}}`"),
            ("Nothing matches", "**not an error**: `routes: []`, `excluded` gives counts per reason, `notes` suggests what to relax"),
        ],
        "side_effects": "None. Reads `data/routes.json` and `data/rules.yaml` (cached in memory). No network.",
        "example": {"section": "yunatstvo", "days": 3, "region": "chornohora", "must_include": ["lake"], "max_results": 2},
    },
    "validate_hike_plan": {
        "purpose": "The rule checker and the engine of the feedback loop. Given a route and the night spots, it splits the "
                   "route into days and computes every number itself (km, ascent, walking hours, arrival vs sunset), then "
                   "checks Plast rules and the group's own limits. Each violation carries a concrete fix, often a better "
                   "waypoint id, which the agent feeds back to the model. Called after every proposal until valid=true.",
        "processing": "Profile slicing between waypoint km; ascent with 10 m hysteresis; walking time by Langmuir's group "
                      "rule (4 km/h + 1 h per 450 m) + 20 % rests; NOAA sunset at the day's end point (Europe/Kyiv); "
                      "rules: km/day, walking hours, ascent, allowed night-spot kind, water within 300 m, ridge (within 30 m "
                      "of the high point in ±1 km), min adults, adult ratio, group size, max days, arrival ≥3 h before sunset. "
                      "Group limits from Obsidian override rules only when stricter.",
        "errors": [
            ("Unknown route id", "`isError: true`, code `ROUTE_NOT_FOUND`, hint shows the id format"),
            ("Waypoint id not on the route", "code `WAYPOINT_NOT_FOUND`, hint gives the valid range w0..wN"),
            ("Night spots out of km order, or start/finish used as a night spot", "code `WAYPOINT_ORDER`"),
            ("Bad `start_time`", "schema pattern `^\\d{2}:\\d{2}$`, or code `INVALID_INPUT` for 24:99"),
            ("Schema violations (participants=0, >6 night spots, id not matching `^r\\d+$`/`^w\\d+$`)",
             "`isError: true`, pydantic validation message"),
            ("Plan breaks rules", "**not an error**: `valid: false` with `violations[]` (severity error/warning)"),
        ],
        "side_effects": "None. Pure computation over the local dataset and rules.",
        "example": {"route_id": "r8613349", "section": "yunatstvo", "participants": 10, "adults": 2,
                    "overnight_waypoint_ids": ["w47"], "start_date": "2026-11-23", "start_time": "08:00",
                    "group_limits": {"max_km_per_day": 12}},
    },
    "assess_weather_risk": {
        "purpose": "Turns a forecast into decisions. For the same plan (route + night spots) and a start date it returns "
                   "a risk level per day with the triggers and what to do. A HIGH result sends the agent back to "
                   "re-plan (other dates or route); the agent also uses it to probe alternative start dates.",
        "processing": "Forecast is taken at the route's highest point (lat, lon, elevation) from Open-Meteo, 16 days. "
                      "A day is 'exposed' if it reaches above the tree line (1400 m). Triggers: thunderstorm and snow "
                      "(high if exposed), wind ≥15/≥25 m/s, precipitation ≥15/≥50 mm, night ≤0 °C at the night spot "
                      "(summit minimum corrected by the 6.5 °C/km lapse rate).",
        "errors": [
            ("Replay mode and no recorded forecast covers the dates", "code `WEATHER_FIXTURE_MISSING`, hint lists the recorded ranges"),
            ("Live mode and dates outside today..today+15", "code `DATE_OUT_OF_RANGE`"),
            ("Open-Meteo down / timeout / HTTP 5xx after 3 tries", "code `WEATHER_UNAVAILABLE`, hint suggests replay mode"),
            ("Unknown route / waypoint / order", "same codes as validate_hike_plan"),
        ],
        "side_effects": "`live`: one HTTPS GET to api.open-meteo.com per route per 30 min (in-memory cache, ≥1 s between calls). "
                        "`record`: same, plus writes `fixtures/open-meteo/<route>_<first-day>.json`. `replay` (default): "
                        "reads fixtures only, no network.",
        "example": {"route_id": "r8613349", "start_date": "2026-11-23", "overnight_waypoint_ids": ["w30", "w56"]},
    },
    "calculate_supplies": {
        "purpose": "Last planning step: what the group must carry. Takes the per-day loads from validate_hike_plan and "
                   "the coldest night from assess_weather_risk and returns food, water, gas, tents, stoves, pots, "
                   "first-aid kits and the pack weight against the section's limit. Every coefficient is echoed in "
                   "`assumptions` with its source, so any number can be traced.",
        "processing": "Food = days × people × kg/day by section (0.9; новацтво 0.72); water = max(3 l, 1 l camp + 0.5 l per "
                      "walking hour); +1.5 l per person per dry night; gas 25 g/person/day ×1.5 with frost; 1 stove + 3 pots "
                      "per 6; 1 first-aid kit per 10; tents by capacity. Pack = food + 1 l water (+ measured base gear); "
                      "limit = min(9 kg, % of the lightest body weight). No invented base-gear default.",
        "errors": [
            ("Schema violations (empty `days`, >7 days, km > 40, tent_capacity not 2/3/4)", "`isError: true`, pydantic message"),
            ("A coefficient missing in rules.yaml", "code `DATA_UNAVAILABLE`, names the missing key"),
        ],
        "side_effects": "None.",
        "example": {"section": "yunatstvo", "participants": 10, "adults": 2,
                    "days": [{"km": 11.3, "ascent_m": 201, "walking_hours": 3.3},
                             {"km": 11.0, "ascent_m": 1087, "walking_hours": 5.2},
                             {"km": 11.0, "ascent_m": 31, "walking_hours": 2.8}],
                    "dry_nights": 0, "night_min_c": -5.9, "lightest_body_weight_kg": 40,
                    "participant_base_gear_kg": 5, "adult_base_gear_kg": 7},
    },
}

OBSIDIAN_NOTES: dict[str, dict[str, Any]] = {
    "vault_read": {
        "role": "Step 1 of every run: reads the leader's request note (free text) that the model then parses. "
                "Its content is the agent's main input; without it the run stops.",
        "errors": "Missing note → `isError: true`, text `File not found: <path>`; the agent then calls vault_list and "
                  "tells the person which requests exist. Wrong key → HTTP 401 at connect; Obsidian closed → connection refused.",
        "side_effects": "None (read-only).",
        "example": {"path": REQUEST_NOTE},
    },
    "vault_list": {
        "role": "Recovery path: when the requested note is missing, the agent lists `Запити/` to show what exists; also `plast-agent --list`.",
        "errors": "Missing folder → `isError: true`. Empty folders are not listed (Obsidian indexes files).",
        "side_effects": "None.",
        "example": {"path": "Запити"},
    },
    "vault_write": {
        "role": "Final step: writes the finished plan note to `Плани/<request> — план.md` (route table, weather, supplies, "
                "documents checklist, the agent's decision log).",
        "errors": "Obsidian unreachable at write time → the agent saves the note to `out/` locally and reports it. "
                  "Binary extensions or NUL bytes are refused by the plugin.",
        "side_effects": "**Creates or overwrites** a file in the vault (parent folders created). The agent writes only under `Плани/`.",
        "example": {"path": PLAN_NOTE, "content": "(the full plan note; this example re-writes the existing note unchanged)"},
    },
    "vault_append": {
        "role": "After writing the plan, appends one status line with a wiki-link to the plan into the request note, "
                "so the leader sees the result where they wrote the request.",
        "errors": "Same as vault_write.",
        "side_effects": "**Appends** text to an existing note (creates it if missing).",
        "example": None,  # not called during doc generation: it would change the demo request
    },
}


# ------------------------------------------------------------------ schema rendering
def resolve(schema: dict, defs: dict) -> dict:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return {**defs.get(name, {}), "_ref": name, **{k: v for k, v in schema.items() if k != "$ref"}}
    return schema


def type_of(schema: dict, defs: dict) -> str:
    s = resolve(schema, defs)
    if "anyOf" in s:
        return " | ".join(type_of(x, defs) for x in s["anyOf"])
    if "enum" in s:
        return "enum: " + " / ".join(f"`{e}`" for e in s["enum"])
    t = s.get("type", "object" if "properties" in s else "any")
    if t == "array":
        return f"array of {type_of(s.get('items', {}), defs)}"
    if s.get("_ref") and t == "object":
        return f"object `{s['_ref']}`"
    if s.get("format"):
        return f"{t} ({s['format']})"
    if t == "object" and isinstance(s.get("additionalProperties"), dict):
        return f"object (string → {type_of(s['additionalProperties'], defs)})"
    return t


def constraints(schema: dict, defs: dict) -> str:
    s = resolve(schema, defs)
    parts = []
    for sub in s.get("anyOf", [s]):
        sub = resolve(sub, defs)
        for key, label in (("minimum", "≥"), ("maximum", "≤"), ("exclusiveMinimum", ">"), ("exclusiveMaximum", "<"),
                           ("minItems", "min items"), ("maxItems", "max items"), ("minLength", "min len"),
                           ("maxLength", "max len"), ("pattern", "pattern")):
            if key in sub:
                parts.append(f"{label} `{sub[key]}`")
        if sub.get("type") == "array" and "items" in sub:
            inner = constraints(sub["items"], defs)
            if inner and inner != "—":
                parts.append(f"items: {inner}")
    return ", ".join(parts) or "—"


def md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def input_table(schema: dict) -> list[str]:
    defs = schema.get("$defs", {})
    req = set(schema.get("required", []))
    out = ["| Field | Type | Required | Constraints | Default | Description |", "|---|---|---|---|---|---|"]
    nested = []
    for name, prop in schema["properties"].items():
        default = f"`{json.dumps(prop['default'], ensure_ascii=False)}`" if "default" in prop else "—"
        desc = prop.get("description") or resolve(prop, defs).get("description", "")
        out.append(f"| `{name}` | {md_cell(type_of(prop, defs))} | {'yes' if name in req else 'no'} | "
                   f"{md_cell(constraints(prop, defs))} | {md_cell(default)} | {md_cell(desc)} |")
        for sub in (prop.get("anyOf") or [prop.get("items", {}), prop]):
            r = resolve(sub, defs)
            if r.get("_ref") and r.get("type") == "object" and r["_ref"] not in nested:
                nested.append(r["_ref"])
    for n in nested:
        d = defs[n]
        out += ["", f"`{n}`: {d.get('description', '')}", "",
                "| Field | Type | Required | Constraints | Default |", "|---|---|---|---|---|"]
        dreq = set(d.get("required", []))
        for fname, fp in d["properties"].items():
            default = f"`{json.dumps(fp['default'])}`" if "default" in fp else "—"
            out.append(f"| `{fname}` | {md_cell(type_of(fp, defs))} | {'yes' if fname in dreq else 'no'} | "
                       f"{md_cell(constraints(fp, defs))} | {default} |")
    return out


def output_table(schema: dict) -> list[str]:
    if not schema:
        return ["No `outputSchema` is declared; the result is a text content block (see example)."]
    defs = schema.get("$defs", {})
    out = ["| Field | Type | Meaning |", "|---|---|---|"]
    for name, prop in schema["properties"].items():
        out.append(f"| `{name}` | {md_cell(type_of(prop, defs))} | {md_cell(prop.get('description', ''))} |")
    for dname, d in defs.items():
        if d.get("type") != "object":
            continue
        out += ["", f"<details><summary><code>{dname}</code></summary>", "",
                "| Field | Type | Meaning |", "|---|---|---|"]
        for fname, fp in d["properties"].items():
            out.append(f"| `{fname}` | {md_cell(type_of(fp, defs))} | {md_cell(fp.get('description', ''))} |")
        out += ["", "</details>"]
    return out


def abbreviate(obj: Any, max_list: int = 2, max_str: int = 220) -> Any:
    if isinstance(obj, dict):
        return {k: abbreviate(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [abbreviate(x, max_list, max_str) for x in obj[:max_list]]
        if len(obj) > max_list:
            items.append(f"... {len(obj) - max_list} more")
        return items
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    return obj


def result_payload(res) -> Any:
    if res.isError:
        return {"isError": True, "text": "".join(getattr(c, "text", "") for c in res.content)}
    if res.structuredContent is not None:
        return res.structuredContent
    text = "".join(getattr(c, "text", "") for c in res.content)
    try:
        return json.loads(text)
    except ValueError:
        return text


def code_block(obj: Any) -> list[str]:
    return ["```json", json.dumps(obj, ensure_ascii=False, indent=2), "```"]


# ------------------------------------------------------------------ main
async def custom_contracts() -> list[str]:
    L = ["# plast-trail: tool contracts", "",
         f"Generated by `scripts/gen_docs.py` from the live server's `tools/list` on {date.today()}. "
         "Names, model-facing descriptions and schemas below are exactly what an MCP client receives; "
         "the example outputs are real calls (lists shortened with `... N more`).", "",
         "## Server", "",
         "- **Process:** separate, started with `uv run plast-trail` (streamable HTTP, `http://127.0.0.1:8770/mcp`; `--stdio` also available).",
         "- **Data:** `data/routes.json` (121 OSM routes + SRTM heights), `data/rules.yaml` (DRAFT norms with sources), "
         "Open-Meteo forecasts (live / record / replay).",
         "- **Server instructions** (sent at `initialize`) describe the typical flow: find → validate (loop) → weather (loop) → supplies.", "",
         "## Error model (all tools)", "",
         "| Situation | What the caller gets |", "|---|---|",
         "| Domain failure (unknown route, missing fixture, …) | `isError: true`; text = `{\"error\": {\"code\": \"...\", \"message\": \"...\", \"hint\": \"...\"}}` |",
         "| Arguments violate the input schema | `isError: true`; FastMCP/pydantic validation message; tool body never runs |",
         "| Unexpected bug | `isError: true`; code `INTERNAL_ERROR`; stack trace only in the server log |",
         "| Valid request, nothing found / plan breaks rules | **success** with an empty list (`routes: []`) or `valid: false` + `violations` |",
         "",
         "Codes: `ROUTE_NOT_FOUND`, `WAYPOINT_NOT_FOUND`, `WAYPOINT_ORDER`, `INVALID_INPUT`, `DATE_OUT_OF_RANGE`, "
         "`WEATHER_UNAVAILABLE`, `WEATHER_FIXTURE_MISSING`, `DATA_UNAVAILABLE`, `INTERNAL_ERROR`.", ""]
    async with streamable_http_client("http://127.0.0.1:8770/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            L.insert(L.index("## Error model (all tools)"), f"Discovered tools: {', '.join(f'`{n}`' for n in tools)}.\n")
            for name, notes in TOOL_NOTES.items():
                t = tools[name]
                res = await s.call_tool(name, notes["example"])
                ann = t.annotations
                L += [f"## `{name}`", "",
                      "| Contract element | Content |", "|---|---|",
                      f"| Name | `{name}` (title: {t.title}) |",
                      f"| Purpose | {md_cell(notes['purpose'])} |",
                      f"| Processing | {md_cell(notes['processing'])} |",
                      f"| Side effects | {md_cell(notes['side_effects'])} |",
                      f"| Annotations | readOnly={ann.readOnlyHint}, idempotent={ann.idempotentHint}, openWorld={ann.openWorldHint} |",
                      "", "**Model-facing description (exact):**", "", f"> {t.description}", "",
                      "**Input schema**", "", *input_table(t.inputSchema), "",
                      "**Output schema** (`structuredContent`; the same JSON is also sent as text)", "",
                      *output_table(t.outputSchema or {}), "",
                      "**Error conditions**", "", "| Condition | Representation |", "|---|---|",
                      *[f"| {md_cell(c)} | {md_cell(rep)} |" for c, rep in notes["errors"]], "",
                      "**Example input**", "", *code_block(notes["example"]), "",
                      "**Example output** (real call, replay mode)", "", *code_block(abbreviate(result_payload(res))), ""]
            # one live error example
            err = await s.call_tool("validate_hike_plan", {**TOOL_NOTES["validate_hike_plan"]["example"], "route_id": "r1"})
            L += ["## Example of an error result", "", "`validate_hike_plan` with `route_id: \"r1\"`:", "",
                  *code_block(result_payload(err)), ""]
            _ = init
    return L


async def obsidian_contract() -> list[str]:
    st = load_settings(need_model=False)
    http = httpx.AsyncClient(headers={"Authorization": f"Bearer {st.obsidian_api_key}"},
                             verify=ssl.create_default_context(cafile=str(st.obsidian_ca_cert)), timeout=30)
    L = ["# Obsidian MCP: the existing server and the tools this project uses", "",
         f"Generated by `scripts/gen_docs.py` from the live server on {date.today()}. Descriptions and input schemas are "
         "copied from `tools/list`; outputs are real calls against the demo vault.", ""]
    async with http, streamable_http_client(st.obsidian_mcp_url, http_client=http) as (r, w, _):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            L += ["## Connection (this project's configuration)", "",
                  f"- **Server:** `{init.serverInfo.name}` {init.serverInfo.version}, built into the Obsidian community plugin "
                  "**Local REST API with MCP** by Adam Coddington (approved project `coddingtonbear/obsidian-local-rest-api`), "
                  "tested with plugin 5.5.2 and Obsidian 1.15.1.",
                  f"- **Transport:** streamable HTTP at `{st.obsidian_mcp_url}` over TLS with the plugin's self-signed certificate "
                  "(verified against `certs/obsidian-local-rest-api.crt`, not disabled).",
                  "- **Auth:** `Authorization: Bearer <OBSIDIAN_API_KEY>` from `.env` (git-ignored).",
                  "- **Vault:** a dedicated `demo-vault/` with fictional data only (`Запити/` requests, `Правила/`, `Плани/` output). "
                  "The plugin exposes only the vault that is open, never the personal one.",
                  f"- **Discovered tools ({len(tools)}):** {', '.join(f'`{n}`' for n in tools)}. The agent uses 4 of them.", "",
                  "## Why this server has a role in the project", "",
                  "A Plast leader already keeps notes; Obsidian is where the request is written in plain words and where the "
                  "finished plan has to land, next to the request, readable and editable by people. The agent therefore "
                  "starts by reading a human-written note (input), and ends by writing a human-readable plan and a link back "
                  "(output). Both ends are real effects on an external environment, and the content read from Obsidian "
                  "(group, dates, limits like «не більше 12 км на день») directly changes the later tool calls: the group "
                  "limits are passed to `validate_hike_plan` and override the rules when stricter.", ""]
            for name, notes in OBSIDIAN_NOTES.items():
                t = tools[name]
                L += [f"## `{name}`", "", "| Contract element | Content |", "|---|---|",
                      f"| Role in this project | {md_cell(notes['role'])} |",
                      f"| Error conditions | {md_cell(notes['errors'])} |",
                      f"| Side effects | {md_cell(notes['side_effects'])} |", "",
                      "**Model-facing description (exact):**", "", f"> {md_cell(t.description)}", "",
                      "**Input schema**", "", *input_table(t.inputSchema), "",
                      "**Output:** " + ("no `outputSchema`; a text block containing JSON." if not t.outputSchema else "structured."), ""]
                ex = notes["example"]
                if ex is None:
                    L += ["Not called during doc generation (it would change the demo request); the agent's call is "
                          "`vault_append(path=<request>, content=\"\\n\\n---\\n**Агент (...):** <status> → [[Плани/...]]\")`.", ""]
                    continue
                if name == "vault_write":
                    current = result_payload(await s.call_tool("vault_read", {"path": PLAN_NOTE}))
                    res = await s.call_tool("vault_write", {"path": PLAN_NOTE, "content": current["content"]})
                else:
                    res = await s.call_tool(name, ex)
                L += ["**Example input**", "", *code_block(ex), "", "**Example output** (real call)", "",
                      *code_block(abbreviate(result_payload(res), max_str=300)), ""]
            miss = await s.call_tool("vault_read", {"path": "Запити/Неіснуючий запит.md"})
            L += ["## Failure behaviour (what the defence demonstrates)", "",
                  "| Failure | What the agent prints | Effect |", "|---|---|---|",
                  "| Obsidian closed / plugin off | `Obsidian MCP is not reachable at https://127.0.0.1:27124/mcp/. Is Obsidian open ...?` | stops before any change |",
                  "| Wrong `OBSIDIAN_API_KEY` | `obsidian rejected the credentials (HTTP 401). Check OBSIDIAN_API_KEY in .env.` | stops before any change |",
                  "| Request note missing | `Запит не знайдено в Obsidian (File not found: ...). Наявні запити: ...` | stops; lists existing requests via vault_list |",
                  "| Obsidian lost before the final write | `Obsidian недоступний при записі ... План збережено локально: out/...` | plan kept locally |",
                  "", "Raw tool result for a missing note:", "", *code_block(result_payload(miss)), ""]
    return L


async def main() -> None:
    DOCS.mkdir(exist_ok=True)
    (DOCS / "tool-contracts.md").write_text("\n".join(await custom_contracts()) + "\n", encoding="utf-8")
    print("wrote docs/tool-contracts.md")
    (DOCS / "obsidian-contract.md").write_text("\n".join(await obsidian_contract()) + "\n", encoding="utf-8")
    print("wrote docs/obsidian-contract.md")


if __name__ == "__main__":
    asyncio.run(main())
