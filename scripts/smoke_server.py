"""Talk to a running plast-trail server over MCP, exactly as the agent will.

Start the server first (separate terminal):  uv run plast-trail
Then:                                       uv run python scripts/smoke_server.py

Shows: discovery (tools/list with input+output schemas), a success path through
the tools, and three failure types (unknown route, bad waypoint order, schema violation).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.environ.get("PLAST_TRAIL_URL", "http://127.0.0.1:8770/mcp")


def show(title: str, result) -> dict | None:
    status = "ERROR" if result.isError else "ok"
    if result.isError:
        text = "".join(getattr(c, "text", "") for c in result.content)
        print(f"\n[{status}] {title}\n  {text[:400]}")
        return None
    data = result.structuredContent or {}
    print(f"\n[{status}] {title}")
    return data


async def main() -> None:
    async with streamable_http_client(URL) as (read, write, _):
        async with ClientSession(read, write) as s:
            init = await s.initialize()
            print(f"connected: {init.serverInfo.name}")
            tools = (await s.list_tools()).tools
            print(f"{len(tools)} tools:")
            for t in tools:
                req = (t.inputSchema or {}).get("required", [])
                out = list((t.outputSchema or {}).get("properties", {}))
                print(f"  - {t.name}: in(required)={req} | out={out[:6]}{'...' if len(out) > 6 else ''}")

            show("resolve_places (misspelt names)", await s.call_tool("resolve_places", {
                "names": ["піпівана", "бребенеско", "mountain lake"]}))
            r = await s.call_tool("find_routes", {"section": "yunatstvo", "days": 2, "region": "chornohora",
                                                  "must_include": ["lake"], "max_results": 3})
            found = show("find_routes(yunatstvo, 2 days, chornohora, lake)", r)
            for route in found["routes"]:
                print(f"  {route['id']} {route['name']} {route['length_km']} km, fit {route['fit_score']}")
            print(f"  excluded: {found['excluded']}")

            plan = {"route_id": "r8613349", "section": "yunatstvo", "participants": 10, "adults": 2,
                    "overnight_waypoint_ids": ["w47"], "start_date": (date.today() + timedelta(days=1)).isoformat(),
                    "group_limits": {"max_km_per_day": 12}}
            v = show("validate_hike_plan (2 days, group limit 12 km/day)", await s.call_tool("validate_hike_plan", plan))
            print(f"  valid={v['valid']}")
            for x in v["violations"]:
                print(f"  {x['severity']} {x['code']} day {x['day']}: {x['suggestion']}")

            plan["overnight_waypoint_ids"] = ["w30", "w56"]
            v = show("validate_hike_plan (3 days after suggestions)", await s.call_tool("validate_hike_plan", plan))
            print(f"  valid={v['valid']} days={[(d['day'], d['km'], d['ascent_m']) for d in v['days']]}")

            wr = show("assess_weather_risk (replay)", await s.call_tool("assess_weather_risk", {
                "route_id": plan["route_id"], "start_date": plan["start_date"],
                "overnight_waypoint_ids": plan["overnight_waypoint_ids"]}))
            if wr:
                print(f"  mode={wr['mode']} overall={wr['overall_risk']} "
                      f"days={[(d['date'], d['risk'], [t['code'] for t in d['triggers']]) for d in wr['days']]}")

            prof = show("get_route_profile", await s.call_tool("get_route_profile", {
                "route_id": plan["route_id"], "overnight_waypoint_ids": plan["overnight_waypoint_ids"]}))
            if prof:
                print("  " + "; ".join(f"day {d['day']}: {d['km']} km, +{d['ascent_m']}/-{d['descent_m']} m, "
                                       f"steep {d['steep_km']} km, {len(d['points'])} points" for d in prof["days"]))

            sup = show("calculate_supplies", await s.call_tool("calculate_supplies", {
                "section": "yunatstvo", "participants": 10, "adults": 2,
                "day_loads": [{"km": d["km"], "ascent_m": d["ascent_m"], "walking_hours": d["walking_hours"]} for d in v["days"]],
                "night_min_c": -0.7}))
            if sup:
                print(json.dumps({k: sup[k] for k in ("food_kg_group", "gas_g_group", "tents", "stoves")}, ensure_ascii=False))
                pk = sup["pack_participant"]
                print(f"  participant: food {pk['food_kg']} kg + water {pk['water_kg']} kg, limit {pk['limit_kg']} kg, "
                      f"left for gear {pk['remaining_for_gear_kg']} kg; warnings: {len(sup['warnings'])}")

            show("ERROR CASE unknown route", await s.call_tool("validate_hike_plan", {**plan, "route_id": "r1"}))
            show("ERROR CASE unknown night spot", await s.call_tool("validate_hike_plan",
                                                                    {**plan, "overnight_waypoint_ids": ["w9999"]}))
            show("ERROR CASE night spots out of order",
                 await s.call_tool("validate_hike_plan", {**plan, "overnight_waypoint_ids": ["w56", "w30"]}))
            show("ERROR CASE schema: days=0", await s.call_tool("find_routes", {"section": "yunatstvo", "days": 0}))
            empty = show("EMPTY RESULT (not an error)", await s.call_tool(
                "find_routes", {"section": "novatstvo", "days": 7, "region": "marmarosy", "must_include": ["lake"]}))
            print(f"  routes={empty['routes']} excluded={empty['excluded']} notes={empty['notes']}")


if __name__ == "__main__":
    asyncio.run(main())
