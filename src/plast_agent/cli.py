"""plast-agent: plan a Plast hike from a request note in Obsidian.

    uv run plast-agent "Запити/Мандрівка гуртка Сокіл.md"
    uv run plast-agent --list            # show request notes in the vault
    uv run plast-agent NOTE --dry-run    # do everything except writing to Obsidian
    uv run plast-agent --chat            # describe the hike in your own words; the agent asks what is missing

Needs two running MCP servers: Obsidian (with the Local REST API plugin) and
plast-trail (`uv run plast-trail` in another terminal).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
import traceback
from pathlib import Path

from plast_agent.config import PROJECT_ROOT, ConfigError, load_settings
from plast_agent.chat import chat
from plast_agent import llm as llm_mod
from plast_agent.llm import LLM, LLMError
from plast_agent.mcp_links import ConnectionFailure, McpCallError, McpLinks, _leaves, explain_connection
from plast_agent.picture import route_gpx, route_svg
from plast_agent.render import render_plan
from plast_agent.workflow import AgentStop, build_graph, now_kyiv

BOLD, RED, GREEN, YELLOW, RESET = "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


def make_llm(settings, local: bool = False) -> LLM:
    """Pick where the model answers from, and say it: quota left today, fallback ready or not."""
    if local:
        usable, local_msg = llm_mod.ollama_status(settings)
        print(f"  Модель: лише локальна ({local_msg})")
        if not usable:
            raise LLMError(local_msg)
        return LLM(providers=llm_mod.build_providers(settings, True), start_with_fallback=True)
    left = llm_mod.free_requests_left(settings)
    usable, local_msg = llm_mod.ollama_status(settings) if settings.ollama_enabled else (False, "запасна модель вимкнена")
    exhausted = llm_mod.quota_exhausted_now()
    low = exhausted or (left is not None and left < llm_mod.MIN_FREE_REQUESTS)
    quota = "невідомо скільки" if left is None else f"{left}"
    print(f"  Модель: openrouter/{settings.openrouter_model} — сьогодні лишилось {quota} безкоштовних запитів"
          + (" (ліміт вичерпано до 02:00 Києва)" if exhausted else ""))
    print(f"  Запасна: {local_msg}")
    if low and usable:
        print(f"  {YELLOW}Запитів замало: працюю на локальній моделі {settings.ollama_model}.{RESET}")
    elif low:
        print(f"  {YELLOW}Запитів замало, а запасної моделі немає: прогін може зупинитися на ліміті.{RESET}")
    return LLM(providers=llm_mod.build_providers(settings, usable), start_with_fallback=low and usable)


def leaflet_enabled(settings) -> bool:
    """OBSIDIAN_LEAFLET=1 in .env: the Obsidian Leaflet plugin is installed, so the plan gets an interactive map.
    The agent cannot find this out itself: the REST API may not read .obsidian/, and the plugin adds no commands
    to command_list (checked with Leaflet 6.0.5)."""
    return (getattr(settings, "obsidian_leaflet", "0") if settings else "0") in ("1", "true", "yes", "on")


async def plan_and_write(links: McpLinks, settings, request_path: str | None, dry_run: bool, llm=None,
                         chat_mode: bool = False, ask=None, local: bool = False) -> int:
    """Everything after both servers are connected. Catches its own errors and returns an exit code."""
    started = time.monotonic()
    try:
        llm = llm or make_llm(settings, local)
    except LLMError as exc:
        print(f"\n{RED}Модель недоступна:{RESET} {exc}")
        return 1
    graph = build_graph(links, llm)
    try:
        start: dict = {"request_path": request_path}
        if chat_mode:  # the request comes from a dialogue, is saved to Obsidian, then planned as usual
            got = await (chat(links, llm, ask) if ask else chat(links, llm))
            if got is None:
                return 0
            request_path, request, notes = got
            start = {"request_path": request_path, "request": request, "search_notes": notes}
        planning_started = time.monotonic()   # the dialogue (the leader typing) is not the agent's time
        state = await graph.ainvoke(start, {"recursion_limit": 80})
    except Exception as exc:  # every failure is explained here, inside both MCP connections
        links.reported = True
        leaf = next((x for x in _leaves(exc) if not isinstance(x, BaseExceptionGroup)), exc)
        if isinstance(leaf, AgentStop):
            print(f"\n{RED}Зупинка:{RESET} {leaf}")
        elif isinstance(leaf, LLMError):
            print(f"\n{RED}Модель недоступна:{RESET} {leaf}\n  Нічого не записано в Obsidian.")
        elif isinstance(leaf, ConnectionFailure):
            print(f"\n{RED}Зв'язок втрачено посеред роботи:{RESET} {leaf}")
        elif isinstance(leaf, McpCallError):
            print(f"\n{RED}Інструмент {leaf.server}.{leaf.tool} повернув помилку:{RESET} {leaf.code}: {leaf.message}")
        else:
            print(f"\n{RED}Внутрішня помилка агента:{RESET} {type(leaf).__name__}: {leaf}")
            traceback.print_exception(leaf, file=sys.stderr)
        return 1

    plan_path = f"Плани/{Path(request_path).stem} — план.md"
    picture = route_svg(state["route_profile"], state.get("validation") or {},
                        (state.get("proposal") or {}).get("start_date"),
                        (state.get("request") or {}).get("place_points")) if state.get("route_profile") else ""
    picture_path = f"Плани/{Path(request_path).stem} — маршрут.svg" if picture else ""
    gpx = route_gpx(state["route_profile"], (state.get("request") or {}).get("place_points"),
                    Path(request_path).stem) if picture else ""
    gpx_path = f"Плани/{Path(request_path).stem} — маршрут.gpx" if gpx else ""
    leaflet = leaflet_enabled(settings) and bool(gpx)
    note = render_plan({**state, "model_switches": getattr(llm, "switches", []), "picture_path": picture_path,
                        "gpx_path": gpx_path, "leaflet": leaflet,
                        "leaflet_id": "plast-" + hashlib.sha1(request_path.encode()).hexdigest()[:10]},
                       request_path, llm.model_name, now_kyiv())
    print(f"\n{BOLD}━━ 9. Записую план в Obsidian: {plan_path}{RESET}")
    if dry_run:
        print("  (dry run: нічого не записано)")
        if picture:
            local = PROJECT_ROOT / "out" / Path(picture_path).name
            local.parent.mkdir(exist_ok=True)
            local.write_text(picture, encoding="utf-8")
            local.with_suffix(".gpx").write_text(gpx, encoding="utf-8")
            print(f"  малюнок і трек маршруту для перегляду: {local} (.gpx поруч)")
    else:
        try:
            if picture:   # the picture and the track first: the note links to them
                await links.call("obsidian", "vault_write", {"path": picture_path, "content": picture})
                await links.call("obsidian", "vault_write", {"path": gpx_path, "content": gpx})
            await links.call("obsidian", "vault_write", {"path": plan_path, "content": note})
            await links.call("obsidian", "vault_append", {
                "path": request_path,
                "content": f"\n\n---\n**Агент ({now_kyiv()}):** {state.get('status')} → "
                           f"[[{plan_path.removesuffix('.md')}]]\n"})
        except (ConnectionFailure, McpCallError) as exc:
            links.reported = True
            backup = PROJECT_ROOT / "out" / Path(plan_path).name
            backup.parent.mkdir(exist_ok=True)
            backup.write_text(note, encoding="utf-8")
            print(f"  {RED}✗ Obsidian недоступний при записі ({exc}).{RESET}\n  План збережено локально: {backup}")
            return 1

    colour = {"ok": GREEN, "attention": YELLOW, "risky": YELLOW, "group": YELLOW}.get(state.get("status"), RED)
    print(f"\n{colour}{BOLD}Готово: {state.get('status')}{RESET} · спроб плану: {len(state.get('attempts', []))} · "
          f"виклики LLM: {llm.calls} ({', '.join(f'{k} {v}' for k, v in getattr(llm, 'calls_by', {}).items())}) · "
          f"виклики MCP: {links.calls} · "
          + (f"діалог {planning_started - started:.0f} с, планування {time.monotonic() - planning_started:.0f} с"
             if chat_mode else f"{time.monotonic() - started:.0f} с"))
    return 0 if state.get("status") in ("ok", "attention", "risky", "group") else 1


async def run(request_path: str | None, list_only: bool, dry_run: bool, chat_mode: bool = False,
              local: bool = False) -> int:
    try:
        settings = load_settings(need_model=not list_only)
    except ConfigError as exc:
        print(f"{RED}Configuration error:{RESET} {exc}")
        return 2

    links = McpLinks(settings)
    print(f"{BOLD}━━ 0. Підключаюся до MCP-серверів (initialize + tools/list){RESET}")
    try:
        async with links.open("obsidian"):
            if list_only:
                listing = await links.call("obsidian", "vault_list", {"path": "Запити"})
                for f in listing.get("files", []):
                    print(f"  Запити/{f}")
                return 0
            try:
                async with links.open("plast-trail"):
                    return await plan_and_write(links, settings, request_path, dry_run, chat_mode=chat_mode,
                                                local=local)
            except Exception as exc:
                if links.reported:
                    return 1
                links.reported = True
                print(f"  {RED}✗ {explain_connection('plast-trail', links.urls['plast-trail'], exc)}{RESET}")
                return 1
    except Exception as exc:
        if links.reported:
            return 1
        print(f"  {RED}✗ {explain_connection('obsidian', links.urls['obsidian'], exc)}{RESET}")
        print(f"\n{RED}Без Obsidian агент не може прочитати запит, тому зупиняюсь. Нічого не змінено.{RESET}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan a Plast hike from an Obsidian request note")
    ap.add_argument("request", nargs="?", help="Vault path of the request note, e.g. 'Запити/....md'")
    ap.add_argument("--list", action="store_true", help="list request notes and exit")
    ap.add_argument("--dry-run", action="store_true", help="do not write the plan to Obsidian")
    ap.add_argument("--chat", action="store_true", help="describe the hike in the terminal; the agent asks what is missing")
    ap.add_argument("--local", action="store_true", help="use only the local Ollama model (no OpenRouter quota)")
    args = ap.parse_args()
    if not args.list and not args.request and not args.chat:
        ap.error("give a request note path, or --chat, or --list")
    sys.exit(asyncio.run(run(args.request, args.list, args.dry_run, args.chat, args.local)))


if __name__ == "__main__":
    main()
