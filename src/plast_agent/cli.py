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
import sys
import time
import traceback
from pathlib import Path

from plast_agent.config import PROJECT_ROOT, ConfigError, load_settings
from plast_agent.chat import chat
from plast_agent.llm import LLM, LLMError
from plast_agent.mcp_links import ConnectionFailure, McpCallError, McpLinks, _leaves, explain_connection
from plast_agent.render import render_plan
from plast_agent.workflow import AgentStop, build_graph, now_kyiv

BOLD, RED, GREEN, YELLOW, RESET = "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


async def plan_and_write(links: McpLinks, settings, request_path: str | None, dry_run: bool, llm=None,
                         chat_mode: bool = False, ask=None) -> int:
    """Everything after both servers are connected. Catches its own errors and returns an exit code."""
    started = time.monotonic()
    llm = llm or LLM(settings)
    graph = build_graph(links, llm)
    try:
        start: dict = {"request_path": request_path}
        if chat_mode:  # the request comes from a dialogue, is saved to Obsidian, then planned as usual
            got = await (chat(links, llm, ask) if ask else chat(links, llm))
            if got is None:
                return 0
            request_path, request, notes = got
            start = {"request_path": request_path, "request": request, "search_notes": notes}
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

    note = render_plan(state, request_path, llm.model_name, now_kyiv())
    plan_path = f"Плани/{Path(request_path).stem} — план.md"
    print(f"\n{BOLD}━━ 9. Записую план в Obsidian: {plan_path}{RESET}")
    if dry_run:
        print("  (dry run: нічого не записано)")
    else:
        try:
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

    colour = {"ok": GREEN, "attention": YELLOW, "risky": YELLOW}.get(state.get("status"), RED)
    print(f"\n{colour}{BOLD}Готово: {state.get('status')}{RESET} · спроб плану: {len(state.get('attempts', []))} · "
          f"виклики LLM: {llm.calls} · виклики MCP: {links.calls} · {time.monotonic() - started:.0f} с")
    return 0 if state.get("status") in ("ok", "attention", "risky") else 1


async def run(request_path: str | None, list_only: bool, dry_run: bool, chat_mode: bool = False) -> int:
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
                    return await plan_and_write(links, settings, request_path, dry_run, chat_mode=chat_mode)
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
    args = ap.parse_args()
    if not args.list and not args.request and not args.chat:
        ap.error("give a request note path, or --chat, or --list")
    sys.exit(asyncio.run(run(args.request, args.list, args.dry_run, args.chat)))


if __name__ == "__main__":
    main()
