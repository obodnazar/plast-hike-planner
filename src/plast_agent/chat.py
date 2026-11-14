"""Chat mode: the leader describes the hike in their own words in the terminal.

The agent shows what it understood, asks about what is missing, lets the leader correct it,
saves the request as a note in Obsidian (vault_write) and then plans exactly as for a note.

    uv run plast-agent --chat
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Awaitable, Callable

from plast_agent.llm import LLM
from plast_agent.mcp_links import McpLinks
from plast_agent.render import request_lines
from plast_agent.workflow import MISSING_UA, understand

BOLD, CYAN, YELLOW, RESET = "\033[1m", "\033[36m", "\033[33m", "\033[0m"

QUESTIONS = {
    "section": "Яка вікова група: новацтво, юнацтво чи старші?",
    "participants": "Скільки учасників (без дорослих)?",
    "adults": "Скільки дорослих виховників іде з групою?",
    "days": "Скільки днів триватиме мандрівка?",
    "start_date": "Коли старт? (наприклад, «21 листопада» або «наступної суботи»)",
}
OPTIONAL = {
    "region": "У якому районі? Чорногора, Горгани, Свидовець, Мармароси, Яремче–Ворохта (Enter — будь-який)",
}
MAX_ROUNDS = 4

Ask = Callable[[str], Awaitable[str]]


async def terminal_ask(prompt: str) -> str:
    """input() in a thread, so the MCP connections stay responsive while the person types."""
    return (await asyncio.to_thread(input, prompt)).strip()


async def read_description(ask: Ask) -> str:
    print(f"\n{BOLD}Опишіть мандрівку своїми словами.{RESET} Хто йде, коли, куди, що хочете побачити, "
          "які обмеження. Порожній рядок — кінець.")
    lines = []
    while True:
        line = await ask("> ")
        if not line:
            if lines:
                return "\n".join(lines)
            continue
        lines.append(line)


def safe_title(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "", text).strip()[:80]


async def chat(links: McpLinks, llm: LLM, ask: Ask = terminal_ask) -> tuple[str, dict, list[str]] | None:
    """Run the dialogue. Returns (vault path of the saved request, parsed request, notes) or None if cancelled."""
    text = await read_description(ask)
    asked_optional: set[str] = set()
    for _ in range(MAX_ROUNDS):
        print(f"\n{CYAN}Думаю над запитом… (безкоштовна модель може думати до хвилини){RESET}")
        request, notes, missing = await understand(llm, text)
        answers = []
        for key in missing:  # ask only what is needed, one question at a time
            answer = await ask(f"{YELLOW}? {QUESTIONS[key]}{RESET} ")
            if answer:
                answers.append(f"- {MISSING_UA[key].capitalize()}: {answer}")
        for key, question in OPTIONAL.items():
            if request.get(key) is None and key not in asked_optional and not missing:
                asked_optional.add(key)
                answer = await ask(f"{YELLOW}? {question}{RESET} ")
                if answer:
                    answers.append(f"- Район: {answer}")
        if answers:
            text += "\n\nУточнення:\n" + "\n".join(answers)
            continue
        print(f"\n{BOLD}Ось що я зрозумів:{RESET}")
        for line in request_lines(request, notes):
            print("  " + line)
        reply = await ask(f"\n{YELLOW}Все правильно?{RESET} Enter — так, «ні» — скасувати, або напишіть, що виправити: ")
        if not reply:
            break
        if reply.lower() in ("ні", "нi", "no", "скасувати"):
            print("Скасовано. Нічого не записано.")
            return None
        text += f"\n\nВиправлення: {reply}"
    else:
        print("Не вдалося зібрати всі дані за кілька уточнень; почніть, будь ласка, спочатку.")
        return None

    title = safe_title(await ask("Назва запиту для Obsidian (Enter — автоматично): ")) or \
        f"Запит {datetime.now():%Y-%m-%d %H-%M}"
    path = f"Запити/{title}.md"
    note = f"# {title}\n\n{text}\n\n_Записано агентом з діалогу {datetime.now():%Y-%m-%d %H:%M}._\n"
    await links.call("obsidian", "vault_write", {"path": path, "content": note})
    print(f"  Запит збережено в Obsidian: {path}")
    return path, request, notes
