"""Chat mode: the leader describes the hike in their own words in the terminal.

The agent shows what it understood, asks about what is missing, lets the leader correct it,
saves the request as a note in Obsidian (vault_write) and then plans exactly as for a note.
The model reads the description once; the answers to the questions are read by code.

    uv run plast-agent --chat
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from typing import Any, Awaitable, Callable

from plast_agent.llm import LLM
from plast_agent.mcp_links import McpLinks
from plast_agent.render import request_lines
from plast_agent.understanding import (RANGE, dont_care, group_consistency, next_weekday, norm, parse_count,
                                       parse_date, parse_days, regions_named)
from plast_agent.workflow import MISSING_UA, resolve_named_places, understand

BOLD, CYAN, YELLOW, RESET = "\033[1m", "\033[36m", "\033[33m", "\033[0m"

QUESTIONS = {
    "section": "Хто йде: новацтво (6–10 років), юнацтво (11–17) чи старші / дорослі (18+)? Можна просто вік.",
    "participants": "Скільки учасників (без дорослих супровідників)?",
    "adults": "Скільки дорослих (виховників, супроводу) іде з групою?",
    "days": "Скільки днів триватиме мандрівка?",
    "start_date": "Коли старт? (наприклад, «21 листопада» або «наступної суботи»)",
}
QUESTIONS_SENIOR = {"participants": "Скільки вас усього?"}
LABELS = {"section": "Група", "participants": "Учасників", "adults": "Дорослих", "days": "Днів", "start_date": "Старт"}
OPTIONAL = {
    "region": "У якому районі? Чорногора, Горгани, Свидовець, Мармароси, Яремче–Ворохта (Enter — будь-який)",
}
MAX_ROUNDS = 6
MAX_EMPTY = 3

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


def guess_section(answer: str) -> str | None:
    """Read the group from a short answer without a model call: a word or an age."""
    a = answer.lower()
    for words, section in ((("новац", "новак", "упн", "рій"), "novatstvo"),
                           (("юнац", "юнак", "юначк", "упю", "гурт", "підліт"), "yunatstvo"),
                           (("старш", "сеньйор", "сенйор", "сенiор", "сеніор", "упс", "усп", "доросл"), "senior")):
        if any(w in a for w in words):
            return section
    ages = [int(n) for n in re.findall(r"\d+", a) if 3 <= int(n) <= 99]
    if ages:
        age = min(ages)
        return "novatstvo" if age <= 10 else "yunatstvo" if age <= 17 else "senior"
    return None


async def ask_required(ask: Ask, question: str) -> str:
    """Ask again on an empty answer: these facts are needed for the plan."""
    for _ in range(MAX_EMPTY):
        answer = await ask(f"{YELLOW}? {question}{RESET} ")
        if answer:
            return answer
        print("  Це потрібно для плану — напишіть, будь ласка, коротко.")
    return ""


def safe_title(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "", text).strip()[:80]


DEFAULT_DAYS = 2
THINKING = f"\n{CYAN}Думаю над запитом… (безкоштовна модель може думати до хвилини){RESET}"


def parse_answer(key: str, answer: str, today: date) -> tuple[Any, str | None, dict[str, Any]]:
    """A short answer -> (value, note for the leader, other fields to set), by code: no model call.
    value None = the code could not read it (then the model re-reads the whole text)."""
    if key == "section":
        return guess_section(answer), None, {}
    if key in ("participants", "adults"):
        n = parse_count(answer)
        if n is None and key == "adults" and norm(answer).strip(" .!") in ("я", "тільки я", "лише я", "я сам", "я сама"):
            n = 1
        if n is not None and RANGE.search(norm(answer)):
            return n, (f"{LABELS[key]}: {n} — з «{answer.strip()}» беру більше число, щоб вистачило запасів і "
                       "дорослих; змініть, якщо треба."), {}
        return n, None, {}
    if key == "days":
        n = parse_days(answer) or parse_count(answer)
        if n and 1 <= n <= 7:
            return n, None, {}
        if dont_care(answer):
            return DEFAULT_DAYS, (f"Днів: {DEFAULT_DAYS} — моє припущення (ви написали, що не важливо); "
                                  "змініть, якщо треба."), {}
        return None, None, {}
    found = parse_date(answer, today)   # key == "start_date"
    if found and found >= today:
        return found.isoformat(), None, {}
    if dont_care(answer):
        saturday = next_weekday(today, 5)
        return saturday.isoformat(), (f"Старт: {saturday:%d.%m} (найближча субота) — моє припущення; "
                                      "агент може зсунути його до 3 днів, якщо погода погана."), {"date_flexibility_days": 3}
    return None, None, {}


async def chat(links: McpLinks, llm: LLM, ask: Ask = terminal_ask,
               today: date | None = None) -> tuple[str, dict, list[str]] | None:
    """Run the dialogue. Returns (vault path of the saved request, parsed request, notes) or None if cancelled.
    The model reads the description once; short answers are read by code (understanding.parse_*), so a
    question costs no model request. The model re-reads only an answer the code cannot read, or a correction."""
    today = today or date.today()

    async def read(text: str) -> tuple[dict, list[str]]:   # the model reads; plast-trail checks the place names
        request, notes, _ = await understand(llm, text, today)
        return request, notes + await resolve_named_places(links, request, text)

    text = await read_description(ask)
    print(THINKING)
    request, notes = await read(text)
    asked_region = False
    for _ in range(MAX_ROUNDS):
        answers, unread = [], False
        for key in MISSING_UA:  # one question at a time, the group first (it changes the next questions)
            if request.get(key) is not None:
                continue
            if key == "adults" and request.get("section") == "senior":
                request["adults"] = 0
                answers.append("- Дорослих супровідників: 0 (група дорослих)")
                continue
            question = QUESTIONS_SENIOR.get(key, QUESTIONS[key]) if request.get("section") == "senior" else QUESTIONS[key]
            answer = await ask_required(ask, question)
            if not answer:
                continue
            answers.append(f"- {LABELS[key]}: {answer}")
            value, note, extra = parse_answer(key, answer, today)
            if value is None:
                if dont_care(answer):  # the group and the people cannot be assumed: the safety rules need them
                    print("  Це потрібно для перевірки норм безпеки — напишіть, будь ласка, число або вік.")
                else:
                    unread = True
                continue
            request[key] = value
            request.update(extra)
            if note:
                notes.append(note)
            if key != "start_date" and request.get("start_date") is None:   # «не важливо, хочемо вийти 19 листопада»
                found = parse_date(answer, today)
                if found and found >= today:
                    request["start_date"] = found.isoformat()
        if answers:
            text += "\n\nУточнення:\n" + "\n".join(answers)
        if unread:
            print(f"{THINKING}\n  (перечитую запит разом з уточненнями)")
            request, notes = await read(text)
            continue
        notes += group_consistency(request, text)
        if any(request.get(k) is None for k in MISSING_UA):
            if not answers:
                print("  Без цих даних план не скласти. Почніть, будь ласка, спочатку.")
                return None
            continue
        if request.get("region") is None and not request.get("places") and not asked_region:
            asked_region = True   # optional, and not needed when the leader named places
            answer = await ask(f"{YELLOW}? {OPTIONAL['region']}{RESET} ")
            if answer:
                text += f"\n\nУточнення:\n- Район: {answer}"
                named = regions_named(answer)
                if len(named) == 1:
                    request["region"] = named.pop()
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
        print(THINKING)
        request, notes = await read(text)
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
