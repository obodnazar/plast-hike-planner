"""One chat in the browser: the leader describes the hike, the agent asks what is missing, shows what it
understood (each field can be changed by a click), plans with the same LangGraph agent as the CLI and answers
with a plan card.

The dialogue follows plast_agent/chat.py step by step (the model reads the description once; short answers
and field edits are read by code; the model re-reads only what code cannot read), but talks in events instead
of print(): the CLI stays exactly as it is for the defence.

A conversation lives on the server, not in the browser tab: every event gets a number and is kept, so a
reloaded tab catches up (hello with last_seq), and a closed tab does not stop a plan that is being made.
Everything the leader reads here is Ukrainian written by code, or passed through plast_web.view.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from datetime import date, timedelta
from typing import Any

from plast_agent import cli
from plast_agent.chat import LABELS, QUESTIONS, QUESTIONS_SENIOR, parse_answer, safe_title
from plast_agent.llm import LLMError
from plast_agent.mcp_links import ConnectionFailure, McpCallError
from plast_agent.render import FEATURE_UA, SECTION_UA
from plast_agent.understanding import (ADULT_WORD, B, NO_ADULTS, PARTICIPANT_WORD, REGION_UA, dont_care, group_consistency,
                                       norm, parse_count, parse_date, parse_days, regions_named)
from plast_agent.workflow import MISSING_UA, kyiv_now, resolve_named_places, today_kyiv, understand
from plast_web.view import MONTHS, clean_note, fmt_date, place_names, present

MAX_TEXT = 4000
# What a message after a plan keeps from that plan when the new reading leaves it empty. Wishes, places and
# limits are not here: «більше не хочу озера» must be able to empty them.
KEEP_AFTER_PLAN = ("section", "participants", "adults", "days", "start_date", "start_time", "region",
                   "date_flexibility_days")
def talked_about(change: str) -> set[str]:
    """The fields a message after a plan talks about («Дорослих буде 0» -> adults). The old plan's value of
    such a field is never put back: if the new reading cannot confirm it, the agent asks."""
    t = norm(change)
    found = set()
    if PARTICIPANT_WORD.search(t):
        found.add("participants")
    if ADULT_WORD.search(t) or NO_ADULTS.search(t):
        found.add("adults")
    if re.search(B + r"(?:дн|день|ноч|одноден)", t) or parse_days(change):
        found.add("days")
    if parse_date(change, today_kyiv()) or re.search(r"старт|вих[іо]д|дат", t):
        found.add("start_date")
    if re.search(B + r"(?:новац|новак|юнац|юнак|старш|сеніор|вік|рок)", t):
        found.add("section")
    return found


SECTION_OPTIONS = [("novatstvo", "новацтво (6–10 років)"), ("yunatstvo", "юнацтво (11–17)"),
                   ("senior", "старші (18+)")]
STEPS = [   # what the leader sees while the graph runs; graph nodes map onto these
    ("read", "Записую й читаю запит в Obsidian"),
    ("search", "Шукаю маршрути"),
    ("plan", "Складаю план і перевіряю норми безпеки"),
    ("weather", "Дивлюся прогноз погоди"),
    ("supplies", "Рахую їжу, воду й спорядження"),
    ("picture", "Готую карту й профіль висот"),
    ("conclude", "Пишу висновок"),
    ("save", "Зберігаю план"),
]
NODE_STEP = {"read_request": "read", "parse_request": "read", "search_routes": "search", "propose": "plan",
             "validate": "plan", "weather": "weather", "give_up": "plan", "supplies": "supplies",
             "picture": "picture", "conclude": "conclude"}
ERRORS_UA = {
    "model": "Модель зараз не відповідає: на сьогодні вичерпано безкоштовний ліміт або немає зв'язку. Спробуйте "
             "пізніше. Якщо встановлено локальну модель Ollama, я перейду на неї сам.",
    "LLMError": "Модель не відповіла вчасно або вичерпано безкоштовний ліміт. Спробуйте ще раз трохи пізніше.",
    "AgentStop": "Жоден маршрут у даних не підходить під ці умови. Спробуйте інший район, менше днів або м'якші "
                 "обмеження.",
    "ConnectionFailure": "Зв'язок із сервером обірвався посеред роботи. Перевірте, що Obsidian і сервер маршрутів "
                         "запущені, і натисніть «Спробувати ще раз».",
    "McpCallError": "Один з інструментів повернув помилку. Подробиці — у Терміналі, де запущено plast-web.",
    "obsidian_write": "План готовий, але записати його в Obsidian не вдалося. Сторінка плану працює.",
}


def human_date(iso: str | None) -> str:
    return fmt_date(iso) if iso else "не вказано"


def card_fields(r: dict[str, Any]) -> list[dict[str, Any]]:
    """The «Ось що я зрозумів» card: values for reading, and how each one can be changed in place."""
    senior = r.get("section") == "senior"
    fields = [
        {"key": "section", "label": "Група", "value": SECTION_UA.get(r.get("section"), "не вказано"),
         "edit": {"kind": "select", "value": r.get("section") or "",
                  "options": [{"value": k, "label": v} for k, v in SECTION_OPTIONS]}},
        {"key": "participants", "label": "Вас усього" if senior else "Учасників",
         "value": str(r.get("participants") or "не вказано"),
         "edit": {"kind": "number", "value": r.get("participants"), "min": 1, "max": 60}},
    ]
    if not senior:
        fields.append({"key": "adults", "label": "Дорослих", "value": str(r.get("adults") if r.get("adults") is not None
                                                                         else "не вказано"),
                       "edit": {"kind": "number", "value": r.get("adults"), "min": 0, "max": 20}})
    fields += [
        {"key": "days", "label": "Днів", "value": str(r.get("days") or "не вказано"),
         "edit": {"kind": "number", "value": r.get("days"), "min": 1, "max": 7}},
        {"key": "start_date", "label": "Старт", "value": f"{human_date(r.get('start_date'))}, "
                                                          f"{r.get('start_time') or '08:00'}",
         "edit": {"kind": "date", "value": r.get("start_date") or ""}},
        {"key": "region", "label": "Район", "value": REGION_UA.get(r.get("region"), "будь-який"),
         "edit": {"kind": "select", "value": r.get("region") or "",
                  "options": [{"value": "", "label": "будь-який"}] +
                             [{"value": k, "label": v} for k, v in REGION_UA.items()]}},
    ]
    if r.get("places"):
        fields.append({"key": "places", "label": "Місця", "value": ", ".join(r["places"]), "edit": None})
    if r.get("must_include"):
        fields.append({"key": "must_include", "label": "Побажання",
                       "value": ", ".join(FEATURE_UA.get(f, f) for f in r["must_include"]), "edit": None})
    if r.get("max_km_per_day"):
        fields.append({"key": "max_km_per_day", "label": "Обмеження", "value": f"до {r['max_km_per_day']:g} км на день",
                       "edit": None})
    return fields


class Conversation:
    def __init__(self, runtime, today: date | None = None, llm=None) -> None:
        self.id = secrets.token_hex(8)
        self.runtime = runtime
        self.today = today or today_kyiv()
        self.llm = llm                                 # tests pass a scripted model; otherwise made on first need
        self.events: list[dict[str, Any]] = []
        self.listeners: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None          # the work in progress; NOT tied to any browser tab
        self.reset()

    def reset(self) -> None:
        self.state = "describe"
        self.text = ""
        self.request: dict[str, Any] = {}
        self.notes: list[str] = []
        self.pending: str | None = None
        self.asked_region = False
        self.card_n = getattr(self, "card_n", 0)
        self.prepared: tuple[str, dict, list[str]] | None = None
        self.trace_from = 0

    # ------------------------------------------------------------------ events
    def emit(self, event: dict[str, Any]) -> None:
        event = {**event, "seq": len(self.events) + 1, "at": int(time.time() * 1000)}
        self.events.append(event)
        for q in list(self.listeners):
            q.put_nowait(event)

    def say(self, text: str, quick: list[str] | None = None) -> None:
        self.emit({"type": "msg", "role": "agent", "text": text, "quick": quick or []})

    def set_state(self, state: str) -> None:
        self.state = state
        placeholder = {"describe": "Опишіть мандрівку: хто, коли, куди, що хочете побачити…",
                       "asking": "Ваша відповідь…", "confirm": "Або напишіть, що виправити…",
                       "busy": "Зачекайте, я працюю…",
                       "done": "Змініть план («а через Говерлу?», «на 2 дні») або натисніть «Новий план»…"}
        self.emit({"type": "state", "state": state, "placeholder": placeholder.get(state, "")})

    def since(self, seq: int) -> list[dict[str, Any]]:
        return [e for e in self.events if e["seq"] > seq]

    # ------------------------------------------------------------------ input from the browser
    def handle(self, msg: dict[str, Any]) -> None:
        """Called for every message of the page. Work runs as a task of the conversation, so a closed tab
        does not cancel it. While it runs, the page shows «зачекайте» and new input is refused."""
        if self.task and not self.task.done():
            self.say("Хвилинку, я ще працюю над попереднім.")
            return
        kind = msg.get("type")
        if kind == "say":
            text = str(msg.get("text") or "").strip()[:MAX_TEXT]
            if text:
                self.emit({"type": "msg", "role": "user", "text": text})
                self.start(self.on_text(text))
        elif kind == "set_field" and self.state == "confirm":
            self.set_field(str(msg.get("key")), msg.get("value"))
        elif kind == "confirm" and self.state == "confirm":
            self.emit({"type": "msg", "role": "user", "text": "Все правильно, плануй"})
            self.start(self.plan())
        elif kind == "cancel" and self.state in ("confirm", "asking"):
            self.emit({"type": "msg", "role": "user", "text": "Скасувати"})
            self.reset()
            self.say("Гаразд, нічого не записую. Опишіть іншу мандрівку, коли будете готові.")
            self.set_state("describe")
        elif kind == "retry" and self.state == "done" and self.prepared:
            self.emit({"type": "msg", "role": "user", "text": "Спробувати ще раз"})
            self.start(self.plan(again=True))

    def start(self, work) -> None:
        async def run():
            try:
                await work
            except Exception as exc:   # never leave the leader with a silent page
                self.emit({"type": "thinking", "on": False})
                print(f"  ✗ помилка розмови: {type(exc).__name__}: {exc}")
                self.say("Щось пішло не так усередині агента. Подробиці — у Терміналі, де запущено plast-web. "
                         "Спробуйте ще раз.")
                self.set_state("confirm" if self.request else "describe")
        self.task = asyncio.create_task(run())

    async def on_text(self, text: str) -> None:
        if self.state == "done" and self.request:   # «а на Говерлу ми зайдем?» changes the plan just made
            before, keep = self.text, dict(self.request)
            self.reset()
            self.text = before + f"\n\nЗміна після плану: {text}"
            self.asked_region = True
            self.trace_from = len(self.runtime.links.trace) if self.runtime.links else 0
            talked = talked_about(text)   # what the change itself is about: never refilled from the old plan
            await self.read("Змінюю план з урахуванням цього…", keep={k: v for k, v in keep.items() if k not in talked})
            return
        if self.state in ("describe", "done"):
            self.reset()
            self.text = text
            self.trace_from = len(self.runtime.links.trace) if self.runtime.links else 0
            await self.read("Читаю ваш опис…")
        elif self.state == "asking":
            await self.answer(text)
        elif self.state == "confirm":
            if text.lower().strip(" .!") in ("ні", "нi", "скасувати", "стоп"):
                self.reset()
                self.say("Гаразд, нічого не записую. Опишіть іншу мандрівку, коли будете готові.")
                self.set_state("describe")
                return
            if text.lower().strip(" .!") in ("так", "все правильно", "ок", "добре", "плануй"):
                await self.plan()
                return
            self.text += f"\n\nВиправлення: {text}"
            await self.read("Перечитую запит з виправленням…")

    # ------------------------------------------------------------------ the dialogue (as in chat.py)
    async def model(self):
        if self.llm is None:
            self.llm = await asyncio.to_thread(cli.make_llm, self.runtime.settings)   # prints the quota
        return self.llm

    async def read(self, label: str, keep: dict[str, Any] | None = None) -> None:
        """The model reads the whole text. keep: the request of the previous plan; what the model leaves empty
        now (the group, the people, the dates) is taken from it, so a follow-up does not ask everything again."""
        self.set_state("busy")
        self.emit({"type": "thinking", "on": True, "label": label})
        try:
            llm = await self.model()
            async with self.runtime.plan_lock:
                request, notes, _ = await understand(llm, self.text, self.today)
            notes += await resolve_named_places(self.runtime.links, request, self.text)
        except LLMError:
            self.emit({"type": "thinking", "on": False})
            self.say(ERRORS_UA["model"])
            self.set_state("describe" if not self.request else "confirm")
            return
        self.emit({"type": "thinking", "on": False})
        for key, value in (keep or {}).items():   # only who / when / where; wishes and limits follow the text
            if key in KEEP_AFTER_PLAN and request.get(key) is None and value is not None:
                request[key] = value
        self.request, self.notes = request, notes
        await self.next_step()

    async def next_step(self) -> None:
        r = self.request
        for key in MISSING_UA:   # one question at a time, the group first (it changes the next questions)
            if r.get(key) is not None:
                continue
            if key == "adults" and r.get("section") == "senior":
                r["adults"] = 0
                continue
            self.pending = key
            question = (QUESTIONS_SENIOR.get(key, QUESTIONS[key]) if r.get("section") == "senior" else QUESTIONS[key])
            self.say(question, self.quick_replies(key))
            self.set_state("asking")
            return
        if r.get("region") is None and not r.get("places") and not self.asked_region:
            self.asked_region, self.pending = True, "region"
            self.say("У якому районі? Якщо не важливо, оберу сам.", [*REGION_UA.values(), "Будь-який"])
            self.set_state("asking")
            return
        for note in group_consistency(r, self.text):
            if note not in self.notes:
                self.notes.append(note)
        self.show_card()

    def quick_replies(self, key: str) -> list[str]:
        if key == "section":
            return ["Новацтво", "Юнацтво", "Старші (18+)"]
        if key == "participants":
            return ["6", "8", "10", "12"]
        if key == "adults":
            return ["1", "2", "3"]
        if key == "days":
            return ["1", "2", "3"]
        if key == "start_date":
            sat = self.today + timedelta(days=(5 - self.today.weekday()) % 7 or 7)
            return [f"{d.day} {MONTHS[d.month - 1]}" for d in (sat, sat + timedelta(days=7))] + ["Не важливо"]
        return []

    async def answer(self, text: str) -> None:
        key, r = self.pending, self.request
        if key == "region":
            named = regions_named(text)
            if len(named) == 1:
                r["region"] = named.pop()
            if not dont_care(text) and text.lower() != "будь-який":
                self.text += f"\n\nУточнення:\n- Район: {text}"
            await self.next_step()
            return
        value, note, extra = parse_answer(key, text, self.today)
        if value is None:
            if dont_care(text):
                self.say("Це потрібно для перевірки норм безпеки. Напишіть, будь ласка, число або вік.",
                         self.quick_replies(key))
                return
            self.text += f"\n\nУточнення:\n- {LABELS[key]}: {text}"   # the code could not read it: the model can
            await self.read("Перечитую запит разом з уточненням…")
            return
        r[key] = value
        r.update(extra)
        if note:
            self.notes.append(note)
        self.text += f"\n\nУточнення:\n- {LABELS[key]}: {text}"
        if key != "start_date" and r.get("start_date") is None:   # «не важливо, хочемо вийти 19 листопада»
            found = parse_date(text, self.today)
            if found and found >= self.today:
                r["start_date"] = found.isoformat()
        await self.next_step()

    def show_card(self) -> None:
        self.card_n += 1
        names = place_names({"request": self.request})
        notes = [n for n in (clean_note(x, names) for x in self.notes) if n]
        self.say("Перевірте, будь ласка, чи я все правильно зрозумів. Будь-яке поле можна змінити натиском.")
        self.emit({"type": "card", "id": f"understood-{self.card_n}", "fields": card_fields(self.request),
                   "notes": notes, "active": True})
        self.set_state("confirm")

    def set_field(self, key: str, value: Any) -> None:
        """A change in the card, read by code (no model call). The note in Obsidian gets the correction too."""
        r = self.request
        if key == "section" and value in SECTION_UA:
            r["section"] = value
            if value == "senior":
                r["adults"] = 0
        elif key in ("participants", "adults", "days"):
            n = parse_count(str(value)) if not isinstance(value, int) else value
            limits = {"participants": (1, 60), "adults": (0, 20), "days": (1, 7)}[key]
            if n is None or not limits[0] <= n <= limits[1]:
                self.say("Це число я не можу прийняти: перевірте, будь ласка.")
                return
            r[key] = n
        elif key == "start_date":
            try:
                d = date.fromisoformat(str(value))
            except ValueError:
                return
            if d < self.today:
                self.say("Ця дата вже минула: оберіть, будь ласка, іншу.")
                return
            r["start_date"] = d.isoformat()
        elif key == "region" and (value in REGION_UA or value == ""):
            r["region"] = value or None
        else:
            return
        shown = next(f["value"] for f in card_fields(r) if f["key"] == key)
        label = next(f["label"] for f in card_fields(r) if f["key"] == key)
        self.text += f"\n\nВиправлення: {label} — {shown}"
        self.emit({"type": "msg", "role": "user", "text": f"{label}: {shown}"})
        self.emit({"type": "card", "id": f"understood-{self.card_n}", "fields": card_fields(r),
                   "notes": [n for n in (clean_note(x, set()) for x in self.notes) if n], "active": True})

    # ------------------------------------------------------------------ planning
    async def plan(self, again: bool = False) -> None:
        rt = self.runtime
        self.set_state("busy")
        self.emit({"type": "card", "id": f"understood-{self.card_n}", "fields": card_fields(self.request),
                   "notes": [], "active": False})
        if not rt.ready():
            self.say("Зараз не можу планувати: " + "; ".join(f"{i['name']} — {i['label']}" for i in
                                                           rt.status()["items"] if i["state"] != "ok") +
                     ". Подробиці й що зробити — вгорі сторінки.")
            self.set_state("confirm" if not again else "done")
            return
        progress = Progress(self)
        try:
            llm = await self.model()
        except LLMError:
            self.say(ERRORS_UA["model"])
            self.set_state("confirm")
            return
        async with rt.plan_lock:
            if not again:
                title = f"Запит {kyiv_now():%Y-%m-%d %H-%M-%S}"
                path = f"Запити/{safe_title(title)}.md"
                note = f"# {title}\n\n{self.text}\n\n_Записано агентом з чату {kyiv_now():%Y-%m-%d %H:%M}._\n"
                try:
                    await rt.links.call("obsidian", "vault_write", {"path": path, "content": note})
                except (ConnectionFailure, McpCallError):
                    progress.finish(ok=False)
                    self.say("Не вдалося записати запит в Obsidian. Перевірте, що Obsidian відкритий, і натисніть "
                             "«Все правильно» ще раз.")
                    self.set_state("confirm")
                    return
                self.prepared = (path, dict(self.request), list(self.notes))
            report: dict[str, Any] = {}
            await cli.plan_and_write(rt.links, rt.settings, self.prepared[0], dry_run=False, llm=llm,
                                     prepared=(self.prepared[0], dict(self.prepared[1]), list(self.prepared[2])),
                                     on_node=progress.node, report=report, trace_from=self.trace_from)
        progress.finish(ok="record" in report)
        record = report.get("record")
        if record:
            view = present(record)
            self.say(view["conclusion"])
            self.emit({"type": "plan", **plan_card(view)})
            if report.get("error"):
                self.say(ERRORS_UA["obsidian_write"])
            self.set_state("done")
            return
        self.say(ERRORS_UA.get(report.get("error", ""), "Щось пішло не так усередині агента. Подробиці — у "
                                                         "Терміналі, де запущено plast-web."),
                 ["Спробувати ще раз"] if report.get("error") in ("ConnectionFailure", "LLMError", "model") else [])
        self.set_state("done")


def plan_card(view: dict[str, Any]) -> dict[str, Any]:
    """The card under the answer: the plan at a glance and the way to the full page."""
    prof = (view.get("map") or {}).get("profile") or []
    step = max(1, len(prof) // 120)
    return {
        "plan_id": view["id"], "title": view["title"], "status": view["status"], "meta": view["meta"],
        "stats": view["stats"], "top": view["safety"][0] if view["safety"] else None,
        "spark": [[round(p[0], 2), round(p[3]), p[4]] for p in prof[::step]],
        "links": {"page_url": f"/plan/{view['id']}", "gpx_url": view["links"]["gpx_url"],
                  "obsidian_url": view["links"]["obsidian_url"]},
    }


class Progress:
    """The progress card, updated in place as the graph runs."""

    def __init__(self, conv: Conversation) -> None:
        self.conv = conv
        self.id = f"progress-{len(conv.events) + 1}"
        self.started = int(time.time() * 1000)
        self.state = {key: "todo" for key, _ in STEPS}
        self.note = {key: "" for key, _ in STEPS}
        self.state["read"] = "now"
        self.attempt = 0
        self.send()

    def send(self, finished: bool = False) -> None:
        self.conv.emit({"type": "progress", "id": self.id, "started": self.started, "finished": finished,
                        "steps": [{"label": label, "state": self.state[k], "note": self.note[k]} for k, label in STEPS]})

    async def node(self, name: str, update: dict[str, Any]) -> None:
        key = NODE_STEP.get(name)
        if not key:
            return
        order = [k for k, _ in STEPS]
        for k in order[:order.index(key)]:
            if self.state[k] != "done":
                self.state[k] = "done"
        if name == "search_routes":
            n = len((update.get("search") or {}).get("routes") or [])
            self.note["search"] = f"знайшов {n}"
        if name == "propose":
            self.attempt += 1
            self.note["plan"] = f"спроба {self.attempt}"
        if name == "weather" and update.get("weather"):
            risk = (update["weather"] or {}).get("overall_risk")
            self.note["weather"] = {"low": "ризик низький", "medium": "ризик середній", "high": "ризик високий",
                                    "unknown": "прогнозу ще немає"}.get(risk, "")
            if risk == "high":
                self.state["weather"] = "done"
                self.state["plan"] = "now"
                self.note["plan"] = f"спроба {self.attempt}: погода ризикована, шукаю інший варіант"
                self.send()
                return
        self.state[key] = "done"
        nxt = order.index(key) + 1
        if name in ("propose",):
            self.state["plan"] = "now"
        elif nxt < len(order) and self.state[order[nxt]] == "todo":
            self.state[order[nxt]] = "now"
        self.send()

    def finish(self, ok: bool) -> None:
        for k in self.state:
            if ok or self.state[k] == "done":
                self.state[k] = "done"
            elif self.state[k] == "now":
                self.state[k] = "failed"
        self.send(finished=True)
