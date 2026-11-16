"""What the chat needs while plast-web runs: settings, the two MCP connections, the plast-trail process.

One supervisor task per MCP server owns its connection: only that task opens, keeps and reopens it (anyio
cancel scopes must be entered and left in the same task). If a server goes away (Obsidian closed), the
supervisor notices on the next ping or failed call and reconnects when it is back; the chat shows the state.
Conversations only call tools on the open sessions; the review checked that one session can serve
concurrent calls.

If nothing listens on the plast-trail port, plast-web starts the server itself (`python -m plast_trail.server`,
one process, no `uv` wrapper) and stops it on exit. A server the leader started in a terminal is used and
left running.
"""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import httpx

from plast_agent.config import PROJECT_ROOT, ConfigError, load_settings
from plast_agent.mcp_links import McpLinks, _leaves, explain_connection

PING_EVERY_S = 15
RETRY_AFTER_S = 5


CAUSE_UA = {   # (server, cause) -> (what is wrong, what to do), for the page
    ("obsidian", "unreachable"): ("не відповідає", "Відкрийте Obsidian з демо-сховищем і плагіном Local REST API; "
                                                   "я підключуся сам."),
    ("obsidian", "key"): ("не прийняв ключ", "Перевірте OBSIDIAN_API_KEY у .env: Obsidian → Settings → Local REST API "
                                            "→ API Key. Потім перезапустіть plast-web."),
    ("obsidian", "certificate"): ("проблема з сертифікатом", "Завантажте сертифікат плагіна заново в certs/ "
                                                             "(README → Installation and configuration)."),
    ("obsidian", "other"): ("не вдається підключитися", "Подробиці — у Терміналі, де запущено plast-web."),
    ("plast-trail", "unreachable"): ("не відповідає", "Запустіть у Терміналі: uv run plast-trail (я перепідключуся сам)."),
    ("plast-trail", "other"): ("не вдається підключитися", "Подробиці — у Терміналі, де запущено plast-web."),
}


def connection_cause(exc: BaseException) -> str:
    """unreachable / key / certificate / other, from the exception tree of a failed connection."""
    for leaf in _leaves(exc):
        if isinstance(leaf, httpx.HTTPStatusError) and leaf.response.status_code in (401, 403):
            return "key"
        if isinstance(leaf, ssl.SSLError):
            return "certificate"
        if isinstance(leaf, (httpx.ConnectError, ConnectionError, OSError)):
            return "unreachable"
    return "other"


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class Runtime:
    def __init__(self) -> None:
        self.settings: Any = None
        self.problem = ""                          # why the chat cannot start (settings), in Ukrainian
        self.links: McpLinks | None = None
        self.state = {"obsidian": "starting", "plast-trail": "starting"}
        self.reason = {"obsidian": "", "plast-trail": ""}   # the technical reason, for the terminal
        self.cause = {"obsidian": "", "plast-trail": ""}    # unreachable / key / certificate / other
        self.tasks: list[asyncio.Task] = []
        self.process: subprocess.Popen | None = None
        self.plan_lock = asyncio.Lock()            # one plan at a time: the free model quota is shared
        self.closing = False

    # ------------------------------------------------------------------ start / stop
    async def start(self) -> None:
        try:
            self.settings = load_settings(need_model=False)
        except ConfigError as exc:
            self.problem = ("Агента ще не налаштовано: " + str(exc) + " Інструкція: README → Installation and "
                            "configuration. Збережені плани переглядати можна й без цього.")
            self.state = {"obsidian": "off", "plast-trail": "off"}
            return
        self.links = McpLinks(self.settings)
        if self.start_plast_trail_if_needed():
            url = urlparse(self.settings.plast_trail_url)
            for _ in range(60):   # the server needs a second or two to start listening
                if port_open(url.hostname or "127.0.0.1", url.port or 8770) or self.process.poll() is not None:
                    break
                await asyncio.sleep(0.25)
        self.tasks = [asyncio.create_task(self.supervise(name)) for name in ("plast-trail", "obsidian")]

    def start_plast_trail_if_needed(self) -> bool:
        url = urlparse(self.settings.plast_trail_url)
        host, port = url.hostname or "127.0.0.1", url.port or 8770
        if host not in ("127.0.0.1", "localhost") or port_open(host, port):
            return False   # someone runs it already (the leader in a terminal) or it is elsewhere: use, do not own
        log = (PROJECT_ROOT / "out" / "plast-trail.log")
        log.parent.mkdir(exist_ok=True)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "plast_trail.server", "--port", str(port)], cwd=PROJECT_ROOT,
            stdout=log.open("a", encoding="utf-8"), stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1"})
        print(f"  plast-trail запущено окремим процесом (порт {port}, журнал {log})")
        return True

    async def stop(self) -> None:
        self.closing = True
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    # ------------------------------------------------------------------ supervisors
    async def supervise(self, name: str) -> None:
        while not self.closing:
            try:
                async with self.links.open(name) as session:
                    if name == "plast-trail" and "find_routes" not in self.links.tools.get(name, []):
                        self.state[name], self.reason[name] = "wrong", "на цьому порту працює інший сервер"
                        raise ConnectionError("not plast-trail")
                    self.state[name], self.reason[name] = "ok", ""
                    while not self.closing:
                        await asyncio.sleep(PING_EVERY_S)
                        await asyncio.wait_for(session.send_ping(), timeout=10)
            except asyncio.CancelledError:
                raise
            except Exception as exc:   # gone, refused, wrong key: say why, try again later
                if self.state[name] != "wrong":
                    self.state[name] = "down"
                    self.reason[name] = explain_connection(name, self.links.urls[name], exc)
                    self.cause[name] = connection_cause(exc)
            self.links.sessions.pop(name, None)
            await asyncio.sleep(RETRY_AFTER_S)

    def ready(self) -> bool:
        return not self.problem and all(v == "ok" for v in self.state.values())

    def status(self) -> dict[str, Any]:
        """For the header and the chat: what works, and what to do about what does not (Ukrainian)."""
        ua = {"ok": "працює", "starting": "підключаюся…", "down": "недоступний", "wrong": "інший сервер на порту",
              "off": "не налаштовано"}
        items = []
        for name, label in (("obsidian", "Obsidian"), ("plast-trail", "Сервер маршрутів plast-trail")):
            st = self.state[name]
            todo, label_ua = "", ua.get(st, st)
            if st == "down":
                label_ua, todo = CAUSE_UA[(name, self.cause[name])] if (name, self.cause[name]) in CAUSE_UA \
                    else CAUSE_UA[(name, "unreachable")]
            elif st == "wrong":
                todo = "Звільніть порт або змініть PLAST_TRAIL_PORT у .env."
            items.append({"name": label, "state": st, "label": label_ua, "todo": todo})
        return {"ready": self.ready(), "problem": self.problem, "items": items}
