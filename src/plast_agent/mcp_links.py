"""MCP client side: one session per server, shared call logging, readable errors.

    obsidian    - the approved existing server (plugin "Local REST API with MCP"),
                  streamable HTTP over TLS with a bearer key
    plast-trail - our custom server, a separate process on 127.0.0.1:8770

Every call goes through McpLinks.call(), which prints one trace line
("server.tool(args) -> ok/ERROR") and turns MCP error results into McpCallError,
so the workflow can tell a failure from an empty result.
"""

from __future__ import annotations

import json
import ssl
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from plast_agent.config import Settings

DIM, GREEN, RED, CYAN, RESET = "\033[2m", "\033[32m", "\033[31m", "\033[36m", "\033[0m"


class McpCallError(RuntimeError):
    """A tool answered with isError=true."""

    def __init__(self, server: str, tool: str, code: str, message: str, hint: str = "") -> None:
        super().__init__(f"{server}.{tool}: {code}: {message}")
        self.server, self.tool, self.code, self.message, self.hint = server, tool, code, message, hint


class ConnectionFailure(RuntimeError):
    """A server could not be reached or rejected us."""


def _leaves(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


def explain_connection(server: str, url: str, exc: BaseException) -> str:
    for leaf in _leaves(exc):
        if isinstance(leaf, httpx.ConnectError):
            if server == "obsidian":
                return (f"Obsidian MCP is not reachable at {url}. Is Obsidian open with the demo vault "
                        "and the Local REST API plugin enabled?")
            return f"plast-trail is not reachable at {url}. Start it in another terminal: uv run plast-trail"
        if isinstance(leaf, httpx.HTTPStatusError):
            code = leaf.response.status_code
            if code in (401, 403):
                return f"{server} rejected the credentials (HTTP {code}). Check OBSIDIAN_API_KEY in .env."
            return f"{server} answered HTTP {code} at {url}."
        if isinstance(leaf, ssl.SSLError):
            return f"TLS problem talking to {server}: {leaf}. Re-download the plugin certificate."
    return f"{server}: {type(exc).__name__}: {exc}"


def _short(args: dict[str, Any], width: int = 110) -> str:
    text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= width else text[: width - 3] + "..."


class McpLinks:
    """Holds both MCP sessions for the duration of one agent run.

    Each connection is a plain nested `async with` (see open()), so if a server is
    down the transport's error comes out of that block in the same task and can be
    explained in one sentence by explain_connection().
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, list[str]] = {}
        self.urls = {"obsidian": settings.obsidian_mcp_url, "plast-trail": settings.plast_trail_url}
        self.calls = 0
        self.trace: list[dict[str, Any]] = []   # every tools/call of this run, for the plan's technical details
        self.reported = False  # a failure was already explained to the person

    def _http(self, name: str) -> httpx.AsyncClient:
        if name == "obsidian":
            return httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.settings.obsidian_api_key}"},
                verify=ssl.create_default_context(cafile=str(self.settings.obsidian_ca_cert)),
                timeout=httpx.Timeout(15.0, read=60.0),
            )
        return httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=120.0))

    @asynccontextmanager
    async def open(self, name: str) -> AsyncIterator[ClientSession]:
        """Connect: initialize + tools/list (discovery). Keeps the session open inside the block."""
        url = self.urls[name]
        http = self._http(name)
        async with http, streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = [t.name for t in (await session.list_tools()).tools]
                self.sessions[name] = session
                self.tools[name] = tools
                print(f"  {GREEN}✓{RESET} {name} ({init.serverInfo.name}) at {url}: {len(tools)} tools")
                print(f"    {DIM}{', '.join(tools)}{RESET}")
                yield session

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        """Call a tool. Returns structured content (or parsed JSON text); raises McpCallError on isError."""
        self.calls += 1
        started = time.monotonic()
        entry = {"server": server, "tool": tool, "args": _short(args), "result": "connection lost", "ms": None}
        self.trace.append(entry)
        print(f"  {CYAN}→ {server}.{tool}{RESET}({_short(args)})")
        try:
            result = await self.sessions[server].call_tool(tool, args)
        except (httpx.HTTPError, ssl.SSLError, ConnectionError, OSError) as exc:
            print(f"    {RED}✗ connection lost{RESET}")
            raise ConnectionFailure(explain_connection(server, self.urls[server], exc)) from exc
        except Exception as exc:
            if any(isinstance(x, (httpx.HTTPError, ConnectionError, OSError)) for x in _leaves(exc)):
                print(f"    {RED}✗ connection lost{RESET}")
                raise ConnectionFailure(explain_connection(server, self.urls[server], exc)) from exc
            raise
        ms = (time.monotonic() - started) * 1000
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            code, message, hint = "TOOL_ERROR", text, ""
            payload = text[text.find("{"):] if "{" in text else ""
            try:
                err = json.loads(payload)["error"]
                code, message, hint = err["code"], err["message"], err.get("hint", "")
            except (ValueError, KeyError, TypeError):
                pass
            print(f"    {RED}✗ {code}{RESET}: {message[:150]} {DIM}({ms:.0f} ms){RESET}")
            entry.update(result=f"error {code}", ms=round(ms))
            raise McpCallError(server, tool, code, message, hint)
        data: Any = result.structuredContent
        if data is None:
            try:
                data = json.loads(text)
            except ValueError:
                data = text
        print(f"    {GREEN}✓{RESET} {DIM}{ms:.0f} ms{RESET}")
        entry.update(result="ok", ms=round(ms))
        return data
