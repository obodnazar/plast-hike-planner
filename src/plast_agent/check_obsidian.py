"""Connection check for the Obsidian MCP server (Part A: discovery + one call).

Run:  uv run check-obsidian

It connects to the MCP endpoint of the "Local REST API with MCP" plugin,
lists the tools the plugin exposes, and reads one note from the demo vault.
Every failure is translated into a sentence that says what to fix.
"""

from __future__ import annotations

import asyncio
import ssl
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from plast_agent.config import ConfigError, load_settings

TEST_NOTE = "Про цей vault.md"


def _leaves(exc: BaseException) -> list[BaseException]:
    """MCP runs inside task groups, so errors can arrive wrapped in an ExceptionGroup."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


def explain(exc: BaseException, url: str) -> str:
    """Turn a low-level error into a message a person can act on."""
    for leaf in _leaves(exc):
        if isinstance(leaf, httpx.ConnectError):
            return (
                f"Obsidian MCP is not reachable at {url}. "
                "Is Obsidian open with the demo vault and the Local REST API plugin enabled?"
            )
        if isinstance(leaf, httpx.HTTPStatusError):
            code = leaf.response.status_code
            if code in (401, 403):
                return f"Obsidian rejected the API key ({code}). Check OBSIDIAN_API_KEY in .env."
            return f"Obsidian MCP answered HTTP {code}."
        if isinstance(leaf, (ssl.SSLError, httpx.ConnectError)):
            return f"TLS problem talking to Obsidian: {leaf}. Re-download the certificate."
    return f"{type(exc).__name__}: {exc}"


async def check() -> int:
    try:
        settings = load_settings(need_model=False)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    tls = ssl.create_default_context(cafile=str(settings.obsidian_ca_cert))
    http = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {settings.obsidian_api_key}"},
        verify=tls,
        timeout=httpx.Timeout(10.0, read=60.0),
    )

    try:
        async with http, streamable_http_client(settings.obsidian_mcp_url, http_client=http) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                print(f"Connected: {init.serverInfo.name} {init.serverInfo.version}")

                tools = (await session.list_tools()).tools
                print(f"\n{len(tools)} tools exposed:")
                for tool in tools:
                    first_line = (tool.description or "").strip().splitlines()[0:1]
                    print(f"  - {tool.name}: {first_line[0] if first_line else ''}")

                read_tool = next((t for t in tools if t.name == "vault_read"), None)
                if read_tool is None:
                    print("\nvault_read not found, skipping the test read.")
                    return 0

                print(f"\nvault_read input schema: {read_tool.inputSchema}")
                result = await session.call_tool("vault_read", {"path": TEST_NOTE})
                status = "ERROR" if result.isError else "OK"
                text = "".join(getattr(block, "text", "") for block in result.content)
                print(f"\nvault_read('{TEST_NOTE}') -> {status}\n{text[:400]}")
                return 1 if result.isError else 0
    except Exception as exc:  # reported to the person, not swallowed
        print(explain(exc, settings.obsidian_mcp_url), file=sys.stderr)
        return 1


def main() -> None:
    sys.exit(asyncio.run(check()))


if __name__ == "__main__":
    main()
