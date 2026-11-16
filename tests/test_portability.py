"""Checks that the project runs on another machine: the server as a real separate process over HTTP,
no cloud key, no system time-zone database. The same tests run in CI on Windows, Linux and macOS.

Run:  uv run pytest -q tests/test_portability.py
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from plast_agent import cli
from plast_agent import llm as L

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_runs_as_a_separate_process_and_answers_over_mcp():
    """`python -m plast_trail.server`: its own process, streamable HTTP, 6 tools, a real call with Ukrainian text."""
    port = free_port()
    env = {**os.environ, "WEATHER_MODE": "replay", "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "plast_trail.server", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(100):   # wait until it listens (about a second)
            if proc.poll() is not None:
                pytest.fail("the server stopped at start-up:\n" + proc.stderr.read().decode("utf-8", "replace")[-2000:])
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            pytest.fail("the server did not start listening in 10 s")

        async def talk():
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
                async with ClientSession(r, w) as session:
                    info = await session.initialize()
                    tools = [t.name for t in (await session.list_tools()).tools]
                    res = await session.call_tool("resolve_places", {"names": ["піпівана"]})
                    return info.serverInfo.name, tools, res
        name, tools, res = asyncio.run(talk())
        assert name == "plast-trail"
        assert tools == ["find_routes", "resolve_places", "validate_hike_plan", "assess_weather_risk",
                         "calculate_supplies", "get_route_profile"]
        assert not res.isError and res.structuredContent["results"][0]["best"]["name"] == "Піп Іван Чорногірський"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_time_zone_data_comes_with_the_project():
    """Windows has no system zone database; without the tzdata package the server fails at import."""
    import tzdata  # noqa: F401
    code = "import plast_trail.validation, plast_trail.weather; print('ok')"
    empty = ROOT / "tests"   # a folder with no zone files: forces the tzdata package to be used
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONTZPATH": str(empty)})
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-800:]


SETTINGS = dict(openrouter_model="x/y:free", openrouter_base_url="https://openrouter.ai/api/v1",
                ollama_base_url="http://localhost:11434/v1", ollama_model="plast-gemma", ollama_enabled=True)


def test_without_a_cloud_key_only_the_local_model_is_used(monkeypatch, capsys):
    settings = SimpleNamespace(openrouter_api_key="", **SETTINGS)
    assert [p.name for p in L.build_providers(settings, True)] == ["ollama"]
    assert L.build_providers(settings, False) == []
    monkeypatch.setattr(L, "ollama_status", lambda s: (True, "Ollama готова: plast-gemma"))
    model = cli.make_llm(settings)
    assert model.provider.name == "ollama"
    assert "працюю лише на локальній моделі" in capsys.readouterr().out


def test_without_any_model_the_message_says_what_to_do(monkeypatch):
    settings = SimpleNamespace(openrouter_api_key="", **SETTINGS)
    monkeypatch.setattr(L, "ollama_status", lambda s: (False, "Ollama не запущена або не встановлена: …"))
    with pytest.raises(L.LLMError) as e:
        cli.make_llm(settings)
    assert "OPENROUTER_API_KEY" in str(e.value) and "openrouter.ai/settings/keys" in str(e.value)


def test_with_a_cloud_key_the_cloud_comes_first():
    settings = SimpleNamespace(openrouter_api_key="test-key-not-real", **SETTINGS)
    assert [p.name for p in L.build_providers(settings, True)] == ["openrouter", "ollama"]


def test_output_is_utf8_even_when_the_console_is_not():
    """Windows with a redirected stdout is cp1251/cp1252: «━━» and Ukrainian must not raise."""
    code = ("from plast_agent.config import utf8_output; utf8_output(); "
            "print('━━ Підключаюся до MCP-серверів ✓')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-500:]
    assert "Підключаюся" in out.stdout.decode("utf-8")
