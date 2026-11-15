"""Settings for the agent, read once from the environment and the local .env file.

A missing key fails here, at start-up, with a message that says what to do,
instead of surfacing later as an opaque 401 in the middle of a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """A required setting is missing or wrong."""


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    obsidian_api_key: str
    obsidian_mcp_url: str
    obsidian_ca_cert: Path
    plast_trail_url: str
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "plast-gemma"
    ollama_enabled: bool = True
    obsidian_leaflet: str = "0"   # 1 = the Obsidian Leaflet plugin is installed: add an interactive map to plans


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_settings(*, need_model: bool = True) -> Settings:
    """Read .env (real environment variables win) and check what the agent needs."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    obsidian_key = _env("OBSIDIAN_API_KEY")
    if not obsidian_key:
        raise ConfigError(
            "OBSIDIAN_API_KEY is empty. Copy it from Obsidian -> Settings -> "
            "Local REST API -> API Key into .env."
        )

    cert = PROJECT_ROOT / _env("OBSIDIAN_CA_CERT", "certs/obsidian-local-rest-api.crt")
    if not cert.exists():
        raise ConfigError(
            f"Obsidian certificate not found at {cert}. With Obsidian open, run:\n"
            "  curl -sk https://127.0.0.1:27124/obsidian-local-rest-api.crt "
            "-o certs/obsidian-local-rest-api.crt"
        )

    openrouter_key = _env("OPENROUTER_API_KEY")
    if need_model and not openrouter_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is empty. Create a key at "
            "https://openrouter.ai/settings/keys and put it into .env."
        )

    host = _env("PLAST_TRAIL_HOST", "127.0.0.1")
    port = _env("PLAST_TRAIL_PORT", "8770")

    return Settings(
        openrouter_api_key=openrouter_key,
        openrouter_model=_env("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        obsidian_api_key=obsidian_key,
        obsidian_mcp_url=_env("OBSIDIAN_MCP_URL", "https://127.0.0.1:27124/mcp/"),
        obsidian_ca_cert=cert,
        plast_trail_url=f"http://{host}:{port}/mcp",
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=_env("OLLAMA_MODEL", "plast-gemma"),
        ollama_enabled=_env("OLLAMA_FALLBACK", "1").lower() not in ("0", "false", "no", "off"),
        obsidian_leaflet=_env("OBSIDIAN_LEAFLET", "0").lower(),
    )
