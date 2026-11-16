"""Saved plans: after every run the agent keeps what it found as one JSON file, so the web page
(`uv run plast-web`) can show the plan without running the agent again.

A record (PlanResult) is the agent's own data: tool results as they came, the model's proposal, the
calls made. It is technical on purpose; the web turns it into human text in plast_web/view.py.

Files live in out/plans/ (not in git: they are the leader's runs). PLAST_PLANS_DIR moves them.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from plast_agent.config import PROJECT_ROOT

VERSION = 1
ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")   # 20261116-140512-a1b2c3: ASCII only, safe in a URL
STATE_KEYS = ("request_text", "request", "search", "search_notes", "attempts", "proposal", "validation",
              "weather", "supplies", "route_profile", "status", "conclusion")


def plans_dir() -> Path:
    return Path(os.environ.get("PLAST_PLANS_DIR") or PROJECT_ROOT / "out" / "plans")


def new_id(now: datetime | None = None) -> str:
    """Date and time first (the list sorts by it), then 6 random hex digits: two runs never collide."""
    now = now or datetime.now(ZoneInfo("Europe/Kyiv"))
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def plan_record(state: dict[str, Any], request_path: str, model: str, generated_at: str, *,
                model_switches: list[str] | None = None, trace: list[dict[str, Any]] | None = None,
                obsidian_plan: str = "", plan_id: str = "") -> dict[str, Any]:
    """Everything the plan page needs, copied from the final workflow state."""
    record = {k: state.get(k) for k in STATE_KEYS}
    record.update(version=VERSION, id=plan_id or new_id(), request_path=request_path, model=model,
                  generated_at=generated_at, model_switches=list(model_switches or []), trace=list(trace or []),
                  obsidian_plan=obsidian_plan)
    return record


def save_plan(record: dict[str, Any], folder: Path | None = None) -> Path:
    folder = folder or plans_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path


def load_plan(plan_id: str, folder: Path | None = None) -> dict[str, Any] | None:
    """None for an unknown id, a malformed id (no ../ tricks) or a broken file."""
    if not ID_PATTERN.match(plan_id or ""):
        return None
    path = (folder or plans_dir()) / f"{plan_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("version") == VERSION else None


def list_plans(folder: Path | None = None) -> list[dict[str, Any]]:
    """Saved plans, newest first (the id starts with the date and time)."""
    folder = folder or plans_dir()
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        record = load_plan(path.stem, folder)
        if record:
            out.append(record)
    return out
