"""Loading the routes dataset and the rules file, plus small geometry helpers.

Both files are read once and kept in memory. A missing or broken file raises
DATA_UNAVAILABLE, which is a tool error, never an empty result.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from plast_trail.errors import DATA_UNAVAILABLE, ROUTE_NOT_FOUND, DomainError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PLAST_TRAIL_DATA_DIR", PROJECT_ROOT / "data"))

ASCENT_HYSTERESIS_M = 10  # same as the dataset build: ignore DEM wiggles under 10 m


@lru_cache(maxsize=1)
def load_routes() -> dict[str, dict[str, Any]]:
    path = DATA_DIR / "routes.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DomainError(DATA_UNAVAILABLE, f"Routes dataset not found at {path}",
                          "Run: uv run python scripts/build_dataset.py") from exc
    except json.JSONDecodeError as exc:
        raise DomainError(DATA_UNAVAILABLE, f"Routes dataset is not valid JSON: {exc}",
                          "Rebuild it with scripts/build_dataset.py") from exc
    return {r["id"]: r for r in data["routes"]}


@lru_cache(maxsize=1)
def dataset_meta() -> dict[str, Any]:
    return json.loads((DATA_DIR / "routes.json").read_text(encoding="utf-8"))["meta"]


@lru_cache(maxsize=1)
def load_rules() -> dict[str, Any]:
    path = DATA_DIR / "rules.yaml"
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DomainError(DATA_UNAVAILABLE, f"Rules file not found at {path}") from exc
    except yaml.YAMLError as exc:
        raise DomainError(DATA_UNAVAILABLE, f"Rules file is not valid YAML: {exc}") from exc


def rules_status() -> str:
    rules = load_rules()
    if rules.get("status") == "approved":
        return f"approved by {rules.get('approved_by')} (version {rules.get('version')})"
    return f"DRAFT, not approved by a Plast leader (version {rules.get('version')})"


def section_rule(section: str, name: str) -> dict[str, Any]:
    """A rule entry {value, severity, source, approved} for a section."""
    return load_rules()["sections"][section].get(name) or {"value": None, "severity": "warning", "source": "absent"}


def common_rule(name: str) -> dict[str, Any]:
    return load_rules()["common"][name]


def get_route(route_id: str) -> dict[str, Any]:
    routes = load_routes()
    if route_id not in routes:
        raise DomainError(
            ROUTE_NOT_FOUND,
            f"No route with id '{route_id}'.",
            "Use an id returned by find_routes, e.g. 'r8613349'.",
        )
    return routes[route_id]


# ------------------------------------------------------------------ geometry
def profile_between(route: dict[str, Any], km_from: float, km_to: float) -> list[list[float]]:
    """Profile points [km, lat, lon, elevation] with km_from <= km <= km_to."""
    return [p for p in route["profile"] if km_from - 1e-6 <= p[0] <= km_to + 1e-6]


def ascent_descent(heights: list[float]) -> tuple[int, int]:
    """Total climb and drop, ignoring changes smaller than the hysteresis (DEM noise)."""
    if not heights:
        return 0, 0
    up = down = 0.0
    ref = heights[0]
    for h in heights[1:]:
        if h - ref >= ASCENT_HYSTERESIS_M:
            up += h - ref
            ref = h
        elif ref - h >= ASCENT_HYSTERESIS_M:
            down += ref - h
            ref = h
    return round(up), round(down)
