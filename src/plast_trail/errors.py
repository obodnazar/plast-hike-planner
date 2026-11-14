"""Errors the tools return to the caller.

Every failure becomes an MCP tool error (`isError: true`) whose text is a JSON object:

    {"error": {"code": "ROUTE_NOT_FOUND", "message": "...", "hint": "..."}}

so a caller can tell a failure apart from a successful empty result
(e.g. find_routes returning `"routes": []`), and a model can read the hint.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp.exceptions import ToolError


class DomainError(Exception):
    """A failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def to_tool_error(self) -> ToolError:
        payload = {"error": {"code": self.code, "message": self.message, "hint": self.hint}}
        return ToolError(json.dumps(payload, ensure_ascii=False))


# Error codes used across the server (documented in docs/tool-contracts.md).
ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"
WAYPOINT_NOT_FOUND = "WAYPOINT_NOT_FOUND"
WAYPOINT_ORDER = "WAYPOINT_ORDER"
INVALID_INPUT = "INVALID_INPUT"
DATE_OUT_OF_RANGE = "DATE_OUT_OF_RANGE"
WEATHER_UNAVAILABLE = "WEATHER_UNAVAILABLE"
WEATHER_FIXTURE_MISSING = "WEATHER_FIXTURE_MISSING"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
INTERNAL_ERROR = "INTERNAL_ERROR"
