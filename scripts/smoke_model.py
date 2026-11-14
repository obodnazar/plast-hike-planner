"""Smoke test: does the configured OpenRouter model answer, and can it call a tool?

Run:  uv run python scripts/smoke_model.py

Sends ONE request (free models allow 50 a day) with one fake tool and checks
that the model asks for that tool with sensible arguments. Nothing is executed.
"""

from __future__ import annotations

import json
import sys
import time

from openai import APIStatusError, OpenAI

from plast_agent.config import ConfigError, load_settings

TOOL = {
    "type": "function",
    "function": {
        "name": "validate_hike_plan",
        "description": "Check a Plast hike plan against safety norms for the group's age section.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": ["novatstvo", "yunatstvo", "senior"]},
                "participants": {"type": "integer", "minimum": 1},
                "adults": {"type": "integer", "minimum": 0},
                "km_per_day": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["section", "participants", "adults", "km_per_day"],
        },
    },
}

PROMPT = (
    "Перевір план: юнацтво, 10 учасників і 2 дорослих, два дні по 14 і 18 км. "
    "Використай інструмент."
)


def main() -> int:
    try:
        s = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2

    client = OpenAI(base_url=s.openrouter_base_url, api_key=s.openrouter_api_key)
    started = time.monotonic()
    try:
        reply = client.chat.completions.create(
            model=s.openrouter_model,
            messages=[{"role": "user", "content": PROMPT}],
            tools=[TOOL],
            temperature=0,
        )
    except APIStatusError as exc:
        hints = {401: "OpenRouter rejected the key. Check OPENROUTER_API_KEY in .env.",
                 402: "Not enough credit for this model.",
                 429: "Rate limit hit (free models: 20/min, 50/day). Try later."}
        print(f"HTTP {exc.status_code}: {hints.get(exc.status_code, exc.message)}")
        return 1
    seconds = time.monotonic() - started

    msg = reply.choices[0].message
    print(f"model: {reply.model}  |  {seconds:.1f}s")
    if not msg.tool_calls:
        print("NO tool call. Model answered with text:\n", (msg.content or "")[:500])
        return 1
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments or "{}")
    print(f"tool call: {call.function.name}({json.dumps(args, ensure_ascii=False)})")
    expected = {"section": "yunatstvo", "participants": 10, "adults": 2, "km_per_day": [14, 18]}
    ok = call.function.name == "validate_hike_plan" and all(
        args.get(k) == v for k, v in expected.items()
    )
    print("arguments match expected:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
