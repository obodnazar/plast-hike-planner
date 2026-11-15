"""Offline check: the agent with the local model and no internet, on several request notes, then every output
checked automatically (status, picture, GPX, interactive-map block, links).

    uv run python scripts/offline_check.py "Запити/Тест — Говерла старші.md" "Запити/Тест — Горгани 3 дні.md"

Each run gets proxy variables that point to a closed port, so any call to the internet fails at once, while
Obsidian, plast-trail and Ollama on this computer are reached directly (NO_PROXY). Needs Obsidian open,
`uv run plast-trail` running and Ollama with plast-gemma. Writes the plans into the vault like a normal run.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT / "demo-vault"
OFFLINE = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9",
           "NO_PROXY": "localhost,127.0.0.1"}
GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}


def check_plan(request_path: str) -> list[str]:
    """Problems found in the outputs of one run (empty = all good)."""
    stem = Path(request_path).stem
    plan_file = VAULT / "Плани" / f"{stem} — план.md"
    if not plan_file.exists():
        return ["plan note was not written"]
    note = plan_file.read_text(encoding="utf-8")
    status = re.search(r"^status: (\w+)", note, re.M).group(1)
    problems = []
    if status == "failed":
        return [f"status failed (no plan to draw): {note.split('## Висновок')[1][:200].strip()}"]
    svg = VAULT / "Плани" / f"{stem} — маршрут.svg"
    gpx = VAULT / "Плани" / f"{stem} — маршрут.gpx"
    if f"![[Плани/{stem} — маршрут.svg]]" not in note:
        problems.append("the picture is not embedded")
    try:
        ET.fromstring(svg.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        problems.append(f"picture unreadable: {exc}")
    try:
        tracks = ET.fromstring(gpx.read_text(encoding="utf-8")).findall("g:trk", GPX_NS)
        days = len(re.findall(r"^\| \d+ \|", note.split("## Маршрут")[1].split("##")[0], re.M))
        if len(tracks) != days:
            problems.append(f"GPX has {len(tracks)} day tracks, the plan {days} days")
    except (OSError, ET.ParseError) as exc:
        problems.append(f"GPX unreadable: {exc}")
    block = re.search(r"```leaflet\n(.*?)```", note, re.S)
    if block is None:
        problems.append("no interactive map block (is OBSIDIAN_LEAFLET=1 in .env?)")
    else:
        try:   # exactly as the plugin reads it: [[links]] hidden, the rest is YAML
            params = yaml.safe_load(re.sub(r"\[\[[^\]]*\]\]", "LINK", block.group(1)))
            if params.get("osmLayer") is not False or not params.get("tileServer") or not params.get("marker"):
                problems.append(f"map block incomplete: {sorted(params)}")
        except yaml.YAMLError as exc:
            problems.append(f"map block is not valid YAML: {str(exc).splitlines()[0]}")
    if note.count("https://mapy.com/fnc/v1/route?") < 1:
        problems.append("no Mapy.com links")
    return problems


def main() -> int:
    requests = sys.argv[1:]
    if not requests:
        print(__doc__)
        return 2
    failures = 0
    for request in requests:
        started = time.monotonic()
        run = subprocess.run(["uv", "run", "plast-agent", request, "--local"], cwd=ROOT, capture_output=True,
                             text=True, env={**os.environ, **OFFLINE})
        summary = next((ln for ln in run.stdout.splitlines() if "Готово" in ln), run.stdout.strip()[-300:])
        summary = re.sub(r"\x1b\[[0-9;]*m", "", summary)
        problems = check_plan(request) if run.returncode in (0, 1) else [f"exit code {run.returncode}"]
        failures += bool(problems)
        print(f"{'✓' if not problems else '✗'} {request}  ({time.monotonic() - started:.0f} s)\n    {summary.strip()}")
        for p in problems:
            print(f"    ✗ {p}")
    subprocess.run(["ollama", "stop", "plast-gemma"], capture_output=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
