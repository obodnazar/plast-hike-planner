"""How well does the agent understand a free-text request? Labelled cases -> per-field scores.

The scoring is plain code (no model), so tests check it offline; scripts/eval_parsing.py runs a
real model over eval/cases/*.yaml and stores the results in eval/results/.

Every field of every case gets one outcome:
  correct  - matches the label (or one of its accepted alternatives);
  missing  - the label has a value, the answer is empty. For the core fields the dialogue then asks,
             so this is the SAFE kind of error;
  invented - the text does not say it, the answer has a value (a guess);
  wrong    - both have a value and they differ.
"invented" and "wrong" are silent errors: nobody is asked, the plan is built on a wrong fact.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from plast_trail.routes import place_stems

FIELDS = ["section", "participants", "adults", "days", "start_date", "date_flexibility_days", "region",
          "must_include", "places", "start_time", "max_km_per_day", "max_ascent_per_day_m",
          "max_walking_hours_per_day", "max_overnight_elevation_m", "lightest_body_weight_kg",
          "participant_base_gear_kg", "adult_base_gear_kg"]
CORE = ["section", "participants", "adults", "days", "start_date"]   # the dialogue asks for these
# What "not stated" looks like after sanitise_request; every other field is None.
EMPTY: dict[str, Any] = {"date_flexibility_days": 0, "start_time": "08:00", "must_include": [], "places": []}
OUTCOMES = ["correct", "missing", "invented", "wrong", "error"]

FIELD_UA = {"section": "група", "participants": "учасників", "adults": "дорослих", "days": "днів",
            "start_date": "дата старту", "date_flexibility_days": "зсув дати", "region": "район",
            "must_include": "побажання (озеро…)", "places": "назви місць", "start_time": "час виходу",
            "max_km_per_day": "ліміт км/день", "max_ascent_per_day_m": "ліміт набору",
            "max_walking_hours_per_day": "ліміт годин", "max_overnight_elevation_m": "висота ночівлі",
            "lightest_body_weight_kg": "вага найлегшого", "participant_base_gear_kg": "рюкзак учасника",
            "adult_base_gear_kg": "рюкзак дорослого"}


@dataclass
class Case:
    id: str
    source: str            # real | demo | handwritten
    style: list[str]
    text: str
    expect: dict[str, list[Any]]   # field -> accepted values (one or more)


# ------------------------------------------------------------------ loading
def _plain(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def load_cases(path: Path) -> tuple[date, list[Case]]:
    """Read one case file. A label for an unknown field is an error: a typo must not pass silently."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    today = data["today"] if isinstance(data["today"], date) else date.fromisoformat(data["today"])
    cases = []
    for raw in data["cases"]:
        expect_raw = raw.get("expect") or {}
        unknown = set(expect_raw) - set(FIELDS)
        if unknown:
            raise ValueError(f"{path.name} {raw['id']}: unknown fields {sorted(unknown)}")
        expect: dict[str, list[Any]] = {}
        for field in FIELDS:
            default = EMPTY.get(field)
            if field == "adults" and expect_raw.get("section") == "senior":
                default = 0   # adults travelling on their own have no accompanying adults
            value = expect_raw.get(field, default)
            options = value["any_of"] if isinstance(value, dict) and "any_of" in value else [value]
            expect[field] = [_plain(o) for o in options]
        cases.append(Case(id=str(raw["id"]), source=raw.get("source", "handwritten"),
                          style=list(raw.get("style") or []), text=raw["text"], expect=expect))
    return today, cases


# ------------------------------------------------------------------ comparing
def is_empty(field: str, value: Any) -> bool:
    if value is None or value == []:
        return True
    return field in EMPTY and value == EMPTY[field]


def _norm(text: str) -> str:
    return text.lower().replace("ґ", "г")


def same_place(got: str, expected: str) -> bool:
    """Would the server find the expected place by the name the agent understood?
    The server's rule: every stem of the given name occurs in the place's name ('Говерлу' ~ 'Говерла').
    A longer official name also counts if it starts its words with the expected stems
    ('Піп Іван Чорногірський' ~ 'Піп Іван'); a glued typo does not ('піпівана' !~ 'Піп Іван')."""
    sg, se = place_stems(got), place_stems(expected)
    if not sg or not se:
        return False
    words = _norm(got).replace("-", " ").split()
    return all(s in _norm(expected) for s in sg) or all(any(w.startswith(s) for w in words) for s in se)


def place_counts(got: list[str], expected: list[str]) -> tuple[int, int, int]:
    """(found, extra, not found) for one case."""
    found = sum(1 for e in expected if any(same_place(g, e) for g in got))
    extra = sum(1 for g in got if not any(same_place(g, e) for e in expected))
    return found, extra, len(expected) - found


def equal(field: str, got: Any, expected: Any) -> bool:
    if field == "places":
        found, extra, missed = place_counts(got or [], expected or [])
        return extra == 0 and missed == 0
    if field == "must_include":
        return set(got or []) == set(expected or [])
    if is_empty(field, got) and is_empty(field, expected):
        return True
    numbers = (int, float)
    if isinstance(got, numbers) and isinstance(expected, numbers) and not isinstance(got, bool):
        return abs(got - expected) < 1e-6
    return got == expected


def outcome(field: str, got: Any, options: list[Any]) -> str:
    if any(equal(field, got, o) for o in options):
        return "correct"
    if is_empty(field, got):
        return "missing"
    if all(is_empty(field, o) for o in options):
        return "invented"
    return "wrong"


def score_case(case: Case, got: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    """One case's record: what the agent understood and the outcome per field."""
    if error is not None or got is None:
        return {"id": case.id, "error": error or "no answer", "got": {},
                "outcomes": {f: "error" for f in FIELDS}, "places": [0, 0, len(case.expect["places"][0])]}
    got = {f: _plain(got.get(f)) for f in FIELDS}
    outcomes = {f: outcome(f, got[f], case.expect[f]) for f in FIELDS}
    # places: count against the closest accepted alternative
    counts = max((place_counts(got["places"] or [], o or []) for o in case.expect["places"]),
                 key=lambda c: c[0] - c[1] - c[2])
    return {"id": case.id, "error": None, "got": got, "outcomes": outcomes, "places": list(counts)}


# ------------------------------------------------------------------ summary
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % interval for a share k/n; honest for small n (unlike +-1.96*sqrt(p(1-p)/n))."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    per_field = {f: {o: 0 for o in OUTCOMES} for f in FIELDS}
    for r in results:
        for f in FIELDS:
            per_field[f][r["outcomes"][f]] += 1
    core_all = sum(all(r["outcomes"][f] == "correct" for f in CORE) for r in results)
    all_fields = sum(all(r["outcomes"][f] == "correct" for f in FIELDS) for r in results)
    core_safe = sum(r["error"] is None and not any(r["outcomes"][f] in ("invented", "wrong") for f in CORE)
                    for r in results)
    silent = sum(o in ("invented", "wrong") for r in results for o in r["outcomes"].values())
    found, extra, missed = (sum(r["places"][i] for r in results) for i in range(3))
    seconds = [r["seconds"] for r in results if r["error"] is None and r.get("seconds") is not None]
    return {
        "n": n,
        "errors": sum(r["error"] is not None for r in results),
        "core_all_correct": core_all, "core_all_correct_ci": wilson(core_all, n),
        "all_fields_correct": all_fields, "all_fields_correct_ci": wilson(all_fields, n),
        "core_safe": core_safe,
        "silent_errors_per_request": round(silent / n, 2) if n else 0.0,
        "field_accuracy": {f: round(per_field[f]["correct"] / n, 3) if n else 0.0 for f in FIELDS},
        "per_field": per_field,
        "places": {"found": found, "extra": extra, "not_found": missed},
        "seconds_median": round(statistics.median(seconds), 1) if seconds else None,
        "seconds_p90": round(sorted(seconds)[max(0, math.ceil(0.9 * len(seconds)) - 1)], 1) if seconds else None,
        "model_calls": sum(r.get("calls", 0) for r in results),
    }


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f} %" if n else "—"


def render_summary(s: dict[str, Any], title: str) -> str:
    n = s["n"]
    lo, hi = s["core_all_correct_ci"]
    lines = [f"### {title}", "",
             f"- Запитів: {n}; збоїв моделі: {s['errors']}; викликів моделі: {s['model_calls']}; "
             f"час на запит: медіана {s['seconds_median']} с, 90-й перцентиль {s['seconds_p90']} с",
             f"- **Основні 5 полів правильні повністю:** {pct(s['core_all_correct'], n)} "
             f"(95 % інтервал {100 * lo:.0f}–{100 * hi:.0f} %)",
             f"- **Без тихих помилок в основних полях** (пропуск дозволено, бо агент перепитає): "
             f"{pct(s['core_safe'], n)}",
             f"- Усі 17 полів правильні: {pct(s['all_fields_correct'], n)}",
             f"- Тихих помилок (вигадано + помилково) на запит: {s['silent_errors_per_request']}",
             f"- Назви місць: знайдено {s['places']['found']}, зайвих {s['places']['extra']}, "
             f"не знайдено {s['places']['not_found']}", "",
             "| Поле | Правильно | Пропущено | Вигадано | Помилково | Збій |",
             "|---|---|---|---|---|---|"]
    for f in FIELDS:
        c = s["per_field"][f]
        mark = " *" if f in CORE else ""
        lines.append(f"| {FIELD_UA[f]}{mark} | {pct(c['correct'], n)} | {c['missing']} | {c['invented']} | "
                     f"{c['wrong']} | {c['error']} |")
    lines.append("")
    lines.append("\\* основні поля: якщо їх немає, агент питає людину.")
    return "\n".join(lines)


def describe_errors(case: Case, record: dict[str, Any]) -> str:
    """One line per wrong field, for the dev set only."""
    if record["error"]:
        return f"збій: {record['error'][:160]}"
    bad = []
    for f in FIELDS:
        o = record["outcomes"][f]
        if o != "correct":
            want = case.expect[f][0] if len(case.expect[f]) == 1 else {"any_of": case.expect[f]}
            bad.append(f"{f}={record['got'][f]!r} ({o}; треба {want!r})")
    return "; ".join(bad) or "усе правильно"


def render_report(runs: list[dict[str, Any]]) -> str:
    """eval/RESULTS.md: one row per run, then field accuracy side by side."""
    runs = sorted(runs, key=lambda r: (r["set"], r["started"]))
    out = ["# Результати перевірки розуміння запитів", "",
           "Згенеровано `scripts/eval_parsing.py --report`. Що означають колонки: `eval/README.md`.",
           "«Перерахунок» — ті самі збережені відповіді моделі, оцінені новим кодом (без нових запитів до моделі).", "",
           "| Набір | Етап | Модель | Запитів | Основні 5 полів повністю (95 % інтервал) | Без тихих помилок в основних | "
           "Усі поля | Тихих помилок / запит | Медіана, с | Дата |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s = r["summary"]
        n = s["n"]
        lo, hi = s["core_all_correct_ci"]
        done = f"{n}/{r['cases_total']}" + ("" if n == r["cases_total"] else " (неповний)")
        label = r["label"] + (" (перерахунок)" if r.get("replay_of") else "")
        out.append(f"| {r['set']} | {label} | {r['provider']}/{r['model']} | {done} | "
                   f"{pct(s['core_all_correct'], n)} ({100 * lo:.0f}–{100 * hi:.0f} %) | {pct(s['core_safe'], n)} | "
                   f"{pct(s['all_fields_correct'], n)} | {s['silent_errors_per_request']} | {s['seconds_median']} | "
                   f"{r['started'][:10]} |")
    for set_name in sorted({r["set"] for r in runs}):
        group = [r for r in runs if r["set"] == set_name]
        out += ["", f"## Точність по полях: {set_name}", "",
                "| Поле | " + " | ".join(f"{r['label']} · {r['model']}" for r in group) + " |",
                "|---|" + "---|" * len(group)]
        for f in FIELDS:
            out.append(f"| {FIELD_UA[f]}{' *' if f in CORE else ''} | " +
                       " | ".join(f"{100 * r['summary']['field_accuracy'][f]:.0f} %" for r in group) + " |")
    out += ["", "\\* основні поля: якщо їх немає, агент питає людину.", ""]
    return "\n".join(out)
