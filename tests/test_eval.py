"""Tests of the evaluation itself (labels and scoring), without any model.

Run:  uv run pytest -q tests/test_eval.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from plast_agent.evaluation import (CORE, FIELDS, Case, load_cases, outcome, place_counts, same_place,
                                    score_case, summarise, wilson)
from plast_agent.llm import ParsedRequest
from plast_agent.workflow import understand

CASES = Path(__file__).resolve().parents[1] / "eval" / "cases"


@pytest.fixture(scope="module")
def sets():
    return {name: load_cases(CASES / f"{name}.yaml") for name in ("dev", "test")}


def test_case_files_are_well_formed(sets):
    ids, texts = [], []
    for today, cases in sets.values():
        assert today.isoformat() == "2026-11-15"
        assert len(cases) >= 30
        for c in cases:
            assert c.text.strip()
            assert set(c.expect) == set(FIELDS)
            ids.append(c.id)
            texts.append(c.text.strip())
    assert len(ids) == len(set(ids)), "case ids must be unique across dev and test"
    assert len(texts) == len(set(texts)), "the same text must not be in both sets"


def test_labels_fit_the_schema(sets):
    """Every accepted value must be something ParsedRequest + sanitise could produce."""
    allowed = {
        "section": {"novatstvo", "yunatstvo", "senior", None},
        "region": {"chornohora", "svydovets", "gorgany", "marmarosy", "yaremche-vorokhta", None},
    }
    features = {"lake", "peak", "water", "hut", "shelter", "campsite"}
    for _, cases in sets.values():
        for c in cases:
            for field, values in allowed.items():
                assert set(c.expect[field]) <= values, (c.id, field)
            for v in c.expect["must_include"]:
                assert set(v) <= features, c.id
            for v in c.expect["date_flexibility_days"]:
                assert 0 <= v <= 3, c.id   # sanitise caps the shift at 3 days
            for v in c.expect["places"]:
                assert len(v) <= 3, c.id   # sanitise keeps at most 3


def test_place_matching_follows_the_server_rule():
    assert same_place("Говерлу", "Говерла")
    assert same_place("Піп Іван Чорногірський", "Піп Іван")
    assert same_place("оз. Несамовите", "Несамовите")
    assert same_place("Бребенеско", "Бребенескул")       # the stem survives the typo
    assert not same_place("піпівана", "Піп Іван")         # glued words: the server would not find it
    assert not same_place("Петрос", "Петраш")
    assert place_counts(["Говерла", "Чорногора"], ["Говерла", "Петрос"]) == (1, 1, 1)


def test_outcomes():
    assert outcome("participants", 10, [10]) == "correct"
    assert outcome("participants", None, [10]) == "missing"
    assert outcome("participants", 13, [None]) == "invented"
    assert outcome("participants", 13, [10]) == "wrong"
    assert outcome("region", None, ["chornohora", None]) == "correct"
    assert outcome("start_time", "08:00", ["07:30"]) == "missing"       # 08:00 is the default
    assert outcome("must_include", ["peak", "lake"], [["lake", "peak"]]) == "correct"
    assert outcome("max_km_per_day", 12.0, [12]) == "correct"


def test_summary_counts_and_interval():
    case = Case(id="x", source="handwritten", style=[], text="t",
                expect={f: [None] for f in FIELDS} | {"date_flexibility_days": [0], "start_time": ["08:00"],
                                                     "must_include": [[]], "places": [[]], "participants": [10]})
    good = score_case(case, {"participants": 10, "date_flexibility_days": 0, "start_time": "08:00",
                             "must_include": [], "places": []})
    bad = score_case(case, None, error="timeout")
    s = summarise([good | {"seconds": 2.0}, bad])
    assert s["n"] == 2 and s["errors"] == 1
    assert s["core_all_correct"] == 1 and s["all_fields_correct"] == 1
    assert s["per_field"]["participants"] == {"correct": 1, "missing": 0, "invented": 0, "wrong": 0, "error": 1}
    lo, hi = wilson(25, 50)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64   # about +-14 points at n=50, p=0.5


class ScriptedLLM:
    def __init__(self, answer: dict) -> None:
        self.answer = answer

    async def structured(self, schema, system: str, user: str):
        assert "2026-11-15" in system   # the fixed evaluation day reaches the prompt
        return schema(**self.answer)


def test_demo_note_scores_perfectly_when_the_model_is_right(sets):
    """The whole path (understand -> sanitise -> score) on the demo note with an ideal answer."""
    today, cases = sets["dev"]
    case = next(c for c in cases if c.id == "d05")
    ideal = {"section": "yunatstvo", "participants": 10, "adults": 2, "days": 3, "start_date": "2026-11-21",
             "date_flexibility_days": 2, "region": "chornohora", "must_include": ["lake"], "start_time": "8:00",
             "max_km_per_day": 12, "max_overnight_elevation_m": 1600, "lightest_body_weight_kg": 40,
             "participant_base_gear_kg": 5, "adult_base_gear_kg": 7}
    ParsedRequest(**ideal)   # the ideal answer is valid for the real schema
    parsed, _, missing = asyncio.run(understand(ScriptedLLM(ideal), case.text, today=today))
    record = score_case(case, parsed)
    assert missing == []
    assert all(o == "correct" for o in record["outcomes"].values()), record
    assert set(CORE) <= set(FIELDS)
