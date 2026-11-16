"""Shared test setup.

The tests must give the same result on any day: the request dates (21–23 Nov 2026) and the expected weather
come from the recording of 14 Nov 2026. So the whole test run has a fixed «today» and sees only that
recording, even after the leader records fresh forecasts before the defence. (The demo note itself may be
moved too: the tests read a frozen copy, tests/data/sokil-request.md.)
"""

import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

TODAY = datetime(2026, 11, 15, 10, 0, tzinfo=ZoneInfo("Europe/Kyiv"))   # the evaluation's day too
RECORDING = "2026-11-14"


@pytest.fixture(autouse=True)
def plans_in_tmp(tmp_path, monkeypatch):
    """Runs of the agent in tests save their plans to a temporary folder, never to out/plans."""
    monkeypatch.setenv("PLAST_PLANS_DIR", str(tmp_path / "plans"))


@pytest.fixture(scope="session", autouse=True)
def fixed_day_and_recording(tmp_path_factory):
    """For the whole session, module fixtures included:
    * the agent's clock (today, «завтра», a past start date, the names of saved requests) reads 15 Nov 2026;
    * replay sees a copy of the forecasts recorded on 14 Nov only (a newer recording would win)."""
    import plast_agent.chat as chat
    import plast_agent.workflow as workflow
    import plast_web.conversation as conversation
    from plast_trail import weather

    folder = tmp_path_factory.mktemp("open-meteo")
    for f in weather.FIXTURES.glob(f"*_{RECORDING}.json"):
        shutil.copy(f, folder / f.name)
    with pytest.MonkeyPatch.context() as mp:
        for module in (workflow, chat, conversation):
            mp.setattr(module, "kyiv_now", lambda: TODAY)
        mp.setattr(weather, "FIXTURES", folder)
        mp.setenv("PLAST_TRAIL_FIXTURES", str(folder))   # for a plast-trail started by a test
        yield
