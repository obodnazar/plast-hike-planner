"""Record genuine Open-Meteo forecasts as replay fixtures (fixtures/open-meteo/<route>_<first-day>.json).

    uv run python scripts/record_weather.py                     # every route
    uv run python scripts/record_weather.py --region chornohora # one region
    uv run python scripts/record_weather.py --route r8613349

Each file holds the exact request and the untouched response body; replay feeds that
body through the same parser as a live call. One request per route, at least 1 s apart
(Open-Meteo free tier: 600 calls/min, 10 000/day). Re-run this shortly before the defence
so the 16-day window covers the demo dates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["WEATHER_MODE"] = "record"

from plast_trail import weather  # noqa: E402
from plast_trail.data import load_routes  # noqa: E402
from plast_trail.errors import DomainError  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region")
    ap.add_argument("--route")
    args = ap.parse_args()
    routes = [r for r in load_routes().values()
              if (not args.region or r["region"] == args.region) and (not args.route or r["id"] == args.route)]
    ok = 0
    for i, r in enumerate(routes, 1):
        try:
            body, fetched = weather.fetch_live(r["id"], weather.forecast_point(r))
            ok += 1
            print(f"[{i}/{len(routes)}] {r['id']} {body['daily']['time'][0]}..{body['daily']['time'][-1]}")
        except DomainError as exc:
            print(f"[{i}/{len(routes)}] {r['id']} FAILED: {exc.message}")
        except Exception as exc:  # one bad answer must not stop the whole recording
            print(f"[{i}/{len(routes)}] {r['id']} FAILED: {type(exc).__name__}: {exc}")
    print(f"recorded {ok}/{len(routes)} into {weather.FIXTURES}")


if __name__ == "__main__":
    main()
