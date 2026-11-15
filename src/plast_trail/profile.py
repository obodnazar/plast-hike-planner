"""The shape of a planned hike, for drawing it: per-day elevation profile with coordinates, and landmarks.

The numbers repeat validate_hike_plan's (same profile, same hysteresis); this tool only adds the points
and the steep stretches, so a client can show how hard each day is.
"""

from __future__ import annotations

from typing import Any

from plast_trail.data import ascent_descent, dataset_meta, get_route, profile_between
from plast_trail.models import Landmark, ProfileDay, RouteProfileResult
from plast_trail.validation import resolve_night_spots

STEEP_GRADE = 0.15   # a display band for "very steep" (15 %), not a Plast rule


def _with_ends(route: dict[str, Any], a: float, b: float) -> list[list[float]]:
    """The day's profile plus the nearest point on each side, so consecutive days join without a gap."""
    prof = route["profile"]
    inside = profile_between(route, a, b)
    before = [p for p in prof if p[0] < a - 1e-6][-1:]
    after = [p for p in prof if p[0] > b + 1e-6][:1]
    return before + inside + after if inside else before + after


def route_profile(route_id: str, overnight_waypoint_ids: list[str]) -> RouteProfileResult:
    route = get_route(route_id)
    spots = resolve_night_spots(route, overnight_waypoint_ids)
    ends = [route["waypoints"][0], *spots, route["waypoints"][-1]]
    days = []
    for n, (a, b) in enumerate(zip(ends, ends[1:]), start=1):
        inside = profile_between(route, a["km"], b["km"])
        up, down = ascent_descent([p[3] for p in inside])
        pts = _with_ends(route, a["km"], b["km"])
        steep = sum(q[0] - p[0] for p, q in zip(pts, pts[1:])
                    if q[0] > p[0] and abs(q[3] - p[3]) / ((q[0] - p[0]) * 1000) > STEEP_GRADE)
        days.append(ProfileDay(day=n, start_km=a["km"], end_km=b["km"], km=round(b["km"] - a["km"], 1),
                               ascent_m=up, descent_m=down, max_elevation_m=round(max(p[3] for p in inside or pts)),
                               steep_km=round(steep, 2), points=[[p[0], p[1], p[2], p[3]] for p in pts]))
    nights = {s["id"]: i for i, s in enumerate(spots, start=1)}
    landmarks = [Landmark(id=w["id"], kind=w["kind"], name=w["name"] or "", km=w["km"], elevation_m=w["elevation_m"],
                          lat=w["lat"], lon=w["lon"], night=nights.get(w["id"]))
                 for w in route["waypoints"]
                 if w["kind"] in ("start", "finish") or w["id"] in nights
                 or (w["name"] and w["kind"] in ("peak", "lake", "hut", "shelter"))]
    meta = dataset_meta()
    return RouteProfileResult(route_id=route["id"], route_name=route["name"], length_km=route["length_km"],
                              steep_grade=STEEP_GRADE, days=days, landmarks=landmarks,
                              source=f"{meta['sources']['routes_and_pois']}; {meta['sources']['elevation']}")
