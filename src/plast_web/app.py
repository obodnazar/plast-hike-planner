"""plast-web: the plans the agent saved, as pages in the browser.

    uv run plast-web                 # http://127.0.0.1:8780, opens the browser
    uv run plast-web --port 8790 --no-browser

Reads out/plans/*.json written by `plast-agent` after each run; needs no model, no Obsidian and no MCP
server. Listens on 127.0.0.1 only. Every answer for the page goes through view.present(), so the page shows
human text, not the agent's ids and codes.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from plast_agent.config import utf8_output
from plast_agent.picture import route_gpx, route_svg
from plast_agent.plan_store import list_plans, load_plan, plans_dir
from plast_web.view import present, summary

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8780
# Scripts only from here and the pinned Leaflet on cdnjs; map tiles only from OpenTopoMap; nothing inline.
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; "
       "style-src 'self' https://cdnjs.cloudflare.com; img-src 'self' data: https://*.tile.opentopomap.org; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


class LocalOnly:
    """Answers only requests addressed to this computer (Host 127.0.0.1 / localhost). A web page elsewhere
    could otherwise reach the server through DNS rebinding. Also sets the security headers on every answer."""

    def __init__(self, app: ASGIApp, allowed_hosts: set[str]) -> None:
        self.app, self.allowed = app, allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "").lower()
        origin = headers.get("origin")
        if host not in self.allowed or (origin and origin.split("://", 1)[-1].lower() not in self.allowed):
            await PlainTextResponse("Доступ лише з цього комп'ютера.", status_code=403)(scope, receive, send)
            return

        async def with_headers(message):
            if message["type"] == "http.response.start":
                extra = [(b"content-security-policy", CSP.encode()), (b"x-content-type-options", b"nosniff"),
                         (b"referrer-policy", b"no-referrer"), (b"x-frame-options", b"DENY")]
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)
        await self.app(scope, receive, with_headers)


def hosts_for(port: int) -> set[str]:
    return {f"127.0.0.1:{port}", f"localhost:{port}"}


async def page(request: Request) -> Response:
    return FileResponse(STATIC / "index.html", headers={"cache-control": "no-cache"})


async def api_plans(request: Request) -> Response:
    return JSONResponse({"plans": [summary(r) for r in list_plans()], "folder": str(plans_dir())})


def record_or_404(request: Request):
    record = load_plan(request.path_params["plan_id"])
    if record is None:
        return None, JSONResponse({"error": "Такого плану немає."}, status_code=404)
    return record, None


async def api_plan(request: Request) -> Response:
    record, missing = record_or_404(request)
    return missing or JSONResponse(present(record))


async def api_gpx(request: Request) -> Response:
    record, missing = record_or_404(request)
    if missing:
        return missing
    if not (record.get("route_profile") or {}).get("days"):
        return JSONResponse({"error": "У цьому плані немає маршруту."}, status_code=404)
    name = (record.get("request_path") or "plan").split("/")[-1].removesuffix(".md")
    gpx = route_gpx(record["route_profile"], (record.get("request") or {}).get("place_points"), name)
    return Response(gpx, media_type="application/gpx+xml", headers={
        "content-disposition": f"attachment; filename=\"plast-{record['id']}.gpx\""})


async def api_svg(request: Request) -> Response:
    """The route picture, for when the map tiles do not load (no internet)."""
    record, missing = record_or_404(request)
    if missing:
        return missing
    if not (record.get("route_profile") or {}).get("days"):
        return JSONResponse({"error": "У цьому плані немає маршруту."}, status_code=404)
    svg = route_svg(record["route_profile"], record.get("validation") or {},
                    (record.get("proposal") or {}).get("start_date"),
                    (record.get("request") or {}).get("place_points"))
    return Response(svg, media_type="image/svg+xml")


def create_app(port: int = DEFAULT_PORT, extra_hosts: set[str] | None = None) -> Starlette:
    routes = [
        Route("/", page), Route("/plan/{plan_id}", page),
        Route("/api/plans", api_plans), Route("/api/plans/{plan_id}", api_plan),
        Route("/api/plans/{plan_id}/route.gpx", api_gpx), Route("/api/plans/{plan_id}/route.svg", api_svg),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    allowed = hosts_for(port) | (extra_hosts or set())
    return Starlette(routes=routes, middleware=[Middleware(LocalOnly, allowed_hosts=allowed)])


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Show the plans the agent saved, in the browser")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PLAST_WEB_PORT", DEFAULT_PORT)))
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    utf8_output()
    url = f"http://127.0.0.1:{args.port}/"
    count = len(list_plans())
    print(f"Стежка: {url}  (планів збережено: {count}; папка {plans_dir()})")
    if not count:
        print("Планів ще немає: запустіть агента, наприклад  uv run plast-agent --chat")
    print("Зупинити: Ctrl+C")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(args.port), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
