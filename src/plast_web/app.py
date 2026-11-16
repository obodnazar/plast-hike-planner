"""plast-web: chat with the agent and the plans it made, in the browser.

    uv run plast-web                 # http://127.0.0.1:8780, opens the browser
    uv run plast-web --port 8790 --no-browser

The chat (/) runs the same agent as `plast-agent --chat`: both MCP servers, the same LangGraph workflow; the
agent's own lines (the MCP calls) are printed in this terminal. It needs Obsidian open and a model; plast-web
starts the plast-trail server itself if it is not running. Saved plans (/plans, /plan/<id>) need none of that.
Listens on 127.0.0.1 only. Every answer for the page goes through view.present() or is written in Ukrainian by
conversation.py, so the page shows human text, not the agent's ids and codes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import contextlib
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
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from plast_agent.config import utf8_output
from plast_agent.picture import route_gpx, route_svg
from plast_agent.plan_store import list_plans, load_plan, plans_dir
from plast_web.conversation import Conversation
from plast_web.view import present, summary

STATIC = Path(__file__).parent / "static"
DEFAULT_PORT = 8780
# Scripts only from here and the pinned Leaflet on cdnjs; map tiles only from OpenTopoMap; nothing inline.
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; "
       "style-src 'self' https://cdnjs.cloudflare.com; img-src 'self' data: https://*.tile.opentopomap.org; "
       "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; "
       "frame-ancestors 'none'")


class LocalOnly:
    """Answers only requests addressed to this computer (Host 127.0.0.1 / localhost). A web page elsewhere
    could otherwise reach the server through DNS rebinding. Also sets the security headers on every answer."""

    def __init__(self, app: ASGIApp, allowed_hosts: set[str]) -> None:
        self.app, self.allowed = app, allowed_hosts
        self.origins = {f"http://{h}" for h in allowed_hosts}   # the page itself is served over plain http

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "").lower()
        origin = headers.get("origin")   # absent for same-origin GETs and non-browser clients: allowed

        async def with_headers(message):   # on every answer, refusals included
            if message["type"] == "http.response.start":
                extra = [(b"content-security-policy", CSP.encode()), (b"x-content-type-options", b"nosniff"),
                         (b"referrer-policy", b"no-referrer"), (b"x-frame-options", b"DENY")]
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        if host not in self.allowed or (origin is not None and origin.lower() not in self.origins):
            if scope["type"] == "websocket":   # a page from elsewhere must not drive the agent or write to Obsidian
                await send({"type": "websocket.close", "code": 1008})
            else:
                await PlainTextResponse("Доступ лише з цього комп'ютера.", status_code=403)(scope, receive, with_headers)
            return
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
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


NO_CHAT = {"ready": False, "problem": "Чат вимкнено в цьому запуску: доступні лише збережені плани.", "items": []}
MAX_CONVERSATIONS = 30


async def api_status(request: Request) -> Response:
    runtime = request.app.state.runtime
    return JSONResponse(runtime.status() if runtime else NO_CHAT)


def conversation_for(app, conversation_id: str | None) -> tuple[Conversation, bool]:
    convs: dict[str, Conversation] = app.state.conversations
    if conversation_id in convs:
        return convs[conversation_id], False
    conv = app.state.new_conversation()
    convs[conv.id] = conv
    while len(convs) > MAX_CONVERSATIONS:   # forget the oldest idle one
        old = next((k for k, c in convs.items() if not (c.task and not c.task.done())), None)
        if old is None:
            break
        convs.pop(old)
    return conv, True


async def ws_chat(ws: WebSocket) -> None:
    """One browser tab. hello{conversation_id, last_seq} -> the events it missed, then live events; the
    tab's messages go to the conversation. Closing the tab does not stop the conversation."""
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    conv = None
    try:
        hello = await ws.receive_json()
        conv, new = conversation_for(ws.app, hello.get("conversation_id"))
        last = 0 if new else int(hello.get("last_seq") or 0)
        conv.listeners.add(queue)   # first listen, then send the backlog: nothing falls in between
        runtime = ws.app.state.runtime
        await ws.send_json({"type": "hello", "conversation_id": conv.id,
                            "status": runtime.status() if runtime else NO_CHAT})
        sent = last
        for event in conv.since(last):
            await ws.send_json(event)
            sent = event["seq"]
        if new:
            conv.say("Привіт! Опишіть мандрівку своїми словами: хто йде, коли, куди, що хочете побачити. "
                     "Я підберу маршрут, перевірю безпеку й погоду та порахую запаси.")
            conv.set_state("describe")

        async def pump():
            while True:
                event = await queue.get()
                if event["seq"] > sent:
                    await ws.send_json(event)

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict):
                    if not (runtime and runtime.links) and msg.get("type") != "status":
                        conv.say(NO_CHAT["problem"] if not runtime else runtime.status()["problem"])
                        continue
                    conv.handle(msg)
        finally:
            sender.cancel()
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        if conv:
            conv.listeners.discard(queue)


def create_app(port: int = DEFAULT_PORT, extra_hosts: set[str] | None = None, runtime=None,
               new_conversation=None) -> Starlette:
    """runtime: plast_web.runtime.Runtime for the chat (main() passes one); without it only saved plans."""
    routes = [
        Route("/", page), Route("/plans", page), Route("/plan/{plan_id}", page),
        Route("/api/status", api_status), WebSocketRoute("/ws", ws_chat),
        Route("/api/plans", api_plans), Route("/api/plans/{plan_id}", api_plan),
        Route("/api/plans/{plan_id}/route.gpx", api_gpx), Route("/api/plans/{plan_id}/route.svg", api_svg),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    allowed = hosts_for(port) | (extra_hosts or set())

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if runtime:
            await runtime.start()
        try:
            yield
        finally:
            if runtime:
                await runtime.stop()

    app = Starlette(routes=routes, middleware=[Middleware(LocalOnly, allowed_hosts=allowed)], lifespan=lifespan)
    app.state.runtime = runtime
    app.state.conversations = {}
    app.state.new_conversation = new_conversation or (lambda: Conversation(runtime))
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Show the plans the agent saved, in the browser")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PLAST_WEB_PORT", DEFAULT_PORT)))
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    utf8_output()
    sys.stdout.reconfigure(line_buffering=True)   # the agent's MCP calls show up at once, even when piped
    url = f"http://127.0.0.1:{args.port}/"
    count = len(list_plans())
    print(f"Стежка: {url}  (планів збережено: {count}; папка {plans_dir()})")
    if not count:
        print("Планів ще немає: запустіть агента, наприклад  uv run plast-agent --chat")
    print("Зупинити: Ctrl+C")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    from plast_web.runtime import Runtime
    uvicorn.run(create_app(args.port, runtime=Runtime()), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
