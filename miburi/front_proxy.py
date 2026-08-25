"""The single public port, served from a process that does no inference.

Why this exists
---------------
Hosting platforms expose one HTTP port, so the Viser viewer and the audio
dashboard have to share it. The obvious way -- proxy Viser from inside
`gest-server` -- puts ~3 MB/s of mesh relay in the *same asyncio event loop*
that must finish a Moshi step every 80 ms. Measured on an A100 with a real
browser attached:

    audio only        Moshi step  73.9 ms mean,  0.3% of steps over budget
    viewer attached   Moshi step  77.1 ms mean, 20.7% over budget (p90 82.1)

Only ~6 ms of slack exists per step, so relaying the heavy stream there eats
a third of it and pushes one step in five past real-time.

This module owns the public port instead, in its own process:

    browser ──► front_proxy ──► viser        (heavy: mesh frames, ~3 MB/s)
                     └───────► gest-server   (light: opus audio, ~20 kB/s)

The inference process now only ever relays audio, and the mesh stream never
touches it. Same URLs, same UI -- purely a change of which process holds the
socket.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import aiohttp
from aiohttp import web

from .client_utils import log
from .single_port import VISER_ROUTE, relay, viser_client_html

# Viser messages can be large; 0 = unlimited on both legs.
_MAX_MSG_SIZE = 0

# Headers that describe a single hop and must not be forwarded verbatim.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
    "sec-websocket-protocol", "host", "content-length",
}


def _clean(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _requested_subprotocols(request: web.Request) -> tuple[str, ...]:
    raw = request.headers.get("Sec-WebSocket-Protocol", "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


class AdaptiveMotionRelay:
    """Drop motion frames when the viewer's link cannot keep up.

    The client-visible symptom of a constrained link is Moshi's latency
    climbing past a second while audio packets go missing: the mesh stream
    and the audio share one connection, and the mesh is ~30x the bytes, so
    it starves the audio.

    We cannot read the browser's latency counter, but we can measure the
    thing that causes it. Writing to a socket the peer is not draining
    blocks once the buffer fills, so the time `send_bytes` takes IS the
    backpressure. If a frame takes longer to hand off than the 40 ms budget
    between frames, this link cannot sustain 25 fps, and we forward a
    proportionate fraction instead.

    Dropping here is free: viser's own buffer coalesces same-node updates
    (newest wins), so a frame this client was never going to render in time
    is one it would have discarded anyway. Audio is untouched -- it rides a
    different socket entirely.
    """

    LARGE = 5_000          # bytes; below this it is scene chatter, never dropped
    FRAME_BUDGET_MS = 40.0 # 25 fps
    ALPHA = 0.15           # EWMA weight

    def __init__(self, min_fps: float = 5.0, label: str = ""):
        self.min_keep = max(0.04, min_fps / 25.0)
        self.label = label
        self.ewma_ms = 0.0
        self.keep = 1.0
        self._credit = 0.0
        self.sent = self.dropped = 0
        self._logged_keep = 1.0

    def _should_send(self) -> bool:
        self._credit += self.keep
        if self._credit >= 1.0:
            self._credit -= 1.0
            return True
        return False

    def _observe(self, ms: float) -> None:
        self.ewma_ms = ms if self.ewma_ms == 0.0 else (
            self.ALPHA * ms + (1 - self.ALPHA) * self.ewma_ms)
        # Sustainable fraction of 25 fps for this link.
        target = 1.0 if self.ewma_ms <= self.FRAME_BUDGET_MS else \
            self.FRAME_BUDGET_MS / self.ewma_ms
        self.keep = max(self.min_keep, min(1.0, 0.8 * self.keep + 0.2 * target))
        if abs(self.keep - self._logged_keep) > 0.15:
            log("info", f"front_proxy: {self.label}link {'throttling' if self.keep < 0.95 else 'recovered'} "
                        f"-- send {self.ewma_ms:.0f} ms/frame, forwarding "
                        f"{self.keep * 100:.0f}% (~{self.keep * 25:.0f} fps), "
                        f"dropped {self.dropped}/{self.sent + self.dropped}")
            self._logged_keep = self.keep

    async def forward(self, data: bytes, send) -> None:
        if len(data) < self.LARGE:
            await send(data)
            return
        if not self._should_send():
            self.dropped += 1
            return
        t0 = time.perf_counter()
        await send(data)
        self.sent += 1
        self._observe((time.perf_counter() - t0) * 1e3)


async def _relay_motion(ws_viser, ws_browser, min_fps: float) -> None:
    """Viser -> browser with adaptive dropping; browser -> viser verbatim."""
    relay_state = AdaptiveMotionRelay(min_fps=min_fps)

    async def down() -> None:
        async for msg in ws_viser:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await relay_state.forward(msg.data, ws_browser.send_bytes)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                await ws_browser.send_str(msg.data)
            else:
                break

    async def up() -> None:
        async for msg in ws_browser:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await ws_viser.send_bytes(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                await ws_viser.send_str(msg.data)
            else:
                break

    tasks = {asyncio.create_task(down()), asyncio.create_task(up())}
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        if relay_state.dropped:
            log("info", f"front_proxy: viewer closed -- forwarded {relay_state.sent}, "
                        f"dropped {relay_state.dropped} motion frames to protect audio")
        for ws in (ws_viser, ws_browser):
            if not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass


async def _proxy_ws(request: web.Request, upstream: str) -> web.WebSocketResponse:
    protocols = _requested_subprotocols(request)
    ws_server = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE, protocols=protocols)
    await ws_server.prepare(request)
    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(upstream, max_msg_size=_MAX_MSG_SIZE,
                                      protocols=protocols, autoping=True) as ws_client:
            await relay(ws_client, ws_server)
    except aiohttp.ClientError as exc:
        log("error", f"front_proxy: upstream ws {upstream} failed ({exc!r})")
    finally:
        await session.close()
        if not ws_server.closed:
            await ws_server.close()
    return ws_server


async def _proxy_http(request: web.Request, upstream: str) -> web.StreamResponse:
    body = await request.read()
    session = aiohttp.ClientSession(auto_decompress=False)
    try:
        async with session.request(
            request.method, upstream, headers=_clean(request.headers),
            params=request.rel_url.query, data=body or None,
            allow_redirects=False,
        ) as up:
            payload = await up.read()
            return web.Response(status=up.status, body=payload,
                                headers=_clean(up.headers))
    except aiohttp.ClientError as exc:
        # gest-server binds after its models load, so early hits are expected.
        log("warning", f"front_proxy: upstream {upstream} not ready ({exc!r})")
        return web.Response(status=503, text="MIBURI backend is still starting.\n")
    finally:
        await session.close()


def build_app(gest_host: str, gest_port: int, viser_host: str, viser_port: int,
              motion_min_fps: float = 5.0, adaptive: bool = True) -> web.Application:
    gest_base = f"http://{gest_host}:{gest_port}"
    viser_ws = f"http://{viser_host}:{viser_port}/"

    async def handle_viser(request: web.Request) -> web.StreamResponse:
        # One route serves both: a plain GET returns Viser's self-contained
        # client bundle, a WS upgrade is relayed to the Viser server. Read
        # from the installed package so the client can never version-skew
        # from the server it talks to.
        protocols = _requested_subprotocols(request)
        probe = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE, protocols=protocols)
        if not probe.can_prepare(request).ok:
            return web.Response(body=viser_client_html(), content_type="text/html",
                                headers={"Cache-Control": "no-store"})
        if not adaptive:
            return await _proxy_ws(request, viser_ws)
        ws_browser = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE, protocols=protocols)
        await ws_browser.prepare(request)
        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(viser_ws, max_msg_size=_MAX_MSG_SIZE,
                                          protocols=protocols, autoping=True) as ws_viser:
                await _relay_motion(ws_viser, ws_browser, motion_min_fps)
        except aiohttp.ClientError as exc:
            log("error", f"front_proxy: viser upstream failed ({exc!r})")
        finally:
            await session.close()
            if not ws_browser.closed:
                await ws_browser.close()
        return ws_browser

    async def handle_rest(request: web.Request) -> web.StreamResponse:
        target = f"{gest_base}{request.rel_url.path}"
        probe = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE,
                                      protocols=_requested_subprotocols(request))
        if probe.can_prepare(request).ok:
            return await _proxy_ws(request, target)
        return await _proxy_http(request, target)

    app = web.Application(client_max_size=0)
    app.router.add_route("*", VISER_ROUTE, handle_viser)
    app.router.add_route("*", "/{tail:.*}", handle_rest)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True, help="the public port")
    p.add_argument("--gest-host", default="127.0.0.1")
    p.add_argument("--gest-port", type=int, required=True)
    p.add_argument("--viser-host", default="127.0.0.1")
    p.add_argument("--viser-port", type=int, required=True)
    p.add_argument("--motion-min-fps", type=float, default=5.0,
                   help="Floor for adaptive motion throttling; the viewer never "
                        "drops below this even on a badly constrained link.")
    p.add_argument("--no-adaptive-motion", action="store_false", dest="adaptive_motion",
                   default=True,
                   help="Forward every motion frame regardless of backpressure.")
    args = p.parse_args()

    log("info", f"front_proxy: adaptive motion "
                f"{'on' if args.adaptive_motion else 'off'} "
                f"(floor {args.motion_min_fps:.0f} fps)")
    log("info", f"front_proxy: public {args.bind}:{args.port} -> "
                f"viser {args.viser_host}:{args.viser_port} (heavy), "
                f"gest {args.gest_host}:{args.gest_port} (light)")
    web.run_app(build_app(args.gest_host, args.gest_port,
                          args.viser_host, args.viser_port,
                          motion_min_fps=args.motion_min_fps,
                          adaptive=args.adaptive_motion),
                host=args.bind, port=args.port, print=None)


if __name__ == "__main__":
    main()
