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


def build_app(gest_host: str, gest_port: int, viser_host: str, viser_port: int) -> web.Application:
    gest_base = f"http://{gest_host}:{gest_port}"
    viser_ws = f"http://{viser_host}:{viser_port}/"

    async def handle_viser(request: web.Request) -> web.StreamResponse:
        # One route serves both: a plain GET returns Viser's self-contained
        # client bundle, a WS upgrade is relayed to the Viser server. Read
        # from the installed package so the client can never version-skew
        # from the server it talks to.
        probe = web.WebSocketResponse(max_msg_size=_MAX_MSG_SIZE,
                                      protocols=_requested_subprotocols(request))
        if probe.can_prepare(request).ok:
            return await _proxy_ws(request, viser_ws)
        return web.Response(body=viser_client_html(), content_type="text/html",
                            headers={"Cache-Control": "no-store"})

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
    args = p.parse_args()

    log("info", f"front_proxy: public {args.bind}:{args.port} -> "
                f"viser {args.viser_host}:{args.viser_port} (heavy), "
                f"gest {args.gest_host}:{args.gest_port} (light)")
    web.run_app(build_app(args.gest_host, args.gest_port,
                          args.viser_host, args.viser_port),
                host=args.bind, port=args.port, print=None)


if __name__ == "__main__":
    main()
