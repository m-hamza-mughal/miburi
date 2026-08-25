"""Serve the Viser motion viewer through the main aiohttp port.

Hugging Face Spaces (and most PaaS) expose exactly one HTTP port. The demo
normally needs two: the aiohttp dashboard (8998) and Viser's own combined
HTTP+WebSocket server (8083). This module collapses the second onto the first.

Two facts make it cheap:

- Viser ships its client as a **single self-contained `index.html`**
  (``viser/client/build/index.html``, ~2.8 MB, zstd-inlined JS/CSS). There are
  no ``/assets/*`` sub-requests to proxy or rewrite.
- The client derives its WebSocket URL from ``window.location`` (page URL,
  query stripped, trailing slash stripped, ``http``->``ws``). So a client
  served at ``https://host/viser`` connects to ``wss://host/viser``.

Hence a *single* aiohttp route at ``/viser`` can serve both: a plain GET
returns the client bundle, and a WebSocket upgrade is proxied to Viser's
internal port.

The one subtlety is the subprotocol. Viser negotiates ``viser-v<version>`` and
**rejects connections whose version does not match the server**
(``viser/infra/_infra.py``), so the proxy has to forward the client's requested
subprotocols upstream and echo the selected one back.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

import aiohttp
from aiohttp import web

from .client_utils import log


# Viser messages can be large (the docstring in _infra.py sets a 50 MB cap for
# e.g. client.get_render()). 0 = unlimited on both legs of the proxy.
_MAX_MSG_SIZE = 0

VISER_ROUTE = "/viser"


@lru_cache(maxsize=1)
def viser_client_html() -> bytes:
    """Bytes of the installed Viser client bundle.

    Read from the *installed* package so the client version can never drift
    from the server version the proxy talks to.
    """
    import viser

    path = Path(viser.__file__).parent / "client" / "build" / "index.html"
    if not path.is_file():  # pragma: no cover - broken viser install
        raise RuntimeError(
            f"Viser client bundle not found at {path}. "
            f"Is `viser` installed correctly?"
        )
    data = path.read_bytes()
    log("info", f"single_port: viser client bundle loaded ({len(data) / 1e6:.1f} MB) from {path}")
    return data


def _requested_subprotocols(request: web.Request) -> tuple[str, ...]:
    raw = request.headers.get("Sec-WebSocket-Protocol", "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


async def relay(ws_a, ws_b) -> None:
    """Relay both directions, tearing down as soon as EITHER side ends.

    Waiting on both directions (``asyncio.gather``) deadlocks: when the
    browser goes away, the browser->server pump finishes but the
    server->browser pump keeps waiting on an upstream that has no idea the
    client left. The upstream socket is never closed, so `gest-server` holds
    its per-conversation lock forever and the *next* connection hangs at the
    handshake -- i.e. Start Over and every reconnect break.
    """
    tasks = {
        asyncio.create_task(_pump(ws_a, ws_b.send_bytes, ws_b.send_str)),
        asyncio.create_task(_pump(ws_b, ws_a.send_bytes, ws_a.send_str)),
    }
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for ws in (ws_a, ws_b):
            if not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass


async def _pump(src, dst_send_bytes, dst_send_str) -> None:
    """Forward every frame from one WebSocket to the other until it closes."""
    async for msg in src:
        if msg.type == aiohttp.WSMsgType.BINARY:
            await dst_send_bytes(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            await dst_send_str(msg.data)
        elif msg.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        ):
            break


def make_viser_handler(viser_port: int, viser_host: str = "127.0.0.1"):
    """Build the aiohttp handler serving Viser on the main port.

    GET            -> the client bundle
    WS upgrade     -> bidirectional proxy to ``ws://{viser_host}:{viser_port}/``
    """
    upstream_url = f"http://{viser_host}:{viser_port}/"

    async def handler(request: web.Request) -> web.StreamResponse:
        protocols = _requested_subprotocols(request)
        ws_server = web.WebSocketResponse(
            max_msg_size=_MAX_MSG_SIZE, protocols=protocols,
        )

        if not ws_server.can_prepare(request).ok:
            # Plain page load.
            return web.Response(
                body=viser_client_html(),
                content_type="text/html",
                headers={"Cache-Control": "no-store"},
            )

        await ws_server.prepare(request)
        log("info", f"single_port: viser ws proxy open (subprotocols={protocols or '-'})")

        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(
                upstream_url,
                max_msg_size=_MAX_MSG_SIZE,
                protocols=protocols,
                autoping=True,
            ) as ws_client:
                await relay(ws_client, ws_server)
        except aiohttp.ClientError as exc:
            log("error", f"single_port: viser upstream unreachable ({exc!r})")
        finally:
            await session.close()
            if not ws_server.closed:
                await ws_server.close()
            log("info", "single_port: viser ws proxy closed")

        return ws_server

    return handler


def attach_viser_routes(app: web.Application, viser_port: int, viser_host: str = "127.0.0.1") -> None:
    """Register the single-port Viser route on an existing aiohttp app."""
    app.router.add_get(VISER_ROUTE, make_viser_handler(viser_port, viser_host))
    log("info", f"single_port: serving viser at {VISER_ROUTE} (proxying {viser_host}:{viser_port})")
