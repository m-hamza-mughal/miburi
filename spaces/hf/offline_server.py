"""Landing page served when the Space has no GPU attached.

This is the Space's resting state. Keeping it on `cpu-basic` costs nothing, so
the public URL stays live -- paper links, teaser, and an honest "the live demo
is offline right now" -- instead of 404-ing between demo windows.

Deliberately dependency-light: aiohttp only, no torch import, so it boots in
about a second.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent / "offline"

PAPER_URL = "https://arxiv.org/abs/2603.03282"
PROJECT_URL = "https://vcai.mpi-inf.mpg.de/projects/MIBURI/"
CODE_URL = "https://github.com/m-hamza-mughal/miburi"

PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>MIBURI</title>
    <style>
      :root {{ color-scheme: dark; }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        background: #070707;
        color: #d7d7d7;
      }}
      .shell {{ min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }}
      .bar {{ padding: 10px 14px; border-bottom: 1px solid #242424; background: #111111; }}
      .title {{ font-size: 18px; font-weight: 600; margin: 0; text-align: center; }}
      .subtitle {{ margin: 4px 0 0 0; font-size: 12px; color: #a0a0a0; text-align: center; }}
      .panes {{ display: grid; place-items: center; padding: 24px 14px 40px; min-height: 0; }}
      .card {{
        width: min(920px, 100%);
        border: 1px solid #242424;
        border-radius: 8px;
        background: #000;
        overflow: hidden;
      }}
      .card img {{ width: 100%; display: block; border-bottom: 1px solid #242424; }}
      .body {{ padding: 18px 20px 22px; }}
      .status {{
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 12px; font-weight: 600; letter-spacing: .04em;
        color: #e8c46a; border: 1px solid #4a3c1c; background: #1a1508;
        border-radius: 999px; padding: 5px 12px; margin-bottom: 14px;
      }}
      .dot {{ width: 7px; height: 7px; border-radius: 50%; background: #e8c46a; }}
      p {{ margin: 0 0 12px 0; font-size: 13px; line-height: 1.65; color: #b9b9b9; }}
      .links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
      a.btn {{
        text-decoration: none; font-size: 12px; font-weight: 600;
        color: #d7d7d7; border: 1px solid #2e2e2e; background: #131313;
        border-radius: 6px; padding: 8px 14px;
      }}
      a.btn:hover {{ border-color: #454545; background: #191919; color: #fff; }}
      code {{ color: #9fc7ef; }}
    </style>
  </head>
  <body>
    <div class="shell">
      <header class="bar">
        <p class="title">MIBURI: Towards Expressive Interactive Gesture Synthesis (CVPR 2026)</p>
        <p class="subtitle">Real-time dialogue-driven full-body gesture and facial-expression synthesis</p>
      </header>
      <main class="panes">
        <section class="card">
          <img src="/offline/teaser.png" alt="MIBURI teaser" />
          <div class="body">
            <span class="status"><span class="dot"></span>LIVE DEMO OFFLINE</span>
            <p>
              The interactive demo runs on a dedicated GPU that is powered on for
              scheduled demo windows rather than left running continuously. When it is
              live, this page becomes the full dashboard: talk into your microphone and
              MIBURI replies with speech while a SMPL-X character gestures along in
              real time at 25&nbsp;fps.
            </p>
            <p>
              In the meantime, the paper, project page (with video results), and the
              full training + inference code are all available. The demo can also be
              run locally on any 24&nbsp;GB GPU with <code>miburi-demo</code>.
            </p>
            <div class="links">
              <a class="btn" href="{project}">Project page</a>
              <a class="btn" href="{paper}">Paper (arXiv)</a>
              <a class="btn" href="{code}">Code (GitHub)</a>
            </div>
          </div>
        </section>
      </main>
    </div>
  </body>
</html>
""".format(project=PROJECT_URL, paper=PAPER_URL, code=CODE_URL)


async def handle_index(_request: web.Request) -> web.Response:
    return web.Response(text=PAGE, content_type="text/html",
                        headers={"Cache-Control": "no-store"})


async def handle_health(_request: web.Request) -> web.Response:
    return web.Response(text="miburi space: offline mode\n", content_type="text/plain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--bind", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_health)
    app.router.add_static("/offline", path=str(HERE), name="offline")
    print(f"[offline_server] listening on {args.bind}:{args.port}", flush=True)
    web.run_app(app, host=args.bind, port=args.port, print=None)


if __name__ == "__main__":
    main()
