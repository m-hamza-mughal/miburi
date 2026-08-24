---
title: MIBURI
emoji: 🚀
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Live Demo for MIBURI
license: cc-by-nc-4.0
---

# MIBURI — live demo

Implementation demo for **MIBURI: Towards Expressive Interactive Gesture
Synthesis** (CVPR 2026) — [paper](https://arxiv.org/abs/2603.03282) ·
[project page](https://vcai.mpi-inf.mpg.de/projects/MIBURI/) ·
[code](https://github.com/m-hamza-mughal/miburi).

Talk into your microphone; a frozen Moshi 7B answers in speech while MIBURI's
Gesture LM drives a SMPL-X character's full body, hands and face at 25 fps in
the viewer next to it.

**Headphones are recommended** — without them your microphone picks up the
model's own voice and it starts replying to itself.

## Demo availability

The demo needs a dedicated ~24 GB GPU and is powered on for scheduled demo
windows rather than left running. When it is off, this URL serves a landing
page with the paper, video results and code. When it is on, the same URL is
the full dashboard.

To run it yourself on any 24 GB GPU, see
[the repo README](https://github.com/m-hamza-mughal/miburi#running-the-demo) —
`miburi-demo` is the same code this Space runs.

## How this Space is built

`Dockerfile` installs the `miburi` package from GitHub and bakes every weight
into the image at build time (build time is unbilled; GPU runtime is not).
`entrypoint.sh` then picks one of two modes based on `$ACCELERATOR`:

| Hardware | What runs | Cost |
| --- | --- | --- |
| `cpu-basic` | `offline_server.py` — static landing page | free |
| GPU | `miburi.gest-server --embed-viser` — the real demo | billed per minute |

The demo normally wants two ports (the aiohttp dashboard and Viser's own
server). Spaces expose one, so `--embed-viser` serves Viser's single-file
client bundle at `/viser` and proxies its WebSocket through the same port.

## Licensing

MIBURI code and weights are **CC BY-NC 4.0** — non-commercial use only.
Third-party components keep their own licenses: Moshi/Mimi (MIT code,
CC-BY-4.0 weights, Kyutai), SMPL-X (Max Planck Society, non-commercial),
Mixamo characters (Adobe Mixamo terms).

```bibtex
@InProceedings{mughal2026miburi,
    title = {MIBURI: Towards Expressive Interactive Gesture Synthesis},
    author = {M. Hamza Mughal and Rishabh Dabral and Vera Demberg and Christian Theobalt},
    booktitle = {Computer Vision and Pattern Recognition (CVPR)},
    year = {2026}
}
```
