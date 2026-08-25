#!/usr/bin/env bash
# Two modes, decided by the hardware the Space is currently running on.
#
#   no GPU  -> offline_server.py: a static landing page. This is the resting
#              state of the Space and it costs nothing, so the URL stays live
#              (paper links, teaser, "demo is offline") between demo windows.
#   GPU     -> the real demo, single-port.
#
# Flip between them from a laptop with tools/space_hardware.py; the Space
# restarts into whichever branch applies.
set -euo pipefail

cd "$HOME/app"
PORT="${PORT:-7860}"
ACCEL="${ACCELERATOR:-none}"

# Ask torch, not $ACCELERATOR. The docs say ACCELERATOR is "none" on CPU-only
# Spaces; the platform actually reports "cpu" on cpu-basic. An exclusion list
# is therefore not safe in either direction -- a value we have not seen would
# be read as "GPU present" and the demo would crash on "Found no NVIDIA
# driver". torch.cuda.is_available() is the authoritative answer, and the
# ~10 s import is paid once per container start.
has_gpu() {
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

if ! has_gpu; then
    echo "[entrypoint] ACCELERATOR='${ACCEL}' and no CUDA device -> serving landing page on :${PORT}"
    exec python offline_server.py --port "${PORT}"
fi

echo "[entrypoint] ACCELERATOR='${ACCEL}' -> starting the live demo on :${PORT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --host is internal wiring only (the motion websocket + the viser subprocess);
# --bind is what faces the platform. Keeping them separate avoids dialling
# 0.0.0.0 as a destination address.
# --public-port hands the public socket to a separate front-proxy process, so
# the ~3 MB/s Viser mesh stream never enters the inference event loop. With it
# relayed in-process the Moshi step ran 77.1 ms with 20.7% of steps over the
# 80 ms budget; with the front proxy it is 67.9 ms and 0.0%. This server then
# listens on localhost only.
exec python -m miburi.gest-server \
    --glm-config experiments/demoexp_release_gtdm3_goodspk/config.yaml \
    --glm-cfg-coef 1.3 \
    --host 127.0.0.1 \
    --port 8998 \
    --public-port "${PORT}" \
    --motion-stream "${MIBURI_MOTION_STREAM:-vertices}" \
    --motion-decimate "${MIBURI_MOTION_DECIMATE:-0}" \
    --motion-stride "${MIBURI_MOTION_STRIDE:-1}" \
    --static assets_dep/demo-static \
    ${MIBURI_EXTRA_ARGS:-}
