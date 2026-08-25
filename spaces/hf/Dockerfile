# MIBURI demo — Hugging Face Space (Docker SDK, on-demand dedicated GPU).
#
# Design notes
# ------------
# * Space **build time is not billed**; only `Starting` and `Running` are. So
#   every byte we can bake into the image is a byte we don't pay to download on
#   a metered GPU. All ~18 GB of weights are therefore baked at build time by
#   default (BAKE_WEIGHTS=1). Set --build-arg BAKE_WEIGHTS=0 to fall back to a
#   runtime download if the image ever gets too large for the builder.
# * The image runs on cpu-basic too: `entrypoint.sh` checks $ACCELERATOR and
#   serves a static landing page instead of loading Moshi when there is no GPU.
#   That is what keeps the Space free at rest.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# DO NOT set NO_TORCH_COMPILE here. Measured on an RTX 3090 (25 s of audio,
# localhost, no tunnel):
#
#   torch.compile ON   Moshi step mean 67.7 ms   0% of steps over the 80 ms
#                      budget -> 15% headroom, latency flat
#   NO_TORCH_COMPILE=1 Moshi step mean 80.3 ms   66% of steps over budget
#                      -> falls behind ~1%/frame, latency grows without bound
#
# rope + gating are in every transformer layer's hot path, so disabling compile
# costs ~16% and flips the loop from real-time to accumulating. It also starves
# the SMPL-X posing in the motion process (2.2 ms -> 40 ms of GPU contention).
# The flag exists only for ZeroGPU, which forbids torch.compile.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as uid 1000 with a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR $HOME/app

# --- torch first, so the (large, stable) layer caches independently ---------
RUN pip install --user \
      torch==2.9.0 torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu126

# --- weights ---------------------------------------------------------------
# ABOVE the miburi install on purpose. bake_weights.py imports only
# huggingface_hub, so this ~18 GB layer does not depend on the application
# code and survives every miburi change -- the difference between a one
# minute rebuild and a seven minute one. Build time is unbilled, GPU runtime
# is not, so a cold start never waits on a download.
# ~18 GB: Moshi 7B bf16 + Mimi + 8 MIBURI checkpoint dirs + SMPL-X + the
# Moshi web UI bundle.
# huggingface_hub normally arrives as a miburi dependency, but miburi is
# installed *below* this layer now, so it has to be explicit here.
RUN pip install --user "huggingface-hub>=0.24,<0.29"
ARG BAKE_WEIGHTS=1
COPY --chown=user bake_weights.py ./
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python bake_weights.py; fi

# --- miburi + demo deps ----------------------------------------------------
# Pin to a commit, never a branch. Docker keys a layer on the instruction
# TEXT, so "git+...@main" stays cache-valid forever even after main moves:
# the image silently keeps an old miburi while the freshly-copied
# entrypoint passes flags it does not have. Bump this on every miburi
# change; it busts the cache correctly and makes the build reproducible.
ARG MIBURI_REF=546ae2bd7ac996f401f6c58934aa524493e23aa8
RUN pip install --user "miburi @ git+https://github.com/m-hamza-mughal/miburi@${MIBURI_REF}"

# The repo ids and filenames in bake_weights.py are duplicated from miburi so
# that the layer above can stay independent of it. This re-checks them against
# the package now that it is installed, so the copies cannot drift silently --
# a stale constant fails the build here instead of the first boot.
RUN python bake_weights.py --verify

# Pin viser AFTER the weight bake so changing it never invalidates that
# 18 GB layer. Its client and server negotiate a "viser-v<version>"
# subprotocol and reject a mismatch, and the client bundle is served from the
# installed package -- so an unpinned viser means the deployed pair is
# whatever was latest at build time: untested, and unaddressable by any
# external client speaking a known version.
RUN pip install --user --force-reinstall --no-deps "viser==1.0.29"

COPY --chown=user entrypoint.sh offline_server.py ./
COPY --chown=user offline ./offline
RUN chmod +x entrypoint.sh

EXPOSE 7860
CMD ["./entrypoint.sh"]
