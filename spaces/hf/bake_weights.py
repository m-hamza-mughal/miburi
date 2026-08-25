"""Pre-fetch every weight the demo needs, at Docker build time.

Two properties this file deliberately has:

1. **Build time is not billed**; `Starting` and `Running` are. So pulling
   ~18 GB here rather than on first boot is free, and it keeps a metered GPU
   from idling through a download on every cold start.

2. **It imports only `huggingface_hub`, never `miburi`.** That is what lets
   the Dockerfile put this layer *above* the `pip install miburi` layer. If it
   imported miburi, every code change would invalidate the 18 GB download and
   turn a one-minute rebuild into a seven-minute one.

The constants below are duplicated from `miburi.models.loaders` and
`miburi.download_{assets,checkpoints}` rather than imported. That duplication
is the price of (2); `--verify` re-checks them against the installed package
so the copies cannot drift silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

APP = Path(__file__).resolve().parent

# --- mirrors miburi.models.loaders ----------------------------------------
MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
MOSHI_FILES = [
    "model.safetensors",                              # MOSHI_NAME
    "tokenizer-e351c8d8-checkpoint125.safetensors",   # MIMI_NAME
    "tokenizer_spm_32k_3.model",                      # TEXT_TOKENIZER_NAME
]
# loaders.hf_get() hardcodes this cache dir, relative to the process CWD.
MOSHI_CACHE = "assets_dep/kyutai_cache"

# --- mirrors miburi.download_assets / download_checkpoints ----------------
ASSETS_REPO = "m-hamza-mughal/miburi-release-assets"
ASSETS_PATTERNS = ["demo-static/**", "mixamo_characters_release/**", "smplx_2020/**"]
# The assets repo stores demo-static/ etc. at its root, so it lands under
# assets_dep/ directly.
ASSETS_DEST = "assets_dep"

CKPT_REPO = "m-hamza-mughal/miburi-release-checkpoints"
CKPT_PATTERNS = ["experiments/**"]
# The checkpoint repo already carries the "experiments/" prefix, so it must be
# unpacked at the app root -- targeting "experiments/" would nest it as
# experiments/experiments/. (miburi.download_checkpoints handles this by
# flattening afterwards; unpacking one level up is the same result without the
# rename dance.)
CKPT_DEST = "."


def _log(msg: str) -> None:
    print(f"[bake] {msg}", flush=True)


def bake() -> int:
    _log(f"assets  <- {ASSETS_REPO}")
    snapshot_download(repo_id=ASSETS_REPO, repo_type="model",
                      local_dir=str(APP / ASSETS_DEST),
                      allow_patterns=ASSETS_PATTERNS)

    _log(f"ckpts   <- {CKPT_REPO}")
    snapshot_download(repo_id=CKPT_REPO, repo_type="model",
                      local_dir=str((APP / CKPT_DEST).resolve()),
                      allow_patterns=CKPT_PATTERNS)

    _log(f"moshi   <- {MOSHI_REPO}")
    cache = APP / MOSHI_CACHE
    for fn in MOSHI_FILES:
        path = hf_hub_download(MOSHI_REPO, fn, cache_dir=str(cache))
        _log(f"  {Path(path).stat().st_size / 1e9:>6.2f} GB  {fn}")

    # Sanity-check the handful of paths the demo dereferences on startup, so a
    # silently partial download fails the build instead of the first boot.
    required = [
        APP / "assets_dep/demo-static/index.html",
        APP / "assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz",
        APP / "experiments/demoexp_release_gtdm3_goodspk/config.yaml",
        APP / "experiments/demoexp_release_gtdm3_goodspk/speaker_id_to_index.json",
        APP / "experiments/demoexp_release_gtdm3_goodspk/tokens_idle_slices/idle_upper_tokens_slices.npz",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        for p in missing:
            print(f"  MISSING {p}", file=sys.stderr)
        return 1
    _log("done")
    return 0


def verify() -> int:
    """Fail loudly if the constants above have drifted from the package.

    Run this after `pip install miburi` -- it is the guard that makes the
    duplication above safe.
    """
    from miburi.models import loaders
    from miburi import download_assets, download_checkpoints

    checks = [
        ("moshi repo", MOSHI_REPO, loaders.DEFAULT_REPO),
        ("moshi file", MOSHI_FILES[0], loaders.MOSHI_NAME),
        ("mimi file", MOSHI_FILES[1], loaders.MIMI_NAME),
        ("tokenizer", MOSHI_FILES[2], loaders.TEXT_TOKENIZER_NAME),
        ("assets repo", ASSETS_REPO, download_assets.DEFAULT_REPO_ID),
        ("ckpt repo", CKPT_REPO, download_checkpoints.DEFAULT_REPO_ID),
    ]
    bad = [(what, mine, theirs) for what, mine, theirs in checks if mine != theirs]
    for what, mine, theirs in bad:
        print(f"  DRIFT {what}: bake_weights has {mine!r}, miburi has {theirs!r}", file=sys.stderr)
    if bad:
        print("bake_weights.py constants are stale -- update them.", file=sys.stderr)
        return 1
    _log(f"constants verified against the installed miburi ({len(checks)} checks)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="only check the constants against the installed miburi")
    args = ap.parse_args()
    sys.exit(verify() if args.verify else bake())
