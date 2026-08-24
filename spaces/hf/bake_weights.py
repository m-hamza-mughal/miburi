"""Pre-fetch every weight the demo needs, at Docker build time.

Spaces build time is **not billed** — only `Starting` and `Running` are. So
pulling ~18 GB here rather than on first boot is free, and it keeps a metered
GPU from idling through a 16 GB download on every cold start.

Fetched:
  * Moshi 7B bf16 + Mimi + the text tokenizer  (kyutai/moshiko-pytorch-bf16)
  * MIBURI release checkpoints                 (~1.7 GB, 8 dirs)
  * MIBURI release assets                      (~200 MB: SMPL-X, Moshi web UI,
                                                 y_bot Mixamo bundle)
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(f"[bake] {msg}", flush=True)


def main() -> int:
    from miburi import download_assets, download_checkpoints
    from miburi.models import loaders

    _log("assets (SMPL-X + Moshi UI + y_bot) -> assets_dep/")
    download_assets.download(local_dir=APP / "assets_dep")

    _log("checkpoints (codecs + GTDM3) -> experiments/")
    download_checkpoints.download(local_dir=APP / "experiments")

    # Reuses the exact resolution logic the demo uses at runtime, so whatever
    # this pulls is what CheckpointInfo.from_hf_repo will find in the cache.
    _log(f"moshi runtime weights <- {loaders.DEFAULT_REPO}")
    info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    for label, path in (
        ("moshi", info.moshi_weights),
        ("mimi", info.mimi_weights),
        ("tokenizer", info.tokenizer),
    ):
        size = Path(path).stat().st_size / 1e9
        _log(f"  {label:<10} {size:>6.2f} GB  {path}")

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
