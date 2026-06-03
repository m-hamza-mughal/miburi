"""Fetch the released MIBURI demo-time assets from Hugging Face.

After ``pip install .``, run::

    miburi-download-assets

This pulls three subtrees (~200 MB total) into ``assets_dep/``:

- ``demo-static/``                — embedded Moshi UI frontend (~1 MB)
- ``mixamo_characters_release/``  — 4 Mixamo characters (.npz) (~8 MB)
- ``smplx_2020/``                 — SMPL-X NEUTRAL 2020 model (~115 MB)

Re-running is a no-op once cached. By downloading you agree to honor the
upstream third-party licenses (SMPL-X, Mixamo, Moshi). See the HF repo's
README for details: https://huggingface.co/m-hamza-mughal/miburi-release-assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "m-hamza-mughal/miburi-release-assets"
DEFAULT_LOCAL_DIR = "assets_dep"

# Files that must exist after a successful download for the demo to launch.
REQUIRED_FILES = [
    "demo-static/index.html",
    "mixamo_characters_release/y_bot.npz",
    "mixamo_characters_release/ch08.npz",
    "mixamo_characters_release/ch31.npz",
    "mixamo_characters_release/remy.npz",
    "smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz",
]

LICENSE_NOTICE = """\
By downloading these assets you agree to honor the upstream licenses:
 - SMPL-X model (smplx_2020/): non-commercial only.
   https://smpl-x.is.tue.mpg.de/modellicense.html
 - Mixamo characters (mixamo_characters_release/): Adobe Mixamo terms of use.
   https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html
 - Moshi weights (auto-fetched by the demo from kyutai/moshiko-pytorch-bf16):
   CC-BY 4.0 (Kyutai).
"""


def _log(msg: str) -> None:
    print(f"[miburi-download-assets] {msg}", flush=True)


def _sanity_check(local_dir: Path) -> list[str]:
    return [r for r in REQUIRED_FILES if not (local_dir / r).is_file()]


def download(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    *,
    force_download: bool = False,
) -> Path:
    local_dir = Path(local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    _log(f"repo: {repo_id}")
    _log(f"target: {local_dir}")
    _log("starting download (~200 MB; re-runs are no-ops once cached)…")
    print(LICENSE_NOTICE, file=sys.stderr)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "demo-static/**",
            "mixamo_characters_release/**",
            "smplx_2020/**",
        ],
        force_download=force_download,
    )

    missing = _sanity_check(local_dir)
    if missing:
        _log("WARNING: download finished but some required files are missing:")
        for r in missing:
            print(f"  - {r}", file=sys.stderr)
        _log("Try re-running with --force-download to re-fetch.")
    else:
        _log(f"all {len(REQUIRED_FILES)} required files present")
    return local_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HF model repo id (default: {DEFAULT_REPO_ID})")
    p.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR,
                   help=f"Where to land the files (default: {DEFAULT_LOCAL_DIR})")
    p.add_argument("--force-download", action="store_true",
                   help="Bypass HF's local cache and re-fetch every file.")
    args = p.parse_args()

    try:
        download(repo_id=args.repo_id, local_dir=args.local_dir,
                 force_download=args.force_download)
    except Exception as exc:
        _log(f"FAILED: {exc!r}")
        return 1

    missing = _sanity_check(Path(args.local_dir).resolve())
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
