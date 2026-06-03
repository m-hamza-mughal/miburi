"""Fetch the released MIBURI checkpoints from Hugging Face.

After ``pip install .``, run::

    miburi-download-checkpoints

This pulls 8 directories (~1.7 GB total) into ``experiments/``: 4 codecs + 1
Gesture LM for each of the two release tracks (demoexp good-speaker subset,
and the 23-speaker all-speaker variant). Re-running is a no-op once the
files match the remote hashes (HF's ``snapshot_download`` is idempotent).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "m-hamza-mughal/miburi-release-checkpoints"
DEFAULT_LOCAL_DIR = "experiments"

# Expected post-download layout (used by the sanity check below). Each entry
# is a relative path under DEFAULT_LOCAL_DIR that must exist for the demo to
# launch.
REQUIRED_FILES = [
    # Codecs — both release tracks.
    "allspk_release_facecodec/config.yaml",
    "allspk_release_facecodec/last_380.safetensors",
    "allspk_release_lowertranscodec/config.yaml",
    "allspk_release_lowertranscodec/last_705.safetensors",
    "allspk_release_uppercodec/config.yaml",
    "allspk_release_uppercodec/last_490.safetensors",
    "demoexp_release_facecodec/config.yaml",
    "demoexp_release_facecodec/last_100.safetensors",
    "demoexp_release_lowercodec/config.yaml",
    "demoexp_release_lowercodec/last_440.safetensors",
    "demoexp_release_uppercodec/config.yaml",
    "demoexp_release_uppercodec/last_180.safetensors",
    # GTDM3 LMs + their startup-time auxiliary files.
    "allspk_release_gtdm3_exp/config.yaml",
    "allspk_release_gtdm3_exp/last_1720.safetensors",
    "allspk_release_gtdm3_exp/speaker_id_to_index.json",
    "allspk_release_gtdm3_exp/tokens_idle_slices/idle_upper_tokens_slices.npz",
    "allspk_release_gtdm3_exp/tokens_idle_slices/idle_face_tokens_slices.npz",
    "demoexp_release_gtdm3_goodspk/config.yaml",
    "demoexp_release_gtdm3_goodspk/last_6500.safetensors",
    "demoexp_release_gtdm3_goodspk/speaker_id_to_index.json",
    "demoexp_release_gtdm3_goodspk/tokens_idle_slices/idle_upper_tokens_slices.npz",
    "demoexp_release_gtdm3_goodspk/tokens_idle_slices/idle_face_tokens_slices.npz",
]


def _log(msg: str) -> None:
    print(f"[miburi-download-checkpoints] {msg}", flush=True)


def _sanity_check(local_dir: Path) -> list[str]:
    """Return a list of missing relative paths (empty list = all present)."""
    return [r for r in REQUIRED_FILES if not (local_dir / r).is_file()]


def download(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    *,
    force_download: bool = False,
) -> Path:
    """Fetch the release checkpoints into ``local_dir``.

    Returns the resolved local directory path.
    """
    local_dir = Path(local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    _log(f"repo: {repo_id}")
    _log(f"target: {local_dir}")
    _log("starting download (~1.7 GB; re-runs are no-ops once cached)…")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["experiments/**"],
        force_download=force_download,
    )

    # snapshot_download lays the repo's `experiments/` subtree directly under
    # `local_dir`, so the dirs end up at `<local_dir>/experiments/<release_dir>/...`
    # when local_dir is something other than `experiments/`. The default value
    # `experiments` would produce `experiments/experiments/...`, which is wrong.
    # Detect and flatten that one level if needed.
    nested = local_dir / "experiments"
    if nested.is_dir() and any(nested.iterdir()) and local_dir.name == "experiments":
        _log("flattening nested experiments/experiments/ layout")
        for entry in nested.iterdir():
            target = local_dir / entry.name
            if target.exists():
                continue
            entry.rename(target)
        try:
            nested.rmdir()
        except OSError:
            pass

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
