"""Fetch the BEAT2-English subset MIBURI's training pipeline expects.

After ``pip install .``, run::

    miburi-download-beatx-dataset

This pulls 4 subdirs from
`m-hamza-mughal/beat2-additional-annotations <https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations>`_
(a fork of `H-Liu1997/BEAT2` that adds annotations from the RAG-Gesture and
MIBURI projects) into ``datasets/beat_english_v2.0.0/``. The 4 subdirs are
exactly what `scripts/trainers/dataloaders/beatx/build_hdf5_beatx.py` and
`scripts/trainers/utils/metrics.py` actually read:

- ``smplxflame_25/`` — 25 fps SMPL-X + FLAME motion (MIBURI annotation)
- ``wave16k/`` — 16 kHz audio (BEAT2 inherited)
- ``whisper_transcription/`` — Whisper transcripts with casing + punctuation (MIBURI)
- ``weights/`` — pretrained CNN motion-autoencoder weights for the FGD metric (BEAT2 inherited)

By downloading these files you agree to honor the upstream licenses: the
base BEAT2 data is Apache 2.0; please cite the original BEAT/EMAGE paper
(Liu et al., CVPR 2024) plus MIBURI / RAG-Gesture where appropriate. See the
dataset card for full citation guidance.

Total download is ~16.6 GB and is intended for *training* MIBURI from
scratch — running the released demo does NOT require this; the demo only
needs ``miburi-download-assets`` + ``miburi-download-checkpoints``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "m-hamza-mughal/beat2-additional-annotations"
DEFAULT_LOCAL_DIR = "datasets/beat_english_v2.0.0"
REPO_PREFIX = "beat_english_v2.0.0"

# Only the 4 subdirs the live MIBURI code path reads.
SUBDIRS = ["smplxflame_25", "wave16k", "whisper_transcription", "weights"]
ALLOW_PATTERNS = [f"{REPO_PREFIX}/{s}/**" for s in SUBDIRS]

# Files that must exist after a successful download. Doesn't enumerate every
# take — just probes that each subdir landed something.
SANITY_PROBES = {
    "smplxflame_25":         "1_wayne_0_1_1.npz",
    "wave16k":               "1_wayne_0_1_1.wav",
    "whisper_transcription": "1_wayne_0_1_1.json",
    "weights":               "AESKConv_240_100.bin",
}

LICENSE_NOTICE = """\
By downloading these files you agree to honor the upstream licenses:
 - BEAT2 (base data): Apache 2.0, https://huggingface.co/datasets/H-Liu1997/BEAT2
 - MIBURI / RAG-Gesture annotations: Apache 2.0 (same as upstream for ease of reuse)
Citation guidance is in the HF dataset card:
 https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations
"""


def _log(msg: str) -> None:
    print(f"[miburi-download-beatx-dataset] {msg}", flush=True)


def _sanity_check(local_dir: Path) -> list[str]:
    missing: list[str] = []
    for sub, probe in SANITY_PROBES.items():
        sub_dir = local_dir / sub
        if not sub_dir.is_dir() or not any(sub_dir.iterdir()):
            missing.append(f"{sub}/ (empty or missing)")
            continue
        # Probe file may or may not exist depending on the take; relax to
        # "any non-zero file present" if the named probe is absent.
        if not (sub_dir / probe).is_file() and not any(p.is_file() for p in sub_dir.rglob("*")):
            missing.append(f"{sub}/<any-file>")
    return missing


def download(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
    *,
    force_download: bool = False,
) -> Path:
    local_dir = Path(local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    _log(f"repo:   {repo_id}")
    _log(f"target: {local_dir}")
    _log("starting download (~16.6 GB; re-runs are no-ops once cached)…")
    print(LICENSE_NOTICE, file=sys.stderr)

    # snapshot_download will land files at <local_dir>/<repo_prefix>/<subdir>/...
    # so users get the right tree if they point --local-dir at the parent of
    # `beat_english_v2.0.0/`. We default --local-dir to that dir itself, so
    # flatten the one extra level of nesting after the download.
    snapshot_local = local_dir.parent  # e.g. datasets/
    snapshot_local.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(snapshot_local),
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        force_download=force_download,
    )

    # If we landed under <parent>/beat_english_v2.0.0/<subdirs>, that's already
    # what the trainer expects. No further flattening needed.
    missing = _sanity_check(local_dir)
    if missing:
        _log("WARNING: download finished but some required subdirs are missing:")
        for r in missing:
            print(f"  - {r}", file=sys.stderr)
        _log("Try re-running with --force-download to re-fetch.")
    else:
        _log(f"all {len(SUBDIRS)} required subdirs present at {local_dir}")
    return local_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})")
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
