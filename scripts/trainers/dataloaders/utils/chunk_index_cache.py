"""Chunk-index cache subsystem.

Each HDF5 database built by the various ``build_hdf5_*.py`` scripts ships with
a sidecar ``*.chunk_index_v{N}.npz`` + ``*.chunk_index_v{N}.meta.json`` pair
that records ``(chunk_id, split, relpath, speaker_id, is_sitting)`` for every
chunk in the file. This module owns reading, writing, validating, and locking
those sidecars.

The ``fallback_speaker_resolver`` argument on the read/scan/build entrypoints
keeps this module dataset-agnostic: callers that need a dataset-specific
chunk-id → speaker-id parser (e.g. the seamless dataloader's
``get_seamlessint_spkids``) pass it in; build scripts that already write
``chunk_speaker_ids`` to the HDF5 root never need it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from loguru import logger

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable.
    tqdm = None  # type: ignore[assignment]


_CHUNK_INDEX_SCHEMA_VERSION = 1
_CHUNK_INDEX_LOG_EVERY = 100_000
_CHUNK_INDEX_LOCK_TIMEOUT_SEC = 60 * 60
_CHUNK_INDEX_LOCK_POLL_SEC = 1.0


@dataclass(frozen=True)
class _SourceStat:
    size_bytes: int
    mtime_ns: int


@dataclass
class ChunkMetaTable:
    chunk_ids: np.ndarray
    chunk_splits: np.ndarray
    chunk_relpaths: np.ndarray
    speaker_ids: np.ndarray
    is_sitting: np.ndarray

    def __post_init__(self) -> None:
        n = int(self.chunk_ids.shape[0])
        if self.chunk_splits.shape[0] != n:
            raise ValueError("chunk_splits length mismatch")
        if self.chunk_relpaths.shape[0] != n:
            raise ValueError("chunk_relpaths length mismatch")
        if self.speaker_ids.shape[0] != n:
            raise ValueError("speaker_ids length mismatch")
        if self.is_sitting.shape[0] != n:
            raise ValueError("is_sitting length mismatch")

    @property
    def num_rows(self) -> int:
        return int(self.chunk_ids.shape[0])


_CHUNK_META_MEMO: dict[tuple[str, str | None, int, int, int | None], ChunkMetaTable] = {}


def _progress_iter(iterable: Iterable[Any], *, total: int | None, desc: str) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        mininterval=1.0,
    )


def _decode_if_bytes(val: Any) -> Any:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def _to_str_list(arr: Iterable[Any]) -> list[str]:
    return [str(_decode_if_bytes(x)) for x in arr]


def _source_stat(path: str) -> _SourceStat:
    st = os.stat(path)
    return _SourceStat(size_bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def _cache_paths_for_hdf5(hdf5_path: str, cache_dir: str | None) -> tuple[str, str, str]:
    abs_hpath = os.path.abspath(hdf5_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha1(abs_hpath.encode("utf-8")).hexdigest()[:12]
        base = os.path.basename(abs_hpath)
        stem = os.path.join(cache_dir, f"{base}.{digest}.chunk_index_v{_CHUNK_INDEX_SCHEMA_VERSION}")
    else:
        stem = f"{abs_hpath}.chunk_index_v{_CHUNK_INDEX_SCHEMA_VERSION}"
    return f"{stem}.npz", f"{stem}.meta.json", f"{stem}.lock"


def _read_meta_json(meta_path: str) -> dict[str, Any] | None:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_meta_json(meta_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".chunk_index_meta_", suffix=".json", dir=os.path.dirname(meta_path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _is_cache_valid(meta: dict[str, Any] | None, source_path: str, source_stat: _SourceStat) -> bool:
    if meta is None:
        return False
    if int(meta.get("version", -1)) != _CHUNK_INDEX_SCHEMA_VERSION:
        return False
    if os.path.abspath(str(meta.get("source_path", ""))) != os.path.abspath(source_path):
        return False
    if int(meta.get("source_size_bytes", -1)) != source_stat.size_bytes:
        return False
    if int(meta.get("source_mtime_ns", -1)) != source_stat.mtime_ns:
        return False
    return True


def _acquire_lock(lock_path: str) -> int:
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            return fd
        except FileExistsError:
            if (time.time() - start) > _CHUNK_INDEX_LOCK_TIMEOUT_SEC:
                raise TimeoutError(f"Timed out waiting for index cache lock: {lock_path}")
            time.sleep(_CHUNK_INDEX_LOCK_POLL_SEC)


def _release_lock(lock_fd: int, lock_path: str) -> None:
    try:
        os.close(lock_fd)
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _empty_chunk_meta_table() -> ChunkMetaTable:
    empty_str = np.asarray([], dtype=np.str_)
    empty_bool = np.asarray([], dtype=np.bool_)
    return ChunkMetaTable(
        chunk_ids=empty_str,
        chunk_splits=empty_str.copy(),
        chunk_relpaths=empty_str.copy(),
        speaker_ids=empty_str.copy(),
        is_sitting=empty_bool,
    )


def _load_cache_table(npz_path: str, meta_path: str) -> tuple[ChunkMetaTable | None, dict[str, Any] | None]:
    if not os.path.exists(npz_path):
        return None, None
    meta = _read_meta_json(meta_path)
    if meta is None:
        return None, None
    try:
        with np.load(npz_path, allow_pickle=False) as cache:
            table = ChunkMetaTable(
                chunk_ids=np.asarray(cache["chunk_ids"]).astype(np.str_),
                chunk_splits=np.asarray(cache["chunk_splits"]).astype(np.str_),
                chunk_relpaths=np.asarray(cache["chunk_relpaths"]).astype(np.str_),
                speaker_ids=np.asarray(cache["speaker_ids"]).astype(np.str_),
                is_sitting=np.asarray(cache["is_sitting"]).astype(np.bool_),
            )
    except Exception:
        return None, None
    return table, meta


def _write_cache_table(
    npz_path: str,
    meta_path: str,
    table: ChunkMetaTable,
    source_path: str,
    source_stat: _SourceStat,
) -> None:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    fd, tmp_npz = tempfile.mkstemp(prefix=".chunk_index_", suffix=".npz", dir=os.path.dirname(npz_path) or ".")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_npz,
            chunk_ids=table.chunk_ids,
            chunk_splits=table.chunk_splits,
            chunk_relpaths=table.chunk_relpaths,
            speaker_ids=table.speaker_ids,
            is_sitting=table.is_sitting,
        )
        os.replace(tmp_npz, npz_path)
    finally:
        if os.path.exists(tmp_npz):
            try:
                os.remove(tmp_npz)
            except OSError:
                pass

    meta = {
        "version": _CHUNK_INDEX_SCHEMA_VERSION,
        "source_path": os.path.abspath(source_path),
        "source_size_bytes": int(source_stat.size_bytes),
        "source_mtime_ns": int(source_stat.mtime_ns),
        "row_count": int(table.num_rows),
        "created_at_unix": float(time.time()),
    }
    _write_meta_json(meta_path, meta)


def _scan_chunk_metadata_from_hdf5(
    hdf5_path: str,
    *,
    max_rows: int | None = None,
    fallback_speaker_resolver: Callable[[str], str] | None = None,
) -> ChunkMetaTable:
    if not os.path.exists(hdf5_path):
        return _empty_chunk_meta_table()

    with h5py.File(hdf5_path, "r") as f:
        if "chunk_ids" not in f:
            return _empty_chunk_meta_table()

        chunk_ids = _to_str_list(f["chunk_ids"][:])
        n_rows_all = len(chunk_ids)
        if max_rows is not None:
            n_rows = min(n_rows_all, max(0, int(max_rows)))
            if n_rows < n_rows_all:
                chunk_ids = chunk_ids[:n_rows]
        else:
            n_rows = n_rows_all
        if n_rows == 0:
            return _empty_chunk_meta_table()

        if "chunk_splits" in f:
            chunk_splits = _to_str_list(f["chunk_splits"][:])
        else:
            chunk_splits = []
        if len(chunk_splits) < n_rows:
            chunk_splits.extend([""] * (n_rows - len(chunk_splits)))
        elif len(chunk_splits) > n_rows:
            chunk_splits = chunk_splits[:n_rows]

        if "chunk_relpaths" in f:
            chunk_relpaths = _to_str_list(f["chunk_relpaths"][:])
        else:
            chunk_relpaths = []
        if len(chunk_relpaths) < n_rows:
            chunk_relpaths.extend([""] * (n_rows - len(chunk_relpaths)))
        elif len(chunk_relpaths) > n_rows:
            chunk_relpaths = chunk_relpaths[:n_rows]

        # Use object dtype during fill; np.str_ with empty init would become <U1 and truncate speaker IDs to 1 char.
        speaker_ids = np.full((n_rows,), "", dtype=object)
        is_sitting = np.zeros((n_rows,), dtype=np.bool_)

        # Fast path: use root metadata arrays when available.
        has_root_speaker = "chunk_speaker_ids" in f
        has_root_sitting = "chunk_is_sitting" in f
        if has_root_speaker:
            root_speaker = _to_str_list(f["chunk_speaker_ids"][:])
            if len(root_speaker) >= n_rows:
                speaker_ids = np.asarray(root_speaker[:n_rows], dtype=np.str_)
            else:
                speaker_ids[: len(root_speaker)] = np.asarray(root_speaker, dtype=np.str_)
        if has_root_sitting:
            root_sitting = np.asarray(f["chunk_is_sitting"][:]).astype(np.bool_)
            if root_sitting.shape[0] >= n_rows:
                is_sitting = root_sitting[:n_rows]
            else:
                is_sitting[: root_sitting.shape[0]] = root_sitting

        need_speaker_fallback = not has_root_speaker or np.any(speaker_ids == "")
        need_sitting_fallback = not has_root_sitting

        if need_speaker_fallback or need_sitting_fallback:
            base = os.path.basename(hdf5_path) or hdf5_path
            iter_chunk_ids = _progress_iter(
                chunk_ids,
                total=n_rows,
                desc=f"scan_chunk_index:{base}",
            )
            for i, cid in enumerate(iter_chunk_ids):
                if i > 0 and i % _CHUNK_INDEX_LOG_EVERY == 0:
                    logger.info("Chunk index scan progress for {}: {}/{}", hdf5_path, i, n_rows)
                if (
                    need_speaker_fallback
                    and (not speaker_ids[i])
                    and fallback_speaker_resolver is not None
                ):
                    try:
                        speaker_ids[i] = fallback_speaker_resolver(cid)
                    except Exception:
                        speaker_ids[i] = ""
                if need_sitting_fallback:
                    grp = f.get(cid)
                    if grp is not None:
                        is_sitting[i] = bool(grp.attrs.get("is_sitting", False))

    return ChunkMetaTable(
        chunk_ids=np.asarray(chunk_ids, dtype=np.str_),
        chunk_splits=np.asarray(chunk_splits, dtype=np.str_),
        chunk_relpaths=np.asarray(chunk_relpaths, dtype=np.str_),
        speaker_ids=np.asarray(speaker_ids, dtype=np.str_),
        is_sitting=is_sitting,
    )


def _derive_meta_path_for_index(npz_path: str) -> str:
    """Mirror the canonical ``<stem>.npz`` / ``<stem>.meta.json`` pairing.
    Used by the explicit ``chunk_index_path`` override so the override
    can point at a file like
    ``database.hdf5.chunk_index_filtered.npz`` and we'll find the
    matching ``.meta.json`` alongside it."""
    if npz_path.endswith(".npz"):
        return npz_path[: -len(".npz")] + ".meta.json"
    return f"{npz_path}.meta.json"


def _load_chunk_metadata_from_explicit_path(
    hdf5_path: str,
    *,
    chunk_index_path: str,
) -> ChunkMetaTable:
    """Load a chunk-index NPZ from an **explicit caller-provided path**,
    bypassing the canonical stem resolution and validity checks.

    A missing file or corrupted contents raise loudly; we deliberately
    do **not** fall back to scanning the HDF5 because falling back
    would silently undo the override the user opted into.
    """
    npz_path = os.path.abspath(chunk_index_path)
    meta_path = _derive_meta_path_for_index(npz_path)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"chunk_index_path points at a missing file: {npz_path}. "
            "Either build the file or unset chunk_index_path to fall back to "
            "the canonical stem resolver."
        )
    table, meta = _load_cache_table(npz_path, meta_path)
    if table is None:
        raise RuntimeError(
            f"chunk_index_path={npz_path} could not be parsed as a chunk index. "
            f"Expected meta.json at {meta_path}."
        )
    logger.info(
        "Chunk index loaded from explicit chunk_index_path: {} ({} rows; hdf5={})",
        npz_path, int(table.num_rows), hdf5_path,
    )
    if meta is not None and isinstance(meta, dict):
        provenance_keys = ("filtered_from_row_count", "filter_schema", "filter_thresholds")
        provenance = {k: meta[k] for k in provenance_keys if k in meta}
        if provenance:
            logger.info("Chunk index provenance: {}", json.dumps(provenance, default=str))
    return table


def _load_chunk_metadata(
    hdf5_path: str,
    *,
    index_cache_mode: str = "auto",
    index_cache_dir: str | None = None,
    scan_max_rows: int | None = None,
    chunk_index_path: str | None = None,
    fallback_speaker_resolver: Callable[[str], str] | None = None,
) -> ChunkMetaTable:
    if chunk_index_path:
        return _load_chunk_metadata_from_explicit_path(
            hdf5_path, chunk_index_path=str(chunk_index_path),
        )

    mode = str(index_cache_mode or "auto").lower()
    if mode not in {"auto", "off", "rebuild", "readonly"}:
        raise ValueError(f"Unsupported index_cache_mode={index_cache_mode!r}")

    if not os.path.exists(hdf5_path):
        logger.warning("HDF5 path does not exist, skipping index load: {}", hdf5_path)
        return _empty_chunk_meta_table()

    source_stat = _source_stat(hdf5_path)
    abs_hpath = os.path.abspath(hdf5_path)
    abs_cache_dir = os.path.abspath(index_cache_dir) if index_cache_dir else None
    scan_max_rows_norm = None if scan_max_rows is None else max(0, int(scan_max_rows))
    memo_scan_key = scan_max_rows_norm if mode == "off" else None
    memo_key = (abs_hpath, abs_cache_dir, source_stat.size_bytes, source_stat.mtime_ns, memo_scan_key)
    if mode != "rebuild" and memo_key in _CHUNK_META_MEMO:
        if mode == "off":
            return _CHUNK_META_MEMO[memo_key]
        npz_path, meta_path, _ = _cache_paths_for_hdf5(hdf5_path, index_cache_dir)
        cached_meta = _read_meta_json(meta_path)
        if os.path.exists(npz_path) and _is_cache_valid(cached_meta, hdf5_path, source_stat):
            return _CHUNK_META_MEMO[memo_key]

    if mode == "off":
        if scan_max_rows_norm is not None:
            logger.info(
                "Index cache mode=off: using early-stop scan_max_rows={} for {}",
                int(scan_max_rows_norm),
                hdf5_path,
            )
        table = _scan_chunk_metadata_from_hdf5(
            hdf5_path,
            max_rows=scan_max_rows_norm,
            fallback_speaker_resolver=fallback_speaker_resolver,
        )
        _CHUNK_META_MEMO[memo_key] = table
        return table

    npz_path, meta_path, lock_path = _cache_paths_for_hdf5(hdf5_path, index_cache_dir)

    table, meta = _load_cache_table(npz_path, meta_path)
    if mode != "rebuild" and table is not None and _is_cache_valid(meta, hdf5_path, source_stat):
        logger.info("Chunk index cache hit: {}", npz_path)
        _CHUNK_META_MEMO[memo_key] = table
        return table

    if mode == "readonly":
        raise RuntimeError(
            f"index_cache_mode=readonly but cache missing/stale for {hdf5_path}. "
            f"Expected valid cache at {npz_path}"
        )
    logger.info("Chunk index cache miss/stale, rebuilding for: {}", hdf5_path)
    lock_fd = _acquire_lock(lock_path)
    try:
        # Another process may have refreshed cache while we waited for the lock.
        if mode != "rebuild":
            table2, meta2 = _load_cache_table(npz_path, meta_path)
            if table2 is not None and _is_cache_valid(meta2, hdf5_path, source_stat):
                logger.info("Chunk index cache became available while waiting: {}", npz_path)
                _CHUNK_META_MEMO[memo_key] = table2
                return table2

        table = _scan_chunk_metadata_from_hdf5(
            hdf5_path,
            fallback_speaker_resolver=fallback_speaker_resolver,
        )
        _write_cache_table(npz_path, meta_path, table, hdf5_path, source_stat)
        logger.info(
            "Chunk index cache written: {} ({} rows)",
            npz_path,
            table.num_rows,
        )
    finally:
        _release_lock(lock_fd, lock_path)

    _CHUNK_META_MEMO[memo_key] = table
    return table


def _normalize_hdf5_paths(hdf5_paths: str | list[str]) -> list[str]:
    if isinstance(hdf5_paths, str):
        return [p.strip() for p in hdf5_paths.split(",") if p.strip()]
    return list(hdf5_paths)


def build_chunk_index_cache(
    hdf5_paths: str | list[str],
    *,
    index_cache_dir: str | None = None,
    force_rebuild: bool = False,
    fallback_speaker_resolver: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Build or refresh chunk-index cache files for one or more HDF5 databases."""
    paths = _normalize_hdf5_paths(hdf5_paths)
    mode = "rebuild" if force_rebuild else "auto"
    results: list[dict[str, Any]] = []
    for path in paths:
        start = time.time()
        table = _load_chunk_metadata(
            path,
            index_cache_mode=mode,
            index_cache_dir=index_cache_dir,
            fallback_speaker_resolver=fallback_speaker_resolver,
        )
        npz_path, meta_path, _ = _cache_paths_for_hdf5(path, index_cache_dir)
        results.append(
            {
                "hdf5_path": path,
                "rows": int(table.num_rows),
                "cache_npz_path": npz_path,
                "cache_meta_path": meta_path,
                "elapsed_sec": float(time.time() - start),
            }
        )
    return results


def write_chunk_index_cache_from_metadata(
    *,
    hdf5_path: str,
    chunk_ids: Iterable[Any],
    chunk_splits: Iterable[Any],
    chunk_relpaths: Iterable[Any],
    chunk_speaker_ids: Iterable[Any],
    chunk_is_sitting: Iterable[Any],
    index_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Write chunk-index cache files from already-collected chunk metadata."""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 path not found for cache write: {hdf5_path}")

    ids = _to_str_list(chunk_ids)
    splits = _to_str_list(chunk_splits)
    relpaths = _to_str_list(chunk_relpaths)
    speakers = _to_str_list(chunk_speaker_ids)
    sitting = np.asarray(list(chunk_is_sitting)).astype(np.bool_)

    n = len(ids)
    if len(splits) != n or len(relpaths) != n or len(speakers) != n or sitting.shape[0] != n:
        raise ValueError(
            "chunk metadata length mismatch: "
            f"ids={n} splits={len(splits)} relpaths={len(relpaths)} "
            f"speakers={len(speakers)} sitting={int(sitting.shape[0])}"
        )

    table = ChunkMetaTable(
        chunk_ids=np.asarray(ids, dtype=np.str_),
        chunk_splits=np.asarray(splits, dtype=np.str_),
        chunk_relpaths=np.asarray(relpaths, dtype=np.str_),
        speaker_ids=np.asarray(speakers, dtype=np.str_),
        is_sitting=sitting,
    )
    npz_path, meta_path, _ = _cache_paths_for_hdf5(hdf5_path, index_cache_dir)
    _write_cache_table(
        npz_path=npz_path,
        meta_path=meta_path,
        table=table,
        source_path=hdf5_path,
        source_stat=_source_stat(hdf5_path),
    )
    return {
        "hdf5_path": hdf5_path,
        "rows": int(table.num_rows),
        "cache_npz_path": npz_path,
        "cache_meta_path": meta_path,
    }
