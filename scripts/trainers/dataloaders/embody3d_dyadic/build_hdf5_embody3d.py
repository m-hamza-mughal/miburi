import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger
from scipy.signal import savgol_filter
from tqdm import tqdm

from ..utils.chunk_index_cache import write_chunk_index_cache_from_metadata
from ..utils.data_tools import (
    align_points_to_first_frame_yaw_only,
    zero_first_frame_pose_yaw_only,
)
from ..utils.interleaver import Interleaver, InterleavedTokenizer
from ..utils.pose_valid import compute_pose_valid
from ..utils.visualize import (
    render_chunk_smplx_joints_side_by_side,
    render_smplx_debug_video,
)
from ..utils import rotation_conversions as Rc


SAFE_INCOMPLETE_EXIT_CODE = 3

# Second-level subdirs under data_dir that are not motion recordings and must
# be excluded from the directory scan.
EMBODY3D_NON_SAMPLE_SUBDIRS = ("audio_raw", "videos")


@dataclass
class FileEntry:
    file_id: str
    split: str
    label: str
    batch_idx: int
    archive_idx: int
    relpath: str
    motion_path: str
    audio_path: str
    text_path: str
    pose_valid_path: str | None


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        import sphn  # type: ignore

        audio, sr = sphn.read(path)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio.astype(np.float32), sr
    except Exception:
        try:
            import soundfile as sf

            audio, sr = sf.read(path, always_2d=True)
            audio = audio.T
            return audio.astype(np.float32), sr
        except Exception as exc:
            raise RuntimeError(f"Failed to read audio: {path}") from exc


def _resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    try:
        import sphn  # type: ignore

        return sphn.resample(audio, sr, target_sr)
    except Exception:
        import librosa

        out = []
        for ch in range(audio.shape[0]):
            out.append(librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr))
        return np.stack(out, axis=0)


def _assign_split(idx: int, n_total: int, train_ratio: float, val_ratio: float) -> str:
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    if idx < n_train:
        return "train"
    if idx < n_train + n_val:
        return "val"
    return "test"


def _build_file_list(args) -> list[FileEntry]:
    entries: list[FileEntry] = []

    progress_every = max(1, int(getattr(args, "filelist_progress_every", 10000)))
    scanned = 0
    kept = 0
    t0 = time.time()
    logger.info("Scanning Embody3D samples under {}", args.data_dir)

    # Embody3D layout: data_dir/<base_sample>/<speaker>/{smplx_*, audio_separated, transcription}/...
    sample_dirs = sorted(glob.glob(os.path.join(args.data_dir, "*", "*")))
    sample_dirs = [
        d for d in sample_dirs
        if os.path.isdir(d) and not any(skip in d for skip in EMBODY3D_NON_SAMPLE_SUBDIRS)
    ]
    n_total = len(sample_dirs)
    logger.info("Discovered {} candidate sample directories", n_total)

    motion_subdir = f"smplx_{args.motion_fps}"

    for idx, sample_dir in enumerate(sample_dirs):
        scanned += 1
        if scanned % progress_every == 0:
            logger.info(
                "File list progress: scanned={} kept={} elapsed={:.1f}s",
                scanned,
                kept,
                time.time() - t0,
            )

        parts = sample_dir.rstrip("/").split("/")
        if len(parts) < 2:
            continue
        base_sample_name = parts[-2]
        speaker_name = parts[-1]

        # file_id encodes <base>+<speaker> so it remains a single string usable as
        # an HDF5 group name; sample_path = file_id.replace("+", "/") inverts it.
        file_id = f"{base_sample_name}+{speaker_name}"

        motion_path = os.path.join(sample_dir, motion_subdir, base_sample_name + ".npz")
        audio_path = os.path.join(sample_dir, "audio_separated", base_sample_name + ".wav")
        text_path = os.path.join(sample_dir, "transcription", base_sample_name + ".json")

        if not os.path.exists(motion_path):
            continue
        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            continue

        split = _assign_split(idx, n_total, args.train_ratio, args.val_ratio)

        kept += 1
        entries.append(
            FileEntry(
                file_id=file_id,
                split=split,
                label=speaker_name,
                batch_idx="-",
                archive_idx="-",
                relpath=file_id,
                motion_path=motion_path,
                audio_path=audio_path,
                text_path=text_path,
                pose_valid_path=None,
            )
        )

    if args.max_files is not None:
        entries = entries[: args.max_files]
    logger.info(
        "File list scan complete: scanned={} kept={} max_files={} elapsed={:.1f}s",
        scanned,
        len(entries),
        str(args.max_files),
        time.time() - t0,
    )
    return entries


def _decode_str_array(values) -> list[str]:
    out: list[str] = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _default_state_json_path(hdf5_path: str) -> str:
    return f"{hdf5_path}.state.json"


def _write_state_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _write_file_status_datasets(f: h5py.File, file_status_map: dict[str, str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    for ds_name in ("file_status_file_ids", "file_status_values"):
        if ds_name in f:
            del f[ds_name]
    if not file_status_map:
        f.create_dataset("file_status_file_ids", data=np.array([], dtype=object), dtype=dt)
        f.create_dataset("file_status_values", data=np.array([], dtype=object), dtype=dt)
        return
    file_ids = np.array(list(file_status_map.keys()), dtype=object)
    statuses = np.array([file_status_map[k] for k in file_status_map.keys()], dtype=object)
    f.create_dataset("file_status_file_ids", data=file_ids, dtype=dt)
    f.create_dataset("file_status_values", data=statuses, dtype=dt)


def _build_run_state_payload(
    *,
    run_complete: bool,
    hdf5_path: str,
    files_total: int,
    file_status_map: dict[str, str],
    files_written: int,
    skipped_quality: int,
    skipped_short: int,
    failed: int,
    skipped_empty: int,
    processed_files: int,
    files_scanned_this_run: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    files_done = len(file_status_map)
    files_remaining = max(0, files_total - files_done)
    return {
        "run_complete": bool(run_complete),
        "hdf5_path": hdf5_path,
        "timestamp": int(time.time()),
        "files_total": int(files_total),
        "files_done": int(files_done),
        "files_remaining": int(files_remaining),
        "files_written": int(files_written),
        "skipped_quality": int(skipped_quality),
        "skipped_short": int(skipped_short),
        "skipped_empty": int(skipped_empty),
        "failed": int(failed),
        "processed_files": int(processed_files),
        "files_scanned_this_run": int(files_scanned_this_run),
        "stop_reason": stop_reason,
    }


def _file_id_from_chunk_id(chunk_id: str) -> str:
    # Chunk ids are "<file_id>_C<chunk_idx>" in chunked mode or "<file_id>" in full-sequence mode.
    return chunk_id.rsplit("_C", 1)[0]


def _load_resume_state(hdf5_path: str) -> dict[str, Any]:
    t0 = time.time()
    state = {
        "chunk_ids": [],
        "chunk_splits": [],
        "chunk_relpaths": [],
        "chunk_bounds": [],
        "chunk_speaker_ids": [],
        "chunk_is_sitting": [],
        "speaker_ids": set(),
        "processed_file_ids": set(),
        "file_status_map": {},
    }
    if not os.path.exists(hdf5_path):
        logger.info("Resume state: HDF5 not found at {} (fresh run).", hdf5_path)
        return state

    logger.info("Resume state: reading existing HDF5 at {} ...", hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        if "chunk_ids" in f:
            state["chunk_ids"] = _decode_str_array(f["chunk_ids"][:])
            logger.info("Resume state: loaded chunk_ids ({})", len(state["chunk_ids"]))
        if "chunk_splits" in f:
            state["chunk_splits"] = _decode_str_array(f["chunk_splits"][:])
            logger.info("Resume state: loaded chunk_splits ({})", len(state["chunk_splits"]))
        if "chunk_relpaths" in f:
            state["chunk_relpaths"] = _decode_str_array(f["chunk_relpaths"][:])
            logger.info("Resume state: loaded chunk_relpaths ({})", len(state["chunk_relpaths"]))
        if "chunk_bounds" in f:
            state["chunk_bounds"] = f["chunk_bounds"][:].tolist()
            logger.info("Resume state: loaded chunk_bounds ({})", len(state["chunk_bounds"]))
        if "chunk_speaker_ids" in f:
            state["chunk_speaker_ids"] = _decode_str_array(f["chunk_speaker_ids"][:])
            logger.info("Resume state: loaded chunk_speaker_ids ({})", len(state["chunk_speaker_ids"]))
        if "chunk_is_sitting" in f:
            state["chunk_is_sitting"] = np.asarray(f["chunk_is_sitting"][:]).astype(bool).tolist()
            logger.info("Resume state: loaded chunk_is_sitting ({})", len(state["chunk_is_sitting"]))
        if "file_status_file_ids" in f and "file_status_values" in f:
            status_ids = _decode_str_array(f["file_status_file_ids"][:])
            status_vals = _decode_str_array(f["file_status_values"][:])
            state["file_status_map"] = dict(zip(status_ids, status_vals))
            logger.info("Resume state: loaded file_status_map ({} terminal statuses)", len(state["file_status_map"]))

        n_chunks = len(state["chunk_ids"])
        if n_chunks > 0:
            if len(state["chunk_speaker_ids"]) != n_chunks:
                if state["chunk_speaker_ids"]:
                    logger.warning(
                        "Resume state: chunk_speaker_ids length mismatch ({} != {}); rebuilding speaker ids from chunk_ids.",
                        len(state["chunk_speaker_ids"]),
                        n_chunks,
                    )
                state["chunk_speaker_ids"] = [get_speaker_id(_file_id_from_chunk_id(cid)) for cid in state["chunk_ids"]]
            if len(state["chunk_is_sitting"]) != n_chunks:
                if state["chunk_is_sitting"]:
                    logger.warning(
                        "Resume state: chunk_is_sitting length mismatch ({} != {}); rebuilding from group attrs.",
                        len(state["chunk_is_sitting"]),
                        n_chunks,
                    )
                sitting_vals: list[bool] = []
                for cid in tqdm(
                    state["chunk_ids"],
                    desc="resume:chunk_is_sitting_from_groups",
                    unit="chunk",
                    mininterval=5.0,
                ):
                    grp = f.get(cid)
                    sitting_vals.append(bool(grp.attrs.get("is_sitting", False)) if grp is not None else False)
                state["chunk_is_sitting"] = sitting_vals

        # Fast path (preferred): status map gives terminal file states directly.
        if state["file_status_map"]:
            state["processed_file_ids"] = set(state["file_status_map"].keys())
            if state["chunk_speaker_ids"]:
                for spk in state["chunk_speaker_ids"]:
                    if spk:
                        state["speaker_ids"].add(spk)
            elif state["chunk_ids"]:
                for chunk_id in tqdm(
                    state["chunk_ids"],
                    desc="resume:speaker_ids_from_chunk_ids",
                    unit="chunk",
                    mininterval=5.0,
                ):
                    state["speaker_ids"].add(get_speaker_id(_file_id_from_chunk_id(chunk_id)))
            logger.info(
                "Resume state: using file_status_map fast path (processed_files={}, speakers={}).",
                len(state["processed_file_ids"]),
                len(state["speaker_ids"]),
            )
            logger.info("Resume state load finished in {:.2f}s", time.time() - t0)
            return state

        # Legacy fallback: derive processed files from chunk ids if status map is absent.
        if state["chunk_ids"]:
            for chunk_id in tqdm(
                state["chunk_ids"],
                desc="resume:derive_status_from_chunk_ids",
                unit="chunk",
                mininterval=5.0,
            ):
                file_id = _file_id_from_chunk_id(chunk_id)
                state["processed_file_ids"].add(file_id)
                state["file_status_map"].setdefault(file_id, "written")
                state["speaker_ids"].add(get_speaker_id(file_id))
            logger.info(
                "Resume state: derived terminal statuses from chunk_ids (legacy path). "
                "processed_files={}, speakers={}",
                len(state["processed_file_ids"]),
                len(state["speaker_ids"]),
            )
            logger.info("Resume state load finished in {:.2f}s", time.time() - t0)
            return state

        # Deep fallback: scan groups only if neither status map nor chunk index exists.
        logger.warning(
            "Resume state: no file_status_map/chunk_ids found. Falling back to group scan (can be slow)."
        )
        group_count = 0
        for key in tqdm(
            f.keys(),
            desc="resume:scan_hdf5_groups",
            unit="group",
            mininterval=5.0,
        ):
            obj = f[key]
            if not isinstance(obj, h5py.Group):
                continue
            group_count += 1
            file_id = obj.attrs.get("file_id")
            if isinstance(file_id, bytes):
                file_id = file_id.decode("utf-8")
            if file_id:
                file_id = str(file_id)
                state["processed_file_ids"].add(file_id)
                state["file_status_map"].setdefault(file_id, "written")
                state["speaker_ids"].add(get_speaker_id(file_id))
            if group_count % 50000 == 0:
                logger.info(
                    "Resume state group scan progress: groups={} processed_files={}",
                    group_count,
                    len(state["processed_file_ids"]),
                )
        logger.info("Resume state: group scan complete (groups={})", group_count)
    logger.info("Resume state load finished in {:.2f}s", time.time() - t0)
    return state


def get_speaker_id(file_id: str) -> str:
    # Embody3D file_id is "<base_sample>+<speaker>"; speaker is the second token.
    try:
        parts = file_id.split("+")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    except Exception:
        pass
    return "UNKNOWN"


def _load_transcript(text_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Load transcription
    with open(text_path, 'r') as f:
        text_data = json.load(f)
        transcript = text_data['segments']

    return transcript


def _extract_alignments(segments: list[dict[str, Any]]) -> list[tuple[str, tuple[float, float], str]]:
    alignments: list[tuple[str, tuple[float, float], str]] = []
    
    prev_start = 0.0
    prev_end = 0.0
    for sentence in segments:
        words = sentence.get("words", [])
        for word in words:
            word_text = word.get("word", "").strip()
            word_start = word.get("start")
            word_end = word.get("end")
            if word_start is None:
                word_start = prev_start
            if word_end is None:
                word_end = prev_end
            alignments.append((word_text, (float(word_start), float(word_end)), "SPEAKER_MAIN"))
            prev_start = float(word_start)
            prev_end = float(word_end)
    return alignments


def _get_text_for_chunk(
    segments: list[dict[str, Any]],
    chunk_startsec: float,
    chunk_endsec: float,
) -> tuple[str, list[tuple[str, float, float]]]:
    words: list[str] = []
    prev_start = 0.0
    prev_end = 0.0
    chunk_segs: list[tuple[str, float, float]] = []
    for sentence in segments:
        words_list = sentence.get("words", [])
        for word in words_list:
            word_start = word.get("start")
            word_end = word.get("end")
            if word_start is None:
                word_start = prev_start
            if word_end is None:
                word_end = prev_end
            if word_end < chunk_startsec or word_start > chunk_endsec:
                continue
            w = word.get("word", "").strip()
            words.append(w)
            chunk_segs.append((w, float(word_start), float(word_end)))
            prev_start = float(word_start)
            prev_end = float(word_end)
    return " ".join(words), chunk_segs


def _compute_vad_bits(vad_info: list[dict[str, Any]], nframes: int, motion_fps: int) -> np.ndarray:
    vad_bits = np.zeros(nframes, dtype=bool)
    for sentence in vad_info:
        # sentence_start = sentence['start']
        # sentence_end = sentence['end']
        for word in sentence['words']:
            start_sec = word.get("start", 0.0)
            end_sec = word.get("end", 0.0)
            start_frame = int(start_sec * motion_fps)
            end_frame = int(end_sec * motion_fps)
            vad_bits[start_frame:end_frame] = True
    return vad_bits


def _compute_contacts(joint_pos: np.ndarray, velocity_thresh: float = 0.01) -> np.ndarray:
    joints = torch.from_numpy(joint_pos)
    foot_joints = joints[:, (7, 8, 10, 11), :]
    feetv = torch.zeros(foot_joints.shape[1], foot_joints.shape[0])
    foot_joints = foot_joints.permute(1, 0, 2)
    feetv[:, :-1] = (foot_joints[:, 1:] - foot_joints[:, :-1]).norm(dim=-1)
    return (feetv < velocity_thresh).numpy().astype(bool).transpose(1, 0)


def _maybe_get(motion_data: dict, key: str, fallback_shape: tuple[int, ...], dtype=np.float32):
    if key in motion_data:
        return motion_data[key]
    return np.zeros(fallback_shape, dtype=dtype)


def _get_valid_savgol_window(num_frames: int, window: int, poly: int) -> int | None:
    if num_frames <= poly:
        return None
    win = int(window)
    if win % 2 == 0:
        win -= 1
    if win <= poly:
        win = poly + 1
        if win % 2 == 0:
            win += 1
    if win > num_frames:
        win = num_frames if num_frames % 2 == 1 else num_frames - 1
    if win <= poly or win < 3:
        return None
    return win


def _savgol_smooth_pose_6d(pose_aa: np.ndarray, window: int, poly: int) -> np.ndarray:
    num_frames = pose_aa.shape[0]
    win = _get_valid_savgol_window(num_frames, window, poly)
    if win is None:
        return pose_aa
    pose_aa_t = torch.from_numpy(pose_aa.reshape(num_frames, -1, 3)).float()
    rot_m = Rc.axis_angle_to_matrix(pose_aa_t)
    rot_6d = Rc.matrix_to_rotation_6d(rot_m)
    rot_6d_np = rot_6d.cpu().numpy()
    rot_6d_np = savgol_filter(rot_6d_np, win, poly, axis=0, mode="interp")
    rot_6d_t = torch.from_numpy(rot_6d_np).float()
    rot_m_s = Rc.rotation_6d_to_matrix(rot_6d_t)
    pose_aa_s = Rc.matrix_to_axis_angle(rot_m_s).cpu().numpy()
    return pose_aa_s.reshape(num_frames, -1)


def _savgol_smooth_transl(transl: np.ndarray, window: int, poly: int) -> np.ndarray:
    num_frames = transl.shape[0]
    win = _get_valid_savgol_window(num_frames, window, poly)
    if win is None:
        return transl
    return savgol_filter(transl, win, poly, axis=0, mode="interp")


def _compute_smplx_joints(
    body_pose: np.ndarray,
    transl: np.ndarray,
    smplx_model,
    betas: np.ndarray | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    device = smplx_model.device
    nframes = body_pose.shape[0]
    if betas is None:
        smplx_betas = torch.zeros(nframes, 300, device=device)
    else:
        betas = np.asarray(betas)
        if betas.ndim == 1:
            smplx_betas = torch.tensor(betas, device=device).float().unsqueeze(0).repeat(nframes, 1)
        else:
            smplx_betas = torch.tensor(betas, device=device).float()
            if smplx_betas.shape[0] != nframes:
                smplx_betas = smplx_betas[:1].repeat(nframes, 1)

    joints_chunks = []
    for start in range(0, nframes, batch_size):
        end = min(start + batch_size, nframes)
        body_pose_t = torch.tensor(body_pose[start:end], dtype=torch.float32, device=device)
        transl_t = torch.tensor(transl[start:end], dtype=torch.float32, device=device)
        zero_exps = torch.zeros((end - start, 100), device=device, dtype=torch.float32)
        betas_t = smplx_betas[start:end]

        with torch.no_grad():
            joints = smplx_model(
                betas=betas_t,
                transl=transl_t,
                expression=zero_exps,
                jaw_pose=body_pose_t[:, 66:69],
                global_orient=body_pose_t[:, :3],
                body_pose=body_pose_t[:, 3 : 21 * 3 + 3],
                left_hand_pose=body_pose_t[:, 25 * 3 : 40 * 3],
                right_hand_pose=body_pose_t[:, 40 * 3 : 55 * 3],
                leye_pose=body_pose_t[:, 69:72],
                reye_pose=body_pose_t[:, 72:75],
                return_joints=True,
            )["joints"][:, :55, :]
        joints_chunks.append(joints.detach().cpu().numpy())

    if not joints_chunks:
        return np.zeros((0, 55, 3), dtype=np.float32)
    return np.concatenate(joints_chunks, axis=0).astype(np.float32, copy=False)


def _compute_sitting_bits(
    joints: np.ndarray,
    height_ratio_thresh: float = 0.75,
    knee_deg_thresh: float = 130.0,
) -> np.ndarray:
    if joints is None or joints.ndim != 3:
        return np.zeros((0,), dtype=bool)

    valid = np.isfinite(joints).all(axis=(1, 2))
    if not np.any(valid):
        return np.zeros((joints.shape[0],), dtype=bool)

    pelvis = joints[:, 0]
    lhip = joints[:, 1]
    rhip = joints[:, 2]
    lknee = joints[:, 4]
    rknee = joints[:, 5]
    lankle = joints[:, 7]
    rankle = joints[:, 8]

    pelvis_h = pelvis[:, 1] - 0.5 * (lankle[:, 1] + rankle[:, 1])
    leg_len = 0.5 * (
        np.linalg.norm(pelvis - lknee, axis=1)
        + np.linalg.norm(lknee - lankle, axis=1)
        + np.linalg.norm(pelvis - rknee, axis=1)
        + np.linalg.norm(rknee - rankle, axis=1)
    )
    ratio = pelvis_h / (leg_len + 1e-6)

    def _knee_angle(hip, knee, ankle):
        v1 = hip - knee
        v2 = ankle - knee
        denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-6
        cosang = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    l_angle = _knee_angle(lhip, lknee, lankle)
    r_angle = _knee_angle(rhip, rknee, rankle)
    knees_bent = (l_angle < knee_deg_thresh) & (r_angle < knee_deg_thresh)

    sitting = (ratio < height_ratio_thresh) | knees_bent
    sitting &= valid
    return sitting.astype(bool)


def _compute_upright_correction(
    global_rot: torch.Tensor,
    tilt_deg_threshold: float = 5.0,
    max_deg: float = 30.0,
) -> torch.Tensor | None:
    if global_rot.ndim != 3:
        return None
    device = global_rot.device
    dtype = global_rot.dtype

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    world_forward = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    world_right = torch.linalg.cross(world_up, world_forward, dim=0)
    if world_right.norm() < 1e-8:
        return None
    world_right = world_right / (world_right.norm() + 1e-8)

    right = torch.matmul(global_rot, world_right)
    up = torch.matmul(global_rot, world_up)

    right_avg = right.mean(0)
    if right_avg.norm() < 1e-8:
        return None
    right_avg = right_avg / (right_avg.norm() + 1e-8)

    up_avg = up.mean(0)
    if up_avg.norm() < 1e-8:
        return None
    up_avg = up_avg / (up_avg.norm() + 1e-8)

    up_proj = up_avg - torch.dot(up_avg, right_avg) * right_avg
    world_up_proj = world_up - torch.dot(world_up, right_avg) * right_avg
    if up_proj.norm() < 1e-8 or world_up_proj.norm() < 1e-8:
        return None
    up_proj = up_proj / (up_proj.norm() + 1e-8)
    world_up_proj = world_up_proj / (world_up_proj.norm() + 1e-8)

    dot = torch.clamp(torch.dot(up_proj, world_up_proj), -1.0, 1.0)
    cross = torch.linalg.cross(up_proj, world_up_proj, dim=0)
    signed_angle = torch.atan2(torch.dot(right_avg, cross), dot)
    angle_deg = signed_angle * (180.0 / math.pi)

    if torch.abs(angle_deg) < tilt_deg_threshold:
        return None

    angle_deg = torch.clamp(angle_deg, -max_deg, max_deg)
    angle_rad = angle_deg * (math.pi / 180.0)
    axis_angle = right_avg * angle_rad
    return Rc.axis_angle_to_matrix(axis_angle)


def _assess_global_orient_quality(
    global_orient_aa: np.ndarray,
    fps: float,
    flip_deg: float = 20.0,
    hard_flip_deg: float = 80.0,
    flip_ratio_thresh: float = 0.01,
    jitter_residual_deg_s: float = 90.0,
    jitter_min_vel_deg_s: float = 30.0,
    jitter_ratio_thresh: float = 0.1,
    jitter_savgol_window: int = 11,
    jitter_savgol_poly: int = 2,
) -> dict[str, float | bool]:
    orient = np.asarray(global_orient_aa, dtype=np.float32)
    nframes = orient.shape[0]
    if nframes < 2:
        return {"is_unusable": False, "max_delta_deg": 0.0, "flip_ratio": 0.0, "jitter_ratio": 0.0}

    orient_t = torch.from_numpy(orient)
    rot = Rc.axis_angle_to_matrix(orient_t)
    rel = torch.matmul(rot[1:], rot[:-1].transpose(1, 2))
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    delta_deg = torch.acos(cos_angle) * (180.0 / math.pi)
    delta_deg_np = delta_deg.cpu().numpy()

    max_delta_deg = float(delta_deg_np.max()) if delta_deg_np.size > 0 else 0.0
    flip_ratio = float(np.mean(delta_deg_np >= flip_deg)) if delta_deg_np.size > 0 else 0.0

    vel_deg_s = delta_deg_np * max(float(fps), 1e-6)
    win = _get_valid_savgol_window(vel_deg_s.shape[0], jitter_savgol_window, jitter_savgol_poly)
    if win is None:
        vel_smooth = vel_deg_s
    else:
        vel_smooth = savgol_filter(vel_deg_s, win, jitter_savgol_poly, axis=0, mode="interp")
    residual = np.abs(vel_deg_s - vel_smooth)
    jitter_steps = (residual >= jitter_residual_deg_s) & (vel_deg_s >= jitter_min_vel_deg_s)
    jitter_ratio = float(np.mean(jitter_steps)) if jitter_steps.size > 0 else 0.0

    is_unusable = (
        (max_delta_deg >= hard_flip_deg)
        or (flip_ratio >= flip_ratio_thresh)
        or (jitter_ratio >= jitter_ratio_thresh)
    )
    return {
        "is_unusable": bool(is_unusable),
        "max_delta_deg": max_delta_deg,
        "flip_ratio": flip_ratio,
        "jitter_ratio": jitter_ratio,
    }


def _load_models(args, device: str):
    from miburi.models import loaders
    import smplx

    t0 = time.time()
    logger.info("Model init on {}: loading Moshi/Mimi...", device)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(loaders.DEFAULT_REPO)
    mimi = checkpoint_info.get_mimi()
    mimi.set_num_codebooks(args.mimi_codebooks)
    mimi = mimi.to(device).eval()

    text_tokenizer = checkpoint_info.get_text_tokenizer()
    interleaver = Interleaver(
        text_tokenizer,
        mimi.frame_rate,
        text_padding=3,
        end_of_text_padding=0,
        zero_padding=-1,
        keep_main_only=True,
        device=device,
    )
    interleaved_tokenizer = InterleavedTokenizer(mimi, interleaver)

    logger.info("Model init on {}: loading SMPL-X model...", device)
    smplx_model = smplx.create(
        args.smplx_model_path,
        model_type="smplx",
        gender=args.smplx_gender,
        flat_hand_mean=True,
        num_betas=args.smplx_num_betas,
        num_expression_coeffs=args.smplx_num_expression_coeffs,
        use_pca=False,
    )
    smplx_model = smplx_model.to(device).eval()
    smplx_model.device = device

    logger.info("Model init on {} complete in {:.1f}s", device, time.time() - t0)
    return {
        "mimi": mimi,
        "interleaved_tokenizer": interleaved_tokenizer,
        "smplx_model": smplx_model,
    }


def _process_entry(entry: FileEntry, args, models: dict[str, Any]) -> dict[str, Any]:
    smplx_model = models["smplx_model"]

    motion_data = np.load(entry.motion_path)
    smplx_body_pose = motion_data[args.pose_key]
    if smplx_body_pose.ndim != 2:
        raise ValueError(f"Expected pose shape (T, J*3); got {smplx_body_pose.shape}")

    smplx_transl = _maybe_get(motion_data, args.trans_key, (smplx_body_pose.shape[0], 3))
    smplx_expr = _maybe_get(
        motion_data,
        args.expr_key,
        (smplx_body_pose.shape[0], args.smplx_num_expression_coeffs),
    )
    smplx_betas = _maybe_get(motion_data, args.betas_key, (args.smplx_num_betas,))
    if smplx_betas.ndim == 2 and smplx_betas.shape[0] == 1:
        smplx_betas = smplx_betas[0]

    full_body_pose = smplx_body_pose.reshape(smplx_body_pose.shape[0], -1, 3)

    # global_orient = torch.from_numpy(full_body_pose[:, 0, :]).float()
    # global_orient_rot = Rc.axis_angle_to_matrix(global_orient)
    # rot_x_180 = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    # global_orient_rot = torch.matmul(rot_x_180, global_orient_rot)
    # global_orient = Rc.matrix_to_axis_angle(global_orient_rot).numpy()
    # full_body_pose[:, 0, :] = global_orient
    # smplx_body_pose[:, :3] = full_body_pose[:, 0, :]

    if args.global_orient_filter_enabled:
        quality = _assess_global_orient_quality(
            smplx_body_pose[:, :3],
            fps=args.motion_fps,
            flip_deg=args.global_orient_flip_deg,
            hard_flip_deg=args.global_orient_hard_flip_deg,
            flip_ratio_thresh=args.global_orient_flip_ratio,
            jitter_residual_deg_s=args.global_orient_jitter_residual_deg_s,
            jitter_min_vel_deg_s=args.global_orient_jitter_min_vel_deg_s,
            jitter_ratio_thresh=args.global_orient_jitter_ratio,
            jitter_savgol_window=args.global_orient_jitter_savgol_window,
            jitter_savgol_poly=args.global_orient_jitter_savgol_poly,
        )
        if bool(quality["is_unusable"]):
            return {
                "status": "skipped_quality",
                "file_id": entry.file_id,
                "metrics": quality,
            }

    # smplx_transl[:, 2] = 0.0
    smplx_transl = smplx_transl - smplx_transl[0:1, :]

    # joint_pos_pre = _compute_smplx_joints(
    #     smplx_body_pose,
    #     smplx_transl,
    #     smplx_model,
    #     betas=smplx_betas,
    #     batch_size=args.smplx_joints_batch_size,
    # )
    # sitting_bits_full = _compute_sitting_bits(
    #     joint_pos_pre,
    #     height_ratio_thresh=args.sitting_height_ratio,
    #     knee_deg_thresh=args.sitting_knee_deg,
    # )
    # file_is_sitting = bool(sitting_bits_full.mean() >= args.sitting_frame_ratio) if sitting_bits_full.size > 0 else False

    # R_corr = None
    # if args.upright_correction and not file_is_sitting:
    #     global_orient = torch.from_numpy(full_body_pose[:, 0, :]).float()
    #     global_orient_rot = Rc.axis_angle_to_matrix(global_orient)
    #     R_corr = _compute_upright_correction(
    #         global_orient_rot,
    #         tilt_deg_threshold=args.upright_tilt_deg,
    #         max_deg=args.upright_max_deg,
    #     )
    #     if R_corr is not None:
    #         global_orient_rot = torch.matmul(R_corr, global_orient_rot)
    #         global_orient = Rc.matrix_to_axis_angle(global_orient_rot).numpy()
    #         full_body_pose[:, 0, :] = global_orient
    #         smplx_body_pose[:, :3] = full_body_pose[:, 0, :]

    # if args.savgol_enabled:
    #     smplx_body_pose = _savgol_smooth_pose_6d(smplx_body_pose, args.savgol_window_pose, args.savgol_poly)
    #     smplx_transl = _savgol_smooth_transl(smplx_transl, args.savgol_window_trans, args.savgol_poly)

    full_body_pose = smplx_body_pose.reshape(smplx_body_pose.shape[0], -1, 3)

    if args.debug_visualize and args.debug_visualize_dir: # and R_corr is not None:
        try:
            os.makedirs(args.debug_visualize_dir, exist_ok=True)
            render_smplx_debug_video(
                smplx_model=smplx_model,
                poses=smplx_body_pose.astype(np.float32),
                transl=smplx_transl.astype(np.float32),
                expressions=smplx_expr.astype(np.float32),
                betas=None, #  smplx_betas.astype(np.float32),
                output_path=os.path.join(args.debug_visualize_dir, f"{entry.file_id}_debug.mp4"),
                fps=args.motion_fps,
                width=args.debug_visualize_width,
                height=args.debug_visualize_height,
                audio_path=entry.audio_path,
            )
        except Exception:
            pass

    nframes = full_body_pose.shape[0]
    # contacts = _compute_contacts(joint_pos_pre)
    joint_pos_post = _compute_smplx_joints(
        smplx_body_pose,
        smplx_transl,
        smplx_model,
        betas=None, #smplx_betas,
        batch_size=args.smplx_joints_batch_size,
    )
    contacts = _compute_contacts(joint_pos_post)

    # Embody3D motion npz ships per-frame tracking validity; trust it as ground
    # truth (matching the legacy embody3d builder semantics). Fall back to the
    # geometric pose_valid heuristic only if the field is missing.
    if "valid_frames" in motion_data.files:
        pose_valid = motion_data["valid_frames"].astype(bool)
        if pose_valid.shape[0] != nframes:
            pose_valid = compute_pose_valid(joint_pos_post, smplx_transl)
    else:
        pose_valid = compute_pose_valid(joint_pos_post, smplx_transl)

    
    # breakpoint()
    segments = _load_transcript(entry.text_path)
    # print(segments)
    vad_bits = _compute_vad_bits(segments, nframes, args.motion_fps)
    # print(vad_bits[:100])
    # breakpoint()

    audio, sr = _load_audio(entry.audio_path)
    sequence_lengthsec = min(audio.shape[1] / sr, nframes / args.motion_fps)
    chunk_lengthsec = args.pose_length / args.motion_fps
    if args.full_sequence:
        num_chunks = 1
        chunk_lengthsec = sequence_lengthsec
    else:
        num_chunks = int(math.floor(sequence_lengthsec / chunk_lengthsec))

    if num_chunks <= 0:
        return {"status": "skipped_short", "file_id": entry.file_id}

    cut_duration = chunk_lengthsec * num_chunks
    if args.process_multimodal_signals:
        mimi = models["mimi"]
        interleaved_tokenizer = models["interleaved_tokenizer"]

        alignments = _extract_alignments(segments)
        cut_audio = _resample_audio(audio, sr, args.mimi_audio_fps)
        cut_audio = cut_audio[:, : int(cut_duration * args.mimi_audio_fps)]

        text_tokens, audio_tokens = interleaved_tokenizer(cut_audio, 0, alignments, cut_duration)
        text_tokens = text_tokens.cpu().numpy()
        audio_tokens = audio_tokens.cpu().numpy()
    else:
        mimi = None
        text_tokens = None
        audio_tokens = None

    sitting_bits_full = np.zeros((nframes,), dtype=bool)

    chunks = []
    for chunk_idx in range(num_chunks):
        chunk_startsec = chunk_idx * chunk_lengthsec
        chunk_endsec = (chunk_idx + 1) * chunk_lengthsec

        start_frame = int(chunk_startsec * args.motion_fps)
        end_frame = int(chunk_endsec * args.motion_fps)
        if end_frame > nframes:
            break

        motion_chunk = full_body_pose[start_frame:end_frame]
        if motion_chunk.shape[0] < args.pose_length:
            logger.info(
                "Skipping chunk {} of file {} due to insufficient frames: got {}, need {}",
                chunk_idx,
                entry.file_id,
                motion_chunk.shape[0],
                args.pose_length,
            )
            continue

        # if pose_valid of this chunk is too low, skip it
        if pose_valid[start_frame:end_frame].mean() < args.pose_valid_frame_ratio:
            logger.info(
                "Skipping chunk {} of file {} due to low pose_valid ratio: got {:.2f}, need {:.2f}",
                chunk_idx,
                entry.file_id,
                pose_valid[start_frame:end_frame].mean(),
                args.pose_valid_frame_ratio,
            )
            continue

        # Yaw-align the chunk to canonical +Z so each stored chunk is heading-normalized.
        # transl and joint_positions are rotated by the same inverse first-frame yaw to
        # keep all spatial fields mutually consistent. Pose pitch/roll are preserved.
        # Mirrors the per-chunk alignment that used to live in the dataloader __getitem__.
        transl_chunk = smplx_transl[start_frame:end_frame]
        joint_pos_chunk = joint_pos_post[start_frame:end_frame]

        transl_chunk = transl_chunk - transl_chunk[0:1, :]

        motion_chunk_t = torch.from_numpy(motion_chunk.copy()).float()
        transl_chunk_t = torch.from_numpy(transl_chunk.copy()).float()
        joint_pos_chunk_t = torch.from_numpy(joint_pos_chunk.copy()).float()
        # Clone before mutation so joint_positions is rotated by the original orient.
        global_orient_orig = motion_chunk_t[:, :1, :].clone()
        global_orient_aligned, transl_chunk_t = zero_first_frame_pose_yaw_only(
            global_orient_orig, transl_chunk_t,
        )
        motion_chunk_t[:, :1, :] = global_orient_aligned
        joint_pos_chunk_t = align_points_to_first_frame_yaw_only(
            global_orient_orig, joint_pos_chunk_t,
        )
        motion_chunk = motion_chunk_t.numpy()
        transl_chunk = transl_chunk_t.numpy()
        joint_pos_chunk = joint_pos_chunk_t.numpy()

        filechunk_id = (
            f"{entry.file_id}_C{chunk_idx}" if not args.full_sequence else entry.file_id
        )

        if args.debug_visualize_chunks and args.debug_visualize_chunks_dir:
            try:
                # Flatten motion back to (T, 165) for the SMPLX forward inside the renderer.
                chunk_poses_flat = motion_chunk.reshape(motion_chunk.shape[0], -1).astype(np.float32)
                render_chunk_smplx_joints_side_by_side(
                    smplx_model=smplx_model,
                    poses=chunk_poses_flat,
                    transl=transl_chunk.astype(np.float32),
                    expressions=smplx_expr[start_frame:end_frame].astype(np.float32),
                    betas=None,
                    joints=joint_pos_chunk.astype(np.float32),
                    output_path=os.path.join(
                        args.debug_visualize_chunks_dir, f"{filechunk_id}_sbs.mp4"
                    ),
                    fps=args.motion_fps,
                    panel_width=args.debug_visualize_width,
                    panel_height=args.debug_visualize_height,
                    audio_path=entry.audio_path,
                    audio_start_sec=float(chunk_startsec),
                    audio_duration_sec=float(chunk_endsec - chunk_startsec),
                )
            except Exception as exc:
                logger.warning(
                    "Per-chunk debug visualization failed for {}: {}", filechunk_id, exc
                )

        # print(f"{entry.file_id}_C{chunk_idx}", entry.file_id, get_speaker_id(entry.file_id))
        chunk_payload: dict[str, Any] = {
            "filechunk_id": filechunk_id,
            "attrs": {
                "file_id": entry.file_id,
                "split": entry.split,
                "label": entry.label,
                "batch_idx": entry.batch_idx,
                "archive_idx": entry.archive_idx,
                "chunk_startsec": float(chunk_startsec),
                "chunk_endsec": float(chunk_endsec),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "speaker_id": get_speaker_id(entry.file_id),
                "is_sitting": False,
            },
            "datasets": {
                "motion": motion_chunk.astype(np.float32),
                "transl": transl_chunk.astype(np.float32),
                "contacts": contacts[start_frame:end_frame].astype(bool),
                "expressions": smplx_expr[start_frame:end_frame].astype(np.float32),
                "betas": smplx_betas.astype(np.float32),
                "pose_valid": pose_valid[start_frame:end_frame].astype(bool),
                "vad_bits": vad_bits[start_frame:end_frame].astype(int),
                "sitting_bits": (
                    sitting_bits_full[start_frame:end_frame]
                    if sitting_bits_full.size > 0
                    else np.zeros((end_frame - start_frame,), dtype=bool)
                ).astype(bool),
                "joint_positions": joint_pos_chunk.astype(np.float32),
            },
            "relpath": entry.relpath,
        }

        if args.process_multimodal_signals:
            chunk_text, _ = _get_text_for_chunk(segments, chunk_startsec, chunk_endsec)
            start_audio_token = int(chunk_startsec * mimi.frame_rate)
            end_audio_token = int(chunk_endsec * mimi.frame_rate)

            chunk_text_tokens = text_tokens[:, :, start_audio_token:end_audio_token]
            chunk_audio_tokens = audio_tokens[:, :, start_audio_token:end_audio_token]

            chunk_payload["attrs"]["text"] = chunk_text
            chunk_payload["datasets"]["moshi_texttokens"] = chunk_text_tokens.astype(int)
            chunk_payload["datasets"]["moshi_audiotokens"] = chunk_audio_tokens.astype(int)
            chunk_payload["chunk_bounds"] = [start_frame, end_frame, start_audio_token, end_audio_token]
        else:
            chunk_payload["chunk_bounds"] = [start_frame, end_frame]

        chunks.append(chunk_payload)

    return {
        "status": "ok",
        "file_id": entry.file_id,
        "chunks": chunks,
        "split": entry.split,
    }


def build_hdf5_database_parallel(args) -> dict[str, Any]:
    args.progress_log_every = max(1, int(args.progress_log_every))
    args.status_flush_every = max(1, int(args.status_flush_every))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.device = device
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    t_build_list = time.time()
    entries_all = _build_file_list(args)
    logger.info("File list loaded: {} candidate files ({:.2f}s)", len(entries_all), time.time() - t_build_list)
    logger.info(
        "Chunk index cache write after finalize: enabled={} cache_dir={}",
        bool(getattr(args, "write_chunk_index_cache", True)),
        str(getattr(args, "index_cache_dir", None)),
    )
    if not entries_all:
        logger.warning("No files found")
        return {"run_complete": True, "files_total": 0}

    all_file_ids = {e.file_id for e in entries_all}
    t_resume = time.time()
    resume_state = _load_resume_state(args.hdf5_path) if args.resume else None
    if args.resume:
        logger.info("Resume preload complete in {:.2f}s", time.time() - t_resume)
    file_status_map: dict[str, str] = {}
    if resume_state is not None:
        file_status_map.update(resume_state["file_status_map"])

    remaining_entries_all = [e for e in entries_all if e.file_id not in file_status_map]
    if args.resume and file_status_map:
        logger.info(
            "Resume enabled: skipping {} terminal-status files, remaining={}",
            len(file_status_map),
            len(remaining_entries_all),
        )

    if not remaining_entries_all:
        state_json_path = args.state_json_path or _default_state_json_path(args.hdf5_path)
        scoped_status = {k: v for k, v in file_status_map.items() if k in all_file_ids}
        payload = _build_run_state_payload(
            run_complete=True,
            hdf5_path=args.hdf5_path,
            files_total=len(entries_all),
            file_status_map=scoped_status,
            files_written=sum(1 for v in scoped_status.values() if v == "written"),
            skipped_quality=sum(1 for v in scoped_status.values() if v == "skipped_quality"),
            skipped_short=sum(1 for v in scoped_status.values() if v == "skipped_short"),
            skipped_empty=sum(1 for v in scoped_status.values() if v == "skipped_empty"),
            failed=sum(1 for v in scoped_status.values() if v == "failed"),
            processed_files=0,
            files_scanned_this_run=0,
            stop_reason="complete",
        )
        state_dir = os.path.dirname(state_json_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        _write_state_json(state_json_path, payload)
        logger.info("Nothing left to process. Output already up to date: {}", args.hdf5_path)
        return payload

    if args.max_files_per_run is not None and args.max_files_per_run > 0:
        entries = remaining_entries_all[: args.max_files_per_run]
    else:
        entries = remaining_entries_all

    files_scanned = len(entries)
    total_files = len(entries_all)
    state_json_path = args.state_json_path or _default_state_json_path(args.hdf5_path)
    hdf_dir = os.path.dirname(args.hdf5_path)
    if hdf_dir:
        os.makedirs(hdf_dir, exist_ok=True)
    state_dir = os.path.dirname(state_json_path)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    chunk_ids: list[str] = list(resume_state["chunk_ids"]) if resume_state else []
    chunk_splits: list[str] = list(resume_state["chunk_splits"]) if resume_state else []
    chunk_relpaths: list[str] = list(resume_state["chunk_relpaths"]) if resume_state else []
    chunk_bounds: list[list[int]] = list(resume_state["chunk_bounds"]) if resume_state else []
    chunk_speaker_ids: list[str] = list(resume_state["chunk_speaker_ids"]) if resume_state else []
    chunk_is_sitting: list[bool] = list(resume_state["chunk_is_sitting"]) if resume_state else []
    speaker_ids: set[str] = set(resume_state["speaker_ids"]) if resume_state else set()

    files_with_written_chunks = 0
    skipped_quality = 0
    skipped_short = 0
    skipped_empty = 0
    failed_files = 0
    processed_files = 0

    run_start_ts = time.time()
    deadline_ts = None
    if args.max_run_seconds is not None and args.max_run_seconds > 0:
        deadline_ts = run_start_ts + float(args.max_run_seconds) - float(max(0, args.graceful_shutdown_buffer_sec))
        if deadline_ts <= run_start_ts:
            deadline_ts = run_start_ts + 1.0

    stop_requested = False
    stop_reason: str | None = None

    def _request_stop(reason: str):
        nonlocal stop_requested, stop_reason
        if not stop_requested:
            stop_requested = True
            stop_reason = reason
            logger.warning("Graceful stop requested: {}", reason)

    def _signal_handler(signum, _frame):
        _request_stop(f"signal_{signum}")

    prev_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    prev_sigint = signal.signal(signal.SIGINT, _signal_handler)

    dt = h5py.string_dtype(encoding="utf-8")
    h5_mode = "a" if args.resume and os.path.exists(args.hdf5_path) else "w"
    models = _load_models(args, device)

    try:
        with h5py.File(args.hdf5_path, h5_mode) as f:
            if "created" not in f.attrs:
                f.attrs["created"] = time.time()
            f.attrs["total_files"] = total_files
            f.attrs["pose_length"] = args.pose_length
            f.attrs["motion_fps"] = args.motion_fps
            for entry in entries:
                if not stop_requested and deadline_ts is not None and time.time() >= deadline_ts:
                    _request_stop("time_budget")
                if stop_requested:
                    break

                result = _process_entry(entry, args, models)
                status = result.get("status")
                file_id = str(result.get("file_id", entry.file_id))
                terminal_status = None

                if status == "skipped_quality":
                    skipped_quality += 1
                    terminal_status = "skipped_quality"
                    m = result.get("metrics", {})
                    logger.warning(
                        "Skipping {} due to quality: max_delta={:.2f}, flip_ratio={:.4f}, jitter_ratio={:.4f}",
                        file_id,
                        float(m.get("max_delta_deg", 0.0)),
                        float(m.get("flip_ratio", 0.0)),
                        float(m.get("jitter_ratio", 0.0)),
                    )
                elif status == "skipped_short":
                    skipped_short += 1
                    terminal_status = "skipped_short"
                else:
                    chunks = result.get("chunks", [])
                    chunks_before = len(chunk_ids)
                    for c in chunks:
                        filechunk_id = c["filechunk_id"]
                        if filechunk_id in f:
                            continue
                        grp = f.create_group(filechunk_id)

                        attrs = c["attrs"]
                        grp.attrs["file_id"] = attrs["file_id"]
                        grp.attrs["split"] = attrs["split"]
                        grp.attrs["label"] = attrs["label"]
                        grp.attrs["batch_idx"] = attrs["batch_idx"]
                        grp.attrs["archive_idx"] = attrs["archive_idx"]
                        grp.attrs["chunk_startsec"] = attrs["chunk_startsec"]
                        grp.attrs["chunk_endsec"] = attrs["chunk_endsec"]
                        grp.attrs["start_frame"] = attrs["start_frame"]
                        grp.attrs["end_frame"] = attrs["end_frame"]
                        grp.attrs["speaker_id"] = str(attrs["speaker_id"]).encode("utf-8")
                        grp.attrs["is_sitting"] = bool(attrs["is_sitting"])
                        if "text" in attrs:
                            grp.attrs["text"] = (attrs["text"] or "").encode("utf-8")

                        for dname, darr in c["datasets"].items():
                            grp.create_dataset(dname, data=darr, compression="gzip")

                        chunk_ids.append(filechunk_id)
                        chunk_splits.append(attrs["split"])
                        chunk_relpaths.append(c["relpath"])
                        chunk_bounds.append(c["chunk_bounds"])
                        speaker_str = str(attrs["speaker_id"])
                        is_sitting_bool = bool(attrs["is_sitting"])
                        chunk_speaker_ids.append(speaker_str)
                        chunk_is_sitting.append(is_sitting_bool)
                        speaker_ids.add(speaker_str)

                    if len(chunk_ids) > chunks_before:
                        files_with_written_chunks += 1
                        terminal_status = "written"
                    else:
                        skipped_empty += 1
                        terminal_status = "skipped_empty"

                if terminal_status is not None and file_id:
                    file_status_map[file_id] = terminal_status

                processed_files += 1

                if processed_files % args.status_flush_every == 0:
                    _write_file_status_datasets(f, file_status_map)
                    f.attrs["last_status_flush"] = time.time()
                    f.flush()
                    scoped_status = {k: v for k, v in file_status_map.items() if k in all_file_ids}
                    partial_payload = _build_run_state_payload(
                        run_complete=False,
                        hdf5_path=args.hdf5_path,
                        files_total=total_files,
                        file_status_map=scoped_status,
                        files_written=files_with_written_chunks,
                        skipped_quality=skipped_quality,
                        skipped_short=skipped_short,
                        skipped_empty=skipped_empty,
                        failed=failed_files,
                        processed_files=processed_files,
                        files_scanned_this_run=files_scanned,
                        stop_reason=stop_reason,
                    )
                    _write_state_json(state_json_path, partial_payload)

                if processed_files % args.progress_log_every == 0 or processed_files == files_scanned:
                    logger.info(
                        "Progress | processed={}/{} | files_written={} | skipped_quality={} | skipped_short={} | skipped_empty={} | failed={}",
                        processed_files,
                        files_scanned,
                        files_with_written_chunks,
                        skipped_quality,
                        skipped_short,
                        skipped_empty,
                        failed_files,
                    )

            for ds_name in ("chunk_ids", "chunk_splits", "chunk_relpaths", "chunk_bounds", "chunk_speaker_ids", "chunk_is_sitting"):
                if ds_name in f:
                    del f[ds_name]
            f.create_dataset("chunk_ids", data=np.array(chunk_ids, dtype=object), dtype=dt)
            f.create_dataset("chunk_splits", data=np.array(chunk_splits, dtype=object), dtype=dt)
            f.create_dataset("chunk_relpaths", data=np.array(chunk_relpaths, dtype=object), dtype=dt)
            f.create_dataset("chunk_bounds", data=np.array(chunk_bounds, dtype=np.int64))
            f.create_dataset("chunk_speaker_ids", data=np.array(chunk_speaker_ids, dtype=object), dtype=dt)
            f.create_dataset("chunk_is_sitting", data=np.array(chunk_is_sitting, dtype=np.bool_))
            _write_file_status_datasets(f, file_status_map)
            f.attrs["num_speakers"] = len(speaker_ids)
            f.attrs["completed"] = time.time()
            f.attrs["total_chunks"] = len(chunk_ids)
            f.flush()
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    scoped_status = {k: v for k, v in file_status_map.items() if k in all_file_ids}
    files_remaining = max(0, total_files - len(scoped_status))
    run_complete = files_remaining == 0
    if run_complete:
        stop_reason = "complete"
    elif stop_reason is None and len(entries) < len(remaining_entries_all):
        stop_reason = "max_files_per_run"

    final_payload = _build_run_state_payload(
        run_complete=run_complete,
        hdf5_path=args.hdf5_path,
        files_total=total_files,
        file_status_map=scoped_status,
        files_written=files_with_written_chunks,
        skipped_quality=skipped_quality,
        skipped_short=skipped_short,
        skipped_empty=skipped_empty,
        failed=failed_files,
        processed_files=processed_files,
        files_scanned_this_run=files_scanned,
        stop_reason=stop_reason,
    )
    _write_state_json(state_json_path, final_payload)

    if bool(getattr(args, "write_chunk_index_cache", True)):
        try:
            cache_out = write_chunk_index_cache_from_metadata(
                hdf5_path=args.hdf5_path,
                chunk_ids=chunk_ids,
                chunk_splits=chunk_splits,
                chunk_relpaths=chunk_relpaths,
                chunk_speaker_ids=chunk_speaker_ids,
                chunk_is_sitting=chunk_is_sitting,
                index_cache_dir=getattr(args, "index_cache_dir", None),
            )
            logger.info(
                "Chunk index cache updated after HDF5 write: npz={} rows={}",
                cache_out["cache_npz_path"],
                int(cache_out["rows"]),
            )
        except Exception as exc:
            logger.exception("Failed to write chunk index cache from in-memory metadata: {}", exc)

    logger.info("Database update complete. Total chunks now: {}", len(chunk_ids))
    logger.info(
        "Run summary | files_scanned={} | processed_files={} | skipped_quality={} | skipped_short={} | skipped_empty={} | failed_files={} | files_with_written_chunks={} | files_remaining={} | run_complete={} | stop_reason={}",
        files_scanned,
        processed_files,
        skipped_quality,
        skipped_short,
        skipped_empty,
        failed_files,
        files_with_written_chunks,
        final_payload["files_remaining"],
        final_payload["run_complete"],
        final_payload["stop_reason"],
    )
    return final_payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/embody_3d_aiagent/")
    parser.add_argument("--hdf5_path", required=True)

    parser.add_argument("--pose_key", default="poses")
    parser.add_argument("--trans_key", default="trans")
    parser.add_argument("--expr_key", default="expressions")
    parser.add_argument("--betas_key", default="betas")

    parser.add_argument("--motion_fps", type=int, default=25)
    parser.add_argument("--mimi_audio_fps", type=int, default=24000)
    parser.add_argument("--pose_length", type=int, default=250)
    parser.add_argument("--full_sequence", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--filelist_progress_every", type=int, default=1000)

    # Splits are assigned positionally over the sorted glob of sample dirs:
    # train = first train_ratio, val = next val_ratio, test = remainder.
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    parser.add_argument("--mimi_codebooks", type=int, default=8)
    parser.add_argument("--process_multimodal_signals", action="store_true", default=False)

    parser.add_argument(
        "--smplx_model_path",
        default="assets_dep/smplx_2020/",
    )
    parser.add_argument("--smplx_gender", default="NEUTRAL_2020")
    parser.add_argument("--smplx_num_betas", type=int, default=300)
    parser.add_argument("--smplx_num_expression_coeffs", type=int, default=100)
    parser.add_argument("--smplx_joints_batch_size", type=int, default=1024)

    parser.add_argument("--savgol_enabled", action="store_true", default=False)
    parser.add_argument("--savgol_window_trans", type=int, default=19)
    parser.add_argument("--savgol_window_pose", type=int, default=7)
    parser.add_argument("--savgol_poly", type=int, default=2)

    parser.add_argument("--upright_correction", action="store_true", default=False)
    parser.add_argument("--upright_tilt_deg", type=float, default=5.0)
    parser.add_argument("--upright_max_deg", type=float, default=30.0)

    parser.add_argument("--global_orient_filter_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global_orient_flip_deg", type=float, default=20.0)
    parser.add_argument("--global_orient_hard_flip_deg", type=float, default=80.0)
    parser.add_argument("--global_orient_flip_ratio", type=float, default=0.01)
    parser.add_argument("--global_orient_jitter_residual_deg_s", type=float, default=90.0)
    parser.add_argument("--global_orient_jitter_min_vel_deg_s", type=float, default=30.0)
    parser.add_argument("--global_orient_jitter_ratio", type=float, default=0.1)
    parser.add_argument("--global_orient_jitter_savgol_window", type=int, default=11)
    parser.add_argument("--global_orient_jitter_savgol_poly", type=int, default=2)

    parser.add_argument("--sitting_height_ratio", type=float, default=0.70)
    parser.add_argument("--sitting_knee_deg", type=float, default=130.0)
    parser.add_argument("--sitting_frame_ratio", type=float, default=0.8)

    parser.add_argument("--pose_valid_frame_ratio", type=float, default=240/250)  # at least this ratio of frames in a chunk must be valid according to pose_valid bits to be included

    parser.add_argument("--debug_visualize", action="store_true", default=False)
    parser.add_argument("--debug_visualize_dir", default=None)
    parser.add_argument("--debug_visualize_width", type=int, default=840)
    parser.add_argument("--debug_visualize_height", type=int, default=1280)
    # Per-chunk side-by-side (SMPLX mesh + 2D joint trajectory) on the
    # post-alignment chunk data, written one mp4 per accepted chunk.
    parser.add_argument("--debug_visualize_chunks", action="store_true", default=False)
    parser.add_argument("--debug_visualize_chunks_dir", default=None)
    parser.add_argument("--progress_log_every", type=int, default=10)
    parser.add_argument("--max_files_per_run", type=int, default=40000)
    parser.add_argument("--max_run_seconds", type=int, default=216000)
    parser.add_argument("--graceful_shutdown_buffer_sec", type=int, default=600)
    parser.add_argument("--status_flush_every", type=int, default=200)
    parser.add_argument("--state_json_path", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write_chunk_index_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index_cache_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting HDF5 build with args: {}", vars(args))
    result = build_hdf5_database_parallel(args)
    if not bool(result.get("run_complete", False)):
        sys.exit(SAFE_INCOMPLETE_EXIT_CODE)
