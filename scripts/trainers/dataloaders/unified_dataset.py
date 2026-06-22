"""Unified multi-source dataset for the new HDF5 schema.

Reads HDF5 files produced by ``build_hdf5_beatx.py`` / ``build_hdf5_embody3d.py``
under ``datasets/data_cache/``. Supports BEATX and Embody3D today via the
``dataset_ratio`` selector; structured to extend to Seamless Interaction.

Key differences from ``EMBODY3DBEATXDataset``:
- Reads the in-HDF5 chunk index (``chunk_ids``, ``chunk_splits``,
  ``chunk_relpaths``, ``chunk_speaker_ids``, ``chunk_is_sitting``) instead of
  the legacy ``*_filechunkid.csv`` sidecar.
- Token keys ``moshi_texttokens`` / ``moshi_audiotokens`` in the HDF5 are
  exposed as ``text_tokens`` / ``audio_tokens`` in the output dict.
- Optional yaw-only first-frame alignment (default off; new HDF5s ship aligned).
- VC chunk-id rewrite is dropped (no VC HDF5s in this pipeline).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils import data

try:
    import sphn  # type: ignore
except Exception:  # pragma: no cover - sphn is optional at import time
    sphn = None  # type: ignore[assignment]

from .utils.chunk_index_cache import _decode_if_bytes, _load_chunk_metadata
from .utils.data_tools import (
    SMPLX_JOINT_NAMES,
    SMPLX_JOINT_NAMES_FACE,
    SMPLX_JOINT_NAMES_HANDS,
    SMPLX_JOINT_NAMES_LOWER,
    SMPLX_JOINT_NAMES_UPPER,
    align_points_to_first_frame_yaw_only,
    check_if_additional_split,
    get_beatx_spkids,
    get_embody3d_spkids,
    zero_first_frame_pose_yaw_only,
)


GOOD_SPK_IDS = ["wayne", "lawrence", "stewart", "solomon"]


@dataclass
class _UnifiedChunkRef:
    hdf5_path: str
    chunk_id: str
    relpath: str
    dataset_name: str
    spk_id: int
    lower_valid: int


class UNIFIEDDataset(data.Dataset):
    def __init__(
        self,
        args,
        split: str,
        only_motion: bool = False,
        dataset_ratio: str = "full_embody3d",
        tiny: bool = False,
        debug: bool = False,
        varying_frame_length: bool = True,
        ret_rawaudio: bool = False,
        ret_vad: bool = False,
        return_joint_positions: bool = False,
        align_first_frame_yaw: bool = False,
        runtime_quality_max_resample_attempts: int = 16,
    ):
        super().__init__()
        self.args = args
        self.loader_type = split

        self.beatx_cache_path = getattr(args, "beatx_cache_path", None)
        self.embody3d_cache_path = getattr(args, "embody3d_cache_path", None)
        self.beatx_data_dir = getattr(args, "beatx_data_path", None)
        self.embody3d_path = getattr(args, "embody3d_path", None)
        self.index_cache_dir = getattr(args, "index_cache_dir", None)

        self.motion_fps = args.motion_fps
        self.frame_chunk_size = getattr(args, "frame_chunk_size", 1)
        self.pose_length = args.pose_length
        self.max_chunk_lengthsec = self.pose_length / self.motion_fps

        body_part = args.body_part
        assert body_part in ["full", "upper", "lower", "face"], (
            "body_part must be one of ['full', 'upper', 'lower', 'face']"
        )
        self.body_part = body_part

        self.only_motion = only_motion
        self.dataset_ratio = dataset_ratio
        self.tiny = tiny
        self.debug = debug
        self.varying_frame_length = varying_frame_length
        self.ret_rawaudio = ret_rawaudio
        self.ret_vad = ret_vad
        self.return_joint_positions = return_joint_positions
        self.align_first_frame_yaw = align_first_frame_yaw
        self.runtime_quality_max_resample_attempts = max(1, int(runtime_quality_max_resample_attempts))

        if self.align_first_frame_yaw:
            logger.warning(
                "UNIFIEDDataset: align_first_frame_yaw=True. "
                "Use this only on HDF5s built before the alignment was moved into the "
                "builder; HDF5s under datasets/data_cache/* are already aligned and "
                "re-applying is a near-no-op."
            )

        self._setup_body_masks()
        self._load_chunk_ids()
        self._h5_handles: dict[str, h5py.File] = {}

        print(
            f"Loaded UNIFIEDDataset with {len(self._chunk_refs)} chunks in '{self.loader_type}' split "
            f"(dataset_ratio={self.dataset_ratio})."
        )

    # ------------------------------------------------------------------ setup
    def _setup_body_masks(self):
        smplx_joint_names = SMPLX_JOINT_NAMES
        self.smplx_joint_names = smplx_joint_names

        upper_mask = [1 if j in SMPLX_JOINT_NAMES_UPPER else 0 for j in smplx_joint_names]
        lower_mask = [1 if j in SMPLX_JOINT_NAMES_LOWER else 0 for j in smplx_joint_names]
        hands_mask = [1 if j in SMPLX_JOINT_NAMES_HANDS else 0 for j in smplx_joint_names]
        face_mask = [1 if j in SMPLX_JOINT_NAMES_FACE else 0 for j in smplx_joint_names]

        self.upper_mask = np.array(upper_mask)
        self.lower_mask = np.array(lower_mask)
        self.hands_mask = np.array(hands_mask)
        self.face_mask = np.array(face_mask)

        self.upper_mask_for_flattened = self.upper_mask.repeat(3, axis=0)
        self.lower_mask_for_flattened = self.lower_mask.repeat(3, axis=0)
        self.hands_mask_for_flattened = self.hands_mask.repeat(3, axis=0)
        self.face_mask_for_flattened = self.face_mask.repeat(3, axis=0)

    def _load_chunk_ids(self):
        # Per-source dataframes built from the in-HDF5 chunk index.
        df_beatx = self._build_source_df(
            self.beatx_cache_path, dataset_name="BEATX", lower_valid_default=1
        )
        df_embody3d = self._build_source_df(
            self.embody3d_cache_path, dataset_name="Embody3D", lower_valid_default=0
        )

        # BEATX-specific speaker filters (idempotent with the build-time filter).
        if df_beatx is not None and not df_beatx.empty:
            df_beatx = self._apply_beatx_lowervalid_filters(df_beatx)

        # Combine according to dataset_ratio.
        self.chunk_id_df = self._dispatch_dataset_ratio(df_beatx, df_embody3d)

        assert self.chunk_id_df["split"].isin(["train", "val", "test"]).all(), (
            "All split values must be 'train', 'val', or 'test'"
        )

        # Speaker remap: anchor on BEATX-lowervalid speakers, then the rest sorted.
        if df_beatx is not None and not df_beatx.empty:
            beatx_lowervalid_spk = (
                df_beatx[df_beatx["lower_valid"] == 1]["spk_id"].unique().tolist()
            )
        else:
            beatx_lowervalid_spk = []

        other_speaker_ids = sorted(
            spk for spk in self.chunk_id_df["spk_id"].unique().tolist()
            if spk not in beatx_lowervalid_spk
        )
        speaker_ids = beatx_lowervalid_spk + other_speaker_ids
        self.speaker_id_to_index = {s: i for i, s in enumerate(speaker_ids)}
        logger.info(
            f"UNIFIEDDataset: found {len(speaker_ids)} unique speakers "
            f"(dataset_ratio={self.dataset_ratio})."
        )
        self.chunk_id_df["spk_id"] = (
            self.chunk_id_df["spk_id"].map(self.speaker_id_to_index).astype(int)
        )

        if self.body_part == "lower":
            self.chunk_id_df = self.chunk_id_df[self.chunk_id_df["lower_valid"] == 1].reset_index(drop=True)

        

        
        
        # 
        self.chunk_id_df = self.chunk_id_df[
            self.chunk_id_df["split"] == self.loader_type
        ].reset_index(drop=True)
        #

        if self.tiny:
            self.chunk_id_df = self.chunk_id_df.iloc[:100].reset_index(drop=True)
        if self.debug:
            cap = 1000 if self.loader_type == "train" else 100
            self.chunk_id_df = self.chunk_id_df.iloc[:cap].reset_index(drop=True)

        self._chunk_refs: list[_UnifiedChunkRef] = [
            _UnifiedChunkRef(
                hdf5_path=row["hdf5_path"],
                chunk_id=row["filechunk_id"],
                relpath=row["relpath"],
                dataset_name=row["dataset_name"],
                spk_id=int(row["spk_id"]),
                lower_valid=int(row["lower_valid"]),
            )
            for _, row in self.chunk_id_df.iterrows()
        ]

    def _build_source_df(
        self,
        hdf5_path: Optional[str],
        *,
        dataset_name: str,
        lower_valid_default: int,
    ) -> Optional[pd.DataFrame]:
        if not hdf5_path:
            return None
        if not os.path.exists(hdf5_path):
            logger.warning(
                f"UNIFIEDDataset: {dataset_name} HDF5 not found at {hdf5_path}; skipping source."
            )
            return None

        table = _load_chunk_metadata(
            hdf5_path,
            index_cache_mode="auto",
            index_cache_dir=self.index_cache_dir,
        )
        if table.num_rows == 0:
            return None

        df = pd.DataFrame({
            "filechunk_id": list(table.chunk_ids),
            "split": list(table.chunk_splits),
            "relpath": list(table.chunk_relpaths),
            "hdf5_path": [hdf5_path] * table.num_rows,
            "dataset_name": [dataset_name] * table.num_rows,
            "lower_valid": [lower_valid_default] * table.num_rows,
        })
        # `dev` was used in some build code paths; normalize to `val` for the dataset.
        df.loc[df["split"] == "dev", "split"] = "val"
        if dataset_name == "BEATX":
            df["spk_id"] = df["filechunk_id"].apply(get_beatx_spkids)
        elif dataset_name == "Embody3D":
            df["spk_id"] = df["filechunk_id"].apply(get_embody3d_spkids)
        else:
            df["spk_id"] = list(table.speaker_ids)
        return df

    @staticmethod
    def _apply_beatx_lowervalid_filters(df: pd.DataFrame) -> pd.DataFrame:
        drop_all = {"itoi", "carla"}
        flag_additional = {"scott", "sophie", "lawrence", "daiki"}

        is_additional = df["filechunk_id"].apply(check_if_additional_split)
        df = df[~((df["spk_id"] == "hailing") & is_additional)].reset_index(drop=True)

        is_additional = df["filechunk_id"].apply(check_if_additional_split)
        df.loc[df["spk_id"].isin(flag_additional) & is_additional, "lower_valid"] = 0
        df.loc[df["spk_id"].isin(drop_all), "lower_valid"] = 0
        return df

    def _dispatch_dataset_ratio(
        self,
        df_beatx: Optional[pd.DataFrame],
        df_embody3d: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        ratio = self.dataset_ratio

        def _need_beatx():
            if df_beatx is None or df_beatx.empty:
                raise ValueError(
                    f"dataset_ratio={ratio!r} requires BEATX but --beatx_cache_path is unset or empty."
                )

        def _need_embody3d():
            if df_embody3d is None or df_embody3d.empty:
                raise ValueError(
                    f"dataset_ratio={ratio!r} requires Embody3D but --embody3d_cache_path is unset or empty."
                )

        if ratio == "full_beatx":
            _need_beatx()
            return df_beatx.reset_index(drop=True)
        if ratio == "full_beatx_lowervalid":
            _need_beatx()
            return df_beatx[df_beatx["lower_valid"] == 1].reset_index(drop=True)
        if ratio == "goodspk_beatx_lowervalid":
            _need_beatx()
            return df_beatx[
                df_beatx["spk_id"].isin(GOOD_SPK_IDS) & (df_beatx["lower_valid"] == 1)
            ].reset_index(drop=True)
        if ratio == "full_beatx_eval":
            _need_beatx()
            return df_beatx.reset_index(drop=True)
        if ratio == "full_beatx_fulllength":
            _need_beatx()
            return df_beatx[~df_beatx["spk_id"].isin(["carla", "itoi"])].reset_index(drop=True)
        if ratio == "scott_beatx_fulllength":
            _need_beatx()
            return df_beatx[df_beatx["spk_id"] == "scott"].reset_index(drop=True)
        if ratio == "goodspk_beatx_fulllength":
            _need_beatx()
            return df_beatx[df_beatx["spk_id"].isin(GOOD_SPK_IDS)].reset_index(drop=True)
        if ratio == "scott_beatx_lowervalid":
            _need_beatx()
            return df_beatx[
                (df_beatx["spk_id"] == "scott") & (df_beatx["lower_valid"] == 1)
            ].reset_index(drop=True)

        if ratio == "full_embody3d":
            _need_embody3d()
            return df_embody3d.reset_index(drop=True)

        if ratio == "66embody_33beatx":
            _need_beatx()
            _need_embody3d()
            emb_len = len(df_embody3d)
            beatx_len = len(df_beatx)
            desired_emb = beatx_len * 2
            if desired_emb <= emb_len:
                emb_part = df_embody3d.iloc[:desired_emb].reset_index(drop=True)
            else:
                oversample = desired_emb // emb_len + 1
                emb_part = pd.concat([df_embody3d] * oversample, ignore_index=True)
                emb_part = emb_part.sample(n=desired_emb, random_state=42).reset_index(drop=True)
            return pd.concat([emb_part, df_beatx], ignore_index=True)
        if ratio == "33embody_66beatx":
            _need_beatx()
            _need_embody3d()
            emb_len = len(df_embody3d)
            beatx_len = len(df_beatx)
            desired_beatx = emb_len * 2
            if desired_beatx <= beatx_len:
                beatx_part = df_beatx.iloc[:desired_beatx].reset_index(drop=True)
            else:
                oversample = desired_beatx // beatx_len + 1
                beatx_part = pd.concat([df_beatx] * oversample, ignore_index=True)
                beatx_part = beatx_part.sample(n=desired_beatx, random_state=42).reset_index(drop=True)
            return pd.concat([df_embody3d, beatx_part], ignore_index=True)
        if ratio == "50embody_50beatx":
            _need_beatx()
            _need_embody3d()
            # Approximate 50/50 by oversampling the smaller pool to match the larger one.
            emb_len = len(df_embody3d)
            beatx_len = len(df_beatx)
            if emb_len < beatx_len:
                oversample = beatx_len // emb_len + 1
                emb_part = pd.concat([df_embody3d] * oversample, ignore_index=True)
                emb_part = emb_part.sample(n=beatx_len, random_state=42).reset_index(drop=True)
                return pd.concat([emb_part, df_beatx], ignore_index=True)
            else:
                oversample = emb_len // max(1, beatx_len) + 1
                beatx_part = pd.concat([df_beatx] * oversample, ignore_index=True)
                beatx_part = beatx_part.sample(n=emb_len, random_state=42).reset_index(drop=True)
                return pd.concat([df_embody3d, beatx_part], ignore_index=True)

        raise ValueError(
            f"Unsupported dataset_ratio={ratio!r}. See unified_dataset.py:_dispatch_dataset_ratio."
        )

    # -------------------------------------------------------------- handles
    def _get_h5(self, path: str) -> h5py.File:
        f = self._h5_handles.get(path)
        if f is not None:
            return f
        try:
            f = h5py.File(path, "r", swmr=True)
        except Exception:
            f = h5py.File(path, "r")
        self._h5_handles[path] = f
        return f

    def close(self):
        for f in self._h5_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_handles.clear()

    # ----------------------------------------------------- resample on bad chunk
    def _chunk_is_valid(self, idx: int) -> bool:
        ref = self._chunk_refs[idx]
        try:
            f = self._get_h5(ref.hdf5_path)
            grp = f[ref.chunk_id]
            if "pose_valid" in grp:
                if not bool(np.asarray(grp["pose_valid"][:]).all()):
                    return False
            if "motion" in grp:
                m = np.asarray(grp["motion"][:])
                if not np.isfinite(m).all():
                    return False
            return True
        except Exception:
            return False

    def _resolve_runtime_quality_idx(self, idx: int) -> int:
        n = len(self._chunk_refs)
        if n == 0:
            raise IndexError("UNIFIEDDataset has no chunks for this split.")
        candidate = int(idx) % n
        for _ in range(self.runtime_quality_max_resample_attempts):
            if self._chunk_is_valid(candidate):
                return candidate
            candidate = int(np.random.randint(0, n))
        logger.warning(
            f"UNIFIEDDataset: exhausted resample attempts for idx={idx}; returning last candidate."
        )
        return candidate

    # ----------------------------------------------------- dunder methods
    def __len__(self) -> int:
        return len(self._chunk_refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # CLAUDE UNCOMMENT START --- IGNORE ---
        # if not self.dataset_ratio.endswith("fulllength"):
            # idx = self._resolve_runtime_quality_idx(int(idx))
        # CLAUDE UNCOMMENT END --- IGNORE ---
        ref = self._chunk_refs[idx]
        f = self._get_h5(ref.hdf5_path)
        grp = f[ref.chunk_id]

        motion = torch.from_numpy(grp["motion"][:]).float()
        transl = torch.from_numpy(grp["transl"][:]).float()

        if "fulllength" not in self.dataset_ratio:
            assert motion.shape[0] == self.pose_length, (
                f"Expected motion length {self.pose_length}, got {motion.shape[0]} at {ref.chunk_id}"
            )
        assert motion.shape[0] == transl.shape[0], (
            f"motion / transl frame mismatch at {ref.chunk_id}: {motion.shape[0]} vs {transl.shape[0]}"
        )
        pose_valid = torch.from_numpy(grp["pose_valid"][:]).bool()
        # print(self.dataset_ratio)
        # CLAUDE UNCOMMENT START --- IGNORE ---
        # if not self.dataset_ratio.endswith("fulllength"):
            # assert pose_valid.all(), f"pose_valid has False frames at {ref.chunk_id}"
        # CLAUDE UNCOMMENT END --- IGNORE ---

        # Clone original global orient: needed both for joint-position alignment
        # below and as a record of what we mutated when alignment is on.
        global_orient_orig = motion[:, :1, :].clone()
        if self.align_first_frame_yaw:
            go_aligned, transl = zero_first_frame_pose_yaw_only(global_orient_orig, transl)
            motion[:, :1, :] = go_aligned

        joint_positions = None
        if self.return_joint_positions and "joint_positions" in grp:
            joint_positions = torch.from_numpy(grp["joint_positions"][:]).float()
            if self.align_first_frame_yaw:
                joint_positions = align_points_to_first_frame_yaw_only(
                    global_orient_orig, joint_positions
                )

        vad_bits = torch.from_numpy(grp["vad_bits"][:]).long() if "vad_bits" in grp else None
        shape_params = torch.from_numpy(grp["betas"][:]).float()
        if shape_params.ndim == 1:
            shape_params = shape_params[None, :].expand(len(motion), -1)

        contacts = None
        expressions = None
        if self.body_part in ["lower", "full"] and "contacts" in grp:
            contacts = torch.from_numpy(grp["contacts"][:]).float()
        if self.body_part in ["face", "full"] and "expressions" in grp:
            expressions = torch.from_numpy(grp["expressions"][:]).float()

        file_id = _decode_if_bytes(grp.attrs.get("file_id", ""))
        split = _decode_if_bytes(grp.attrs.get("split", ""))
        chunk_startsec = float(grp.attrs.get("chunk_startsec", 0.0))
        chunk_endsec = float(grp.attrs.get("chunk_endsec", 0.0))

        if not self.only_motion:
            audio_tokens = torch.from_numpy(grp["moshi_audiotokens"][:]).long().squeeze(0)
            text_tokens = torch.from_numpy(grp["moshi_texttokens"][:]).long().squeeze(0)
            text = _decode_if_bytes(grp.attrs.get("text", ""))
            spk_id = torch.tensor(ref.spk_id, dtype=torch.long)
        else:
            audio_tokens = None
            text_tokens = None
            text = None
            spk_id = None

        raw_audio = None
        if self.ret_rawaudio and not self.only_motion:
            raw_audio = self._load_raw_audio(ref, file_id, chunk_startsec, chunk_endsec)

        if torch.any(torch.isnan(motion)):
            raise ValueError(f"NaN in motion at {ref.chunk_id}")

        motion_upper = motion[:, self.upper_mask == 1, :]
        motion_lower = motion[:, self.lower_mask == 1, :]
        motion_hands = motion[:, self.hands_mask == 1, :]
        motion_face = motion[:, self.face_mask == 1, :]

        return {
            "file_id": file_id,
            "filechunk_id": ref.chunk_id,
            "chunk_startsec": chunk_startsec,
            "chunk_endsec": chunk_endsec,
            "split": split,
            "motion": motion,
            "motion_upper": motion_upper,
            "motion_lower": motion_lower,
            "motion_hands": motion_hands,
            "motion_face": motion_face,
            "beta": shape_params,
            "transl": transl,
            "contact": contacts,
            "expressions": expressions,
            "text": text,
            "audio": raw_audio,
            "audio_tokens": audio_tokens,
            "text_tokens": text_tokens,
            "joint_positions": joint_positions,
            "spk_id": spk_id,
            "vad_bits": vad_bits,
            "lower_valid_mask": torch.tensor(int(ref.lower_valid), dtype=torch.int),
            "dataset_name": ref.dataset_name,
        }

    # ----------------------------------------------------- raw audio
    def _load_raw_audio(
        self,
        ref: _UnifiedChunkRef,
        file_id: str,
        chunk_startsec: float,
        chunk_endsec: float,
    ) -> Optional[np.ndarray]:
        if sphn is None:
            raise RuntimeError("sphn is required for raw-audio loading but is not installed.")

        if ref.dataset_name == "BEATX":
            if not self.beatx_data_dir:
                raise ValueError("--beatx_data_path is required to load BEATX raw audio.")
            audio_path = os.path.join(self.beatx_data_dir, "wave16k", f"{file_id}.wav")
        elif ref.dataset_name == "Embody3D":
            if not self.embody3d_path:
                raise ValueError("--embody3d_path is required to load Embody3D raw audio.")
            sample_path = file_id.replace("+", "/")
            base = sample_path.split("/")[0]
            audio_path = os.path.join(self.embody3d_path, sample_path, "audio_separated", f"{base}.wav")
        else:
            raise ValueError(f"Unsupported dataset_name for raw audio: {ref.dataset_name}")

        audio_each_file, sr = sphn.read(audio_path)
        audio_fps = getattr(self.args, "audio_fps", sr)
        if audio_fps < sr and sr % audio_fps == 0:
            audio_each_file = audio_each_file[:, :: sr // audio_fps]
        elif audio_fps != sr:
            audio_each_file = sphn.resample(audio_each_file, sr, audio_fps)

        start = int(chunk_startsec * audio_fps)
        end = int(chunk_endsec * audio_fps)
        return audio_each_file[0, start:end]

    # ------------------------------------------------------------ collate
    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if "fulllength" in self.dataset_ratio:
            assert len(batch) == 1, "Batch size must be 1 for fulllength dataset_ratio"

        stacked_body_pose = torch.stack([item["motion"] for item in batch], dim=0)
        stacked_body_pose_upper = torch.stack([item["motion_upper"] for item in batch], dim=0)
        stacked_body_pose_lower = torch.stack([item["motion_lower"] for item in batch], dim=0)
        stacked_body_pose_hands = torch.stack([item["motion_hands"] for item in batch], dim=0)
        stacked_body_pose_face = torch.stack([item["motion_face"] for item in batch], dim=0)
        stacked_transl = torch.stack([item["transl"] for item in batch], dim=0)
        stacked_contacts = (
            torch.stack([item["contact"] for item in batch], dim=0)
            if self.body_part in ["lower", "full"] and batch[0]["contact"] is not None
            else None
        )
        stacked_expressions = (
            torch.stack([item["expressions"] for item in batch], dim=0)
            if self.body_part in ["face", "full"] and batch[0]["expressions"] is not None
            else None
        )
        stacked_betas = torch.stack([item["beta"] for item in batch], dim=0)
        stacked_joints = (
            torch.stack([item["joint_positions"] for item in batch], dim=0)
            if batch[0]["joint_positions"] is not None
            else None
        )

        if self.only_motion and "fulllength" not in self.dataset_ratio:
            min_frame_chunk_size = self.frame_chunk_size
            if self.varying_frame_length:
                rand_multiplier = np.random.randint(
                    1, ((self.max_chunk_lengthsec * self.motion_fps) // min_frame_chunk_size) + 1
                )
                rand_multiplier = (
                    rand_multiplier
                    if np.random.rand() < 0.75
                    else int((self.max_chunk_lengthsec * self.motion_fps) // min_frame_chunk_size)
                )
            else:
                rand_multiplier = int((self.max_chunk_lengthsec * self.motion_fps) // min_frame_chunk_size)

            rand_chunk_length = self.frame_chunk_size * rand_multiplier
            if rand_chunk_length == self.max_chunk_lengthsec * self.motion_fps:
                rand_start_point = 0
            else:
                rand_start_point = np.random.randint(
                    0, (self.max_chunk_lengthsec * self.motion_fps) - rand_chunk_length
                )

            stacked_body_pose = stacked_body_pose[:, rand_start_point : rand_start_point + rand_chunk_length, :]
            stacked_body_pose_upper = stacked_body_pose_upper[
                :, rand_start_point : rand_start_point + rand_chunk_length, :
            ]
            stacked_body_pose_lower = stacked_body_pose_lower[
                :, rand_start_point : rand_start_point + rand_chunk_length, :
            ]
            stacked_body_pose_hands = stacked_body_pose_hands[
                :, rand_start_point : rand_start_point + rand_chunk_length, :
            ]
            stacked_body_pose_face = stacked_body_pose_face[
                :, rand_start_point : rand_start_point + rand_chunk_length, :
            ]
            stacked_transl = stacked_transl[:, rand_start_point : rand_start_point + rand_chunk_length, :]
            if stacked_contacts is not None:
                stacked_contacts = stacked_contacts[
                    :, rand_start_point : rand_start_point + rand_chunk_length, :
                ]
            if stacked_expressions is not None:
                stacked_expressions = stacked_expressions[
                    :, rand_start_point : rand_start_point + rand_chunk_length, :
                ]
            if stacked_joints is not None:
                stacked_joints = stacked_joints[
                    :, rand_start_point : rand_start_point + rand_chunk_length, ...
                ]
            stacked_betas = stacked_betas[:, rand_start_point : rand_start_point + rand_chunk_length, :]

        bs, nframes, _, _ = stacked_body_pose.shape
        stacked_body_pose = stacked_body_pose.reshape(bs, nframes, -1)
        stacked_body_pose_upper = stacked_body_pose_upper.reshape(bs, nframes, -1)
        stacked_body_pose_lower = stacked_body_pose_lower.reshape(bs, nframes, -1)
        stacked_body_pose_hands = stacked_body_pose_hands.reshape(bs, nframes, -1)
        stacked_body_pose_face = stacked_body_pose_face.reshape(bs, nframes, -1)

        if not self.only_motion:
            audio_tokens = torch.stack([item["audio_tokens"] for item in batch], dim=0)
            text_tokens = torch.stack([item["text_tokens"] for item in batch], dim=0)
            spk_ids = torch.stack([item["spk_id"] for item in batch], dim=0)
            vad_bits = (
                torch.stack([item["vad_bits"] for item in batch], dim=0)
                if batch[0]["vad_bits"] is not None
                else None
            )
        else:
            audio_tokens = None
            text_tokens = None
            spk_ids = None
            vad_bits = None

        lower_valid_mask = torch.stack([item["lower_valid_mask"] for item in batch], dim=0)

        return {
            "file_id": [item["file_id"] for item in batch],
            "filechunk_id": [item["filechunk_id"] for item in batch],
            "chunk_startsec": [item["chunk_startsec"] for item in batch],
            "chunk_endsec": [item["chunk_endsec"] for item in batch],
            "split": [item["split"] for item in batch],
            "motion": stacked_body_pose,
            "motion_upper": stacked_body_pose_upper,
            "motion_lower": stacked_body_pose_lower,
            "motion_hands": stacked_body_pose_hands,
            "motion_face": stacked_body_pose_face,
            "beta": stacked_betas,
            "transl": stacked_transl,
            "contact": stacked_contacts,
            "expressions": stacked_expressions,
            "raw_audio": (
                np.stack([item["audio"] for item in batch], axis=0)
                if (not self.only_motion) and self.ret_rawaudio and batch[0]["audio"] is not None
                else None
            ),
            "raw_text": [item["text"] for item in batch] if not self.only_motion else None,
            "audio_tokens": audio_tokens,
            "text_tokens": text_tokens,
            "joint_positions": stacked_joints,
            "speaker_id": spk_ids,
            "vad_bits": vad_bits,
            "lower_valid_mask": lower_valid_mask,
            "dataset_name": [item["dataset_name"] for item in batch],
        }

    @staticmethod
    def inverse_selection_tensor(filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
