# Datasets

This directory holds the raw source datasets MIBURI trains on, plus the HDF5 caches the unified loader actually reads at training and evaluation time.

> **You only need this directory populated for training and evaluation.** Running the released demo (`miburi-demo`) does **not** require any dataset — the published checkpoints under [`experiments/`](../experiments/) ship everything the realtime server needs.

---

## Layout

```
datasets/
├── beat_english_v2.0.0/         # raw BEAT2 (English) checkout
├── embody_3d_aiagent/           # raw Embody3D dyadic checkout
└── data_cache/                  # HDF5 caches produced by the builders below
    ├── beatx/                   # ← training cache (fixed-length chunks)
    ├── beatx_eval/              # ← evaluation cache (full sequences)
    ├── embody3d_dyadic/         # ← training cache
    └── embody3d_dyadic_eval/    # ← evaluation cache
```

Two source datasets are supported: **BEATX (BEAT2 English)** and **embody3d_dyadic**. The training-time loader is **unified** — it reads one or both HDF5 caches and mixes them according to the `dataset_ratio` field in your training config (e.g. `full_beatx`, `33embody_66beatx`, `goodspk_beatx_lowervalid`). The build-whichever-you-have, mix-at-train-time pattern means you can start with a single dataset and add more later without retraining the codecs.

---

## 1. Place the raw datasets

### BEATX (BEAT2 English)

Two ways to populate [`datasets/beat_english_v2.0.0/`](beat_english_v2.0.0/):

**Option A — fetch from Hugging Face (recommended).** A fork at [`m-hamza-mughal/beat2-additional-annotations`](https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations) combines the upstream [`H-Liu1997/BEAT2`](https://huggingface.co/datasets/H-Liu1997/BEAT2) data with the additional annotations contributed by the MIBURI + [RAG-Gesture](https://vcai.mpi-inf.mpg.de/projects/RAG-Gesture/) projects. The MIBURI repo ships a one-line download for the 4 subdirs the pipeline actually reads (~16.6 GB):

```bash
miburi-download-beatx-dataset
```

**Option B — bring your own BEAT2-English checkout.** Point it at [`datasets/beat_english_v2.0.0/`](beat_english_v2.0.0/) directly, with the layout below. The MIBURI annotations (`smplxflame_25/`, `whisper_transcription/`) aren't in upstream BEAT2 — you'll have to either download those subdirs from the Hugging Face fork above (e.g. via `huggingface-cli download m-hamza-mughal/beat2-additional-annotations --include 'beat_english_v2.0.0/smplxflame_25/*'`) or generate them yourself from the 30 fps source.

```
datasets/beat_english_v2.0.0/
├── smplxflame_25/               # per-take SMPL-X motion .npz @ 25 fps
├── wave16k/                     # per-take 16 kHz mono audio .wav
├── whisper_transcription/       # per-take Whisper transcripts .json
├── weights/                     # CNN motion autoencoder weights for FGD
└── train_test_split.csv         # official split file (BEAT2)
```

By downloading these files you agree to honor the upstream licenses: BEAT2 is Apache 2.0. Citation guidance lives in the [HF dataset card](https://huggingface.co/datasets/m-hamza-mughal/beat2-additional-annotations).

### Embody3D dyadic (AIAgent)

Point your Embody3D AIAgent checkout at [`datasets/embody_3d_aiagent/`](embody_3d_aiagent/). The expected layout is per-speaker, with separated audio and motion `.npz`s inside:

```
datasets/embody_3d_aiagent/
├── <speaker_id>/
│   ├── audio_separated/         # per-utterance .wav
│   └── <motion>.npz             # SMPL-X motion
└── ...
```

<!-- ### Seamless Interaction *(optional, not used by the released demo)*

Seamless Interaction is wired through `scripts/trainers/dataloaders/seamlessinteraction/` for ablation runs. The released MIBURI checkpoints are not trained on it, so you can skip this for the standard recipe. -->

---

## 2. Build HDF5 caches

MIBURI's unified loader reads HDF5 caches, not raw files. Two cache flavours per dataset are needed:

| Cache | Used by | Built with |
| --- | --- | --- |
| **training** | `scripts/train.py` | default builder invocation (fixed-length chunks of `--pose_length`) |
| **evaluation** | `scripts/test.py` | adds `--full_sequence` (one HDF5 entry per source file) + `--process_multimodal_signals` (precomputes audio/text payloads the metric loops consume) |

Build whichever subset you need.

### Training caches (fixed-length chunks)

**BEATX:**

```bash
python -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx \
    --data_dir datasets/beat_english_v2.0.0/ \
    --file_list_path datasets/beat_english_v2.0.0/train_test_split.csv \
    --hdf5_path datasets/data_cache/beatx/database.hdf5 \
    --motion_fps 25 \
    --pose_length 250
```

**Embody3D dyadic:**

```bash
python -m scripts.trainers.dataloaders.embody3d_dyadic.build_hdf5_embody3d \
    --data_dir datasets/embody_3d_aiagent/ \
    --hdf5_path datasets/data_cache/embody3d_dyadic/database.hdf5 \
    --motion_fps 25 \
    --pose_length 250
```

### Evaluation caches (full-sequence variant)

`scripts/test.py` operates on **full sequences**, not the fixed-length `pose_length` chunks training uses. Two extra flags change the cache layout:

- `--full_sequence` — keeps each source file as one HDF5 entry (no `_C<chunk>` suffixes), so eval iterates over whole takes.
- `--process_multimodal_signals` — pre-computes the audio + text alignment payloads consumed by the metric loops (BeatAlign, FGD, L1-Div, Facial-MSE).
- `--pose_length 250` is reduced to a *minimum-length filter* in this mode rather than a chunk size.

**BEATX:**

```bash
python -m scripts.trainers.dataloaders.beatx.build_hdf5_beatx \
    --data_dir datasets/beat_english_v2.0.0/ \
    --file_list_path datasets/beat_english_v2.0.0/train_test_split.csv \
    --hdf5_path datasets/data_cache/beatx_eval/database.hdf5 \
    --full_sequence \
    --motion_fps 25 \
    --pose_length 250 \
    --process_multimodal_signals
```

**Embody3D dyadic:**

```bash
python -m scripts.trainers.dataloaders.embody3d_dyadic.build_hdf5_embody3d \
    --data_dir datasets/embody_3d_aiagent/ \
    --hdf5_path datasets/data_cache/embody3d_dyadic_eval/database.hdf5 \
    --full_sequence \
    --motion_fps 25 \
    --pose_length 250 \
    --process_multimodal_signals
```

The released demo checkpoints' YAMLs ship with `dataset_ratio` values ending in `_eval` / `_fulllength` (e.g. `full_beatx_eval`, `full_beatx_fulllength`) — those expect the eval HDF5 layout above.

---

## 3. Wire caches into the training / eval configs

Every YAML in [`configs/`](../configs/) has a `cache_path` (or per-dataset `beatx_cache_path` / `embody3d_cache_path`) field that the trainer reads at startup. Two ways to point at the right cache:

- **Edit the YAML** before training, replacing the default with your absolute path:
  ```yaml
  beatx_cache_path: datasets/data_cache/beatx/database.hdf5
  embody3d_cache_path: datasets/data_cache/embody3d_dyadic/database.hdf5
  ```
- **Override on the CLI** (preferred for eval, since the training cache and eval cache live in different folders):
  ```bash
  python scripts/test.py \
      --config experiments/<experiment-dir>/config.yaml \
      --beatx_cache_path datasets/data_cache/beatx_eval/database.hdf5 \
      --dataset_ratio full_beatx_fulllength
  ```

---

## Common flags

Both builders share the same surface; defaults are sane for the canonical MIBURI recipe.

| Flag | Default | Notes |
| --- | --- | --- |
| `--data_dir` | `datasets/<dataset>/` | Root of the raw dataset. |
| `--hdf5_path` | *(required)* | Output `.hdf5` file. Parent dirs are auto-created. |
| `--motion_fps` | `25` | MIBURI runs at 25 fps end-to-end. |
| `--pose_length` | `250` | Chunk length (frames) in training mode; minimum-length filter in `--full_sequence` mode. |
| `--full_sequence` | off | Switch to one-entry-per-source-file layout for eval. |
| `--process_multimodal_signals` | off | Pre-compute audio + Whisper-transcript alignment payloads. Needed for eval. |
| `--mimi_audio_fps` | `24000` | Mimi's audio sample rate; do not change. |
| `--mimi_codebooks` | `8` | Audio codebook count; do not change. |
| `--smplx_gender` | `NEUTRAL_2020` | Matches the SMPL-X model under [`assets_dep/smplx_2020/`](../assets_dep/smplx_2020/). |
| `--smplx_num_betas` | `300` | Identity-shape dim. |
| `--smplx_num_expression_coeffs` | `100` | Expression dim. |
| `--max_files` | `None` | Cap the number of source files; useful for dry-runs. |

BEATX-only:

| Flag | Default | Notes |
| --- | --- | --- |
| `--file_list_path` | `datasets/beat_english_v2.0.0/train_test_split.csv` | BEAT2 official split CSV. |
| `--filter_lowervalid_speakers` | off | Drops takes from speakers with unreliable lower-body motion (matches the `goodspk_beatx_lowervalid` dataset_ratio). |

Embody3D-only:

| Flag | Default | Notes |
| --- | --- | --- |
| `--train_ratio` / `--val_ratio` | `0.9` / `0.05` | Per-speaker split; the rest is `test`. |
| `--savgol_enabled` | off | Optional Savitzky–Golay smoothing of the source pose/trans before chunking. |

For the complete list, see each builder's `argparse` block:
- [`scripts/trainers/dataloaders/beatx/build_hdf5_beatx.py`](../scripts/trainers/dataloaders/beatx/build_hdf5_beatx.py)
- [`scripts/trainers/dataloaders/embody3d_dyadic/build_hdf5_embody3d.py`](../scripts/trainers/dataloaders/embody3d_dyadic/build_hdf5_embody3d.py)

---

## Cross-reference

The training and evaluation workflows that consume these caches are documented in the main [README.md](../README.md):

- [Training → Step 1 (build caches)](../README.md#step-1--build-per-dataset-hdf5-caches)
- [Training → Step 2 (train)](../README.md#step-2--train)
- [Evaluation](../README.md#evaluation)
