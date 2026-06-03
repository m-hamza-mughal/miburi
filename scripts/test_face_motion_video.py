"""Face-focused motion comparison video: SMPL-X reference vs retargeted Mixamo.

Sweeps:
  - Phase 1: jaw open  (head_pose[0] = 0 -> 0.6 -> 0)
  - Phase 2: expr0     (0 -> 4 -> 0)
  - Phase 3: expr1     (0 -> 3 -> 0)
  - Phase 4: combined  (jaw + expr0 + expr3 simultaneously)

Side-by-side SMPL-X / Mixamo with finger-color-coded vert scatter, zoomed
to the face area. Output: PNG strip + MP4 (or GIF) animated video.

Usage:
    python scripts/test_face_motion_video.py \\
        --character_npz assets_dep/mixamo_characters_release/remy.npz \\
        --out_video assets_dep/mixamo_characters_release/remy_face_motion.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miburi.utils.mixamo_character import load_mixamo_character, pose_mixamo_character


def _make_motion_keyframes(n_expr: int, device) -> list[tuple[str, dict]]:
    """Sequence of (label, kwargs) frames for the video."""
    seq: list[tuple[str, dict]] = []

    def _ramp(n_steps: int):
        return np.concatenate([
            np.linspace(0.0, 1.0, n_steps),
            np.linspace(1.0, 0.0, n_steps),
        ])

    steps = 20
    # --- Phase 1: jaw open ---
    for t in _ramp(steps):
        amp = 0.7 * t  # max ~40 deg jaw open
        hp = torch.zeros(1, 9, device=device); hp[0, 0] = amp
        seq.append((f"jaw open  {amp:.2f}", dict(head_pose=hp,
                                                  expression=torch.zeros(1, n_expr, device=device))))

    # --- Phase 2: expr0 (FLAME dominant mode -- mostly mouth/cheek) ---
    for t in _ramp(steps):
        amp = 5.0 * t
        e = torch.zeros(1, n_expr, device=device); e[0, 0] = amp
        seq.append((f"expr0     {amp:.2f}", dict(head_pose=torch.zeros(1, 9, device=device),
                                                   expression=e)))

    # --- Phase 3: expr1 ---
    for t in _ramp(steps):
        amp = 4.0 * t
        e = torch.zeros(1, n_expr, device=device); e[0, 1] = amp
        seq.append((f"expr1     {amp:.2f}", dict(head_pose=torch.zeros(1, 9, device=device),
                                                   expression=e)))

    # --- Phase 4: combined jaw + expr0 + expr3 (speech-like) ---
    for t in _ramp(steps):
        e = torch.zeros(1, n_expr, device=device)
        e[0, 0] = 4.0 * t
        e[0, 3] = 2.5 * t
        hp = torch.zeros(1, 9, device=device); hp[0, 0] = 0.5 * t
        seq.append((f"combined  {t:.2f}", dict(head_pose=hp, expression=e)))

    return seq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--character_npz", type=Path, required=True)
    p.add_argument("--smplx_model", type=Path,
                   default=Path("assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz"))
    p.add_argument("--out_video", type=Path, required=True)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio is required: pip install imageio[ffmpeg]")
        sys.exit(1)

    device = torch.device(args.device)
    from miburi.utils.viser_scene import make_smplx_model
    from smplx.vertex_ids import vertex_ids as _smplx_vertex_ids
    smplx = make_smplx_model(args.smplx_model, gender="NEUTRAL_2020").to(device)
    char = load_mixamo_character(args.character_npz, device=device)
    print(f"Loaded {char.name}: V={char.num_verts}, face verts={char.face_vert_idx.shape[0] if char.face_vert_idx is not None else 0}")

    # Load saved Mixamo region labels (one int per Mixamo vert, indices into
    # FACE_REGION_NAMES) and the per-region palette. Same regions get the
    # same color on the SMPL-X panel via runtime Voronoi assignment, so a
    # mis-aligned K-NN is visible as Mixamo verts colored wrong vs SMPL-X.
    _bundle = np.load(args.character_npz, allow_pickle=True)
    char_region_per_vert = (_bundle["face_region_per_vert"]
                            if "face_region_per_vert" in _bundle.files else None)
    if "face_region_names" in _bundle.files:
        FACE_REGION_NAMES = list(_bundle["face_region_names"])
    else:
        FACE_REGION_NAMES = ["brow", "nose", "leye", "reye", "mouth", "lear", "rear", "chin"]
    REGION_PALETTE = {
        "brow":  [0.55, 0.20, 0.75],   # purple
        "nose":  [0.10, 0.65, 0.20],   # green
        "leye":  [0.10, 0.45, 0.85],   # blue
        "reye":  [0.10, 0.45, 0.85],   # blue
        "lear":  [0.95, 0.85, 0.0],    # yellow (side of face)
        "rear":  [0.95, 0.85, 0.0],    # yellow
        "mouth": [1.0, 0.55, 0.0],     # orange
        "chin":  [0.85, 0.10, 0.10],   # red
    }
    # SMPL-X landmark vertex IDs for Voronoi labels on the SMPL-X panel.
    _smplx_lm_ids = _smplx_vertex_ids["smplx"]

    # Identify face-region masks for visualization. For SMPL-X: any vert
    # with nonzero exprdirs. For Mixamo: the saved face_vert_mask plus a
    # spatial expansion for context (so the rendered region isn't just the
    # face mask -- include surrounding head for spatial anchor).
    smplx_expr_mass = torch.linalg.norm(
        smplx.exprdirs.reshape(smplx.exprdirs.shape[0], -1), dim=1
    ).cpu().numpy()
    smplx_face_mask = smplx_expr_mass > 1e-6

    # For Mixamo, render the entire head region so face landmarks are
    # visible alongside hair/scalp. We use the head joint position to
    # threshold: any vert within 30cm below the head joint Y is "head".
    char_verts_tpose = char.verts_tpose.cpu().numpy()
    char_y = char_verts_tpose[:, 1]
    # Conservative threshold: top 25% by Y (covers face + hair + neck)
    head_thresh = np.percentile(char_y, 75)
    char_face_mask = (char_y > head_thresh)

    # Highlight face-deforming verts in a distinct color.
    char_active_face = np.zeros(char.verts_tpose.shape[0], dtype=bool)
    if char.face_vert_idx is not None:
        char_active_face[char.face_vert_idx.cpu().numpy()] = True

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keyframes = _make_motion_keyframes(smplx.NUM_EXPR_COEFFS, device)
    print(f"rendering {len(keyframes)} frames at {args.fps} fps ...")

    frames: list[np.ndarray] = []

    def _region_colors(verts_np, head_y, mouth_y):
        """Color-code each vert by anatomical face region. Bands defined
        in NORMALIZED face-Y so corresponding colors land on the same
        anatomical parts of any character regardless of head size. The
        normalized coord is fy in [0, 1]: 0 = mouth_y - 5 cm (below the
        chin), 1 = head_y (crown). X is normalized to the per-character
        face half-width."""
        n = verts_np.shape[0]
        c = np.array([[0.7, 0.7, 0.7]] * n)   # default gray
        y = verts_np[:, 1]; x = verts_np[:, 0]; z = verts_np[:, 2]
        # Face Y normalized so chin region sits below 0, mouth ~0.1, crown ~1.
        face_y0 = mouth_y - 0.05         # 5 cm below the lip line
        face_y1 = head_y
        face_height = max(face_y1 - face_y0, 1e-4)
        fy = (y - face_y0) / face_height
        # Lateral half-width: 90th-percentile |X| within the face band so
        # the cheek thresholds adapt to the character's actual head width.
        in_face = (fy > 0.0) & (fy < 1.05)
        if in_face.sum() > 50:
            half_w = float(np.percentile(np.abs(x[in_face]), 90))
        else:
            half_w = 0.07
        half_w = max(half_w, 0.04)
        x_norm = np.abs(x) / half_w     # 0 = center, 1 = side
        # Forward-Z normalized similarly so "Z > 0.03" stays scale-correct
        if in_face.sum() > 50:
            z_med = float(np.median(z[in_face]))
            z_norm = z - z_med
        else:
            z_norm = z
        # chin: below the lip line
        chin = fy < 0.08
        c[chin] = [0.85, 0.10, 0.10]            # red
        # lips: a narrow band around the lip line
        lips = (fy >= 0.08) & (fy < 0.20)
        c[lips] = [1.0, 0.55, 0.0]              # orange
        # cheek: mid-face, lateral
        cheek = (fy >= 0.20) & (fy < 0.50) & (x_norm > 0.55)
        c[cheek] = [0.95, 0.85, 0.0]            # yellow
        # nose: mid-face, central, forward
        nose = (fy >= 0.20) & (fy < 0.55) & (x_norm < 0.50) & (z_norm > 0.0)
        c[nose] = [0.10, 0.65, 0.20]            # green
        # eyes: above nose
        eyes = (fy >= 0.55) & (fy < 0.75)
        c[eyes] = [0.10, 0.45, 0.85]            # blue
        # brow / forehead
        brow = fy >= 0.75
        c[brow] = [0.55, 0.20, 0.75]            # purple
        return c

    def _zoom_face(ax, verts_np, mask, color, *, center, half, title,
                   region_colors=None, **kwargs):
        v = verts_np[mask]
        if region_colors is not None:
            cm = region_colors[mask]
            ax.scatter(v[:, 0], v[:, 1], s=4, c=cm, alpha=0.85, edgecolors="none")
        else:
            ax.scatter(v[:, 0], v[:, 1], s=3, c=color, alpha=0.5, edgecolors="none")
        ax.set_xlim(center[0] - half, center[0] + half)
        # Symmetric Y window — characters vary in face height (Mixamo Ch08
        # has a 22 cm tall face; SMPL-X only 15 cm). Keep both head-top
        # and chin in frame.
        ax.set_ylim(center[1] - half * 1.4, center[1] + half * 1.4)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    for label, kw in keyframes:
        fwd = {
            "shape": torch.zeros(1, smplx.NUM_BETAS, device=device),
            "expression": kw.get("expression", torch.zeros(1, smplx.NUM_EXPR_COEFFS, device=device)),
            "body_pose": torch.zeros(1, 63, device=device),
            "hand_pose": torch.zeros(1, 90, device=device),
            "head_pose": kw.get("head_pose", torch.zeros(1, 9, device=device)),
            "global_rotation": torch.zeros(1, 3, device=device),
            "global_translation": torch.zeros(1, 3, device=device),
        }
        with torch.no_grad():
            so = smplx(**fwd)
            smplx_v = so["vertices"][0].cpu().numpy()
            head_smplx = so["joints"][0, 15].cpu().numpy()
            mixamo_v, _ = pose_mixamo_character(char, smplx, fwd)
            mixamo_v_np = mixamo_v[0].cpu().numpy()

        # Per-panel camera center. The Mixamo character may live in a
        # Y-shifted frame (chamfer drift can place verts head_top up to
        # 30 cm below SMPL-X's joint frame), so the camera Y is anchored
        # to each character's OWN 99th-percentile Y (head top) minus an
        # offset for the mid-face.
        def _face_center(verts):
            head_top_y = float(np.percentile(verts[:, 1], 99))
            # Face band: 25 cm below the crown (covers brow -> chin for
            # both compact SMPL-X faces and taller Mixamo faces).
            band = (verts[:, 1] > head_top_y - 0.25) & (verts[:, 1] < head_top_y + 0.02)
            if band.sum() < 50:
                return np.array([0.0, head_top_y - 0.10, 0.0], dtype=np.float32)
            return verts[band].mean(axis=0).astype(np.float32)

        cam_smplx = _face_center(smplx_v)
        cam_char = _face_center(mixamo_v_np)

        # Color-code by region using NEUTRAL Y values; cache them on the
        # first frame. Each character uses its OWN mouth-Y derived from its
        # neutral face geometry, so region colors reflect actual anatomy.
        # Mouth-Y heuristic: the Y at which the face shrinks back from
        # max-forward (i.e. the chin transitions to the lip line). We pick
        # it as the 30th-percentile Y of the most-forward face verts (those
        # with Z >= 80th percentile within the head Y band).
        # Each character has its own head_top (Mixamo may be Y-shifted from
        # SMPL-X by ~30cm if chamfer drifted). Detect head_top + mouth Y
        # from the character's own verts.
        def _detect_head_mouth_y(verts):
            head_y = float(np.percentile(verts[:, 1], 99))
            band = (verts[:, 1] > head_y - 0.25) & (verts[:, 1] < head_y - 0.05)
            if band.sum() < 50:
                return head_y, head_y - 0.18
            zb = verts[band, 2]
            z_thr = np.percentile(zb, 80)
            forward = band & (verts[:, 2] >= z_thr)
            if forward.sum() < 20:
                return head_y, head_y - 0.18
            return head_y, float(np.percentile(verts[forward, 1], 30))

        if "_neutral_v_char" not in locals():
            _neutral_v_char = mixamo_v_np.copy()
            _neutral_v_smplx = smplx_v.copy()
            _head_y_smplx, _mouth_y_smplx = _detect_head_mouth_y(_neutral_v_smplx)
            _head_y_char, _mouth_y_char = _detect_head_mouth_y(_neutral_v_char)
            print(f"[video] head_y/mouth_y SMPL-X: {_head_y_smplx:.3f}/"
                  f"{_mouth_y_smplx:.3f}; Mixamo: {_head_y_char:.3f}/"
                  f"{_mouth_y_char:.3f}")

            # Landmark-Voronoi labels on SMPL-X using FLAME's 51 face
            # landmarks grouped into anatomical regions. Same scheme as
            # the saved Mixamo region labels -- if K-NN works, same color
            # lands on same body part in both panels.
            _expr_mass = np.linalg.norm(
                smplx.exprdirs.detach().cpu().numpy().reshape(
                    smplx.exprdirs.shape[0], -1
                ), axis=1,
            )
            _smplx_anchor_mask = _expr_mass > 0.2 * _expr_mass.max()
            _smplx_region_names = list(FACE_REGION_NAMES)
            _FLAME_GROUPS = {
                "brow":  list(range(0, 10)),
                "nose":  list(range(10, 19)),
                "leye":  list(range(19, 25)),
                "reye":  list(range(25, 31)),
                "mouth": list(range(31, 51)),
            }
            _raw_npz = np.load(args.smplx_model, allow_pickle=True)
            _lmk_faces_idx = np.asarray(_raw_npz["lmk_faces_idx"], dtype=np.int64)
            _lmk_bary = np.asarray(_raw_npz["lmk_bary_coords"], dtype=np.float32)
            _faces_raw = np.asarray(_raw_npz["f"], dtype=np.int64)
            _lmk_tri_verts = _neutral_v_smplx[_faces_raw[_lmk_faces_idx]]   # (51, 3, 3)
            _flame_lmk_pos = np.einsum("lij,li->lj", _lmk_tri_verts, _lmk_bary).astype(np.float32)
            # Also include lear / rear single-vertex landmarks so ear/side
            # regions get their own Voronoi cell (otherwise the mouth's
            # 20 landmarks absorb cheek/ear-area verts).
            _extra_lmks = []
            for _name in ("lear", "rear"):
                if _name in _smplx_lm_ids:
                    _extra_lmks.append((_neutral_v_smplx[_smplx_lm_ids[_name]], _name))
            _all_lmk = (np.concatenate(
                [_flame_lmk_pos] + [p[None, :] for p, _ in _extra_lmks], axis=0,
            ) if _extra_lmks else _flame_lmk_pos.copy())
            _per_lmk_region = np.full(_all_lmk.shape[0], -1, dtype=np.int32)
            for _ri, _name in enumerate(_smplx_region_names):
                if _name in _FLAME_GROUPS:
                    for _li in _FLAME_GROUPS[_name]:
                        if 0 <= _li < _flame_lmk_pos.shape[0]:
                            _per_lmk_region[_li] = _ri
            for _i, (_pos, _name) in enumerate(_extra_lmks):
                if _name in _smplx_region_names:
                    _per_lmk_region[_flame_lmk_pos.shape[0] + _i] = _smplx_region_names.index(_name)
            _smplx_region_per_vert = np.full(_neutral_v_smplx.shape[0], -1, dtype=np.int32)
            _anchor_verts = _neutral_v_smplx[_smplx_anchor_mask]
            _d2 = ((_anchor_verts[:, None, :] - _all_lmk[None, :, :])
                   ** 2).sum(axis=-1)
            _nearest_lmk = np.argmin(_d2, axis=-1)
            _smplx_region_per_vert[_smplx_anchor_mask] = _per_lmk_region[_nearest_lmk]
            # Below-mouth anchors -> chin
            if "chin" in _smplx_region_names:
                _chin_idx = _smplx_region_names.index("chin")
                _is_chin = (_neutral_v_smplx[:, 1] < (_mouth_y_smplx - 0.015)) & _smplx_anchor_mask
                _smplx_region_per_vert[_is_chin] = _chin_idx

        # Build per-vert colour arrays using landmark labels (or fall back
        # to heuristic colours for verts without a label).
        def _palette_colors(verts_np, region_per_vert):
            n = verts_np.shape[0]
            c = np.array([[0.55, 0.55, 0.55]] * n, dtype=np.float32)  # gray default
            for ri, name in enumerate(_smplx_region_names):
                rgb = REGION_PALETTE.get(name, [0.7, 0.7, 0.7])
                mask = region_per_vert == ri
                c[mask] = rgb
            return c
        smplx_region_colors = _palette_colors(_neutral_v_smplx, _smplx_region_per_vert)
        if char_region_per_vert is not None:
            char_region_colors = _palette_colors(_neutral_v_char, char_region_per_vert)
        else:
            char_region_colors = _region_colors(_neutral_v_char, _head_y_char, _mouth_y_char)

        # Render mask = all verts in head region (Y within 25cm of camera Y).
        smplx_render_mask = np.abs(smplx_v[:, 1] - cam_smplx[1]) < 0.25
        char_render_mask = np.abs(mixamo_v_np[:, 1] - cam_char[1]) < 0.25

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        _zoom_face(axes[0], smplx_v, smplx_render_mask, "tab:gray",
                   center=cam_smplx, half=0.13, title=f"SMPL-X  |  {label}",
                   region_colors=smplx_region_colors)
        _zoom_face(axes[1], mixamo_v_np, char_render_mask, "tab:gray",
                   center=cam_char, half=0.13,
                   title=f"{char.name}  |  {label}",
                   region_colors=char_region_colors)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close(fig)

    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    out = str(args.out_video)
    if out.endswith(".gif"):
        iio.imwrite(out, frames, duration=int(1000 / args.fps), loop=0)
    else:
        # PyAV codec needs RGB (3 channels) and even W/H.
        rgb_frames = [f[..., :3] for f in frames]
        h0, w0 = rgb_frames[0].shape[:2]
        pad_h = h0 % 2; pad_w = w0 % 2
        if pad_h or pad_w:
            rgb_frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                                  mode="edge") for f in rgb_frames]
        iio.imwrite(out, rgb_frames, fps=args.fps, codec="libx264")
    print(f"Wrote {out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
