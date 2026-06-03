"""Face-focused verification: render SMPL-X face vs Mixamo character face
side-by-side across a sweep of FLAME expression coefficients, so we can
visually confirm the blendshape transfer is animating the Mixamo face
consistently with SMPL-X.

Output: a PNG strip + an MP4/GIF video (if imageio is available).

Usage:
    python scripts/test_mixamo_face.py \\
        --character_npz assets_dep/mixamo_characters_release/ch08.npz \\
        --out_png assets_dep/mixamo_characters_release/ch08_face_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miburi.utils.mixamo_character import load_mixamo_character, pose_mixamo_character


def _make_expression_sequence(n_expr: int, device, n_frames: int = 12) -> list[tuple[str, torch.Tensor]]:
    """Strong-amplitude FLAME expression sweeps so face deformation is
    clearly visible at a moderate zoom level."""
    seqs = []
    seqs.append(("neutral", torch.zeros(n_expr, device=device)))
    # Drive single coefficients at strong amplitudes (FLAME exprs scale
    # linearly; coefficients up to ~5-6 are physically meaningful).
    for amp in [3.0, 5.0]:
        e = torch.zeros(n_expr, device=device); e[0] = amp
        seqs.append((f"expr0={amp:.1f}", e))
    for amp in [3.0, -3.0]:
        e = torch.zeros(n_expr, device=device); e[1] = amp
        seqs.append((f"expr1={amp:+.1f}", e))
    for amp in [3.0, -3.0]:
        e = torch.zeros(n_expr, device=device); e[2] = amp
        seqs.append((f"expr2={amp:+.1f}", e))
    # Big mixed expression
    e = torch.zeros(n_expr, device=device); e[0] = 4.0; e[3] = 3.0
    seqs.append(("e0+e3 strong", e))
    return seqs


def _face_mask_for_smplx(smplx_model) -> np.ndarray:
    expr_mass = torch.linalg.norm(
        smplx_model.exprdirs.reshape(smplx_model.exprdirs.shape[0], -1), dim=1
    ).cpu().numpy()
    return expr_mass > 1e-6


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--character_npz", type=Path, required=True)
    p.add_argument("--smplx_model", type=Path,
                   default=Path("assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz"))
    p.add_argument("--out_png", type=Path, required=True)
    p.add_argument("--out_video", type=Path, default=None,
                   help="Optional .mp4 / .gif animated output of the same frames.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    from miburi.utils.viser_scene import make_smplx_model
    smplx = make_smplx_model(args.smplx_model, gender="NEUTRAL_2020").to(device)
    char = load_mixamo_character(args.character_npz, device=device)
    if char.face_vert_idx is None:
        print(f"[face_test] character {char.name} has no face blendshapes -- nothing to verify")
        return
    print(f"[face_test] character {char.name}: {char.face_vert_idx.shape[0]} face verts, "
          f"expr_dirs {tuple(char.expr_dirs_face.shape)}")

    smplx_face_mask = _face_mask_for_smplx(smplx)
    mixamo_face_mask = torch.zeros(char.verts_tpose.shape[0], dtype=torch.bool, device=device)
    mixamo_face_mask[char.face_vert_idx] = True
    mixamo_face_mask_np = mixamo_face_mask.cpu().numpy()
    print(f"[face_test] SMPL-X face verts: {smplx_face_mask.sum()}/{smplx_face_mask.size}")

    seqs = _make_expression_sequence(smplx.NUM_EXPR_COEFFS, device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seqs)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 6))
    rgba_frames = []  # for video

    def _render_face(ax, verts_np, face_mask, color, *, center, title, half=0.06):
        v = verts_np[face_mask]
        # Front-ish view (head sits at top, faces forward at +Z)
        ax.scatter(v[:, 0], v[:, 1], s=4, c=color, alpha=0.75, edgecolors="none")
        ax.set_xlim(center[0] - half, center[0] + half)
        # Shift y-extent slightly down so we see mouth + chin not just forehead
        ax.set_ylim(center[1] - half * 1.2, center[1] + half * 0.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for col, (name, expr) in enumerate(seqs):
        fwd = {
            "shape": torch.zeros(1, smplx.NUM_BETAS, device=device),
            "expression": expr.unsqueeze(0),
            "body_pose": torch.zeros(1, 63, device=device),
            "hand_pose": torch.zeros(1, 90, device=device),
            "head_pose": torch.zeros(1, 9, device=device),
            "global_rotation": torch.zeros(1, 3, device=device),
            "global_translation": torch.zeros(1, 3, device=device),
        }
        with torch.no_grad():
            smplx_out = smplx(**fwd)
            smplx_v = smplx_out["vertices"][0].cpu().numpy()
            head_pos_smplx = smplx_out["joints"][0, 15].cpu().numpy()  # β=0 frame
            mixamo_v, _ = pose_mixamo_character(char, smplx, fwd)
            mixamo_v_np = mixamo_v[0].cpu().numpy()
            # The Mixamo character lives in a β=fitted frame internally; use
            # the Ch08 face-vert centroid as the camera center for that row.
            mixamo_face_center = mixamo_v_np[mixamo_face_mask_np].mean(axis=0)

        _render_face(axes[0, col], smplx_v, smplx_face_mask, "tab:gray",
                     center=head_pos_smplx, title=f"SMPL-X / {name}")
        _render_face(axes[1, col], mixamo_v_np, mixamo_face_mask_np, "tab:orange",
                     center=mixamo_face_center, title=f"{char.name} / {name}")

    fig.suptitle(f"Face-blendshape transfer check  -  {char.name}", fontsize=11)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {args.out_png}")

    # Optional: also render an animated MP4/GIF with per-frame side-by-side.
    if args.out_video is not None:
        try:
            import imageio.v3 as iio
        except ImportError:
            print(f"[face_test] imageio not installed; skipping video output")
            return
        # Use the same `seqs` plus a smoother linspace sweep on coefficient 0 so
        # the resulting video shows clear mouth opening / closing motion.
        n_steps = 30
        frames = []
        amps = np.concatenate([np.linspace(0, 5.0, n_steps),
                               np.linspace(5.0, 0, n_steps)])
        for amp in amps:
            expr = torch.zeros(smplx.NUM_EXPR_COEFFS, device=device); expr[0] = float(amp)
            fwd = {
                "shape": torch.zeros(1, smplx.NUM_BETAS, device=device),
                "expression": expr.unsqueeze(0),
                "body_pose": torch.zeros(1, 63, device=device),
                "hand_pose": torch.zeros(1, 90, device=device),
                "head_pose": torch.zeros(1, 9, device=device),
                "global_rotation": torch.zeros(1, 3, device=device),
                "global_translation": torch.zeros(1, 3, device=device),
            }
            with torch.no_grad():
                so = smplx(**fwd)
                sv = so["vertices"][0].cpu().numpy()
                hp_smplx = so["joints"][0, 15].cpu().numpy()
                mv, _ = pose_mixamo_character(char, smplx, fwd)
                mv_np = mv[0].cpu().numpy()
                mxc = mv_np[mixamo_face_mask_np].mean(axis=0)

            fig2, axes2 = plt.subplots(1, 2, figsize=(8, 5))
            _render_face(axes2[0], sv, smplx_face_mask, "tab:gray",
                         center=hp_smplx, title=f"SMPL-X  expr0={amp:.2f}")
            _render_face(axes2[1], mv_np, mixamo_face_mask_np, "tab:orange",
                         center=mxc, title=f"{char.name}  expr0={amp:.2f}")
            fig2.tight_layout()
            fig2.canvas.draw()
            frame = np.asarray(fig2.canvas.renderer.buffer_rgba())
            frames.append(frame)
            plt.close(fig2)

        if str(args.out_video).endswith(".gif"):
            iio.imwrite(args.out_video, frames, duration=80, loop=0)
        else:
            iio.imwrite(args.out_video, frames, fps=15)
        print(f"Wrote {args.out_video}")


if __name__ == "__main__":
    main()
