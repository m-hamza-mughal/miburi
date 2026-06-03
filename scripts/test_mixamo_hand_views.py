"""Large, side-by-side hand-only verification for a Mixamo character.

For each test pose, renders 3 views (side / top / front) of both SMPL-X and
the Mixamo character at the same scale, with finger-colored verts so the
palm orientation is unambiguous.

Usage:
    python scripts/test_mixamo_hand_views.py \\
        --character_npz assets_dep/mixamo_characters_release/y_bot.npz \\
        --out_png assets_dep/mixamo_characters_release/y_bot_hand_views.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miburi.utils.mixamo_character import load_mixamo_character, pose_mixamo_character


RIGHT_FINGER_JOINTS = {
    "index":  [40, 41, 42], "middle": [43, 44, 45],
    "pinky":  [46, 47, 48], "ring":   [49, 50, 51], "thumb":  [52, 53, 54],
}
FINGER_COLORS = {"index": "tab:red", "middle": "tab:orange",
                 "pinky": "tab:green", "ring": "tab:purple", "thumb": "tab:blue"}


def _make_pose(device, joint_aa, hand_curls=None, dtype=torch.float32):
    body_pose = torch.zeros(1, 63, device=device, dtype=dtype)
    for j, aa in joint_aa.items():
        body_pose[0, j * 3 : (j + 1) * 3] = torch.tensor(aa, device=device, dtype=dtype)
    hand_pose = torch.zeros(1, 90, device=device, dtype=dtype)
    if hand_curls is not None:
        order = ["index", "middle", "pinky", "ring", "thumb"]
        for f, ang in hand_curls.items():
            base = 45 + order.index(f) * 9  # right hand starts at slot 45
            for seg in range(3):
                hand_pose[0, base + seg * 3 + 2] = ang
    return body_pose, hand_pose


def _finger_colors_for(lbs_weights: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    rh = lbs_weights[:, 40:55].cpu().numpy()
    dom_local = rh.argmax(axis=1)
    dom_global = dom_local + 40
    in_rh = rh.sum(axis=1) > 0.5
    fj = {j: f for f, joints in RIGHT_FINGER_JOINTS.items() for j in joints}
    colors = np.array([FINGER_COLORS[fj[j]] for j in dom_global])
    return in_rh, colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character_npz", type=Path, required=True)
    parser.add_argument("--smplx_model", type=Path,
                        default=Path("assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--out_png", type=Path, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    from miburi.utils.viser_scene import make_smplx_model
    smplx = make_smplx_model(args.smplx_model, gender="NEUTRAL_2020").to(device)
    char = load_mixamo_character(args.character_npz, device=device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smplx_mask, smplx_colors = _finger_colors_for(smplx.lbs_weights)
    mixamo_mask, mixamo_colors = _finger_colors_for(char.lbs_weights)

    # Three poses: T-pose (arm extended), arm-forward-open, arm-forward-fist.
    # T-pose isolates pure bind orientation. Forward poses show how the palm
    # orients after a shoulder rotation -- the bind-roll fix matters most here.
    poses = []
    # 1. T-pose, fingers open -- pure bind orientation.
    bp, hp = _make_pose(device, {})
    poses.append(("T-pose / open", bp, hp))
    # 2. T-pose, fingers fisted -- finger curl in bind orientation.
    bp, hp = _make_pose(device, {}, {"index":1.3,"middle":1.3,"pinky":1.3,"ring":1.3,"thumb":0.7})
    poses.append(("T-pose / fist", bp, hp))
    # 3. Arm-forward, open palm.
    bp, hp = _make_pose(device, {16: [0.0, -1.5, -0.2], 18: [0.0, -1.4, 0.0]})
    poses.append(("Arm-forward / open", bp, hp))
    # 4. Arm-forward, fist.
    bp, hp = _make_pose(device, {16: [0.0, -1.5, -0.2], 18: [0.0, -1.4, 0.0]},
                        {"index":1.3,"middle":1.3,"pinky":1.3,"ring":1.3,"thumb":0.7})
    poses.append(("Arm-forward / fist", bp, hp))

    n_poses = len(poses)
    # Layout: 6 rows (3 views x 2 chars) x n_poses cols.
    fig, axes = plt.subplots(6, n_poses, figsize=(5 * n_poses, 22))

    def _scatter(ax, v_world, mask, colors, *, center, half, view, title):
        v = v_world.cpu().numpy()[mask]
        c = colors[mask]
        if view == "side":
            xs, ys = v[:, 2], v[:, 1]; xc, yc = center[2], center[1]
            ax.set_xlabel("Z forward"); ax.set_ylabel("Y up")
        elif view == "top":
            xs, ys = v[:, 0], v[:, 2]; xc, yc = center[0], center[2]
            ax.set_xlabel("X (left/right)"); ax.set_ylabel("Z (fwd)")
        elif view == "front":
            xs, ys = v[:, 0], v[:, 1]; xc, yc = center[0], center[1]
            ax.set_xlabel("X (left/right)"); ax.set_ylabel("Y up")
        ax.scatter(xs, ys, s=8, c=c, alpha=0.85, edgecolors="none")
        ax.set_xlim(xc - half, xc + half); ax.set_ylim(yc - half, yc + half)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    for col, (name, body_pose, hand_pose) in enumerate(poses):
        fwd = {
            "shape": torch.zeros(1, 300, device=device),
            "expression": torch.zeros(1, 100, device=device),
            "body_pose": body_pose,
            "hand_pose": hand_pose,
            "head_pose": torch.zeros(1, 9, device=device),
            "global_rotation": torch.zeros(1, 3, device=device),
            "global_translation": torch.zeros(1, 3, device=device),
        }
        with torch.no_grad():
            out = smplx(**fwd)
            smplx_v = out["vertices"][0]
            mixamo_v, _ = pose_mixamo_character(char, smplx, fwd)
        center = out["joints"][0, 21].cpu().numpy()  # right_wrist

        for r_idx, view in enumerate(["side", "top", "front"]):
            half_s = 0.15
            _scatter(axes[r_idx * 2, col], smplx_v, smplx_mask, smplx_colors,
                     center=center, half=half_s, view=view,
                     title=f"SMPL-X | {name} | {view}")
            _scatter(axes[r_idx * 2 + 1, col], mixamo_v[0], mixamo_mask, mixamo_colors,
                     center=center, half=half_s, view=view,
                     title=f"{char.name} | {name} | {view}")

    legend_handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f, markersize=8)
                      for f, c in FINGER_COLORS.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Right-hand palm orientation check  -  {char.name}  "
                 f"(top: SMPL-X reference / bottom of each pair: {char.name})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=110)
    plt.close(fig)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
