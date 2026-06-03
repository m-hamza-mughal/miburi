"""Offline sanity check for the Mixamo runtime swap.

Renders a strip of poses driven by SMPL-X, side-by-side: default SMPL-X vs
Mixamo-character output. The Mixamo body should deform identically to
SMPL-X for each pose (within barycentric-transfer drift, ~mm-level).

Usage:
    python scripts/test_mixamo_runtime.py \\
        --character_npz assets_dep/mixamo_characters_release/y_bot.npz \\
        --out_png assets_dep/mixamo_characters_release/y_bot_runtime_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miburi.utils.mixamo_character import load_mixamo_character, pose_mixamo_character


def _make_body_poses(device, dtype=torch.float32):
    """Body-only poses, demoed full-body."""
    def _pose(joint_aa):
        p = torch.zeros(1, 63, device=device, dtype=dtype)
        for joint_idx, aa in joint_aa.items():
            p[0, joint_idx * 3 : (joint_idx + 1) * 3] = torch.tensor(aa, device=device, dtype=dtype)
        return p

    # SMPL-X body_pose joint indices (0-based, excluding pelvis):
    #   15 -> left_shoulder    16 -> right_shoulder
    #   17 -> left_elbow       18 -> right_elbow
    #    2 -> left_knee         5 -> right_knee  (approx; for twist demo)
    return [
        ("rest",      _pose({})),
        ("arms_up",   _pose({15: [0.0, 0.0, 1.2], 16: [0.0, 0.0, -1.2]})),
        ("right_fwd", _pose({16: [0.0, -1.0, 0.0]})),
        ("elbows",    _pose({17: [0.0, 1.5, 0.0], 18: [0.0, -1.5, 0.0]})),
        ("twist",     _pose({2: [0.0, 0.6, 0.0], 5: [0.0, 0.6, 0.0]})),
    ]


def _make_hand_poses(device, dtype=torch.float32):
    """Hand poses (right hand). Returned as (body_pose, hand_pose) tuples so
    we can lift the arms out of the way to make the hand visible.

    SMPL-X hand_pose layout (90 = 15+15 finger joints * 3 dof each, MANO order):
        L finger MCP/PIP/DIP for: index(0..2), middle(3..5), pinky(6..8),
                                  ring(9..11), thumb(12..14)
        R finger ditto, offset by 15.
    Within each joint, the curl axis (closes the finger) is approximately
    +Z in MANO's joint-local frame.
    """
    # Helper: build a 45-dof hand pose by setting per-finger curls.
    def _hand(curls: dict[str, float]) -> torch.Tensor:
        # curls maps finger name -> curl angle (radians, around local +Z).
        # We apply the angle to all 3 segments of that finger.
        order = ["index", "middle", "pinky", "ring", "thumb"]
        h = torch.zeros(45, device=device, dtype=dtype)
        for f, ang in curls.items():
            base = order.index(f) * 9    # 3 segments * 3 dof
            for seg in range(3):
                h[base + seg * 3 + 2] = ang
        return h

    # Lift arms forward so the hand is in front of the body, easy to see.
    body_pose = torch.zeros(1, 63, device=device, dtype=dtype)
    # Right shoulder: rotate forward + slightly down so the hand is in
    # front of the chest. Joint index 16 -> elements 48:51.
    body_pose[0, 16 * 3 : 16 * 3 + 3] = torch.tensor([0.0, -1.5, -0.2], device=device, dtype=dtype)
    # Right elbow (joint 18): bend ~90 degrees.
    body_pose[0, 18 * 3 : 18 * 3 + 3] = torch.tensor([0.0, -1.4, 0.0], device=device, dtype=dtype)

    curl = 1.3
    poses = []
    finger_specs = [
        ("open",       {}),
        ("fist",       {"index": curl, "middle": curl, "pinky": curl,
                        "ring": curl, "thumb": 0.7}),
        ("thumbs_up",  {"index": curl, "middle": curl, "pinky": curl, "ring": curl}),
        ("peace_v",    {"pinky": curl, "ring": curl, "thumb": 0.7}),
        ("index_pt",   {"middle": curl, "pinky": curl, "ring": curl, "thumb": 0.6}),
        ("pinky_up",   {"index": curl, "middle": curl, "ring": curl, "thumb": 0.6}),
    ]
    for name, curls in finger_specs:
        h_right = _hand(curls)
        full = torch.zeros(1, 90, device=device, dtype=dtype)
        full[0, 45:90] = h_right
        poses.append((name, body_pose, full))
    return poses


def _right_hand_mask(lbs_weights: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Boolean mask selecting verts whose dominant LBS weight lies on right-hand
    joints. SMPL-X joints 40..54 are the right hand (15 finger joints)."""
    right_hand_weight = lbs_weights[:, 40:55].sum(dim=-1)
    return right_hand_weight > threshold


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
    print(f"Loaded character {char.name}: V={char.num_verts}, F={char.num_faces}, "
          f"betas_l2={char.betas.norm().item():.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body_poses = _make_body_poses(device)
    hand_poses = _make_hand_poses(device)
    n_body, n_hand = len(body_poses), len(hand_poses)
    n = max(n_body, n_hand)

    # Layout:
    #   rows 0,1     -> body strip (SMPL-X, Mixamo) full-body view
    #   rows 2,3,4,5 -> hand strip: 3 views (side Z-Y, top X-Z, front X-Y)
    #                  for both SMPL-X (row 2,3,4 ... no, alternating)
    # Use: row 2,4,6 = SMPL-X (side, top, front), row 3,5,7 = Y-Bot
    fig, axes = plt.subplots(8, n, figsize=(3 * n, 22))

    def _scatter(ax, verts, color, title, *, xlim, ylim):
        v = verts.cpu().numpy()
        rng = np.random.default_rng(0)
        # For hand-zoom views, keep more verts so finger detail survives.
        max_pts = 8000 if (xlim[1] - xlim[0]) < 1.0 else 4000
        idx = rng.choice(v.shape[0], size=min(v.shape[0], max_pts), replace=False)
        ax.scatter(v[idx, 0], v[idx, 1], s=2, c=color, alpha=0.6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # --- rows 0, 1: body poses (full body view) ---
    # When there are more hand-pose cols (6) than body-pose cols (5), pad.
    for r in range(8):
        for c in range(n):
            axes[r, c].axis("off")
    body_xlim, body_ylim = (-1.5, 1.5), (-1.5, 1.5)
    for col in range(n_body):
        axes[0, col].axis("on"); axes[1, col].axis("on")
    for col in range(n_body):
        name, body_pose = body_poses[col]
        fwd = {
            "shape": torch.zeros(1, 300, device=device),
            "expression": torch.zeros(1, 100, device=device),
            "body_pose": body_pose,
            "hand_pose": torch.zeros(1, 90, device=device),
            "head_pose": torch.zeros(1, 9, device=device),
            "global_rotation": torch.zeros(1, 3, device=device),
            "global_translation": torch.zeros(1, 3, device=device),
        }
        with torch.no_grad():
            smplx_v = smplx(**fwd)["vertices"][0]
            mixamo_v, _ = pose_mixamo_character(char, smplx, fwd)
        _scatter(axes[0, col], smplx_v, "tab:gray",
                 f"SMPL-X / {name}", xlim=body_xlim, ylim=body_ylim)
        _scatter(axes[1, col], mixamo_v[0], "tab:orange",
                 f"{char.name} / {name}", xlim=body_xlim, ylim=body_ylim)
    for col in range(n_body, n):
        axes[0, col].axis("off"); axes[1, col].axis("off")

    # --- rows 2, 3: hand poses (zoomed view around the right hand) ---
    # Use a finer point size + per-finger coloring so finger separation is
    # visible despite the mm-scale gaps between digits.
    finger_joints = {
        "index":  [40, 41, 42], "middle": [43, 44, 45],
        "pinky":  [46, 47, 48], "ring":   [49, 50, 51], "thumb":  [52, 53, 54],
    }
    finger_colors = {"index": "tab:red", "middle": "tab:orange",
                     "pinky": "tab:green", "ring": "tab:purple", "thumb": "tab:blue"}

    def _finger_color_array(lbs_weights: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask, color_array) where mask selects right-hand verts and
        color_array assigns each selected vert a color based on its dominant
        right-hand joint."""
        rh = lbs_weights[:, 40:55].cpu().numpy()                # (V, 15)
        dom_local = rh.argmax(axis=1)                            # 0..14
        dom_global = dom_local + 40                              # 40..54
        in_rh = rh.sum(axis=1) > 0.5                             # right-hand verts
        finger_of_joint = {j: f for f, joints in finger_joints.items() for j in joints}
        colors = np.array([finger_colors[finger_of_joint[j]] for j in dom_global])
        return in_rh, colors

    smplx_rh_mask, smplx_finger_colors = _finger_color_array(smplx.lbs_weights)
    mixamo_rh_mask, mixamo_finger_colors = _finger_color_array(char.lbs_weights)

    def _scatter_hand(ax, verts, rh_mask, colors, title, *, center, half, view):
        """view: 'side' (Z-Y), 'top' (X-Z), 'front' (X-Y)."""
        v = verts.cpu().numpy()
        v_rh = v[rh_mask]
        c_rh = colors[rh_mask]
        if view == "side":
            xs, ys = v_rh[:, 2], v_rh[:, 1]
            ax.set_xlim(center[2] - half, center[2] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_xlabel("Z (fwd)", fontsize=7); ax.set_ylabel("Y (up)", fontsize=7)
        elif view == "top":
            xs, ys = v_rh[:, 0], v_rh[:, 2]
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[2] - half, center[2] + half)
            ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Z (fwd)", fontsize=7)
        elif view == "front":
            xs, ys = v_rh[:, 0], v_rh[:, 1]
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y (up)", fontsize=7)
        ax.scatter(xs, ys, s=2, c=c_rh, alpha=0.85, edgecolors="none")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for col in range(n_hand):
        name, body_pose, hand_pose = hand_poses[col]
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
            smplx_out = smplx(**fwd)
            smplx_v = smplx_out["vertices"][0]
            mixamo_v, _ = pose_mixamo_character(char, smplx, fwd)

        center_smplx = smplx_out["joints"][0, 21].cpu().numpy()
        # 3 views x 2 characters = 6 rows (rows 2..7).
        # row order: side SMPL-X, side Mixamo, top SMPL-X, top Mixamo, front SMPL-X, front Mixamo.
        for r_idx, view in enumerate(["side", "top", "front"]):
            row_smplx = 2 + r_idx * 2
            row_mixamo = 3 + r_idx * 2
            axes[row_smplx, col].axis("on"); axes[row_mixamo, col].axis("on")
            _scatter_hand(axes[row_smplx, col], smplx_v, smplx_rh_mask, smplx_finger_colors,
                          f"SMPL-X / {name} / {view}", center=center_smplx, half=0.12, view=view)
            _scatter_hand(axes[row_mixamo, col], mixamo_v[0], mixamo_rh_mask, mixamo_finger_colors,
                          f"{char.name} / {name} / {view}", center=center_smplx, half=0.10, view=view)

    # Legend in the right-most empty cell.
    legend_handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f)
                      for f, c in finger_colors.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Mixamo runtime check  -  character={char.name}  "
                 f"(top: body, bottom: right-hand zoom)", fontsize=12)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=110)
    plt.close(fig)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
