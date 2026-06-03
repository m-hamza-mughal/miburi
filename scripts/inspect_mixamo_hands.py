"""Visualize the right-hand region of each Mixamo FBX before any fitting.

Selects right-hand verts by spatial filtering (rightmost X extreme, with arms
outstretched in the bind pose), then renders front + side views per character
to show whether the hand has finger-shaped geometry.

Run from repo root:
    python scripts/inspect_mixamo_hands.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.smplx_fitting.fbx_to_mesh import load_fbx_mesh


FBX_LIST = [
    ("Y Bot", "assets_dep/mixamo_characters/Y Bot.fbx"),
    ("Ch08",  "assets_dep/mixamo_characters/Ch08_nonPBR.fbx"),
    ("Ch31",  "assets_dep/mixamo_characters/Ch31_nonPBR.fbx"),
    ("Remy",  "assets_dep/mixamo_characters/Remy.fbx"),
]


def _select_right_hand_box(verts: np.ndarray) -> np.ndarray:
    """Heuristic: right hand sits at the X extreme (either +X or -X depending
    on character orientation), roughly at shoulder height (upper third of
    Y-extent). We pick the X side with the LARGER spatial extent in X (arms
    out -> hand is the farthest-out cluster) and take a 12 cm cube around
    the extreme."""
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    # If the rightmost extreme is farther from the origin, the right hand is
    # at -X. Mixamo bind poses usually center the character at X=0, so we
    # take whichever side has the farther extent.
    if abs(x_min) >= abs(x_max):
        x_center = x_min
    else:
        x_center = x_max
    # Hand bounding box: 12 cm cube around (x_center, y_at_x_extreme, z_at_x_extreme).
    # Y-band: top of hand cluster sits below shoulder; we keep upper 50% of Y range.
    near_mask = np.abs(verts[:, 0] - x_center) < 0.25
    if near_mask.sum() == 0:
        return np.zeros(verts.shape[0], dtype=bool)
    near_verts = verts[near_mask]
    y_med = float(np.median(near_verts[:, 1]))
    z_med = float(np.median(near_verts[:, 2]))
    box_size = 0.15  # 15 cm half-extent
    mask = (
        (np.abs(verts[:, 0] - x_center) < box_size)
        & (np.abs(verts[:, 1] - y_med) < box_size)
        & (np.abs(verts[:, 2] - z_med) < box_size)
    )
    return mask


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(FBX_LIST)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for col, (name, p) in enumerate(FBX_LIST):
        m = load_fbx_mesh(p)
        v = m.verts
        # Normalize to ~1.7m height for visualization parity across characters
        # (Remy ships ~3.74m, others ~1.8m). This is purely cosmetic for QA.
        height = v[:, 1].max() - v[:, 1].min()
        scale = 1.7 / max(height, 1e-6)
        v_norm = v * scale

        mask = _select_right_hand_box(v_norm)
        hand_v = v_norm[mask]
        if hand_v.shape[0] == 0:
            print(f"[{name}] no hand verts found")
            continue

        # Front view (X-Y)
        ax = axes[0, col]
        ax.scatter(hand_v[:, 0], hand_v[:, 1], s=3, c="tab:blue", alpha=0.7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}: front  (V_hand={hand_v.shape[0]})", fontsize=10)

        # Top view (X-Z)
        ax = axes[1, col]
        ax.scatter(hand_v[:, 0], hand_v[:, 2], s=3, c="tab:blue", alpha=0.7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}: top (xz)", fontsize=10)

        print(f"[{name}] mesh={m.mesh_name}  total V={v.shape[0]}  "
              f"hand-box V={hand_v.shape[0]}  bbox_norm Y=[{v_norm[:,1].min():.2f}, {v_norm[:,1].max():.2f}]")

    fig.suptitle("Mixamo character right-hand geometry (raw FBX, normalized to 1.7m height)", fontsize=11)
    fig.tight_layout()
    out_path = Path("assets_dep/mixamo_characters/_hand_geometry_comparison.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
