"""Offline fitting tool: produce a Mixamo character bundle for the MIBURI demo.

Given a Mixamo FBX (e.g. Y Bot.fbx) and the MIBURI SMPL-X 2020 model, this
script:

    1. Imports the FBX (Blender headless) and extracts the rest-pose mesh.
    2. Fits SMPL-X (beta, global R, t, per-axis scale, arm A-pose) to the
       Mixamo mesh via two-phase chamfer Adam.
    3. Transfers SMPL-X per-vertex LBS weights onto the Mixamo vertices via
       barycentric projection.
    4. Writes:
           assets_dep/mixamo_characters/<slug>.npz   -- runtime bundle
           assets_dep/mixamo_characters/<slug>_qa.png -- QA render

The runtime bundle is consumed by miburi.utils.mixamo_character.
Usage:
    python -m scripts.fit_mixamo_character \\
        --fbx_path assets_dep/mixamo_characters/'Y Bot.fbx' \\
        --character_name y_bot

The --with_face flag is a v2 placeholder; v1 ships body-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# Allow `python scripts/fit_mixamo_character.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.smplx_fitting.fbx_to_mesh import load_fbx_mesh
from scripts.utils.smplx_fitting.chamfer_fit import fit_smplx_to_mesh
from scripts.utils.smplx_fitting.transfer_lbs_weights import transfer_lbs_weights
from scripts.utils.smplx_fitting.transfer_face_blendshapes import transfer_face_blendshapes
from scripts.utils.smplx_fitting.add_jaw_skinning import add_jaw_skinning
from scripts.utils.smplx_fitting.hand_correction import correct_hand_bind_pose


DEFAULT_SMPLX_MODEL = "assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz"
DEFAULT_OUT_DIR = "assets_dep/mixamo_characters"


def _slug_from_fbx(fbx_path: Path) -> str:
    stem = fbx_path.stem
    return stem.lower().replace(" ", "_").replace("_nonpbr", "")


def _qa_render(
    out_png: Path,
    mixamo_verts: np.ndarray,
    mixamo_faces: np.ndarray,
    smplx_verts: np.ndarray,
    smplx_faces: np.ndarray,
    title: str,
) -> None:
    """Three views (front / side / overlap) of source Mixamo vs fitted SMPL-X."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 10))

    def _draw(ax, verts, color, alpha, label, plane):
        rng = np.random.default_rng(0)
        idx = rng.choice(verts.shape[0], size=min(verts.shape[0], 5000), replace=False)
        v = verts[idx]
        if plane == "front":   # x-y
            ax.scatter(v[:, 0], v[:, 1], s=2, c=color, label=label, alpha=alpha)
            ax.set_xlabel("X (left/right)"); ax.set_ylabel("Y (up)")
        elif plane == "side":  # z-y
            ax.scatter(v[:, 2], v[:, 1], s=2, c=color, label=label, alpha=alpha)
            ax.set_xlabel("Z (forward)"); ax.set_ylabel("Y (up)")
        elif plane == "top":   # x-z
            ax.scatter(v[:, 0], v[:, 2], s=2, c=color, label=label, alpha=alpha)
            ax.set_xlabel("X"); ax.set_ylabel("Z")
        ax.set_aspect("equal")

    for col, (plane, name) in enumerate([("front", "front (x-y)"),
                                          ("side", "side (z-y)"),
                                          ("top", "top (x-z)")]):
        ax = fig.add_subplot(2, 3, col + 1)
        _draw(ax, mixamo_verts, "tab:blue", 0.6, "Mixamo", plane)
        ax.set_title(f"Mixamo - {name}")

        ax = fig.add_subplot(2, 3, col + 4)
        _draw(ax, mixamo_verts, "tab:blue", 0.35, "Mixamo", plane)
        _draw(ax, smplx_verts, "tab:orange", 0.35, "SMPL-X fit", plane)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Overlap - {name}")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fbx_path", required=True, type=Path)
    p.add_argument("--character_name", type=str, default=None,
                   help="Output slug. Defaults to a slug derived from the FBX filename.")
    p.add_argument("--smplx_model", type=Path, default=Path(DEFAULT_SMPLX_MODEL))
    p.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument("--with_face", action="store_true",
                   help="v2: also transfer FLAME expression blendshapes. (Not yet implemented in v1.)")
    p.add_argument("--n_iters_align", type=int, default=100)
    p.add_argument("--n_iters_joint", type=int, default=800)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--blender_bin", type=str, default=None)
    p.add_argument("--no_apose_init", action="store_true",
                   help="Skip the A-pose shoulder initialization (use this for T-pose characters).")
    p.add_argument("--optimize_body_pose", action="store_true",
                   help="Let chamfer fit also optimize body_pose. Off by default -- the "
                        "saved verts_tpose must live in SMPL-X rest frame for runtime LBS.")
    p.add_argument("--hand_loss_weight", type=float, default=4.0,
                   help="Per-vertex weight bonus on SMPL-X hand region during chamfer "
                        "(wrist + finger joints). 0 disables.")
    p.add_argument("--skinning", choices=["native", "barycentric"], default="native",
                   help="Source of per-vertex LBS weights. 'native' (default) uses the "
                        "FBX armature's own bone weights via name mapping; 'barycentric' "
                        "falls back to closest-point-on-SMPL-X transfer.")
    p.add_argument("--no_hand_bind_correction", action="store_true",
                   help="Skip the per-wrist bind-pose roll correction. By default we rotate "
                        "each hand around the wrist so the Mixamo palm orientation matches "
                        "SMPL-X's T-pose -- this makes gesture-LM hand poses transfer cleanly.")
    p.add_argument("--no_skeleton_retarget", action="store_true",
                   help="Skip the per-vertex skeleton retarget. By default we shift each "
                        "vertex so it sits at the correct offset from SMPL-X J_regressor "
                        "joints rather than Mixamo's bone positions; this is the standard "
                        "LBS retargeting step needed for articulation to be faithful.")
    p.add_argument("--no_jaw_skinning", action="store_true",
                   help="Skip transferring jaw-joint skinning weight to the lower-face "
                        "region. By default we use SMPL-X's own jaw weights as the template "
                        "so head_pose[0] (jaw open) mechanically opens the Mixamo mouth.")
    p.add_argument("--lip_method",
                   choices=["flame_tps", "geometric", "open_boundary", "sharp_crease"],
                   default=None,
                   help="How to detect the Mixamo lip line for jaw skinning. "
                        "Default falls back to CHARACTER_DEFAULTS[slug]['lip_method'] "
                        "(sharp_crease for known Mixamo characters). "
                        "flame_tps: TPS-warp SMPL-X FLAME lip landmarks. "
                        "geometric: midpoint of nose tip + chin tip. "
                        "open_boundary: modal Y of open-boundary edges in central face. "
                        "sharp_crease: modal Y of edges with >60deg dihedral angle.")
    # Per-character mouth / jaw geometry. Defaults are pulled from
    # CHARACTER_DEFAULTS below when --character_name is in the table; CLI
    # values always override.
    p.add_argument("--mouth_y_below_head_top", type=float, default=None,
                   help="Vertical offset from the rest-frame head_top Y to the lip line "
                        "(in meters). Larger = mouth sits lower on the head. Typical 0.18-0.27.")
    p.add_argument("--mouth_radius_m", type=float, default=None,
                   help="Radius of the jaw-weight Gaussian sphere centered on the lip line. "
                        "Larger = jaw influence extends further into the cheek/chin. "
                        "Typical 0.06-0.12.")
    p.add_argument("--jaw_y_upper_offset", type=float, default=None,
                   help="Verts above (mouth_center.Y + this offset) get NO jaw weight. "
                        "Positive keeps slightly-above-lip-line verts pinned. Typical 0.003.")
    p.add_argument("--jaw_y_lower_offset", type=float, default=None,
                   help="Verts below (mouth_center.Y + this offset) get FULL jaw weight. "
                        "Negative number; smooth ramp between upper and lower. Typical -0.010.")
    p.add_argument("--jaw_pivot_y_above_mouth", type=float, default=None,
                   help="TMJ pivot Y offset above mouth_center (m). Typical 0.02.")
    p.add_argument("--jaw_pivot_z_behind_mouth", type=float, default=None,
                   help="TMJ pivot Z offset behind mouth_center (m). Typical 0.04.")
    p.add_argument("--expr_dir_scale", type=float, default=None,
                   help="Extra multiplicative scale on FLAME expression displacements, on top "
                        "of the automatic bbox scaling. 1.0 = no extra scale, 2.0 = doubled "
                        "mouth motion magnitude.")
    return p.parse_args()


# Per-character tunables. Each character's face has slightly different
# proportions (head size, chin length, lip-line position relative to the
# crown). These defaults were tuned by inspecting the QA render + face
# motion video; override on the CLI for fine-tuning. Keys must match the
# --character_name slug.
CHARACTER_DEFAULTS: dict[str, dict[str, float]] = {
    # NOTE: all Mixamo characters here use T-pose (arms straight out), so
    # apose_init=False is the right default -- the script's default A-pose
    # init rotates SMPL-X shoulders DOWN by ~57 deg, which fights the
    # outstretched Mixamo arms and causes the chamfer to drift t_y up by
    # 0.5-1.0 m to maximise torso overlap, giving 0.34 m RMS misalignment.
    # With apose_init=False the same characters fit at ~0.18 m RMS.
    "y_bot": {
        "mouth_y_below_head_top": 0.18,
        "mouth_radius_m": 0.08,
        "jaw_y_upper_offset": 0.003,
        "jaw_y_lower_offset": -0.010,
        "jaw_pivot_y_above_mouth": 0.02,
        "jaw_pivot_z_behind_mouth": 0.04,
        "expr_dir_scale": 1.0,
        "apose_init": False,
        "lip_method": "sharp_crease",
    },
    "ch08": {
        # Ch08 has a tall, broad head with a long chin; mouth sits ~25 cm
        # below the crown. The Y ramp is shifted down so the (relatively
        # tall) upper lip stays pinned and only the lower lip + chin drop.
        "mouth_y_below_head_top": 0.25,
        "mouth_radius_m": 0.10,
        "jaw_y_upper_offset": 0.003,    # 3mm above the detected lip line = no jaw
        "jaw_y_lower_offset": -0.005,   # 5mm below = full jaw (so lower lip rotates fully)
        "jaw_pivot_y_above_mouth": 0.025,
        "jaw_pivot_z_behind_mouth": 0.05,
        "expr_dir_scale": 1.3,
        "apose_init": False,
        "lip_method": "sharp_crease",
    },
    "ch31": {
        # Ch31 is similar to Ch08 but slightly shorter chin.
        "mouth_y_below_head_top": 0.23,
        "mouth_radius_m": 0.10,
        "jaw_y_upper_offset": 0.003,
        "jaw_y_lower_offset": -0.005,
        "jaw_pivot_y_above_mouth": 0.022,
        "jaw_pivot_z_behind_mouth": 0.05,
        "expr_dir_scale": 1.3,
        "apose_init": False,
        "lip_method": "sharp_crease",
    },
    "remy": {
        # Remy is smaller / leaner; mouth sits ~22 cm below crown.
        "mouth_y_below_head_top": 0.22,
        "mouth_radius_m": 0.08,
        "jaw_y_upper_offset": 0.003,
        "jaw_y_lower_offset": -0.005,
        "jaw_pivot_y_above_mouth": 0.020,
        "jaw_pivot_z_behind_mouth": 0.04,
        "expr_dir_scale": 1.3,
        "apose_init": False,
        "lip_method": "sharp_crease",
    },
}


def resolve_char_params(args: argparse.Namespace, slug: str) -> dict:
    """Merge CLI args, CHARACTER_DEFAULTS for slug, and fallback defaults."""
    fallback = {
        "mouth_y_below_head_top": 0.22,
        "mouth_radius_m": 0.08,
        "jaw_y_upper_offset": 0.003,
        "jaw_y_lower_offset": -0.010,
        "jaw_pivot_y_above_mouth": 0.02,
        "jaw_pivot_z_behind_mouth": 0.04,
        "expr_dir_scale": 1.0,
        "apose_init": True,
        "lip_method": "sharp_crease",
    }
    char_defaults = CHARACTER_DEFAULTS.get(slug, {})
    resolved = {**fallback, **char_defaults}
    for k in resolved:
        v = getattr(args, k, None)
        if v is not None:
            resolved[k] = v
    return resolved


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slug = args.character_name or _slug_from_fbx(args.fbx_path)
    out_npz = args.out_dir / f"{slug}.npz"
    out_png = args.out_dir / f"{slug}_qa.png"
    params = resolve_char_params(args, slug)
    print(f"[fit] FBX           : {args.fbx_path}")
    print(f"[fit] character slug: {slug}")
    print(f"[fit] output npz    : {out_npz}")
    print(f"[fit] per-char params: {params}")

    # ---- 1. Extract Mixamo rest-pose mesh ----
    print("[fit] loading FBX via Blender headless ...")
    fbx_mesh = load_fbx_mesh(args.fbx_path, blender_bin=args.blender_bin)
    print(f"[fit]   verts={fbx_mesh.verts.shape}  faces={fbx_mesh.faces.shape}  "
          f"uv={'yes' if fbx_mesh.uv is not None else 'no'}")

    # ---- 2. Build SMPL-X (mirrors basecausalcodec_trainer settings) ----
    from miburi.utils.viser_scene import make_smplx_model
    smplx_model = make_smplx_model(args.smplx_model, gender="NEUTRAL_2020")

    # ---- 3. Chamfer fit ----
    # Use ONLY the humanoid body submesh as the chamfer target. The other
    # submeshes (hair, hoodie, pants, shoes, beard, ...) don't conform to
    # SMPL-X's body silhouette and would blow up beta if included. They get
    # the same scale/R/t transform applied below, so they stay glued to the
    # body.
    if fbx_mesh.body_vert_range is not None:
        bv0, bv1 = int(fbx_mesh.body_vert_range[0]), int(fbx_mesh.body_vert_range[1])
        chamfer_target = fbx_mesh.verts[bv0:bv1]
        print(f"[fit] chamfer target = body submesh verts [{bv0}:{bv1}) "
              f"({chamfer_target.shape[0]}/{fbx_mesh.verts.shape[0]} of combined)")
    else:
        chamfer_target = fbx_mesh.verts
        print(f"[fit] chamfer target = all {fbx_mesh.verts.shape[0]} verts (no body submesh hint)")

    # Resolve apose_init: CLI --no_apose_init forces False; otherwise use
    # CHARACTER_DEFAULTS (which sets False for all Mixamo T-pose chars).
    _apose_init = (False if args.no_apose_init else params["apose_init"])
    print(f"[fit] running chamfer fit (apose_init={_apose_init}) ...")
    fit = fit_smplx_to_mesh(
        smplx_model,
        target_verts=chamfer_target,
        device=args.device,
        n_iters_align=args.n_iters_align,
        n_iters_joint=args.n_iters_joint,
        apose_init=_apose_init,
        optimize_body_pose=args.optimize_body_pose,
        hand_loss_weight=args.hand_loss_weight,
    )
    print(f"[fit] final chamfer = {fit.final_chamfer:.6f}  "
          f"sqrt = {np.sqrt(fit.final_chamfer):.4f} m")

    # The fit absorbed a per-axis scale on the *target* (Mixamo body submesh)
    # around its body centroid. Apply the same scaling to ALL combined verts
    # (body + clothes + hair) so they stay aligned.
    centroid = chamfer_target.mean(axis=0, keepdims=True)
    mixamo_verts_aligned = centroid + (fbx_mesh.verts - centroid) * fit.scale[None, :]

    # ---- 4. LBS weights ----
    # Two sources: 'native' uses the FBX rig's own per-bone weights (mapped
    # to SMPL-X joints via mixamo_bone_map), 'barycentric' falls back to
    # closest-point on the fitted SMPL-X surface.
    smplx_faces_np = smplx_model.faces.cpu().numpy().astype(np.int32)
    smplx_lbs_np = smplx_model.lbs_weights.cpu().numpy().astype(np.float32)

    if args.skinning == "native":
        if fbx_mesh.native_lbs_weights is None:
            print("[fit] WARNING: --skinning=native requested but the FBX has no "
                  "vertex_groups / armature. Falling back to barycentric.")
            skinning_used = "barycentric"
        else:
            skinning_used = "native"
    else:
        skinning_used = "barycentric"

    if skinning_used == "native":
        lbs_weights = fbx_mesh.native_lbs_weights.astype(np.float32)
        print(f"[fit] using native FBX skinning  (rows sum range: "
              f"[{lbs_weights.sum(1).min():.3f}, {lbs_weights.sum(1).max():.3f}])")
    else:
        print("[fit] transferring LBS weights via barycentric on SMPL-X surface ...")
        lbs_weights = transfer_lbs_weights(
            target_verts=mixamo_verts_aligned,
            smplx_verts=fit.smplx_verts_fit,
            smplx_faces=smplx_faces_np,
            smplx_lbs_weights=smplx_lbs_np,
        )

    # ---- 4b. Un-globalize Mixamo to SMPL-X rest frame ----
    # SMPL-X forward at (beta, body_pose=0, global=0) produces v_shaped in
    # its internal rest frame. The fitted global R, t mapped that frame to
    # the Mixamo world. To make `verts_tpose` interchangeable with
    # `j_t = J_regressor @ v_shaped` at runtime, apply the inverse here.
    def _rodrigues_np(aa: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(aa)
        if theta < 1e-8:
            return np.eye(3, dtype=np.float32)
        k = aa / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
        return (np.eye(3, dtype=np.float32) + np.sin(theta) * K
                + (1.0 - np.cos(theta)) * K @ K).astype(np.float32)

    R_fit = _rodrigues_np(fit.global_rotation.astype(np.float32))    # (3, 3)
    t_fit = fit.global_translation.astype(np.float32)                # (3,)
    # Inverse of v_world = R @ v_rest + t  ->  v_rest = R^T @ (v_world - t)
    # In numpy row-vector form: v_rest = (v_world - t) @ R
    mixamo_verts_rest = ((mixamo_verts_aligned - t_fit[None, :]) @ R_fit).astype(np.float32)

    # ---- 4b.5. Per-vertex skeleton retarget (the key articulation fix) ----
    # Mixamo verts in the bind pose are positioned relative to Mixamo's
    # ACTUAL bone positions, not SMPL-X's J_regressor predictions. Standard
    # LBS expects (v_rest - j_rest) to be the rest offset from a vert to its
    # joint. We use SMPL-X's J_regressor for j_rest at runtime, so the
    # offset is wrong by (j_smplx - j_mixamo) for each vert/joint pair.
    # Shift each vertex by the weighted sum of those offsets so the math
    # works out.
    if not args.no_skeleton_retarget and fbx_mesh.bone_heads is not None:
        import torch as _torch
        # Apply same Y-up + chamfer-scale + un-globalize chain to the
        # Mixamo bone positions that we applied to verts. fbx_mesh.bone_heads
        # is already Y-up (load_fbx_mesh did the swap).
        bones_yup = fbx_mesh.bone_heads.astype(np.float32)
        bones_aligned = centroid + (bones_yup - centroid) * fit.scale[None, :]
        bones_rest = ((bones_aligned - t_fit[None, :]) @ R_fit).astype(np.float32)

        # j_mixamo[j] = position of the Mixamo bone mapped to SMPL-X joint j.
        # If multiple bones map to the same SMPL-X joint (via parent-chain
        # fallback for unmapped tips), prefer the bone whose name maps
        # directly. We score: direct name match = 0, parent-fallback = 1.
        j_mixamo = np.zeros((55, 3), dtype=np.float32)
        j_mixamo_set = np.zeros(55, dtype=bool)
        score = np.full(55, 999, dtype=np.int32)
        from scripts.utils.smplx_fitting.mixamo_bone_map import (
            MIXAMO_TO_SMPLX, strip_prefix,
        )
        for bi, name in enumerate(fbx_mesh.bone_names):
            j = int(fbx_mesh.bone_smplx_joint_idx[bi])
            direct = MIXAMO_TO_SMPLX.get(strip_prefix(str(name))) is not None
            s = 0 if direct else 1
            if s < score[j]:
                j_mixamo[j] = bones_rest[bi]
                j_mixamo_set[j] = True
                score[j] = s

        # SMPL-X joint positions in the runtime frame (beta=fitted, since
        # pose_mixamo_character internally uses char.betas at runtime).
        with _torch.no_grad():
            shape_comps = _torch.cat([
                _torch.from_numpy(fit.betas).unsqueeze(0).to(args.device),
                _torch.zeros(1, smplx_model.NUM_EXPR_COEFFS, device=args.device),
            ], dim=-1)
            sd = _torch.cat([
                smplx_model.shapedirs[:, :, :fit.betas.shape[0]],
                smplx_model.exprdirs[:, :, :smplx_model.NUM_EXPR_COEFFS],
            ], dim=-1)
            v_shaped_fit = (smplx_model.v_template
                            + _torch.einsum("bi,nki->bnk", shape_comps, sd))[0]
            j_smplx = _torch.einsum("ik,ji->jk", v_shaped_fit, smplx_model.J_regressor)
        j_smplx_np = j_smplx.cpu().numpy().astype(np.float32)

        # For SMPL-X joints with no Mixamo counterpart, set j_mixamo = j_smplx
        # so the offset for those joints is zero (no shift). This covers
        # SMPL-X-only joints like jaw / eyes (22-24).
        j_mixamo[~j_mixamo_set] = j_smplx_np[~j_mixamo_set]

        # Per-vertex offset: shift[i] = sum_j w_ij * (j_smplx[j] - j_mixamo[j]).
        # Only apply the retarget to FINGER joints (25..54). Body joints
        # (0..21) were already aligned by the chamfer fit at the surface
        # level; forcing joint-position offsets there fights the fit and
        # introduces visible gaps at knees/hips/spine. Head/jaw/eye joints
        # (22..24) are usually absent in Mixamo and don't matter.
        joint_offsets = j_smplx_np - j_mixamo                          # (55, 3)
        retarget_mask = np.zeros(55, dtype=bool)
        retarget_mask[25:55] = True                                     # finger joints only
        joint_offsets[~retarget_mask] = 0.0
        vertex_shifts = lbs_weights @ joint_offsets                     # (Nm, 3)
        mixamo_verts_rest = (mixamo_verts_rest + vertex_shifts).astype(np.float32)

        # Diagnostic: how much each vert was moved on average (overall, and
        # restricted to verts dominantly weighted to finger joints).
        shift_mag = np.linalg.norm(vertex_shifts, axis=1)
        finger_dom = (lbs_weights[:, 25:55].sum(1) > 0.5)
        print(f"[fit] skeleton retarget (finger joints only): "
              f"all_verts mean={shift_mag.mean()*1000:.1f}mm max={shift_mag.max()*1000:.1f}mm; "
              f"finger_verts ({finger_dom.sum()}) mean={shift_mag[finger_dom].mean()*1000:.1f}mm "
              f"max={shift_mag[finger_dom].max()*1000:.1f}mm")
    elif fbx_mesh.bone_heads is None and not args.no_skeleton_retarget:
        print("[fit] WARNING: no bone_heads in FBX bundle -- skipping skeleton retarget. "
              "Articulation will be off.")

    # ---- 4b.6. Transfer jaw-joint skinning weight to lower-face verts ----
    # Mixamo armatures don't have a jaw bone; without this fix the whole
    # head is rigid even when head_pose[0] (jaw open) varies. We use
    # SMPL-X's own jaw weight as the template for each Mixamo vert via
    # closest-vert lookup.
    if not args.no_jaw_skinning and skinning_used == "native":
        import torch as _torch
        if "v_shaped_fit" not in locals():
            with _torch.no_grad():
                shape_comps = _torch.cat([
                    _torch.from_numpy(fit.betas).unsqueeze(0).to(args.device),
                    _torch.zeros(1, smplx_model.NUM_EXPR_COEFFS, device=args.device),
                ], dim=-1)
                sd = _torch.cat([
                    smplx_model.shapedirs[:, :, :fit.betas.shape[0]],
                    smplx_model.exprdirs[:, :, :smplx_model.NUM_EXPR_COEFFS],
                ], dim=-1)
                v_shaped_fit = (smplx_model.v_template
                                + _torch.einsum("bi,nki->bnk", shape_comps, sd))[0]
        v_shaped_fit_np = v_shaped_fit.cpu().numpy().astype(np.float32)
        smplx_lbs_full_local = smplx_model.lbs_weights.cpu().numpy().astype(np.float32)
        # Derive mouth_center from the SMPL-X FLAME lip landmarks (FLAME
        # indices 31-50 are the 20 lip landmarks on the actual SMPL-X lip
        # ring), TPS-warped into Mixamo space. The geometric (nose+chin)/2
        # estimate lands at upper-chin for Mixamo characters with long
        # chins; the FLAME lip ring is the actual lip seam.
        from scripts.utils.smplx_fitting.transfer_face_blendshapes import (
            detect_face_landmarks_geometric, detect_mixamo_lip_line,
        )
        _head_y = float(np.percentile(mixamo_verts_rest[:, 1], 99))
        _face_band_mask = ((mixamo_verts_rest[:, 1] > _head_y - 0.30)
                           & (mixamo_verts_rest[:, 1] < _head_y))
        _mx_face_pool = mixamo_verts_rest[_face_band_mask] if _face_band_mask.any() else mixamo_verts_rest
        # Align SMPL-X anchor verts to the Mixamo rest frame by matching
        # head-top Y. When the chamfer fit drifts t_fit, the saved Mixamo
        # rest frame and the SMPL-X v_shaped joint frame differ in Y by up
        # to ~30 cm — K-NN across them then has no candidates within
        # max_dist. Shifting v_shaped_fit by the head-top delta brings
        # them into a common frame for the spatial K-NN.
        _smplx_head_y = float(np.percentile(v_shaped_fit_np[:, 1], 99))
        _frame_y_shift = _head_y - _smplx_head_y
        v_shaped_fit_aligned = v_shaped_fit_np.copy()
        v_shaped_fit_aligned[:, 1] = v_shaped_fit_aligned[:, 1] + _frame_y_shift
        print(f"[fit] frame Y-shift (SMPL-X -> Mixamo rest): {_frame_y_shift:+.3f} "
              f"(SMPL-X head_top={_smplx_head_y:.3f})")
        # Dispatch lip detection based on --lip_method (CLI > per-char default).
        mouth_center_for_jaw = None
        _method = args.lip_method if args.lip_method is not None else params["lip_method"]
        print(f"[fit] lip detection method: {_method}")
        if _method == "flame_tps":
            try:
                _raw_npz = np.load(args.smplx_model, allow_pickle=True)
                _lmk_faces_idx_raw = np.asarray(_raw_npz["lmk_faces_idx"], dtype=np.int64)
                _lmk_bary_raw = np.asarray(_raw_npz["lmk_bary_coords"], dtype=np.float32)
                _faces_raw = np.asarray(_raw_npz["f"], dtype=np.int64)
                _all_flame_lmks = np.einsum(
                    "lij,li->lj",
                    v_shaped_fit_aligned[_faces_raw[_lmk_faces_idx_raw]],
                    _lmk_bary_raw,
                ).astype(np.float32)
                _flame_lip_lmks = _all_flame_lmks[31:51]
                _smplx_expr_mass = np.linalg.norm(
                    smplx_model.exprdirs.detach().cpu().numpy().reshape(
                        smplx_model.exprdirs.shape[0], -1), axis=1)
                _smplx_anchor_mask_for_lip = _smplx_expr_mass > 0.2 * _smplx_expr_mass.max()
                _smplx_anchors_for_lip = v_shaped_fit_aligned[_smplx_anchor_mask_for_lip]
                _lip_info = detect_mixamo_lip_line(
                    _smplx_anchors_for_lip, _flame_lip_lmks, _mx_face_pool,
                )
                if _lip_info:
                    mouth_center_for_jaw = np.array(
                        [0.0, _lip_info["lip_line_y"], _lip_info["lip_line_z"]],
                        dtype=np.float32,
                    )
                    print(f"[fit] mouth_center (flame_tps): {mouth_center_for_jaw.round(3)}")
            except Exception as _e:
                print(f"[fit] flame_tps failed: {_e!r}")
        elif _method == "geometric":
            _mx_geom = detect_face_landmarks_geometric(_mx_face_pool)
            if "mouth" in _mx_geom:
                mouth_center_for_jaw = _mx_geom["mouth"].copy()
                print(f"[fit] mouth_center (geometric): {mouth_center_for_jaw.round(3)}")
        elif _method == "open_boundary":
            from collections import Counter, defaultdict
            _edges = np.vstack([fbx_mesh.faces[:, [0, 1]], fbx_mesh.faces[:, [1, 2]], fbx_mesh.faces[:, [2, 0]]])
            _e_s = np.sort(_edges, axis=1)
            _cnt = Counter(map(tuple, _e_s))
            _bnd_edges = np.array([list(x) for x, c in _cnt.items() if c == 1], dtype=np.int64)
            if _bnd_edges.shape[0]:
                _vidx = np.unique(_bnd_edges.flatten())
                _pos = mixamo_verts_rest[_vidx]
                _in_face = ((_pos[:, 1] > _head_y - 0.30) & (_pos[:, 1] < _head_y - 0.05)
                            & (np.abs(_pos[:, 0]) < 0.05))
                _face_pool_z = mixamo_verts_rest[(mixamo_verts_rest[:, 1] > _head_y - 0.30)
                                                  & (mixamo_verts_rest[:, 1] < _head_y - 0.05)
                                                  & (np.abs(mixamo_verts_rest[:, 0]) < 0.05), 2]
                if _face_pool_z.size:
                    _in_face = _in_face & (_pos[:, 2] > np.percentile(_face_pool_z, 60))
                _pos_f = _pos[_in_face]
                if _pos_f.shape[0] > 5:
                    _bins = defaultdict(int)
                    for y in _pos_f[:, 1]:
                        _bins[round(y * 100) / 100] += 1
                    _m = max(_bins, key=_bins.get)
                    _modal_v = _pos_f[np.abs(_pos_f[:, 1] - _m) < 0.015]
                    _mz = float(_modal_v[:, 2].mean()) if _modal_v.size else float(_pos_f[:, 2].mean())
                    mouth_center_for_jaw = np.array([0.0, _m, _mz], dtype=np.float32)
                    print(f"[fit] mouth_center (open_boundary): {mouth_center_for_jaw.round(3)} (count={len(_modal_v)})")
        elif _method == "sharp_crease":
            from collections import defaultdict
            _fv = mixamo_verts_rest[fbx_mesh.faces]
            _fn = np.cross(_fv[:, 1] - _fv[:, 0], _fv[:, 2] - _fv[:, 0])
            _fn = _fn / (np.linalg.norm(_fn, axis=-1, keepdims=True) + 1e-9)
            _e2f = defaultdict(list)
            for _fi, (_a, _b, _c) in enumerate(fbx_mesh.faces):
                for _u, _w in ((_a, _b), (_b, _c), (_c, _a)):
                    _e2f[tuple(sorted((_u, _w)))].append(_fi)
            _sharp_v = []
            for _k, _fs in _e2f.items():
                if len(_fs) == 2:
                    _cos = float(np.clip(np.dot(_fn[_fs[0]], _fn[_fs[1]]), -1.0, 1.0))
                    if np.degrees(np.arccos(_cos)) > 60:
                        _sharp_v.extend(_k)
            if _sharp_v:
                _vidx = np.unique(np.array(_sharp_v, dtype=np.int64))
                _pos = mixamo_verts_rest[_vidx]
                _in_face = ((_pos[:, 1] > _head_y - 0.30) & (_pos[:, 1] < _head_y - 0.05)
                            & (np.abs(_pos[:, 0]) < 0.05))
                _face_pool_z = mixamo_verts_rest[(mixamo_verts_rest[:, 1] > _head_y - 0.30)
                                                  & (mixamo_verts_rest[:, 1] < _head_y - 0.05)
                                                  & (np.abs(mixamo_verts_rest[:, 0]) < 0.05), 2]
                if _face_pool_z.size:
                    _in_face = _in_face & (_pos[:, 2] > np.percentile(_face_pool_z, 60))
                _pos_f = _pos[_in_face]
                if _pos_f.shape[0] > 5:
                    _bins = defaultdict(int)
                    for y in _pos_f[:, 1]:
                        _bins[round(y * 100) / 100] += 1
                    _m = max(_bins, key=_bins.get)
                    _modal_v = _pos_f[np.abs(_pos_f[:, 1] - _m) < 0.015]
                    _mz = float(_modal_v[:, 2].mean()) if _modal_v.size else float(_pos_f[:, 2].mean())
                    mouth_center_for_jaw = np.array([0.0, _m, _mz], dtype=np.float32)
                    print(f"[fit] mouth_center (sharp_crease): {mouth_center_for_jaw.round(3)} (count={len(_modal_v)})")
        # Final fallback: parametric.
        if mouth_center_for_jaw is None:
            _mx_geom = detect_face_landmarks_geometric(_mx_face_pool)
            mouth_center_for_jaw = (_mx_geom["mouth"].copy()
                if "mouth" in _mx_geom
                else np.array([0.0, _head_y - 0.22, 0.0], dtype=np.float32))
            print(f"[fit] mouth_center (FALLBACK to geometric): {mouth_center_for_jaw.round(3)}")
        lbs_weights = add_jaw_skinning(
            mixamo_verts=mixamo_verts_rest,
            mixamo_lbs=lbs_weights,
            smplx_verts=v_shaped_fit_aligned,
            smplx_lbs=smplx_lbs_full_local,
            mouth_center=mouth_center_for_jaw,
            mouth_radius_m=params["mouth_radius_m"],
            jaw_y_upper_offset=params["jaw_y_upper_offset"],
            jaw_y_lower_offset=params["jaw_y_lower_offset"],
        )

        # Anatomical jaw pivot for runtime jaw rotation. SMPL-X's J_regressor
        # puts joint 22 at the top of the head (Y=0.27) which is the wrong
        # pivot for "mouth open" -- rotating lip verts around it produces a
        # wide downward swing (face tilts forward) instead of lips parting.
        # The actual TMJ hinge is at ear level (Y just above the ear, Z
        # slightly back). Save this pivot in the bundle; the runtime helper
        # will apply jaw rotation around it manually for verts marked with
        # jaw LBS weight.
        anatomical_jaw_pivot = np.array([
            0.0,
            mouth_center_for_jaw[1] + params["jaw_pivot_y_above_mouth"],
            mouth_center_for_jaw[2] - params["jaw_pivot_z_behind_mouth"],
        ], dtype=np.float32)
        print(f"[fit] anatomical jaw pivot: {anatomical_jaw_pivot.round(3)} "
              f"(mouth_center was {mouth_center_for_jaw.round(3)})")

    # ---- 4c. Per-wrist bind-pose correction ----
    # Mixamo bind poses often have palms rolled relative to SMPL-X's T-pose.
    # Rotate each hand around the wrist so thumb->pinky aligns with SMPL-X's.
    # Only runs when native skinning is in use (the joint mapping is what
    # tells us which verts belong to which hand).
    if not args.no_hand_bind_correction and skinning_used == "native":
        # Use SMPL-X at beta=0 as the reference -- gesture-LM hand poses are
        # interpreted in this canonical frame at runtime (motion_vis_server
        # always passes shape=zeros). Using beta=fitted here would introduce
        # a ~15deg bias between bind orientation and what the LM expects.
        import torch as _torch
        with _torch.no_grad():
            v_shaped = smplx_model.v_template                                       # (V, 3)
            j_rest = _torch.einsum("ik,ji->jk", v_shaped, smplx_model.J_regressor)  # (J, 3)
        v_shaped_np = v_shaped.cpu().numpy().astype(np.float32)
        j_rest_np = j_rest.cpu().numpy().astype(np.float32)
        smplx_lbs_full = smplx_model.lbs_weights.cpu().numpy().astype(np.float32)

        for side in ("left", "right"):
            # Align the whole hand orientation (palm normal) to SMPL-X via a
            # full 3D basis match around the wrist. No per-finger spread or
            # end-effector correction -- Y-Bot's fingers keep their bind-pose
            # spatial arrangement.
            mixamo_verts_rest, _ = correct_hand_bind_pose(
                mixamo_verts=mixamo_verts_rest,
                mixamo_lbs=lbs_weights,
                smplx_v_shaped=v_shaped_np,
                smplx_lbs=smplx_lbs_full,
                smplx_j_rest=j_rest_np,
                side=side,
            )
    print(f"[fit]   lbs_weights = {lbs_weights.shape}  row sums in "
          f"[{lbs_weights.sum(1).min():.3f}, {lbs_weights.sum(1).max():.3f}]")

    # ---- 5. Save bundle ----
    # verts_tpose lives in SMPL-X rest frame; runtime LBS uses (verts_tpose - j_t).
    save_kwargs = dict(
        betas=fit.betas.astype(np.float32),
        verts_tpose=mixamo_verts_rest,
        faces=fbx_mesh.faces.astype(np.int32),
        lbs_weights=lbs_weights.astype(np.float32),
        body_pose_fit=fit.body_pose.astype(np.float32),
        global_rotation_fit=fit.global_rotation.astype(np.float32),
        global_translation_fit=fit.global_translation.astype(np.float32),
        scale_fit=fit.scale.astype(np.float32),
        smplx_model_path=str(args.smplx_model),
        source_fbx=str(args.fbx_path),
        skinning_source=skinning_used,
    )
    if fbx_mesh.uv is not None:
        save_kwargs["uv_coords"] = fbx_mesh.uv.astype(np.float32)
    if fbx_mesh.vertex_colors is not None:
        save_kwargs["vertex_colors"] = fbx_mesh.vertex_colors.astype(np.uint8)
    if fbx_mesh.submesh_names is not None:
        save_kwargs["submesh_names"] = fbx_mesh.submesh_names
        save_kwargs["submesh_ranges"] = fbx_mesh.submesh_ranges.astype(np.int32)
    try:
        save_kwargs["anatomical_jaw_pivot"] = anatomical_jaw_pivot
    except NameError:
        pass
    try:
        save_kwargs["lip_method"] = np.array(_method)
        save_kwargs["mouth_center"] = mouth_center_for_jaw
    except NameError:
        pass

    if args.with_face:
        print("[fit] --with_face: transferring FLAME expression blendshapes onto Mixamo face verts ...")
        smplx_exprdirs_np = smplx_model.exprdirs.detach().cpu().numpy().astype(np.float32)
        # Use beta-FITTED SMPL-X surface as the reference: pose_mixamo_character
        # internally uses shape=char.betas at runtime so the Mixamo verts
        # share the beta-fitted SMPL-X frame. Closest-point on beta=0 SMPL-X
        # would mis-pair Mixamo head verts (their nearest beta=0 SMPL-X vert
        # could be tens of cm away in the chest region).
        import torch as _torch
        with _torch.no_grad():
            shape_comps = _torch.cat([
                _torch.from_numpy(fit.betas).unsqueeze(0).to(args.device),
                _torch.zeros(1, smplx_model.NUM_EXPR_COEFFS, device=args.device),
            ], dim=-1)
            sd = _torch.cat([
                smplx_model.shapedirs[:, :, :fit.betas.shape[0]],
                smplx_model.exprdirs[:, :, :smplx_model.NUM_EXPR_COEFFS],
            ], dim=-1)
            smplx_v_shaped_for_face = (smplx_model.v_template
                                       + _torch.einsum("bi,nki->bnk", shape_comps, sd))[0]
        smplx_v_for_face = smplx_v_shaped_for_face.cpu().numpy().astype(np.float32)
        # Align SMPL-X verts to the Mixamo rest frame by head-top Y (same
        # rationale as the jaw-skinning call). Without this alignment, the
        # K-NN spatial match has no Mixamo candidates within max_dist of
        # any SMPL-X anchor when chamfer's t_fit drifts.
        _sm_head_y_face = float(np.percentile(smplx_v_for_face[:, 1], 99))
        _mx_head_y_face = float(np.percentile(mixamo_verts_rest[:, 1], 99))
        _frame_y_shift_face = _mx_head_y_face - _sm_head_y_face
        smplx_v_for_face_aligned = smplx_v_for_face.copy()
        smplx_v_for_face_aligned[:, 1] = (
            smplx_v_for_face_aligned[:, 1] + _frame_y_shift_face
        )
        # Use Mixamo rest-frame geometry to derive mouth_center.
        head_y = _mx_head_y_face
        # Forward extent of mesh near head height (approx mouth/nose Z)
        face_y_near_head = (mixamo_verts_rest[:, 1] > head_y - 0.30) & \
                           (mixamo_verts_rest[:, 1] < head_y)
        if face_y_near_head.any():
            forward_z = float(np.percentile(mixamo_verts_rest[face_y_near_head, 2], 95))
            head_z = float(mixamo_verts_rest[
                mixamo_verts_rest[:, 1] > head_y - 0.05, 2
            ].mean())
        else:
            head_z = 0.0
            forward_z = head_z + 0.08
        mouth_z = 0.5 * (head_z + forward_z)
        mouth_y = head_y - params["mouth_y_below_head_top"]
        smplx_mouth_pos_for_face = np.array([0.0, mouth_y, mouth_z], dtype=np.float32)
        print(f"[fit] mouth_center for face transfer: {smplx_mouth_pos_for_face.round(3)} "
              f"(rest-frame head_top_Y={head_y:.3f}; frame Y-shift={_frame_y_shift_face:+.3f})")
        # SMPL-X face landmarks (vertex IDs from smplx.vertex_ids). These
        # let us label each face vert by anatomical region (nose, eyes,
        # ears, mouth, chin) and propagate the labels through the K-NN to
        # the Mixamo face — used for diagnostic visualization so we can
        # see whether the K-NN matched chin->chin, nose->nose, etc.
        from smplx.vertex_ids import vertex_ids as _smplx_vertex_ids
        _lm_ids = _smplx_vertex_ids["smplx"]
        _lm_world = {
            name: smplx_v_for_face_aligned[_lm_ids[name]].astype(np.float32)
            for name in ("nose", "leye", "reye", "lear", "rear")
            if name in _lm_ids
        }
        # 51 FLAME static landmarks (brows + nose + eyes + mouth) from the
        # SMPL-X npz. Much richer anatomical Voronoi than 5 single-vertex
        # landmarks. Loaded directly from the model npz since the standard
        # `smplx` package does not expose `lmk_faces_idx` / `lmk_bary_coords`.
        try:
            _raw = np.load(args.smplx_model, allow_pickle=True)
            _lmk_faces_idx_np = np.asarray(_raw["lmk_faces_idx"], dtype=np.int64)
            _lmk_bary_np = np.asarray(_raw["lmk_bary_coords"], dtype=np.float32)
            _faces_np_raw = np.asarray(_raw["f"], dtype=np.int64)
            # Compute landmark positions on the Y-shifted SMPL-X mesh.
            _lmk_tri = _faces_np_raw[_lmk_faces_idx_np]            # (51, 3)
            _lmk_vert_pos = smplx_v_for_face_aligned[_lmk_tri]      # (51, 3, 3)
            _flame_lmk_positions = np.einsum(
                "lij,li->lj", _lmk_vert_pos, _lmk_bary_np,
            ).astype(np.float32)                                    # (51, 3)
        except Exception as e:
            print(f"[fit] couldn't load FLAME landmarks: {e!r}; falling back to 5-lmk")
            _flame_lmk_positions = None
        # Only consider BODY submesh verts as face candidates -- hair,
        # clothing, and accessory submeshes shouldn't deform with FLAME
        # expressions even if their verts are spatially near the face.
        if fbx_mesh.body_vert_range is not None:
            bv0, bv1 = int(fbx_mesh.body_vert_range[0]), int(fbx_mesh.body_vert_range[1])
            body_verts_for_face = mixamo_verts_rest[bv0:bv1]
            (body_face_mask, body_expr_dirs,
             sm_region_per_anchor, body_region_per_vert) = transfer_face_blendshapes(
                target_verts=body_verts_for_face,
                smplx_verts=smplx_v_for_face_aligned,
                smplx_faces=smplx_faces_np,
                smplx_exprdirs=smplx_exprdirs_np,
                mouth_center=smplx_mouth_pos_for_face,
                expr_dir_scale=params["expr_dir_scale"],
                landmark_world_pos=_lm_world,
                flame_landmark_positions=_flame_lmk_positions,
            )
            # Expand back to the full combined-mesh index space: False for
            # all non-body verts; True only for body verts marked face.
            face_vert_mask = np.zeros(mixamo_verts_rest.shape[0], dtype=bool)
            face_vert_mask[bv0:bv1] = body_face_mask
            expr_dirs_face = body_expr_dirs
            mixamo_region_per_vert = np.full(mixamo_verts_rest.shape[0], -1, dtype=np.int32)
            if body_region_per_vert is not None:
                mixamo_region_per_vert[bv0:bv1] = body_region_per_vert
        else:
            (face_vert_mask, expr_dirs_face,
             sm_region_per_anchor, mixamo_region_per_vert) = transfer_face_blendshapes(
                target_verts=mixamo_verts_rest,
                smplx_verts=smplx_v_for_face_aligned,
                smplx_faces=smplx_faces_np,
                smplx_exprdirs=smplx_exprdirs_np,
                mouth_center=smplx_mouth_pos_for_face,
                expr_dir_scale=params["expr_dir_scale"],
                landmark_world_pos=_lm_world,
                flame_landmark_positions=_flame_lmk_positions,
            )
        save_kwargs["face_vert_mask"] = face_vert_mask
        save_kwargs["expr_dirs_face"] = expr_dirs_face
        if mixamo_region_per_vert is not None:
            save_kwargs["face_region_per_vert"] = mixamo_region_per_vert
            save_kwargs["face_region_names"] = np.array(
                ["brow", "nose", "leye", "reye", "mouth", "lear", "rear", "chin"]
            )
        print(f"[fit]   saved face_vert_mask ({face_vert_mask.sum()} verts) + "
              f"expr_dirs_face {expr_dirs_face.shape}")

    np.savez(out_npz, **save_kwargs)
    print(f"[fit] wrote {out_npz}")

    # ---- 6. QA render: overlap source Mixamo and fitted SMPL-X in the
    # *fitted-world* frame, plus the un-globalized Mixamo against SMPL-X
    # v_shaped (sanity check that the rest-frame transform was correct). ----
    print(f"[fit] writing QA render to {out_png}")
    _qa_render(
        out_png,
        mixamo_verts=mixamo_verts_aligned,
        mixamo_faces=fbx_mesh.faces,
        smplx_verts=fit.smplx_verts_fit,
        smplx_faces=smplx_faces_np,
        title=f"{slug}: chamfer={fit.final_chamfer:.5f}  sqrt={np.sqrt(fit.final_chamfer):.4f}m",
    )
    print("[fit] done.")


if __name__ == "__main__":
    main()
