"""Add jaw-joint skinning to a Mixamo character.

Mixamo armatures don't have a jaw bone -- the whole head (including lower
lip, chin) is skinned to ``mixamorig:Head`` which we map to SMPL-X joint 15
(head). So when the gesture LM emits a jaw rotation in head_pose, no
Mixamo vert responds, and the mouth never opens mechanically.

Fix: transfer LBS weight from joint 15 (head) to joint 22 (jaw) for Mixamo
verts that fall in the lower-jaw region. We use SMPL-X's OWN per-vertex
jaw weights as the template -- for each Mixamo vert, look up the jaw
weight at its closest SMPL-X vertex (in the beta-fitted SMPL-X frame),
then transfer that fraction from head to jaw.

After this fix, head_pose[0] (jaw open/close) correctly opens the mouth on
Mixamo characters.
"""

from __future__ import annotations

import numpy as np


HEAD_JOINT = 15
JAW_JOINT = 22


def _knn_chunked_idx(target: np.ndarray, source: np.ndarray, k: int = 1,
                     tile: int = 1024) -> np.ndarray:
    """Returns indices (Nt, k) of K nearest rows in source per row of target."""
    nt = target.shape[0]
    src_sq = (source ** 2).sum(axis=-1).astype(np.float32)
    out_i = np.empty((nt, k), dtype=np.int64)
    for s in range(0, nt, tile):
        e = min(s + tile, nt)
        chk = target[s:e].astype(np.float32)
        chk_sq = (chk ** 2).sum(axis=-1, keepdims=True)
        d2 = chk_sq + src_sq[None, :] - 2.0 * (chk @ source.T)
        d2 = np.maximum(d2, 0.0)
        cand = np.argpartition(d2, k, axis=-1)[:, :k]
        cand_d = np.take_along_axis(d2, cand, axis=-1)
        order = np.argsort(cand_d, axis=-1)
        out_i[s:e] = np.take_along_axis(cand, order, axis=-1)
    return out_i


def add_jaw_skinning(
    mixamo_verts: np.ndarray,        # (Nm, 3) in SMPL-X rest frame (after un-globalize)
    mixamo_lbs: np.ndarray,          # (Nm, 55) current weights (native FBX, mapped)
    smplx_verts: np.ndarray,         # (V_smplx, 3) beta-fitted SMPL-X T-pose verts
    smplx_lbs: np.ndarray,           # (V_smplx, 55) SMPL-X's own per-vertex weights
    mouth_center: np.ndarray | None = None,  # (3,) anatomical mouth center for spatial mask
    mouth_radius_m: float = 0.04,    # only verts within this sphere get jaw skinning
    jaw_y_upper_offset: float = 0.003,
    jaw_y_lower_offset: float = -0.010,
    verbose: bool = True,
) -> np.ndarray:
    """Returns a new (Nm, 55) LBS weights array with jaw weight transferred
    from head joint to jaw joint for verts in SMPL-X's jaw-influenced region.

    Per-vertex: jaw_w_added = SMPL-X jaw weight at closest SMPL-X vert,
    GATED by spatial proximity to the lip/mouth region. Without the spatial
    gate, the SMPL-X jaw weight distribution extends to chin/cheek verts,
    which makes Mixamo's whole lower face rotate downward when jaw opens
    (SMPL-X mesh tolerates this only because its lip topology creates a
    visible gap; Mixamo's closed-mouth mesh just looks like a face-tilt).

    Per-vertex: transfer = SMPL_X_jaw_weight x spatial_gate where the gate
    is 1.0 inside `mouth_radius_m` of `mouth_center` and 0 outside (hard
    sphere). This concentrates the effect on the actual lips."""
    assert mixamo_lbs.shape[1] == smplx_lbs.shape[1] == 55
    new_lbs = mixamo_lbs.copy()
    del smplx_verts, smplx_lbs  # no longer used; spatial-only assignment

    # Spatial assignment of jaw weight. We don't rely on K-NN matching to
    # SMPL-X's per-vertex jaw distribution (which is fragile when the two
    # meshes are in different frames or have different face geometry):
    # instead, build a smooth Gaussian falloff centered on the mouth, gated
    # to verts strictly at or below the upper-lip line so jaw rotation
    # opens the mouth instead of tilting the whole lower face forward.
    Nm = mixamo_verts.shape[0]
    if mouth_center is None or mouth_radius_m <= 0:
        if verbose:
            print("[jaw_skinning] no mouth_center supplied -- skipping jaw transfer")
        return new_lbs.astype(np.float32)

    d2 = ((mixamo_verts - mouth_center[None, :]) ** 2).sum(axis=1)
    # Gaussian centered on mouth: sigma = mouth_radius_m / 2 so weight
    # falls to ~0.14 at the gate boundary.
    sigma2 = (mouth_radius_m / 2.0) ** 2
    gauss = np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)
    # Hard cutoff at mouth_radius_m (no jaw influence outside the lip sphere).
    sphere_gate = (d2 < mouth_radius_m ** 2).astype(np.float32)
    # Lower-lip-only Y ramp: jaw weight is 0 above (mouth_y + upper_offset),
    # ramps smoothly to 1 below (mouth_y + lower_offset). Tuning the upper
    # boundary down (more negative offset) keeps the upper lip pinned more
    # aggressively, which is needed when the detected mouth_center sits
    # near the lip line of a character with a thick upper lip.
    y_upper = mouth_center[1] + jaw_y_upper_offset
    y_lower = mouth_center[1] + jaw_y_lower_offset
    y = mixamo_verts[:, 1]
    y_ramp = np.clip((y_upper - y) / (y_upper - y_lower + 1e-6),
                     0.0, 1.0).astype(np.float32)
    jaw_target = gauss * sphere_gate * y_ramp
    if verbose:
        active = (jaw_target > 0.05).sum()
        print(f"[jaw_skinning] spatial assignment: {int(active)} verts get "
              f"jaw weight > 0.05 (Gaussian sigma={mouth_radius_m/2*100:.1f}cm "
              f"within {mouth_radius_m*100:.1f}cm of mouth, "
              f"Y ramp [{y_upper:.3f} -> {y_lower:.3f}])")

    # Cap by available head weight (can't move more than is there).
    head_w_have = new_lbs[:, HEAD_JOINT]
    transfer = np.minimum(jaw_target, head_w_have)                   # (Nm,)
    new_lbs[:, JAW_JOINT] = new_lbs[:, JAW_JOINT] + transfer
    new_lbs[:, HEAD_JOINT] = new_lbs[:, HEAD_JOINT] - transfer

    # Re-normalize rows (should already be ~1 since transfer is zero-sum,
    # but float drift can accumulate over many verts).
    row_sums = new_lbs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    new_lbs = new_lbs / row_sums

    if verbose:
        n_with_jaw = (transfer > 0.05).sum()
        print(f"[jaw_skinning] {n_with_jaw} Mixamo verts received jaw weight "
              f"(transfer > 0.05); max transfer = {transfer.max():.3f}, "
              f"mean (over jaw verts) = {transfer[transfer > 0.05].mean():.3f}")

    return new_lbs.astype(np.float32)
