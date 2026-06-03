"""Per-wrist bind-pose correction: align Mixamo hand orientation to SMPL-X T-pose.

Mixamo character bind poses don't always have the palms in SMPL-X's canonical
T-pose orientation. Y-Bot's right palm, for example, is rolled ~30 deg from
SMPL-X T-pose. The LBS math is still correct (per-finger isolation passes),
but every gesture the LM emits assumes SMPL-X palm convention, so the
character ends up with a rotated palm.

Fix: at fit time, rotate each hand's verts around its wrist position by the
angle that maps Mixamo's thumb->pinky vector onto SMPL-X's thumb->pinky
vector, with the rotation constrained to the wrist's "roll" axis (the axis
along the hand from wrist to fingertip). The hand keeps its shape; only its
orientation about the forearm axis changes.

We rotate verts whose dominant LBS joint is a finger joint of that hand
(joints 25..39 for left, 40..54 for right). Wrist verts (joint 20/21) are
left untouched so the wrist-cuff seam stays continuous.
"""

from __future__ import annotations

import numpy as np


LEFT_THUMB_JOINTS  = [37, 38, 39]
LEFT_PINKY_JOINTS  = [31, 32, 33]
LEFT_MIDDLE_JOINTS = [28, 29, 30]
LEFT_FINGER_JOINTS = list(range(25, 40))
LEFT_WRIST = 20

RIGHT_THUMB_JOINTS  = [52, 53, 54]
RIGHT_PINKY_JOINTS  = [46, 47, 48]
RIGHT_MIDDLE_JOINTS = [43, 44, 45]
RIGHT_FINGER_JOINTS = list(range(40, 55))
RIGHT_WRIST = 21


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + s * K + (1.0 - c) * K @ K


def _signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from `a` to `b` around `axis`. All 3-vectors.

    Uses atan2(sin, cos) so the sign respects the right-hand rule about
    `axis`. The vectors are first projected off `axis`."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a = a - np.dot(a, axis) * axis
    b = b - np.dot(b, axis) * axis
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    cos = float(np.dot(a, b))
    sin = float(np.dot(np.cross(a, b), axis))
    return float(np.arctan2(sin, cos))


def _hand_joint_indices(side: str) -> tuple[list[int], list[int], list[int], list[int], int]:
    if side == "left":
        return LEFT_THUMB_JOINTS, LEFT_PINKY_JOINTS, LEFT_MIDDLE_JOINTS, LEFT_FINGER_JOINTS, LEFT_WRIST
    if side == "right":
        return RIGHT_THUMB_JOINTS, RIGHT_PINKY_JOINTS, RIGHT_MIDDLE_JOINTS, RIGHT_FINGER_JOINTS, RIGHT_WRIST
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def _build_hand_basis(thumb_c: np.ndarray, pinky_c: np.ndarray, middle_c: np.ndarray,
                      wrist_pos: np.ndarray) -> np.ndarray:
    """Build an orthonormal hand basis from anatomical landmarks.

    Returns a 3x3 matrix whose columns are:
        e1 = wrist -> middle-finger  (the hand's "axial" direction)
        e2 = thumb -> pinky, projected perpendicular to e1
        e3 = e1 x e2 (palm normal direction)
    """
    e1 = middle_c - wrist_pos
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    raw_e2 = pinky_c - thumb_c
    # Gram-Schmidt: orthogonalize e2 against e1.
    e2 = raw_e2 - np.dot(raw_e2, e1) * e1
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    e3 = np.cross(e1, e2)
    e3 = e3 / max(np.linalg.norm(e3), 1e-12)
    return np.stack([e1, e2, e3], axis=1).astype(np.float64)


def correct_hand_bind_pose(
    mixamo_verts: np.ndarray,        # (Nm, 3) in SMPL-X rest frame
    mixamo_lbs: np.ndarray,          # (Nm, 55)
    smplx_v_shaped: np.ndarray,      # (V_smplx, 3) beta-shaped SMPL-X T-pose (no global, no pose)
    smplx_lbs: np.ndarray,           # (V_smplx, 55)
    smplx_j_rest: np.ndarray,        # (55, 3) SMPL-X T-pose joint positions
    side: str,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Rotate the Mixamo verts of `side`'s hand around the wrist to align its
    hand basis to SMPL-X's. Matches BOTH the thumb->pinky direction and the
    wrist->middle-finger direction simultaneously, so the palm normal also
    aligns. Returns (corrected_verts, residual_normal_angle_deg).
    """
    thumb_j, pinky_j, middle_j, finger_j, wrist_j = _hand_joint_indices(side)

    # Mixamo centroids
    mx_thumb_mask = mixamo_lbs[:, thumb_j].sum(axis=1) > 0.5
    mx_pinky_mask = mixamo_lbs[:, pinky_j].sum(axis=1) > 0.5
    mx_middle_mask = mixamo_lbs[:, middle_j].sum(axis=1) > 0.5
    if (mx_thumb_mask.sum() == 0 or mx_pinky_mask.sum() == 0
            or mx_middle_mask.sum() == 0):
        if verbose:
            print(f"[hand_correction/{side}] missing thumb/pinky/middle verts -- skipping")
        return mixamo_verts, 0.0
    mx_thumb_c = mixamo_verts[mx_thumb_mask].mean(axis=0)
    mx_pinky_c = mixamo_verts[mx_pinky_mask].mean(axis=0)
    mx_middle_c = mixamo_verts[mx_middle_mask].mean(axis=0)

    # SMPL-X centroids in the same frame
    sx_thumb_c = smplx_v_shaped[smplx_lbs[:, thumb_j].sum(axis=1) > 0.5].mean(axis=0)
    sx_pinky_c = smplx_v_shaped[smplx_lbs[:, pinky_j].sum(axis=1) > 0.5].mean(axis=0)
    sx_middle_c = smplx_v_shaped[smplx_lbs[:, middle_j].sum(axis=1) > 0.5].mean(axis=0)

    wrist_pos = smplx_j_rest[wrist_j].astype(np.float64)

    # Build orthonormal bases for each character.
    B_mx = _build_hand_basis(mx_thumb_c, mx_pinky_c, mx_middle_c, wrist_pos)
    B_sx = _build_hand_basis(sx_thumb_c, sx_pinky_c, sx_middle_c, wrist_pos)

    # Rotation that maps Mixamo basis vectors onto SMPL-X basis vectors.
    # Since both are orthonormal, R = B_sx @ B_mx^T satisfies R @ e_mx_i = e_sx_i.
    R = (B_sx @ B_mx.T).astype(np.float32)

    # Rotate verts whose dominant joint is a finger joint of this hand. Wrist
    # verts stay put so the wrist-cuff seam doesn't open.
    dom = mixamo_lbs.argmax(axis=1)
    finger_set = set(finger_j)
    affected = np.array([j in finger_set for j in dom])

    corrected = mixamo_verts.copy()
    pivot = wrist_pos.astype(np.float32)
    delta = corrected[affected] - pivot[None, :]
    corrected[affected] = delta @ R.T + pivot[None, :]

    # Diagnostic: residual angle between palm normals after rotation.
    new_thumb_c = corrected[mx_thumb_mask].mean(axis=0)
    new_pinky_c = corrected[mx_pinky_mask].mean(axis=0)
    new_middle_c = corrected[mx_middle_mask].mean(axis=0)
    B_mx_after = _build_hand_basis(new_thumb_c, new_pinky_c, new_middle_c, wrist_pos)
    cos = float(np.clip(np.dot(B_mx_after[:, 2], B_sx[:, 2]), -1.0, 1.0))
    residual = float(np.degrees(np.arccos(cos)))

    if verbose:
        # Report applied rotation as the angle of `R` (Frobenius-derived).
        applied_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"[hand_correction/{side}] full 3D basis match: "
              f"applied={applied_angle:+.2f} deg, residual palm-normal angle={residual:.2f} deg, "
              f"affected={affected.sum()}")

    return corrected, residual
