"""Mixamo armature bone name -> SMPL-X joint index mapping.

Mixamo characters use the standard ``mixamorig:<BoneName>`` armature.
SMPL-X has 55 joints in a fixed order; we map each Mixamo bone to its
nearest SMPL-X counterpart. Unknown / extra bones (tips like
``HeadTop_End``, extra finger segments like ``LeftHandIndex4``) are handled
by walking up the parent chain to the nearest known ancestor in the
Blender extraction step.

SMPL-X joint ordering (NUM_JOINTS=55):
    0  pelvis
    1  left_hip          2  right_hip
    3  spine1
    4  left_knee         5  right_knee
    6  spine2
    7  left_ankle        8  right_ankle
    9  spine3
    10 left_foot         11 right_foot
    12 neck
    13 left_collar       14 right_collar
    15 head
    16 left_shoulder     17 right_shoulder
    18 left_elbow        19 right_elbow
    20 left_wrist        21 right_wrist
    22 jaw               23 left_eye        24 right_eye
    25-39 left hand (15, MANO order)
    40-54 right hand (15, MANO order)

MANO finger order within each hand:
    index1, index2, index3, middle1..3, pinky1..3, ring1..3, thumb1..3
"""

from __future__ import annotations

NUM_SMPLX_JOINTS = 55


def _hand_finger_idx(side: str, finger: str, seg: int) -> int:
    """side in {'left','right'}, finger in {index,middle,pinky,ring,thumb}, seg in {1,2,3}."""
    base = 25 if side == "left" else 40
    finger_offsets = {"index": 0, "middle": 3, "pinky": 6, "ring": 9, "thumb": 12}
    return base + finger_offsets[finger] + (seg - 1)


MIXAMO_TO_SMPLX: dict[str, int] = {
    # ---- root / spine ----
    "Hips":      0,
    "Spine":     3,    # spine1
    "Spine1":    6,    # spine2
    "Spine2":    9,    # spine3
    "Neck":     12,
    "Head":     15,
    "HeadTop_End": 15,  # tip vertex group -> head

    # ---- left arm ----
    "LeftShoulder": 13,   # left_collar
    "LeftArm":      16,   # left_shoulder (yes, naming differs)
    "LeftForeArm":  18,   # left_elbow
    "LeftHand":     20,   # left_wrist

    # ---- right arm ----
    "RightShoulder": 14,  # right_collar
    "RightArm":      17,  # right_shoulder
    "RightForeArm":  19,  # right_elbow
    "RightHand":     21,  # right_wrist

    # ---- left leg ----
    "LeftUpLeg":   1,   # left_hip
    "LeftLeg":     4,   # left_knee
    "LeftFoot":    7,   # left_ankle
    "LeftToeBase": 10,  # left_foot
    "LeftToe_End": 10,  # tip -> left_foot

    # ---- right leg ----
    "RightUpLeg":   2,
    "RightLeg":     5,
    "RightFoot":    8,
    "RightToeBase": 11,
    "RightToe_End": 11,
}

# Add finger bones. Mixamo uses LeftHandIndex1..3 (and sometimes a 4th tip
# bone). Tip bones (idx 4 and above) are mapped to seg 3 of the same finger.
for _side, _smplx_side in [("Left", "left"), ("Right", "right")]:
    for _finger in ["Index", "Middle", "Pinky", "Ring", "Thumb"]:
        for _seg in (1, 2, 3):
            MIXAMO_TO_SMPLX[f"{_side}Hand{_finger}{_seg}"] = _hand_finger_idx(
                _smplx_side, _finger.lower(), _seg
            )
        for _seg_extra in (4, 5):
            MIXAMO_TO_SMPLX[f"{_side}Hand{_finger}{_seg_extra}"] = _hand_finger_idx(
                _smplx_side, _finger.lower(), 3
            )


def strip_prefix(bone_name: str) -> str:
    """Strip the conventional ``mixamorig:`` prefix from a Mixamo bone name."""
    if ":" in bone_name:
        return bone_name.split(":", 1)[1]
    return bone_name


def map_bone_to_smplx_joint(bone_name: str) -> int | None:
    """Return the SMPL-X joint index for a Mixamo bone, or None if unmapped."""
    return MIXAMO_TO_SMPLX.get(strip_prefix(bone_name))
