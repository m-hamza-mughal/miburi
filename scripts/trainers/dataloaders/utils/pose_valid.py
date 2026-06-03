import numpy as np


SMPLX_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
    [0, 3, 6, 9, 12, 15],  # spine
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
    [20, 25, 26, 27],  # left index finger
    [20, 28, 29, 30],  # left middle finger
    [20, 31, 32, 33],  # left pinky finger
    [20, 34, 35, 36],  # left ring finger
    [20, 37, 38, 39],  # left thumb
    [21, 40, 41, 42],  # right index finger
    [21, 43, 44, 45],  # right middle finger
    [21, 46, 47, 48],  # right pinky finger
    [21, 49, 50, 51],  # right ring finger
    [21, 52, 53, 54],  # right thumb
]


def _compute_bone_pairs(kinematic_chain: list[list[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            pairs.append((chain[i], chain[i + 1]))
    return pairs


def _choose_height_axis(joints: np.ndarray, head_idx: int, foot_indices: tuple[int, ...]) -> int:
    axis_scores = []
    for axis in range(3):
        head = joints[:, head_idx, axis]
        feet = joints[:, foot_indices, axis]
        foot_min = np.min(feet, axis=1)
        heights = head - foot_min
        med = np.median(np.abs(heights[np.isfinite(heights)]))
        axis_scores.append(med)
    return int(np.argmax(axis_scores))


def compute_pose_valid(
    joints: np.ndarray,
    transl: np.ndarray | None = None,
    kinematic_chain: list[list[int]] | None = None,
    head_idx: int = 15,
    foot_indices: tuple[int, ...] = (10, 11, 7, 8),
    height_axis: int | None = None,
    height_min_ratio: float = 0.5,
    height_max_ratio: float = 1.5,
    bone_rel_dev_thresh: float = 0.35,
    min_pose_std: float = 1e-8,
    max_abs_value: float = 1e6,
    require_nonzero: bool = True,
) -> np.ndarray:
    """Return per-frame pose validity bits using joint positions.

    Heuristics:
    - finite values only
    - absolute values below a sane max
    - not all-zeros / near-constant (optional)
    - height consistency (head-to-feet)
    - bone length consistency across kinematic chains
    - transl checks if provided
    """
    if joints.ndim != 3:
        raise ValueError(f"Expected joints to be (T, J, 3); got {joints.shape}")

    finite_ok = np.isfinite(joints).all(axis=(1, 2))
    abs_ok = (np.abs(joints) < max_abs_value).all(axis=(1, 2))
    valid = finite_ok & abs_ok

    if require_nonzero:
        hand_joints = joints[:, 22:, :]  # ignore body joints for pose constancy check
        pose_std = hand_joints.reshape(joints.shape[0], -1).std(axis=1)
        valid &= pose_std > min_pose_std

    if transl is not None:
        if transl.ndim == 2:
            finite_t = np.isfinite(transl).all(axis=1)
            abs_t = (np.abs(transl) < max_abs_value).all(axis=1)
        else:
            finite_t = np.isfinite(transl).all(axis=(1, 2))
            abs_t = (np.abs(transl) < max_abs_value).all(axis=(1, 2))
        valid &= finite_t & abs_t
        if require_nonzero:
            t_std = transl.reshape(transl.shape[0], -1).std(axis=1)
            valid &= t_std > min_pose_std

    # Height consistency
    if height_axis is None:
        height_axis = _choose_height_axis(joints, head_idx, foot_indices)
    # breakpoint() # check height axis choice

    head = joints[:, head_idx, height_axis]
    feet = joints[:, foot_indices, height_axis]
    foot_min = np.min(feet, axis=1)
    heights = head - foot_min
    height_med = np.median(heights[np.isfinite(heights)])
    if height_med <= 0:
        # height_med = np.median(np.abs(heights[np.isfinite(heights)]))

        # here we assume that skeleton is +y-up and 
        # that head should be above feet
        height_ok = False
        valid &= height_ok
    else:
        min_h = height_min_ratio * height_med
        max_h = height_max_ratio * height_med
        height_ok = (heights > min_h) & (heights < max_h)
        valid &= height_ok

    # breakpoint() # check height consistency heuristic
    
    # Bone length stability
    if kinematic_chain is None:
        kinematic_chain = SMPLX_KINEMATIC_CHAIN
    pairs = _compute_bone_pairs(kinematic_chain)
    if pairs:
        lengths = []
        for a, b in pairs:
            seg = joints[:, b, :] - joints[:, a, :]
            lengths.append(np.linalg.norm(seg, axis=1))
        lengths = np.stack(lengths, axis=1)  # (T, B)
        # breakpoint() # check bone length distributions
        med = np.median(lengths, axis=0)
        denom = np.where(med > 1e-8, med, 1.0)
        rel_dev = np.abs(lengths - med) / denom
        mean_rel_dev = np.mean(rel_dev, axis=1)
        valid &= mean_rel_dev < bone_rel_dev_thresh

    # breakpoint() # check validity heuristics

    return valid.astype(bool)
