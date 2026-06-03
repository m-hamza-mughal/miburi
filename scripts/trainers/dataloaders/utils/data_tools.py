import torch
from . import rotation_conversions as Rc


SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

SMPLX_JOINT_NAMES_UPPER = [
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

SMPLX_JOINT_NAMES_LOWER = [
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
]

SMPLX_JOINT_NAMES_HANDS = [
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

SMPLX_JOINT_NAMES_FACE = [
    "jaw",
]


def get_seamlessint_spkids(file_id: str) -> str:
    parts = file_id.split("_")
    if len(parts) > 3:
        return parts[3]
    raise ValueError("File ID format is incorrect")

def get_beatxint_spkids(file_id: str) -> str:
    try:
        parts = file_id.split("_")
        return parts[0] + "_" + parts[1]
    except Exception:
        pass
    return "UNKNOWN"


def get_beatx_spkids(file_id: str) -> str:
    """Speaker name from a BEATX file_id ('9_miranda_1_9_9_C10' -> 'miranda')."""
    parts = file_id.split("_")
    return parts[1]


def get_embody3d_spkids(file_id: str) -> str:
    """Speaker id from an Embody3D dyadic file_id (after the '+' separator)."""
    return file_id.split("+")[1].split("_")[0]


def check_if_additional_split(file_id: str) -> bool:
    """Whether a BEATX file_id is from the 'additional' split.

    Example: '9_miranda_1_9_9_C10' -> True; '9_miranda_0_9_9_C10' -> False.
    The third underscore-separated field encodes the split.
    """
    parts = file_id.split("_")
    if len(parts) > 2:
        return parts[2] == "1"
    raise ValueError("File ID format is incorrect")


def rotate_axis_angle_by_180_x_torch(axis_angle_tensor: torch.Tensor) -> torch.Tensor:
    original_shape = axis_angle_tensor.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        reshaped_input = axis_angle_tensor.squeeze(1)
    else:
        reshaped_input = axis_angle_tensor

    rot_mat = Rc.axis_angle_to_matrix(reshaped_input)
    rot_180_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], device=axis_angle_tensor.device)
    rot_180_x = rot_180_x.unsqueeze(0).expand(rot_mat.shape[0], -1, -1)
    rotated_mat = torch.matmul(rot_180_x, rot_mat)
    rotated_aa = Rc.matrix_to_axis_angle(rotated_mat)
    if len(original_shape) == 3 and original_shape[1] == 1:
        rotated_aa = rotated_aa.unsqueeze(1)
    return rotated_aa


def zero_first_frame_pose(axis_angle_tensor: torch.Tensor, translation_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = axis_angle_tensor.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        reshaped_input = axis_angle_tensor.squeeze(1)
    else:
        reshaped_input = axis_angle_tensor

    first_rot_mat = Rc.axis_angle_to_matrix(reshaped_input[0:1])
    inv_first_rot_mat = first_rot_mat.transpose(1, 2)
    all_rot_mats = Rc.axis_angle_to_matrix(reshaped_input)
    inv_first_rot_mat = inv_first_rot_mat.expand(all_rot_mats.shape[0], -1, -1)
    aligned_rot_mats = torch.matmul(inv_first_rot_mat, all_rot_mats)
    aligned_axis_angles = Rc.matrix_to_axis_angle(aligned_rot_mats)
    if len(original_shape) == 3 and original_shape[1] == 1:
        aligned_axis_angles = aligned_axis_angles.unsqueeze(1)

    aligned_translations = torch.matmul(inv_first_rot_mat, translation_tensor.unsqueeze(-1)).squeeze(-1)
    return aligned_axis_angles, aligned_translations


def _inverse_first_frame_yaw_matrix(axis_angle_tensor: torch.Tensor) -> torch.Tensor:
    """Build inverse first-frame heading rotation around world up (+Y).

    The first-frame heading is derived from the rotated forward axis projected
    to the ground plane, so pitch/roll do not contribute to the extracted yaw.
    """
    original_shape = axis_angle_tensor.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        reshaped_input = axis_angle_tensor.squeeze(1)
    else:
        reshaped_input = axis_angle_tensor

    if reshaped_input.ndim != 2 or reshaped_input.shape[-1] != 3:
        raise ValueError(
            "_inverse_first_frame_yaw_matrix expects axis-angle rotations with shape [T, 3] or [T, 1, 3], "
            f"got shape={tuple(axis_angle_tensor.shape)}"
        )

    first_rot_mat = Rc.axis_angle_to_matrix(reshaped_input[0:1])[0]
    forward = first_rot_mat[:, 2]
    forward_xz = torch.stack((forward[0], forward[2]))
    forward_xz_norm = torch.linalg.norm(forward_xz)
    if float(forward_xz_norm.item()) <= 1e-8:
        return torch.eye(3, dtype=first_rot_mat.dtype, device=first_rot_mat.device)

    yaw = torch.atan2(forward_xz[0], forward_xz[1])
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    inv_yaw = torch.stack(
        (
            torch.stack((cos_yaw, torch.zeros_like(cos_yaw), -sin_yaw)),
            torch.stack((torch.zeros_like(cos_yaw), torch.ones_like(cos_yaw), torch.zeros_like(cos_yaw))),
            torch.stack((sin_yaw, torch.zeros_like(cos_yaw), cos_yaw)),
        )
    )
    return inv_yaw


def zero_first_frame_pose_yaw_only(
    axis_angle_tensor: torch.Tensor,
    translation_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove only first-frame heading around world up (+Y).

    Unlike zero_first_frame_pose, this preserves the first-frame pitch/roll and
    only aligns the body to face the canonical +Z direction.
    """
    original_shape = axis_angle_tensor.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        reshaped_input = axis_angle_tensor.squeeze(1)
    else:
        reshaped_input = axis_angle_tensor

    inv_first_yaw = _inverse_first_frame_yaw_matrix(axis_angle_tensor)
    all_rot_mats = Rc.axis_angle_to_matrix(reshaped_input)
    inv_first_yaw = inv_first_yaw.unsqueeze(0).expand(all_rot_mats.shape[0], -1, -1)
    aligned_rot_mats = torch.matmul(inv_first_yaw, all_rot_mats)
    aligned_axis_angles = Rc.matrix_to_axis_angle(aligned_rot_mats)
    if len(original_shape) == 3 and original_shape[1] == 1:
        aligned_axis_angles = aligned_axis_angles.unsqueeze(1)

    if translation_tensor.ndim != 2 or translation_tensor.shape[-1] != 3:
        raise ValueError(
            "zero_first_frame_pose_yaw_only expects translations with shape [T, 3], "
            f"got shape={tuple(translation_tensor.shape)}"
        )
    aligned_translations = torch.matmul(inv_first_yaw, translation_tensor.unsqueeze(-1)).squeeze(-1)
    return aligned_axis_angles, aligned_translations


def align_points_to_first_frame(axis_angle_tensor: torch.Tensor, points_tensor: torch.Tensor) -> torch.Tensor:
    """Rotate point trajectories by inverse first-frame global orientation.

    Args:
        axis_angle_tensor: [T, 1, 3] or [T, 3] axis-angle rotations.
        points_tensor: [T, N, 3] or [T, 3] points in global frame.
    Returns:
        Rotated points with same shape as input.
    """
    original_shape = axis_angle_tensor.shape
    if len(original_shape) == 3 and original_shape[1] == 1:
        reshaped_input = axis_angle_tensor.squeeze(1)
    else:
        reshaped_input = axis_angle_tensor

    first_rot_mat = Rc.axis_angle_to_matrix(reshaped_input[0:1])  # [1, 3, 3]
    inv_first_rot_mat = first_rot_mat.transpose(1, 2)[0]  # [3, 3]

    if points_tensor.ndim == 2 and points_tensor.shape[-1] == 3:
        return torch.matmul(points_tensor, inv_first_rot_mat.T)
    if points_tensor.ndim == 3 and points_tensor.shape[-1] == 3:
        return torch.matmul(points_tensor, inv_first_rot_mat.T)
    raise ValueError(f"align_points_to_first_frame expects [...,3] points, got shape={tuple(points_tensor.shape)}")


def align_points_to_first_frame_yaw_only(axis_angle_tensor: torch.Tensor, points_tensor: torch.Tensor) -> torch.Tensor:
    """Rotate point trajectories by inverse first-frame heading only."""
    inv_first_yaw = _inverse_first_frame_yaw_matrix(axis_angle_tensor)

    if points_tensor.ndim == 2 and points_tensor.shape[-1] == 3:
        return torch.matmul(points_tensor, inv_first_yaw.T)
    if points_tensor.ndim == 3 and points_tensor.shape[-1] == 3:
        return torch.matmul(points_tensor, inv_first_yaw.T)
    raise ValueError(
        f"align_points_to_first_frame_yaw_only expects [...,3] points, got shape={tuple(points_tensor.shape)}"
    )
