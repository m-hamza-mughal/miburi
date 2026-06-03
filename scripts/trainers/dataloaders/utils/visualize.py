import logging
import os
import subprocess
from typing import Optional

import numpy as np
import torch

from .data_tools import (
    SMPLX_JOINT_NAMES,
    SMPLX_JOINT_NAMES_HANDS,
    SMPLX_JOINT_NAMES_LOWER,
    SMPLX_JOINT_NAMES_UPPER,
)


LOG = logging.getLogger(__name__)


def _render_selected_joint_indices(body_part: str) -> list[int]:
    bp = str(body_part or "all").strip().lower()
    if bp == "upper":
        names = list(SMPLX_JOINT_NAMES_UPPER)
    elif bp == "upper+hands":
        names = list(dict.fromkeys(SMPLX_JOINT_NAMES_UPPER + SMPLX_JOINT_NAMES_HANDS))
    elif bp == "upper+hands+lower":
        names = list(dict.fromkeys(SMPLX_JOINT_NAMES_UPPER + SMPLX_JOINT_NAMES_HANDS + SMPLX_JOINT_NAMES_LOWER))
    elif bp == "lower":
        names = list(SMPLX_JOINT_NAMES_LOWER)
    elif bp == "hands":
        names = list(SMPLX_JOINT_NAMES_HANDS)
    else:
        names = list(SMPLX_JOINT_NAMES)
    sel = set(names)
    return [i for i, n in enumerate(SMPLX_JOINT_NAMES) if n in sel]


def _body_part_uses_translation(body_part: str) -> bool:
    bp = str(body_part or "all").strip().lower()
    if bp in {"all", "full", "fullbody", "whole", "upper+hands+lower"}:
        return True
    return "lower" in bp


def apply_body_part_render_filter(
    poses: np.ndarray,
    transl: np.ndarray,
    body_part: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only joints covered by `body_part` for rendering and optionally zero translation.

    - For upper-only style parts (e.g. `upper`, `upper+hands`), lower joints are zeroed and transl is zeroed.
    - For lower-inclusive parts (e.g. `upper+hands+lower`), full translation is retained.
    """
    pose_np = np.asarray(poses, dtype=np.float32)
    transl_np = np.asarray(transl, dtype=np.float32)
    n_joints = len(SMPLX_JOINT_NAMES)

    pose_flat_in = (pose_np.ndim == 2)
    if pose_np.ndim == 2:
        if pose_np.shape[1] != n_joints * 3:
            LOG.warning(
                "apply_body_part_render_filter: expected flattened pose dim=%d, got %s; skipping joint mask.",
                n_joints * 3,
                str(tuple(pose_np.shape)),
            )
            pose_out = pose_np
        else:
            pose3 = pose_np.reshape(pose_np.shape[0], n_joints, 3).copy()
            keep_idx = _render_selected_joint_indices(body_part)
            keep_mask = np.zeros((n_joints,), dtype=bool)
            keep_mask[np.asarray(keep_idx, dtype=np.int64)] = True
            pose3[:, ~keep_mask, :] = 0.0
            pose_out = pose3.reshape(pose_np.shape[0], n_joints * 3)
    elif pose_np.ndim == 3 and pose_np.shape[1] == n_joints and pose_np.shape[2] == 3:
        pose3 = pose_np.copy()
        keep_idx = _render_selected_joint_indices(body_part)
        keep_mask = np.zeros((n_joints,), dtype=bool)
        keep_mask[np.asarray(keep_idx, dtype=np.int64)] = True
        pose3[:, ~keep_mask, :] = 0.0
        pose_out = pose3
    else:
        LOG.warning(
            "apply_body_part_render_filter: unsupported pose shape %s; skipping joint mask.",
            str(tuple(pose_np.shape)),
        )
        pose_out = pose_np

    transl_out = transl_np.copy()
    if not _body_part_uses_translation(body_part):
        transl_out[...] = 0.0

    if pose_flat_in and pose_out.ndim == 3:
        pose_out = pose_out.reshape(pose_out.shape[0], -1)
    return pose_out.astype(np.float32), transl_out.astype(np.float32)


def create_checkerboard_floor(
    y: float = 0.0,
    length: float = 10.0,
    tile_size: float = 1.0,
    color_a: tuple[int, int, int, int] = (170, 170, 170, 255),
    color_b: tuple[int, int, int, int] = (120, 120, 120, 255),
):
    import trimesh

    half = length * 0.5
    nx = max(1, int(length / tile_size))
    nz = max(1, int(length / tile_size))

    vertices = []
    faces = []
    face_colors = []
    idx = 0
    for ix in range(nx):
        for iz in range(nz):
            x0 = -half + ix * tile_size
            x1 = x0 + tile_size
            z0 = -half + iz * tile_size
            z1 = z0 + tile_size
            vertices.extend(
                [
                    [x0, y, z0],
                    [x1, y, z0],
                    [x1, y, z1],
                    [x0, y, z1],
                ]
            )
            faces.extend(
                [
                    [idx + 0, idx + 2, idx + 1],
                    [idx + 0, idx + 3, idx + 2],
                ]
            )
            c = color_a if ((ix + iz) % 2 == 0) else color_b
            face_colors.extend([c, c])
            idx += 4

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        face_colors=np.asarray(face_colors, dtype=np.uint8),
        process=False,
    )


def mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
) -> str:
    if not audio_path or not os.path.exists(audio_path):
        LOG.warning("Audio file missing for muxing: %s", audio_path)
        return video_path

    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_audio_tmp.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, video_path)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found, keeping silent video: %s", video_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg mux failed for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as exc:
        LOG.warning("Unexpected mux error for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)

    return video_path


def ensure_vscode_compatible_video(
    video_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Re-encode video to H.264/yuv420p for broad editor playback support."""
    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_vscode_tmp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(output_path, video_path)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found, keeping original video codec: %s", video_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg re-encode failed for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as exc:
        LOG.warning("Unexpected re-encode error for %s: %s", video_path, exc)
        if os.path.exists(output_path):
            os.remove(output_path)
    return video_path


def render_joint_debug_video(
    joints: np.ndarray,
    output_path: str,
    fps: int,
    width: int = 640,
    height: int = 480,
    title: str | None = None,
    point_color: tuple[int, int, int] = (20, 90, 220),
    bone_color: tuple[int, int, int] = (40, 120, 240),
    bg_color: tuple[int, int, int] = (235, 235, 235),
    joint_indices: Optional[list[int]] = None,
) -> str:
    """Render a lightweight 2D joint trajectory debug video from 3D joints [T, J, 3]."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints shape [T, J, 3], got {joints.shape}")
    if joints.shape[0] <= 0:
        raise ValueError("Cannot render empty joint sequence.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from .pose_valid import SMPLX_KINEMATIC_CHAIN
    import cv2

    # 2D projection: x-horizontal, y-vertical (flip y for image coordinates).
    xy = joints[..., [0, 1]]
    x_vals = xy[..., 0]
    y_vals = xy[..., 1]
    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    x_span = max(1.0e-6, x_max - x_min)
    y_span = max(1.0e-6, y_max - y_min)
    pad_ratio = 0.08
    x_min -= x_span * pad_ratio
    x_max += x_span * pad_ratio
    y_min -= y_span * pad_ratio
    y_max += y_span * pad_ratio

    # Keep fixed scale across frames.
    sx = (width - 1) / max(1.0e-6, (x_max - x_min))
    sy = (height - 1) / max(1.0e-6, (y_max - y_min))
    scale = min(sx, sy)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    px_center = 0.5 * (width - 1)
    py_center = 0.5 * (height - 1)

    def _project(p3: np.ndarray) -> tuple[int, int]:
        x = float(p3[0])
        y = float(p3[1])
        px = int(round((x - x_center) * scale + px_center))
        py = int(round((-y + y_center) * scale + py_center))
        return px, py

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    bone_pairs: list[tuple[int, int]] = []
    if joint_indices is None:
        for chain in SMPLX_KINEMATIC_CHAIN:
            for i in range(len(chain) - 1):
                bone_pairs.append((int(chain[i]), int(chain[i + 1])))
    else:
        index_map = {int(full_idx): int(local_idx) for local_idx, full_idx in enumerate(joint_indices)}
        for chain in SMPLX_KINEMATIC_CHAIN:
            for i in range(len(chain) - 1):
                a_full = int(chain[i])
                b_full = int(chain[i + 1])
                if a_full in index_map and b_full in index_map:
                    bone_pairs.append((index_map[a_full], index_map[b_full]))

    try:
        for t_idx in range(joints.shape[0]):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            pts = joints[t_idx]

            # Draw bones first.
            for a, b in bone_pairs:
                if a >= pts.shape[0] or b >= pts.shape[0]:
                    continue
                pa = _project(pts[a])
                pb = _project(pts[b])
                cv2.line(frame, pa, pb, bone_color, thickness=1, lineType=cv2.LINE_AA)

            # Draw joints on top.
            for j in range(pts.shape[0]):
                pj = _project(pts[j])
                cv2.circle(frame, pj, radius=2, color=point_color, thickness=-1, lineType=cv2.LINE_AA)

            if title:
                cv2.putText(
                    frame,
                    title,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()

    return ensure_vscode_compatible_video(output_path)


def stitch_videos_hstack(video_paths: list[str], output_path: str) -> str:
    """Horizontally stack videos using ffmpeg. Returns output_path on success."""
    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos to hstack.")
    for p in video_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input video for hstack: {p}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in video_paths:
        cmd.extend(["-i", p])
    inputs = "".join(f"[{i}:v]" for i in range(len(video_paths)))
    filter_complex = f"{inputs}hstack=inputs={len(video_paths)}[v]"
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]
    )

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        LOG.warning("ffmpeg not found; cannot stitch videos: %s", output_path)
    except subprocess.CalledProcessError as exc:
        LOG.warning("ffmpeg hstack failed for %s: %s", output_path, exc)
    except Exception as exc:
        LOG.warning("Unexpected hstack error for %s: %s", output_path, exc)
    return output_path


def _prepare_betas(betas: Optional[np.ndarray], nframes: int, device: torch.device):
    if betas is None:
        return torch.zeros((nframes, 300), device=device, dtype=torch.float32)

    betas_np = np.asarray(betas)
    if betas_np.ndim == 1:
        betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32).unsqueeze(0)
        return betas_t.repeat(nframes, 1)

    betas_t = torch.tensor(betas_np, device=device, dtype=torch.float32)
    if betas_t.shape[0] == 1:
        return betas_t.repeat(nframes, 1)
    if betas_t.shape[0] != nframes:
        return betas_t[:1].repeat(nframes, 1)
    return betas_t


def _smplx_vertices_from_params(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    batch_size: int = 256,
) -> np.ndarray:
    device = next(smplx_model.parameters()).device
    nframes = poses.shape[0]
    expr_dim = 100 if expressions is None else int(expressions.shape[1])
    verts_all = []

    for start in range(0, nframes, batch_size):
        end = min(start + batch_size, nframes)
        p = torch.tensor(poses[start:end], device=device, dtype=torch.float32)
        t = torch.tensor(transl[start:end], device=device, dtype=torch.float32)
        if expressions is None:
            e = torch.zeros((end - start, expr_dim), device=device, dtype=torch.float32)
        else:
            e = torch.tensor(expressions[start:end], device=device, dtype=torch.float32)
        b = _prepare_betas(betas, nframes, device)[start:end]

        with torch.no_grad():
            out = smplx_model(
                betas=b,
                transl=t,
                expression=e,
                jaw_pose=p[:, 66:69],
                global_orient=p[:, :3],
                body_pose=p[:, 3 : 21 * 3 + 3],
                left_hand_pose=p[:, 25 * 3 : 40 * 3],
                right_hand_pose=p[:, 40 * 3 : 55 * 3],
                leye_pose=p[:, 69:72],
                reye_pose=p[:, 72:75],
                return_verts=True,
            )
        verts_all.append(out.vertices.detach().cpu().numpy())

    return np.concatenate(verts_all, axis=0)


def render_smplx_debug_video(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    output_path: str,
    fps: int,
    width: int = 640,
    height: int = 480,
    audio_path: Optional[str] = None,
    mesh_color: tuple[int, int, int, int] = (36, 73, 156, 255),
    camera_pose: Optional[np.ndarray] = None,
    only_face: bool = False,
) -> str:
    """Render an SMPL-X mesh sequence to an mp4.

    Camera framing:
      * Default (full body): camera at (x=0, y=-0.1, z=2.0) with a -8 deg
        pitch -- tuned for SMPL-X v_template feet at y~=-1.05.
      * `only_face=True`: tighter framing on the head, taken from the old
        `visualize_smpl(only_face=True)` (translation (0, 0.285, 0.25),
        no pitch).
      * `camera_pose=<4x4 ndarray>`: caller-supplied pose, overrides both.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import pyrender
    import trimesh

    vertices = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=poses,
        transl=transl,
        expressions=expressions,
        betas=betas,
    )
    faces = smplx_model.faces

    scene = pyrender.Scene(
        bg_color=np.array([0.75, 0.75, 0.75, 1.0]),
        ambient_light=np.array([0.35, 0.35, 0.35]),
    )

    floor_y = float(vertices[..., 1].min()) - 0.02
    floor_mesh = create_checkerboard_floor(y=floor_y, length=12.0, tile_size=1.0)
    scene.add(pyrender.Mesh.from_trimesh(floor_mesh, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / float(height))
    if camera_pose is None:
        if only_face:
            # Tight head framing -- matches old visualize_smpl(only_face=True):
            # translation (0, 0.285, 0.25), no pitch. With SMPL-X v_template
            # head at y~=0.55 (above origin in body-local frame), z=0.25 sits
            # ~25 cm in front of the head, y=0.285 lifts the camera to head
            # height.
            camera_pose = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.285],
                    [0.0, 0.0, 1.0, 0.25],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            cam_pitch_deg = -8.0
            cam_pitch = np.deg2rad(cam_pitch_deg)
            c = float(np.cos(cam_pitch))
            s = float(np.sin(cam_pitch))
            camera_pose = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, c, -s, -0.1],
                    [0.0, s, c, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
    else:
        camera_pose = np.asarray(camera_pose, dtype=np.float32)
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(key_light, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[0, 3] = 1.5
    fill_pose[1, 3] = 2.0
    scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for fidx in range(vertices.shape[0]):
            mesh = trimesh.Trimesh(vertices=vertices[fidx], faces=faces, process=False)
            mesh.visual.vertex_colors = np.tile(
                np.asarray(mesh_color, dtype=np.uint8),
                (mesh.vertices.shape[0], 1),
            )
            mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            color, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            scene.remove_node(mesh_node)

            # if fidx == 10:
            #     break 

    finally:
        writer.release()
        renderer.delete()

    if audio_path:
        return mux_audio_into_video(output_path, audio_path)
    return ensure_vscode_compatible_video(output_path)


def render_smplx_side_by_side_video(
    smplx_model,
    gt_poses: np.ndarray,
    gt_transl: np.ndarray,
    pred_poses: np.ndarray,
    pred_transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    output_path: str,
    fps: int,
    width: int = 1280,
    height: int = 480,
    audio_path: Optional[str] = None,
    gt_color: tuple[int, int, int, int] = (36, 73, 156, 255),
    pred_color: tuple[int, int, int, int] = (180, 54, 54, 255),
    horizontal_gap: float = 1.2,
) -> str:
    """Render GT (left) and reconstruction (right) in one side-by-side video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import pyrender
    import trimesh

    gt_vertices = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=gt_poses,
        transl=gt_transl,
        expressions=expressions,
        betas=betas,
    )
    pred_vertices = _smplx_vertices_from_params(
        smplx_model=smplx_model,
        poses=pred_poses,
        transl=pred_transl,
        expressions=expressions,
        betas=betas,
    )
    nframes = min(gt_vertices.shape[0], pred_vertices.shape[0])
    if nframes <= 0:
        raise ValueError("No frames to render in side-by-side video.")

    faces = smplx_model.faces

    scene = pyrender.Scene(
        bg_color=np.array([0.75, 0.75, 0.75, 1.0]),
        ambient_light=np.array([0.35, 0.35, 0.35]),
    )

    floor_y = float(min(gt_vertices[..., 1].min(), pred_vertices[..., 1].min())) - 0.02
    floor_mesh = create_checkerboard_floor(y=floor_y, length=12.0, tile_size=1.0)
    scene.add(pyrender.Mesh.from_trimesh(floor_mesh, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / float(height))
    cam_pitch_deg = -8.0
    cam_pitch = np.deg2rad(cam_pitch_deg)
    c = float(np.cos(cam_pitch))
    s = float(np.sin(cam_pitch))
    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, -0.1],
            [0.0, s, c, 2.2],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scene.add(camera, pose=camera_pose)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    scene.add(key_light, pose=camera_pose)
    fill_pose = camera_pose.copy()
    fill_pose[0, 3] = 1.5
    fill_pose[1, 3] = 2.0
    scene.add(fill_light, pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    left_shift = -horizontal_gap * 0.5
    right_shift = horizontal_gap * 0.5

    try:
        for fidx in range(nframes):
            gt_v = gt_vertices[fidx].copy()
            pred_v = pred_vertices[fidx].copy()
            gt_v[:, 0] += left_shift
            pred_v[:, 0] += right_shift

            gt_mesh = trimesh.Trimesh(vertices=gt_v, faces=faces, process=False)
            gt_mesh.visual.vertex_colors = np.tile(np.asarray(gt_color, dtype=np.uint8), (gt_mesh.vertices.shape[0], 1))
            pred_mesh = trimesh.Trimesh(vertices=pred_v, faces=faces, process=False)
            pred_mesh.visual.vertex_colors = np.tile(np.asarray(pred_color, dtype=np.uint8), (pred_mesh.vertices.shape[0], 1))

            gt_node = scene.add(pyrender.Mesh.from_trimesh(gt_mesh, smooth=True))
            pred_node = scene.add(pyrender.Mesh.from_trimesh(pred_mesh, smooth=True))
            color, _ = renderer.render(scene)
            writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            scene.remove_node(gt_node)
            scene.remove_node(pred_node)
    finally:
        writer.release()
        renderer.delete()

    if audio_path:
        return mux_audio_into_video(output_path, audio_path)
    return ensure_vscode_compatible_video(output_path)


def render_chunk_smplx_joints_side_by_side(
    smplx_model,
    poses: np.ndarray,
    transl: np.ndarray,
    expressions: Optional[np.ndarray],
    betas: Optional[np.ndarray],
    joints: np.ndarray,
    output_path: str,
    fps: int,
    panel_width: int = 840,
    panel_height: int = 1280,
    audio_path: Optional[str] = None,
    audio_start_sec: float = 0.0,
    audio_duration_sec: Optional[float] = None,
    mesh_color: tuple[int, int, int, int] = (36, 73, 156, 255),
    joint_indices: Optional[list[int]] = None,
) -> str:
    """Render a SMPLX mesh panel and a 2D joint trajectory panel side-by-side.

    Intermediate per-panel videos are written to a tempdir; only the final
    stitched (and optionally audio-muxed) mp4 lands at ``output_path``.

    ``audio_start_sec`` and ``audio_duration_sec`` slice the audio source so
    chunk videos stay in sync when the audio_path covers the entire parent
    sample. When the slice args are omitted, audio is muxed from time 0
    and truncated to video length via ffmpeg's ``-shortest``.
    """
    import shutil
    import tempfile

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="chunk_sbs_") as tmpdir:
        smplx_path = os.path.join(tmpdir, "smplx.mp4")
        joints_path = os.path.join(tmpdir, "joints.mp4")
        stitched_path = os.path.join(tmpdir, "stitched.mp4")

        render_smplx_debug_video(
            smplx_model=smplx_model,
            poses=poses,
            transl=transl,
            expressions=expressions,
            betas=betas,
            output_path=smplx_path,
            fps=fps,
            width=panel_width,
            height=panel_height,
            audio_path=None,
            mesh_color=mesh_color,
        )
        render_joint_debug_video(
            joints=joints,
            output_path=joints_path,
            fps=fps,
            width=panel_width,
            height=panel_height,
            joint_indices=joint_indices,
        )
        stitch_videos_hstack([smplx_path, joints_path], stitched_path)
        if not os.path.exists(stitched_path):
            raise RuntimeError(f"Side-by-side stitch failed; no output at {stitched_path}")

        shutil.move(stitched_path, output_path)

        if audio_path and os.path.exists(audio_path):
            chunk_audio_path = audio_path
            need_slice = audio_start_sec > 0.0 or audio_duration_sec is not None
            if need_slice:
                chunk_audio_path = os.path.join(tmpdir, "chunk_audio.wav")
                slice_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{float(audio_start_sec):.4f}",
                ]
                if audio_duration_sec is not None:
                    slice_cmd.extend(["-t", f"{float(audio_duration_sec):.4f}"])
                slice_cmd.extend(["-i", audio_path, "-vn", "-acodec", "pcm_s16le", chunk_audio_path])
                try:
                    subprocess.run(slice_cmd, check=True)
                except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                    LOG.warning("ffmpeg audio slice failed for %s: %s", chunk_audio_path, exc)
                    chunk_audio_path = audio_path  # fall back to full audio
            mux_audio_into_video(output_path, chunk_audio_path)
    return output_path
