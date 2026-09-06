import argparse
import asyncio
from collections import deque
import contextlib
import struct
import time
from pathlib import Path

import aiohttp
from aiohttp import web
import numpy as np
import torch
import viser
import trimesh
from PIL import Image

from .client_utils import log
from .utils.viser_scene import (
    SceneState,
    create_checker_floor,
    make_smplx_model,
    setup_scene,
    write_mesh,
)
from .utils.decimate import face_vertex_mask, load_or_build
from .utils import foot_locking as fl
from .utils.mixamo_character import (
    MixamoCharacter,
    load_mixamo_character,
    pose_mixamo_character,
    prepare_runtime_caches,
    build_smplx_fast_cache,
    smplx_forward_fast,
)


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle -> xyzw quaternion, the convention the locker uses."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = aa / theta
    half = 0.5 * theta
    return np.array([*(axis * np.sin(half)), np.cos(half)], dtype=np.float32)


def _quat_to_aa(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> axis-angle, taking the shorter arc."""
    q = np.asarray(q, dtype=np.float32)
    if q[3] < 0.0:
        q = -q
    xyz = q[:3]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    theta = 2.0 * np.arctan2(sin_half, float(q[3]))
    return (xyz / sin_half * theta).astype(np.float32)


def _aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues for a single axis-angle vector."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = aa / theta
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K)).astype(np.float32)


def _sample_texture_at_uvs(uvs: np.ndarray, tex_np: np.ndarray) -> np.ndarray:
    """Per-vertex RGBA color from a per-vertex UV array and an in-memory image.

    Used for Mixamo character bundles where the UVs ride directly with the
    mesh (no separate uv_faces table). UV (0,0) is bottom-left in GL
    convention, so the V axis is flipped before indexing.
    """
    H, W = tex_np.shape[:2]
    u = uvs[:, 0]
    v = 1.0 - uvs[:, 1]
    x = np.clip((u * (W - 1)).round().astype(np.int32), 0, W - 1)
    y = np.clip((v * (H - 1)).round().astype(np.int32), 0, H - 1)
    colors = tex_np[y, x]  # (M, 3) or (M, 4)
    if colors.shape[-1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=colors.dtype)
        colors = np.concatenate([colors, alpha], axis=1)
    return colors


class MotionVisualizer:
    HEADER_FMT = ">IIII"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)

    def __init__(
        self,
        smplx_model: torch.nn.Module,
        viser_server: viser.ViserServer,
        shared_state,
        mesh_name: str = "/mesh_smplx",
        queue_maxsize: int = 32,
        mixamo_character_path: str | Path | None = None,
        motion_stream: str = "vertices",
        motion_stride: int = 1,
        ground_character: bool = False,
        ground_seconds: float = 1.5,
        foot_locking: bool = False,
        foot_lock_blend: float = 0.5,
        decimate: int = 0,
        decimate_face_detail: float = 6.0,
        # uvmap_path: str | None = None,
    ):
        self.smplx_model = smplx_model
        self.viser_server = viser_server
        self.shared_state = shared_state
        self.mesh_name = mesh_name
        # "vertices": add the mesh once, then push only the per-frame vertex
        #   buffer (viser sends a partial SceneNodeUpdateMessage carrying just
        #   that prop). ~3x less traffic than "glb" and no per-frame encode.
        # "glb":      legacy add_mesh_trimesh() every frame. viser exports the
        #   whole mesh to GLB on each call, so this re-encodes and re-ships
        #   vertices AND faces 25x/s. Only viable on a LAN, but it is the only
        #   path that carries per-vertex colours (Mixamo textures).
        self.motion_stream = motion_stream
        # Render (and therefore send) only every Nth motion frame.
        #
        # Viser's message buffer already coalesces same-node updates, so a
        # client that can only drain ~12 fps silently discards half of a
        # 25 fps stream. Producing those frames is wasted twice over: the
        # SMPL-X pose costs real GPU (35.8 ms/frame on the Space's A100,
        # contending with Moshi) and the bytes cost bandwidth the client
        # does not have. Skipping them outright reclaims both, and costs
        # nothing the viewer was actually displaying.
        self.motion_stride = max(1, int(motion_stride))
        # Decimation table: a fixed subset of the character's own vertices
        # plus a retriangulated face list. Built once from the rest pose, so
        # the per-frame cost is one gather. See utils/decimate.py.
        self.decimate = decimate
        self.decimate_face_detail = decimate_face_detail
        # Foot locking. MIBURI predicts lower-body pose and translation
        # velocity independently, so a planted foot drifts a few cm/s across
        # the floor. This removes that online, correcting only leg rotations
        # -- root translation is left exactly as the model produced it.
        self._foot_locker = None
        self._foot_contacts = None
        self._foot_rig = None
        self._foot_lock_enabled = bool(foot_locking)
        self._foot_lock_blend = foot_lock_blend
        # How far the toes travel per frame before vs after locking, so the
        # live demo can report whether it is actually doing anything on real
        # motion rather than only on synthetic tests.
        self._slide_raw = 0.0
        self._slide_locked = 0.0
        self._slide_n = 0
        self._mesh_bottom = None
        # Vertical grounding. Nothing in the pipeline ties the character to
        # the floor: velocity2position_mixeddiff integrates a predicted
        # translation velocity from an arbitrary starting height, so any
        # constant bias in it just persists and the feet sit above or below
        # the drawn floor. This closes the loop -- measure the lowest drawn
        # vertex against the floor and feed the residual back into the next
        # frame's translation.
        self._ground_enabled = bool(ground_character)
        self._ground_offset = 0.0
        self._ground_hist: deque[float] = deque(
            maxlen=max(1, int(round(ground_seconds * 25.0))))
        self._floor_y = 0.0
        self._prev_toes_raw = None
        self._prev_toes_locked = None
        self._keep_idx = None
        self._faces_dec = None
        self._mesh_handle = None
        # Rolling per-frame timing, logged every _STAT_EVERY frames. Splits
        # the cost into pose (GPU LBS + device->host) vs push (viser
        # serialise + enqueue) so a slow viewer can be told apart from a
        # slow forward pass.
        self._stat_n = 0
        self._stat_pose = 0.0
        self._stat_push = 0.0
        self._stat_t0 = None
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._viz_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.mframe_idx = 0

        # default solid RGBA color (pastel blue) as 0-1 floats; trimesh
        # broadcasts to per-vertex on assignment to `mesh.visual.vertex_colors`.
        self.vertex_colors = [80/255, 150/255, 250/255, 1.0]
        if not torch.cuda.is_available():
            log("warning", "CUDA not available, motion visualization can have high latency.")

        self.smplx_model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Build the SMPL-X fast-LBS cache for the zero-betas case used by
        # this demo. Saves ~30% per frame in the SMPL-X-only path by
        # skipping the shape-blendshape einsum and the J_regressor matmul.
        self._smplx_fast_cache = build_smplx_fast_cache(self.smplx_model)
        # Warm CUDA kernels on the path actually used at runtime
        # (smplx_forward_fast), not the upstream forward we no longer call.
        self.warmup_smplx()

        self.mixamo_character: MixamoCharacter | None = None
        if mixamo_character_path is not None:
            device = self.smplx_model.device
            self.mixamo_character = load_mixamo_character(
                mixamo_character_path, device=device,
            )
            # Precompute the LBS caches (j_rest, T_rest_inv, v_rest_h,
            # parents) so that pose_mixamo_character skips the kinematic
            # chain Python loop, the SMPL-X vertex skinning, and the per-
            # frame `v_offset` broadcast on each call.
            prepare_runtime_caches(self.mixamo_character, self.smplx_model)
            # Prefer pre-baked per-vertex RGBA when present (per-submesh
            # textures already sampled in Blender). Fall back to single
            # texture + UV bake, or solid color, in that order.
            mc = self.mixamo_character
            if mc.vertex_colors is not None:
                self.vertex_colors = mc.vertex_colors
                log("info", f"motion_vis_server: using pre-baked per-vertex "
                           f"colors ({mc.vertex_colors.shape[0]} verts)")
            elif mc.uv_coords is not None and mc.texture_png is not None:
                try:
                    import io
                    tex_img = Image.open(io.BytesIO(mc.texture_png)).convert("RGBA")
                    self.vertex_colors = _sample_texture_at_uvs(
                        mc.uv_coords.cpu().numpy(), np.array(tex_img)
                    )
                    log("info", f"motion_vis_server: baked texture "
                               f"({tex_img.size[0]}x{tex_img.size[1]}) onto "
                               f"{mc.num_verts} vertices")
                except Exception as exc:
                    log("warning", f"motion_vis_server: texture bake failed ({exc}); "
                                   "falling back to solid color")
                    self.vertex_colors = [200/255, 200/255, 200/255, 1.0]
            else:
                self.vertex_colors = [200/255, 200/255, 200/255, 1.0]
            log("info", f"motion_vis_server: rendering Mixamo character "
                       f"'{mc.name}' (V={mc.num_verts}, F={mc.num_faces})")

        if self._foot_lock_enabled:
            try:
                self._build_foot_locker()
            except Exception as exc:
                log("warning", f"motion_vis_server: foot locking disabled ({exc!r})")
                self._foot_lock_enabled = False

        if self.decimate and self.motion_stream == "vertices":
            try:
                self._build_decimation()
            except Exception as exc:
                log("warning", f"motion_vis_server: decimation disabled ({exc!r})")
                self._keep_idx = self._faces_dec = None

        # Move the world floor below the character's feet by rebuilding
        # /floor at the character's feet Y. setup_scene already added the
        # initial floor at Y=0; this just replaces it once we know the
        # runtime feet height.
        try:
            self._reposition_floor()
        except Exception as exc:
            log("warning", f"motion_vis_server: floor reposition skipped: {exc!r}")

        # uv_coords, _ = load_smplx_uv_from_obj(uv_template_obj)
        # assert uv_coords.shape[0] == vertices.shape[1], "UV count must match vertex count."

    def _reset_session_state(self) -> None:
        """Drain any motion packets left in the queue from a previous
        audio session, zero the per-session frame index, and remove the
        currently-rendered character mesh from the scene. Called from
        websocket_handler at the start of every new connection."""
        drained = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.queue.task_done()
                drained += 1
        self.mframe_idx = 0
        if self._foot_locker is not None:
            self._foot_locker.reset()
            self._foot_contacts.reset()
        self._stat_n = 0
        self._stat_pose = 0.0
        self._stat_push = 0.0
        self._stat_t0 = None
        # Remove the stale character mesh(es) so the viewer doesn't keep
        # showing the previous pose until the first new frame lands. The
        # node lives in Viser's scene tree as "/mesh_smplx" (the displayed
        # name "mesh_smplx" in the viewer drops the leading slash). We
        # sweep Viser's authoritative handle map and remove any node whose
        # name matches `self.mesh_name` exactly OR sits under that path,
        # then drop our bookkeeping dict entry.
        self.shared_state.body_meshes.pop("smplx", None)
        # Drop the cached streaming handle so the next session re-adds the
        # mesh instead of pushing vertices at a node that no longer exists.
        self._mesh_handle = None
        handles_map = getattr(
            self.viser_server.scene, "_handle_from_node_name", {}
        )
        removed_names = []
        target = self.mesh_name  # "/mesh_smplx"
        for name, handle in list(handles_map.items()):
            if name != target and not name.startswith(target + "/"):
                continue
            try:
                if getattr(handle, "_impl", None) is not None and handle._impl.removed:
                    continue
                handle.remove()
                removed_names.append(name)
            except Exception as exc:
                log("warning", f"motion_vis_server: mesh remove on reset failed for {name!r}: {exc!r}")
        if drained or removed_names:
            log("info", f"motion_vis_server: reset (drained={drained}, removed={removed_names})")

    def _reposition_floor(self) -> None:
        """Replace the world floor with big checker squares, positioned
        just below the character's *runtime* feet height.

        How the runtime feet Y is estimated:
            The motion data from gest-server-face2 produces a `transl`
            calibrated for the SMPL-X v_template (feet rest-Y ~= -1.302
            m). For SMPL-X that lands feet at world Y ~= 0. A Mixamo
            character with a different rest-frame feet Y will land at:
                runtime_feet_y ~= verts_tpose.Y_min - smplx_v_template.Y_min
            which is 0 for SMPL-X, +0.21 m for y_bot (taller-rest-frame
            offset), -0.10 m for ch31 (lower feet), etc.
            We place the floor `margin` below that:
                margin = 10 cm for Mixamo (shoes / mesh extends a bit
                                  below the SMPL-X-equivalent feet line)
                       = 2 cm for SMPL-X (bare feet)
        """
        smplx_v_y_min = float(self.smplx_model.v_template[:, 1].min().item())
        if self.mixamo_character is not None:
            verts_y_min = float(self.mixamo_character.verts_tpose[:, 1].min().item())
            if self.mixamo_character.name == "y_bot":
                margin = -0.001
            else:
                margin = 0.0001
        else:
            verts_y_min = smplx_v_y_min
            margin = -0.001 # 001
        runtime_feet_y = verts_y_min - smplx_v_y_min
        floor_y = runtime_feet_y - margin
        floor_handle = getattr(self.shared_state, "floor", None)
        if floor_handle is None:
            log("info", f"motion_vis_server: floor handle not present yet "
                       f"(skipping; would have moved to Y={floor_y:.3f})")
            return
        try:
            new_floor = create_checker_floor(size=8.0, n_squares=32)
            new_handle = self.viser_server.scene.add_mesh_trimesh(
                name="/floor",
                mesh=new_floor,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, floor_y, 0.0),
                visible=True,
            )
            # Keep state.floor pointing at the live handle so any future
            # floor toggle reads the live handle.
            self.shared_state.floor = new_handle
            self._floor_y = floor_y
            log("info", f"motion_vis_server: floor at Y={floor_y:.3f} "
                       f"(runtime feet Y~={runtime_feet_y:.3f}, margin={margin:.2f}; "
                       f"{'mixamo' if self.mixamo_character is not None else 'smplx'})")
        except Exception as exc:
            # Fall back to just repositioning the existing floor.
            log("warning", f"motion_vis_server: floor mesh replace failed "
                          f"({exc!r}); just repositioning.")
            floor_handle.position = (0.0, floor_y, 0.0)
        
    def get_smplx_mesh(self, forward_kwargs: dict[str, torch.Tensor]) -> trimesh.Trimesh:
        if self.mixamo_character is not None:
            with torch.no_grad():
                verts, faces = pose_mixamo_character(
                    self.mixamo_character, self.smplx_model, forward_kwargs,
                )
            mesh = trimesh.Trimesh(
                vertices=verts.squeeze(0).cpu().numpy(),
                faces=faces.cpu().numpy(),
                process=False,
            )
            mesh.visual.vertex_colors = self.vertex_colors
            return mesh

        with torch.no_grad():
            verts_t, faces_t = smplx_forward_fast(
                self._smplx_fast_cache, forward_kwargs,
            )
        vertices = verts_t
        faces = faces_t.cpu().numpy() if isinstance(faces_t, torch.Tensor) else faces_t

        mesh = trimesh.Trimesh(
            vertices=vertices.squeeze(0).cpu().numpy(),
            faces=faces,
            process=False,
        )
        mesh.visual.vertex_colors = self.vertex_colors
        return mesh
    
    def _build_foot_locker(self) -> None:
        """Set up the causal locker against this demo's own SMPL-X rig."""
        full = fl.build_foot_rig(self.smplx_model, world_up_axis="y")
        self._foot_rig = fl.reduce_to_legs(full)
        motion_fps = 25.0
        self._foot_locker = fl.OnlineFootLocker(
            self._foot_rig, fps=motion_fps, blend_time=self._foot_lock_blend,
        )
        self._foot_contacts = fl.CausalFootContacts(
            fps=motion_fps, up=self._foot_rig.up,
            toe_rest_height=self._foot_rig.legs[0].toe_height,
        )
        # Root position is translation + R @ rest_pelvis: smplx_forward_fast
        # leaves the pelvis rotation identity inside the chain and applies
        # global_rotation about the origin afterwards, so the pelvis lands at
        # the rotated rest position plus the translation.
        self._rest_root = full.offsets[0].astype(np.float32).copy()
        log("info", f"motion_vis_server: foot locking on -- "
                    f"{len(self._foot_rig.parents)}-joint leg rig, "
                    f"blend {self._foot_lock_blend:.2f}s, floor auto-calibrated")

    def _lock_feet(self, motion_frame: torch.Tensor, transl: torch.Tensor) -> torch.Tensor:
        """Correct one frame's leg rotations in place-ish; returns a new frame.

        `motion_frame` is (1, 165) axis-angle for 55 SMPL-X joints, matching
        what the demo already unpacks; only the eight leg joints change.
        """
        aa = motion_frame[0].detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        rig = self._foot_rig
        chain = rig.chain

        quats = np.stack([_aa_to_quat(aa[j]) for j in chain])
        R = _aa_to_matrix(aa[0])
        root = transl[0].detach().cpu().numpy().astype(np.float32) + R @ self._rest_root

        toes = fl.toe_positions(rig, quats, root)
        contacts = self._foot_contacts.step(toes)
        corrected = self._foot_locker.step(
            quats, root, contacts, ground=self._foot_contacts.floor,
        )

        # Only count frames where a foot is actually claimed to be planted --
        # a swinging foot is *supposed* to move, so including it would dilute
        # the measurement into meaninglessness.
        planted = contacts > self._foot_locker.contact_threshold
        if planted.any():
            locked_toes = fl.toe_positions(rig, corrected, root)
            if self._prev_toes_raw is not None:
                raw = np.linalg.norm(toes - self._prev_toes_raw, axis=1)[planted]
                got = np.linalg.norm(locked_toes - self._prev_toes_locked, axis=1)[planted]
                self._slide_raw += float(raw.mean())
                self._slide_locked += float(got.mean())
                self._slide_n += 1
            self._prev_toes_raw, self._prev_toes_locked = toes, locked_toes
        else:
            self._prev_toes_raw = self._prev_toes_locked = None

        # Only hips, knees, ankles and toes are ever written; the root
        # rotation is untouched by the solver, so it is left alone here.
        out = aa.copy()
        for slot, joint in enumerate(chain):
            if joint == 0:
                continue
            out[joint] = _quat_to_aa(corrected[slot])
        return torch.from_numpy(out.reshape(1, -1)).to(motion_frame.device)

    def _build_decimation(self) -> None:
        """Precompute the vertex subset for whichever character we render.

        Works for both paths: the Mixamo bundle carries its own rest verts,
        faces and LBS weights, and at 27850 verts it benefits even more than
        SMPL-X does.
        """
        if self.mixamo_character is not None:
            mc = self.mixamo_character
            verts = mc.verts_tpose.detach().cpu().numpy()
            faces = mc.faces.detach().cpu().numpy()
            w = mc.lbs_weights.detach().cpu().numpy()
            from .utils.decimate import FACE_JOINTS
            cols = [j for j in FACE_JOINTS if j < w.shape[1]]
            fmask = w[:, cols].sum(axis=1) > 0.5
            name = mc.name
        else:
            verts = self.smplx_model.v_template.detach().cpu().numpy()
            f = self.smplx_model.faces
            faces = f if isinstance(f, np.ndarray) else f.detach().cpu().numpy()
            fmask = face_vertex_mask(self.smplx_model)
            name = "smplx"

        cache = Path("assets_dep") / "decimation_cache" / \
            f"{name}_{self.decimate}_{self.decimate_face_detail:g}.npz"
        table = load_or_build(
            cache, verts, faces, face_mask=fmask,
            target_verts=self.decimate, face_detail=self.decimate_face_detail,
        )
        self._keep_idx = np.asarray(table["keep_idx"], dtype=np.int64)
        self._faces_dec = np.asarray(table["faces"], dtype=np.uint32)
        full = len(verts) * 3 * 4
        cut = len(self._keep_idx) * 3 * 4
        log("info", f"motion_vis_server: decimation on -- {len(verts)} -> "
                    f"{len(self._keep_idx)} verts, {full/1024:.0f} -> {cut/1024:.0f} kB/frame "
                    f"({full/cut:.1f}x less at the same 25 fps)")

    def get_posed_vertices(
        self, forward_kwargs: dict[str, torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posed vertices + faces as numpy, skipping the trimesh round-trip.

        Same maths as `get_smplx_mesh`, but returns raw arrays: the
        "vertices" stream path never needs a trimesh object, and building one
        per frame just to read `.vertices` back out is wasted work.
        """
        with torch.no_grad():
            if self.mixamo_character is not None:
                verts, faces = pose_mixamo_character(
                    self.mixamo_character, self.smplx_model, forward_kwargs,
                )
            else:
                verts, faces = smplx_forward_fast(
                    self._smplx_fast_cache, forward_kwargs,
                )
        verts_np = verts.squeeze(0).detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
        return verts_np, faces_np.astype(np.uint32)

    _STAT_EVERY = 50

    def _record_frame_timing(self, t0: float, t1: float, t2: float) -> None:
        if self._stat_t0 is None:
            self._stat_t0 = t0
        self._stat_n += 1
        self._stat_pose += (t1 - t0) * 1e3
        self._stat_push += (t2 - t1) * 1e3
        if self._stat_n < self._STAT_EVERY:
            return
        wall = t2 - self._stat_t0
        n = self._stat_n
        slide = ""
        if self._slide_n:
            raw_mm = self._slide_raw / self._slide_n * 1e3
            got_mm = self._slide_locked / self._slide_n * 1e3
            cut = (1.0 - got_mm / raw_mm) * 100.0 if raw_mm > 1e-9 else 0.0
            churn = self._foot_locker.transitions if self._foot_locker else 0
            slide = (f" | planted-foot slide {raw_mm:5.2f} -> {got_mm:5.2f} mm/frame "
                     f"({cut:4.1f}% removed, n={self._slide_n}, "
                     f"lock flips {churn}) | floor est {self._foot_contacts.floor:+.3f} m")
        if self._mesh_bottom is not None:
            slide += (f" | mesh bottom {self._mesh_bottom:+.3f} vs floor "
                      f"{self._floor_y:+.3f} m")
            if self._ground_enabled:
                slide += f" (ground offset {self._ground_offset:+.3f})"
            self._mesh_bottom = None
            if self._foot_locker is not None:
                self._foot_locker.transitions = 0
            self._slide_raw = self._slide_locked = 0.0
            self._slide_n = 0
        log("info",
            f"motion: {n / wall:5.1f} fps produced | "
            f"pose {self._stat_pose / n:5.1f} ms | "
            f"push {self._stat_push / n:5.1f} ms | "
            f"wall {wall / n * 1e3:5.1f} ms/frame | "
            f"queue {self.queue.qsize()}{slide}")
        self._stat_n = 0
        self._stat_pose = 0.0
        self._stat_push = 0.0
        self._stat_t0 = None

    def _solid_color(self) -> tuple[int, int, int]:
        """Collapse `self.vertex_colors` to the single RGB tuple viser's
        non-GLB meshes accept.

        `add_mesh_simple` has no per-vertex colour channel (only `add_glb` /
        `add_mesh_trimesh` do), so a textured Mixamo bundle renders as its
        median colour in "vertices" mode. SMPL-X is already a solid colour, so
        the default demo is pixel-identical either way.
        """
        vc = self.vertex_colors
        if isinstance(vc, np.ndarray) and vc.ndim == 2:
            med = np.median(vc[:, :3], axis=0)
            return tuple(int(c) for c in med)
        return tuple(int(round(float(c) * 255)) for c in list(vc)[:3])

    def warmup_smplx(self):
        device = self.smplx_model.device
        dummy_kwargs = {
            "shape": torch.zeros(1, self.smplx_model.NUM_BETAS, device=device),
            "expression": torch.zeros(1, self.smplx_model.NUM_EXPR_COEFFS, device=device),
            "body_pose": torch.zeros(1, 63, device=device),
            "hand_pose": torch.zeros(1, 45 * 2, device=device),
            "head_pose": torch.zeros(1, 9, device=device),
            "global_rotation": torch.zeros(1, 3, device=device),
            "global_translation": torch.zeros(1, 3, device=device),
        }
        log("info", "Warming up SMPLX model...")
        with torch.no_grad():
            for _ in range(4):
                smplx_forward_fast(self._smplx_fast_cache, dummy_kwargs)

    async def start(self, app: web.Application) -> None:
        if self._viz_task is None or self._viz_task.done():
            self._stop_event.clear()
            self._viz_task = asyncio.create_task(self._visualization_loop())

    async def stop(self, app: web.Application) -> None:
        if self._viz_task is not None:
            self._stop_event.set()
            self._viz_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._viz_task

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        log("info", "motion visualization websocket connected")
        # A new connection here implies a fresh audio session on the
        # gest-server side (Start Over reloads the page, which closes and
        # reopens both WS). Drop any frames left from the previous session
        # and reset the per-session counter so the new conversation starts
        # clean.
        self._reset_session_state()
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue
                try:
                    frame_batch = self._parse_message(msg.data)
                except Exception as exc:  # pragma: no cover
                    log("error", f"failed to decode motion packet: {exc}")
                    continue

                await self.queue.put(frame_batch)
        finally:
            log("info", "motion visualization websocket disconnected")
            # Also reset on disconnect so the stale mesh is removed
            # immediately when the audio session ends -- not only when the
            # next one starts. Shortens the window in which the user can
            # see the previous pose after clicking Start Over.
            self._reset_session_state()
        return ws

    def _parse_message(self, message: bytes) -> dict[str, object]:
        if len(message) < self.HEADER_SIZE:
            raise ValueError("motion packet too short for header")
        header = message[: self.HEADER_SIZE]
        payload = message[self.HEADER_SIZE :]
        nframes, motion_frame_dim, exp_dim, transl_dim = struct.unpack(self.HEADER_FMT, header)
        expected = nframes * (motion_frame_dim + exp_dim + transl_dim) * 4
        if len(payload) != expected:
            raise ValueError(
                "unexpected payload length",
                len(payload),
                expected,
            )

        frames = np.frombuffer(payload, dtype=np.float32).reshape(nframes, -1)
        # print("Received motion frames:", frames.shape)
        return {
            "frames": frames,
            "motion_frame_dim": motion_frame_dim,
            "exp_dim": exp_dim,
            "transl_dim": transl_dim,
        }

    async def _visualization_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                batch = await self.queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._handle_batch(batch)
            finally:
                self.queue.task_done()

    async def _handle_batch(self, batch: dict[str, object]) -> None:
        frames: np.ndarray = batch["frames"]  # type: ignore
        motion_frame_dim: int = batch["motion_frame_dim"]  # type: ignore
        exp_dim: int = batch["exp_dim"]  # type: ignore
        transl_dim: int = batch["transl_dim"]  # type: ignore

        total_motion_dim = motion_frame_dim + exp_dim
        device = self.smplx_model.device

        for frame in frames:
            frame_tensor = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0).to(device)
            motion_transexp = frame_tensor[:, : total_motion_dim]
            transl_frame = frame_tensor[:, total_motion_dim : total_motion_dim + transl_dim]
            motion_frame = motion_transexp[:, :motion_frame_dim]
            exp_frame = motion_transexp[:, motion_frame_dim:]

            # print("Visualizing motion frame:", motion_frame.shape, exp_frame.shape, transl_frame.shape)

            if self._foot_lock_enabled and self._foot_locker is not None:
                try:
                    motion_frame = self._lock_feet(motion_frame, transl_frame)
                except Exception as exc:
                    log("warning", f"motion_vis_server: foot locking failed, "
                                   f"passing the frame through ({exc!r})")
                    self._foot_lock_enabled = False

            smplx_body_pose = motion_frame[:, 3 : 22 * 3]
            smplx_global_orient = motion_frame[:, :3]
            smplx_lefthand = motion_frame[:, 25 * 3 : 40 * 3]
            smplx_righthand = motion_frame[:, 40 * 3 : 55 * 3]
            smplx_handpose = torch.cat([smplx_lefthand, smplx_righthand], dim=1)
            smplx_transl = transl_frame
            smplx_head_pose = motion_frame[:, 22 * 3 : 25 * 3]
            expression = exp_frame
            smplx_betas = torch.zeros(motion_frame.shape[0], self.smplx_model.NUM_BETAS, device=device)

            if self._ground_enabled:
                smplx_transl = smplx_transl.clone()
                smplx_transl[:, 1] -= self._ground_offset

            forward_kwargs = {
                "shape": smplx_betas,
                "expression": expression,
                "body_pose": smplx_body_pose,
                "hand_pose": smplx_handpose,
                "head_pose": smplx_head_pose,
                "global_rotation": smplx_global_orient,
                "global_translation": smplx_transl,
            }

            # Cheap skip: bail before the SMPL-X forward, not after, so the
            # GPU work is saved too rather than just the socket write.
            if self.motion_stride > 1 and (self.mframe_idx % self.motion_stride):
                self.mframe_idx += 1
                continue

            _t0 = time.perf_counter()
            if self.motion_stream == "vertices":
                verts, faces = self.get_posed_vertices(forward_kwargs)
                if self._keep_idx is not None:
                    verts = verts[self._keep_idx]
                    faces = self._faces_dec
                # Lowest drawn vertex: what a viewer actually judges as
                # "standing on the floor". Runs for every stream mode, not
                # only the decimated one.
                low = float(verts[:, 1].min())
                self._mesh_bottom = low if self._mesh_bottom is None else \
                    min(self._mesh_bottom, low)
                if self._ground_enabled:
                    # Trailing minimum, not the instantaneous value: a foot
                    # lifted mid-step is legitimately high, and chasing that
                    # would drive the whole body downwards.
                    self._ground_hist.append(low - self._floor_y)
                    residual = min(self._ground_hist)
                    self._ground_offset += 0.15 * residual
                _t1 = time.perf_counter()
                if self._mesh_handle is None:
                    self._mesh_handle = self.viser_server.scene.add_mesh_simple(
                        name=self.mesh_name,
                        vertices=verts,
                        faces=faces,
                        color=self._solid_color(),
                        flat_shading=False,
                        side="front",
                    )
                    self.shared_state.body_meshes["smplx"] = self._mesh_handle
                else:
                    self._mesh_handle.vertices = verts
            else:
                mesh = self.get_smplx_mesh(forward_kwargs)
                _t1 = time.perf_counter()
                self.shared_state.body_meshes["smplx"] = write_mesh(
                    self.viser_server, mesh, self.mesh_name,
                )
            _t2 = time.perf_counter()
            self.mframe_idx += 1
            self._record_frame_timing(_t0, _t1, _t2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--smplx-dir", type=str, required=True)
    parser.add_argument("--uvmap-path", type=str, default=None)
    parser.add_argument("--motion-stream", type=str, default="vertices",
                        choices=["vertices", "glb"],
                        help="How each motion frame reaches the browser. "
                             "'vertices' (default) adds the mesh once and then "
                             "streams only the vertex buffer. 'glb' re-exports "
                             "the whole mesh every frame (legacy; ~3x the "
                             "traffic, but keeps per-vertex Mixamo colours).")
    parser.add_argument("--ground-character",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Keep the character standing on the floor. The "
                             "translation velocity is integrated from an "
                             "arbitrary starting height, so the feet can end "
                             "up above or below the floor plane; this "
                             "measures the lowest drawn vertex against the "
                             "floor and corrects the offset.")
    parser.add_argument("--foot-locking", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Remove foot sliding online. MIBURI predicts "
                             "lower-body pose and translation velocity "
                             "independently, so a planted foot drifts a few "
                             "cm/s. Corrects leg rotations only; root "
                             "translation is untouched.")
    parser.add_argument("--foot-lock-blend", type=float, default=0.5,
                        help="Seconds over which a foot eases into and out of "
                             "a lock. Lower is crisper but can pop.")
    parser.add_argument("--motion-stride", type=int, default=1,
                        help="Render only every Nth motion frame. 1 (default) "
                             "= all 25 fps. 2 halves both the GPU posing cost "
                             "and the viewer bandwidth; because viser coalesces "
                             "updates, a client that could not drain 25 fps was "
                             "discarding those frames anyway.")
    parser.add_argument("--motion-decimate", type=int, default=0,
                        help="Stream a decimated mesh of about N vertices "
                             "instead of the full one. 0 (default) = off. "
                             "The motion is unchanged; only the surface gets "
                             "coarser, which cuts per-frame bytes roughly "
                             "linearly. SMPL-X 10475 -> 3500 is ~3x less.")
    parser.add_argument("--decimate-face-detail", type=float, default=6.0,
                        help="How much finer the facial grid is than the body "
                             "grid when decimating. Higher keeps more of the "
                             "face, where expression lives.")
    parser.add_argument("--mixamo-character", type=str, default=None,
                        help="Path to a Mixamo character .npz bundle "
                             "(e.g. assets_dep/mixamo_characters_release/y_bot.npz). "
                             "When set, the demo renders this character driven "
                             "by SMPL-X joint transforms instead of the default "
                             "SMPL-X mesh.")
    args = parser.parse_args()

    
    smplx_path = Path(args.smplx_dir).resolve()
    print(f"Loading SMPLX model from: {smplx_path}")
    smplx_model = make_smplx_model(smplx_path, gender="NEUTRAL_2020")
    log("info", "Init SMPLX...")

    viser_server = viser.ViserServer(port=args.viser_port)
    state = setup_scene(viser_server)
    log("info", "Init Viser...")

    log("info", "Starting motion visualizer...")
    visualizer = MotionVisualizer(
        smplx_model, viser_server, state,
        mixamo_character_path=args.mixamo_character,
        motion_stream=args.motion_stream,
        motion_stride=args.motion_stride,
        ground_character=args.ground_character,
        foot_locking=args.foot_locking,
        foot_lock_blend=args.foot_lock_blend,
        decimate=args.motion_decimate,
        decimate_face_detail=args.decimate_face_detail,
    )
    app = web.Application()
    app.router.add_get("/ws/motion", visualizer.websocket_handler)
    app.on_startup.append(visualizer.start)
    app.on_cleanup.append(visualizer.stop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
