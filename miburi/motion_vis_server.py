import argparse
import asyncio
import contextlib
import struct
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
from .utils.mixamo_character import (
    MixamoCharacter,
    load_mixamo_character,
    pose_mixamo_character,
    prepare_runtime_caches,
    build_smplx_fast_cache,
    smplx_forward_fast,
)


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
        # uvmap_path: str | None = None,
    ):
        self.smplx_model = smplx_model
        self.viser_server = viser_server
        self.shared_state = shared_state
        self.mesh_name = mesh_name
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
        # Remove the stale character mesh(es) so the viewer doesn't keep
        # showing the previous pose until the first new frame lands. The
        # node lives in Viser's scene tree as "/mesh_smplx" (the displayed
        # name "mesh_smplx" in the viewer drops the leading slash). We
        # sweep Viser's authoritative handle map and remove any node whose
        # name matches `self.mesh_name` exactly OR sits under that path,
        # then drop our bookkeeping dict entry.
        self.shared_state.body_meshes.pop("smplx", None)
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

            smplx_body_pose = motion_frame[:, 3 : 22 * 3]
            smplx_global_orient = motion_frame[:, :3]
            smplx_lefthand = motion_frame[:, 25 * 3 : 40 * 3]
            smplx_righthand = motion_frame[:, 40 * 3 : 55 * 3]
            smplx_handpose = torch.cat([smplx_lefthand, smplx_righthand], dim=1)
            smplx_transl = transl_frame
            smplx_head_pose = motion_frame[:, 22 * 3 : 25 * 3]
            expression = exp_frame
            smplx_betas = torch.zeros(motion_frame.shape[0], self.smplx_model.NUM_BETAS, device=device)

            forward_kwargs = {
                "shape": smplx_betas,
                "expression": expression,
                "body_pose": smplx_body_pose,
                "hand_pose": smplx_handpose,
                "head_pose": smplx_head_pose,
                "global_rotation": smplx_global_orient,
                "global_translation": smplx_transl,
            }

            mesh = self.get_smplx_mesh(forward_kwargs)
            self.shared_state.body_meshes["smplx"] = write_mesh(
                self.viser_server, mesh, self.mesh_name,
            )
            self.mframe_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--smplx-dir", type=str, required=True)
    parser.add_argument("--uvmap-path", type=str, default=None)
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
    )
    app = web.Application()
    app.router.add_get("/ws/motion", visualizer.websocket_handler)
    app.on_startup.append(visualizer.start)
    app.on_cleanup.append(visualizer.stop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
