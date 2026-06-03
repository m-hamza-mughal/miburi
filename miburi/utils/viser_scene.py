"""Pure-viser scene helpers used by motion_vis_server.

Bundles the SMPL-X body model construction (standard `smplx` PyPI + a small
attribute shim for `miburi.utils.mixamo_character`), a checker-pattern floor
mesh, the per-frame mesh write, and the small bit of shared mutable scene
state the visualizer mutates between frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import smplx as _smplx
import torch
import trimesh
import viser
from PIL import Image

from ..client_utils import log


SMPLX_NUM_JOINTS = 55  # body 22 + jaw + 2 eyes + 30 hand


def make_smplx_model(
    model_path: str | Path,
    *,
    gender: str = "NEUTRAL_2020",
    num_betas: int = 300,
    num_expression_coeffs: int = 100,
) -> torch.nn.Module:
    """Build a standard `smplx.SMPLX` model and patch on the attribute aliases
    that `miburi.utils.mixamo_character` expects.

    `mixamo_character` only reads these attrs at cache-build time (it never
    calls `.forward()` at runtime), so a small attribute shim is sufficient
    — we don't need to wrap the forward pass.

    `model_path` may be a directory (e.g. `assets_dep/smplx_2020/`, the form
    `smplx.create` expects) or a specific npz file (e.g.
    `assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz`); in the latter
    case the function walks up to the directory `smplx.create` expects.
    """
    model_path = Path(model_path)
    if model_path.is_file():
        # smplx.create takes the parent of the `smplx/` subdir.
        model_dir = model_path.parent.parent if model_path.parent.name == "smplx" else model_path.parent
    else:
        model_dir = model_path

    model = _smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        flat_hand_mean=True,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        use_pca=False,
    ).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Class-constant style attrs the cache builder reads as `smplx_model.NUM_*`.
    model.NUM_BETAS = num_betas
    model.NUM_EXPR_COEFFS = num_expression_coeffs
    model.NUM_JOINTS = SMPLX_NUM_JOINTS

    # Standard smplx stores expression dirs in `expr_dirs`; mixamo_character
    # reads `exprdirs`. Alias so either name resolves.
    if not hasattr(model, "exprdirs"):
        model.exprdirs = model.expr_dirs

    # `.device` is not a default nn.Module attribute. Bind a property so
    # `self.smplx_model.device` keeps working at every call site.
    if not isinstance(getattr(type(model), "device", None), property):
        type(model).device = property(lambda self: self.v_template.device)

    # Sanity-check the attribute surface mixamo_character relies on. Fail
    # at boot rather than at first frame if any aliasing missed.
    required = (
        "NUM_BETAS", "NUM_EXPR_COEFFS", "NUM_JOINTS",
        "shapedirs", "exprdirs", "v_template", "J_regressor",
        "parents", "lbs_weights", "faces", "device",
    )
    missing = [a for a in required if not hasattr(model, a)]
    if missing:
        raise AttributeError(
            f"viser_scene.make_smplx_model: SMPL-X model is missing "
            f"attributes required by mixamo_character: {missing}"
        )

    return model


def _checker_texture(
    resolution: int = 1024,
    n_squares: int = 16,
    contrast: float = 1.0,
    base_color: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Square checker PIL image used as the floor texture."""
    x = np.linspace(0, n_squares, resolution, endpoint=False)
    y = np.linspace(0, n_squares, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y)
    mask = ((X.astype(int) + Y.astype(int)) % 2)[..., None]
    base = np.array(base_color, float)[None, None, :]
    hi = np.clip(base * (1 + contrast), 100, 255)
    lo = np.clip(base * (1 - contrast), 100, 255)
    tex = mask * hi + (1 - mask) * lo
    return Image.fromarray(tex.astype(np.uint8))


def create_checker_floor(
    size: float = 8.0,
    n_squares: int = 32,
    resolution: int = 1024,
    contrast: float = 1.0,
    base_color: tuple[int, int, int] = (128, 128, 128),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Single textured quad in the XZ plane at Y = center[1]. Drop-in for the
    floor mesh motion_vis_server places under `/floor`."""
    cx, cy, cz = center
    hx, hz = size / 2, size / 2
    vertices = np.array([
        [cx - hx, cy, cz - hz],
        [cx + hx, cy, cz - hz],
        [cx - hx, cy, cz + hz],
        [cx + hx, cy, cz + hz],
    ])
    faces = np.array([[0, 2, 1], [1, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    texture = _checker_texture(resolution, n_squares, contrast, base_color)
    material = trimesh.visual.material.SimpleMaterial(image=texture)
    visual = trimesh.visual.TextureVisuals(uv=uv, image=texture, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)


@dataclass
class SceneState:
    """Mutable scene state motion_vis_server passes to the visualizer: the
    floor handle (rebuilt by `_reposition_floor`) and the current body-mesh
    handle keyed by `"smplx"`."""
    body_meshes: dict[str, viser.SceneNodeHandle] = field(default_factory=dict)
    floor: Optional[viser.SceneNodeHandle] = None


def setup_scene(
    server: viser.ViserServer,
    *,
    floor_size: float = 8.0,
    floor_n_squares: int = 32,
    panel_label: str = "MIBURI",
    # Tuned to frame an SMPL-X character standing at the world origin
    # (feet Y=0, head Y~=1.7) full-body in the viewport, with a slight
    # downward tilt. Both defaults expressed relative to character
    # height for clarity.
    camera_position: tuple[float, float, float] = (0.0, 1.6, 3.8),
    camera_look_at: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> SceneState:
    """Configure theme, lighting, add `/floor`, install the on-connect
    camera pose + a 'Frame Character' GUI button, and return a
    `SceneState` for the visualizer to mutate.

    Lighting is set to the viser client-side defaults (`hdri='city'`,
    `environment_intensity=1.0`, default lights enabled). These are
    called explicitly so the messages are in the scene-serialization
    stream and replay to every new client deterministically — avoids
    stale browser state from any prior session that overrode them.
    """
    server.gui.configure_theme(dark_mode=True, control_layout="floating")
    try:
        server.gui.set_panel_label(panel_label)
    except Exception:
        # `set_panel_label` is best-effort across viser versions.
        pass

    # Scene up-axis: SMPL-X is Y-up. Viser defaults to +Z. Setting this is
    # load-bearing for *lighting*, not just camera ergonomics — per viser's
    # own docs, "default lights and environment map are oriented to match
    # the scene's up direction." Without this, the floor's up face is
    # treated as a side face and the IBL contribution misses it, producing
    # a too-dark high-contrast look.
    try:
        server.scene.set_up_direction(direction="+y")
    except Exception as exc:
        log("warning", f"viser_scene: set_up_direction failed ({exc!r})")

    # Lock the env map + default lights to the viser client-side defaults
    # (== viser client defaults). The HDRI texture for 'city' is bundled
    # locally in the viser client (potsdamer_platz_1k.jpg), so there's no
    # network fetch race — only a brief decode delay on first paint.
    try:
        server.scene.configure_environment_map(
            hdri="city",
            background=False,
            environment_intensity=1.0,
        )
    except Exception as exc:
        log("warning", f"viser_scene: configure_environment_map failed ({exc!r})")
    try:
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    except Exception as exc:
        log("warning", f"viser_scene: configure_default_lights failed ({exc!r})")

    state = SceneState()
    floor_mesh = create_checker_floor(size=floor_size, n_squares=floor_n_squares)
    state.floor = server.scene.add_mesh_trimesh(
        name="/floor", mesh=floor_mesh,
        wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
        visible=True,
    )

    def _frame_character(client: viser.ClientHandle) -> None:
        client.camera.position = camera_position
        client.camera.look_at = camera_look_at
        client.camera.up_direction = (0.0, 1.0, 0.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        _frame_character(client)

    # GUI: a single "Frame Character" button that snaps every connected
    # client back to the character-facing pose. Useful after orbiting.
    try:
        frame_btn = server.gui.add_button("Frame Character")

        @frame_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is not None:
                _frame_character(client)
    except Exception as exc:
        log("warning", f"viser_scene: frame-character button install failed ({exc!r})")

    log("info", f"viser_scene: scene initialised "
                f"(floor={floor_size}m/{floor_n_squares}sq, "
                f"cam={camera_position}->{camera_look_at}, panel='{panel_label}')")
    return state


def write_mesh(
    server: viser.ViserServer,
    mesh: trimesh.Trimesh,
    name: str,
    visible: bool = True,
) -> viser.SceneNodeHandle:
    """Per-frame mesh write. `add_mesh_trimesh` already overwrites a node at
    the same name, so this is just a thin alias for symmetry with the rest
    of the helpers."""
    return server.scene.add_mesh_trimesh(name=name, mesh=mesh, visible=visible)
