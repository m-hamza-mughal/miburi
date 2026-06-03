"""FBX -> rest-pose mesh extraction.

The Mixamo FBX files store the body as a skinned mesh. We need the verts in
their bind pose (no animation evaluated). The cleanest reliable path is a
Blender headless subprocess — `blender --background --python` — which uses
Blender's own FBX importer and reads the rest pose via the depsgraph.

We deliberately do not try `trimesh.load_mesh(..., force='mesh')` here:
Trimesh's FBX path requires the `assimp` shared lib or `pyassimp`, neither of
which is in the `miburi` env. Blender is on $PATH and works out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np


_BLENDER_HELPER = Path(__file__).with_name("_blender_fbx_export.py")


@dataclass
class MixamoFBXMesh:
    verts: np.ndarray                      # (Nm, 3) float32, world-coords in Y-up frame
    faces: np.ndarray                      # (Fm, 3) int32
    uv: np.ndarray | None                  # (Nm, 2) float32 or None
    mesh_name: str                         # name of the source Blender mesh
    source_fbx: Path
    native_lbs_weights: np.ndarray | None  # (Nm, 55) float32, rows sum to ~1; from the FBX rig
    # Per-bone armature data, also in Y-up world-coords. Used for skeleton
    # retargeting (per-vertex shift so verts sit correctly relative to SMPL-X
    # J_regressor positions rather than Mixamo's bone positions).
    bone_names: np.ndarray | None          # (Nb,) object array of bone names
    bone_heads: np.ndarray | None          # (Nb, 3) bone head world positions
    bone_smplx_joint_idx: np.ndarray | None  # (Nb,) SMPL-X joint index per bone
    # Embedded texture (PNG-encoded bytes) and the material name that owned it
    texture_png: bytes | None
    texture_material: str | None
    # Pre-baked per-vertex RGBA colors (uint8) from each submesh's diffuse
    # texture. When the FBX has multiple submeshes (body + clothes + hair),
    # this is the only practical way to keep textures since each submesh has
    # its own texture image and UV layout.
    vertex_colors: np.ndarray | None       # (Nm, 4) uint8 RGBA or None
    submesh_names: np.ndarray | None       # (Nsub,) object array
    submesh_ranges: np.ndarray | None      # (Nsub, 2) [vstart, vend) per submesh
    # Which submesh is the humanoid body silhouette -- used by the chamfer
    # fit so hair/hoodie/etc. don't blow up beta. (vstart, vend) into verts.
    body_vert_range: np.ndarray | None     # (2,)


def _find_blender_binary(override: str | None) -> str:
    if override is not None:
        if not Path(override).is_file():
            raise FileNotFoundError(f"--blender-bin not found: {override}")
        return override
    found = shutil.which("blender")
    if found is None:
        raise RuntimeError(
            "`blender` not found on PATH. Install Blender (>=3.0) or pass "
            "blender_bin=<path>."
        )
    return found


def _zup_to_yup(verts: np.ndarray) -> np.ndarray:
    """Rotate Blender's Z-up right-handed frame into SMPL-X's Y-up right-handed
    frame: (x, y, z) -> (x, z, -y). This is a -90 degree rotation about +X."""
    out = np.empty_like(verts)
    out[..., 0] = verts[..., 0]
    out[..., 1] = verts[..., 2]
    out[..., 2] = -verts[..., 1]
    return out


def load_fbx_mesh(
    fbx_path: str | Path,
    *,
    blender_bin: str | None = None,
    keep_npz: bool = False,
    convert_to_yup: bool = True,
) -> MixamoFBXMesh:
    """Extract rest-pose verts + faces (+ UV if present) from an FBX.

    Args:
        fbx_path: path to the input .fbx
        blender_bin: optional override for the Blender executable
        keep_npz: if True, leaves the intermediate NPZ next to the FBX for
            debugging (uses `<fbx_basename>.blender.npz`); otherwise written
            to a tempfile and deleted.
        convert_to_yup: rotate the verts from Blender Z-up to SMPL-X Y-up.
            Default True — keeps downstream code agnostic of the import frame.

    Returns:
        MixamoFBXMesh with arrays as numpy.
    """
    fbx_path = Path(fbx_path).resolve()
    if not fbx_path.is_file():
        raise FileNotFoundError(f"FBX not found: {fbx_path}")
    if not _BLENDER_HELPER.is_file():
        raise FileNotFoundError(
            f"Blender helper script missing: {_BLENDER_HELPER}"
        )

    blender = _find_blender_binary(blender_bin)

    if keep_npz:
        tmp_npz = fbx_path.with_suffix(".blender.npz")
        cleanup = False
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".npz", prefix="fbx_mesh_")
        os.close(tmp_fd)
        tmp_npz = Path(tmp_path)
        cleanup = True

    cmd = [
        blender,
        "--background",
        "--python", str(_BLENDER_HELPER),
        "--",
        str(fbx_path),
        str(tmp_npz),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not tmp_npz.is_file():
            raise RuntimeError(
                f"Blender FBX export failed (rc={result.returncode}).\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        # Forward diffuse-image selection logs to the caller for debugging.
        for line in result.stdout.splitlines():
            if any(t in line for t in ("[diffuse]", "[fbx_export]", "[mat_images]",
                                        "[color]", "[uv_layers]")):
                print(line)
        data = np.load(tmp_npz, allow_pickle=False)
        verts = data["verts"].astype(np.float32)
        faces = data["faces"].astype(np.int32)
        uv = data["uv"].astype(np.float32) if "uv" in data.files else None
        mesh_name = str(data["mesh_name"]) if "mesh_name" in data.files else ""
        native_w = (data["native_lbs_weights"].astype(np.float32)
                    if "native_lbs_weights" in data.files else None)

        bone_names = data["bone_names"] if "bone_names" in data.files else None
        bone_heads = (data["bone_heads"].astype(np.float32)
                      if "bone_heads" in data.files else None)
        bone_smplx_idx = (data["bone_smplx_joint_idx"].astype(np.int64)
                          if "bone_smplx_joint_idx" in data.files else None)
        texture_png = (bytes(data["texture_png"].tobytes())
                       if "texture_png" in data.files else None)
        texture_material = (str(data["texture_material"])
                            if "texture_material" in data.files else None)
        vertex_colors = (data["vertex_colors"].astype(np.uint8)
                         if "vertex_colors" in data.files else None)
        submesh_names = (data["submesh_names"] if "submesh_names" in data.files else None)
        submesh_ranges = (data["submesh_ranges"].astype(np.int64)
                          if "submesh_ranges" in data.files else None)
        body_vert_range = (data["body_vert_range"].astype(np.int64)
                           if "body_vert_range" in data.files else None)

        if convert_to_yup:
            verts = _zup_to_yup(verts)
            if bone_heads is not None:
                bone_heads = _zup_to_yup(bone_heads)
    finally:
        if cleanup and tmp_npz.is_file():
            try:
                tmp_npz.unlink()
            except OSError:
                pass

    return MixamoFBXMesh(
        verts=verts, faces=faces, uv=uv, mesh_name=mesh_name, source_fbx=fbx_path,
        native_lbs_weights=native_w,
        bone_names=bone_names, bone_heads=bone_heads,
        bone_smplx_joint_idx=bone_smplx_idx,
        texture_png=texture_png, texture_material=texture_material,
        vertex_colors=vertex_colors,
        submesh_names=submesh_names, submesh_ranges=submesh_ranges,
        body_vert_range=body_vert_range,
    )
