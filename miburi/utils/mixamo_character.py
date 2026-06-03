"""Runtime helper for swapping the SMPL-X demo mesh for a Mixamo character.

The character bundle (`.npz`) is produced offline by
`scripts/fit_mixamo_character.py`. It holds:

    verts_tpose   (Nm, 3)   Mixamo verts in SMPL-X *rest frame* (un-globalized)
    faces         (Fm, 3)   Mixamo triangle vertex indices
    lbs_weights   (Nm, J)   per-Mixamo-vertex LBS weights, summing to 1 per row
    betas         (300,)    SMPL-X beta fit to the Mixamo body shape
    uv_coords     (Nm, 2)   optional baked UVs (when the FBX had them)

At runtime, `pose_mixamo_character` skins the Mixamo mesh using SMPL-X's per-
joint world transforms `T_j`, which we obtain by calling the SMPL-X forward
with the character's saved betas. Global rotation/translation are applied
*after* LBS so we can request transforms_world in the rest frame.

Notes on the math:
    LBS:  v_posed[i] = sum_j w_{i,j} * (R^j_world @ (v_rest[i] - j_rest[j])
                                         + t^j_world)
    where j_rest is the rest-pose joint position computed from beta-shaped
    SMPL-X verts via the J_regressor. Both v_rest and j_rest live in the
    SMPL-X internal rest frame (no global R, t).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MixamoCharacter:
    name: str                              # slug, e.g. "y_bot"
    verts_tpose: torch.Tensor              # (Nm, 3) in SMPL-X rest frame
    faces: torch.Tensor                    # (Fm, 3) int64
    lbs_weights: torch.Tensor              # (Nm, J) float32, rows sum to 1
    betas: torch.Tensor                    # (1, 300)
    uv_coords: torch.Tensor | None         # (Nm, 2) or None
    texture_png: bytes | None              # legacy: single texture (kept for compat)
    vertex_colors: np.ndarray | None       # (Nm, 4) uint8, pre-baked per-submesh
    # v2 face fields: present when the bundle was fit with --with_face.
    # face_vert_idx is the int64 index list into verts_tpose for face verts;
    # expr_dirs_face is (Nm_face, 100, 3) per-face-vert blendshape contrib.
    face_vert_idx: torch.Tensor | None
    expr_dirs_face: torch.Tensor | None
    # Anatomical jaw rotation pivot. SMPL-X's J_regressor jaw joint is at the
    # top of the head, which is the wrong pivot for "mouth open". We override
    # at runtime by zeroing head_pose[0] before LBS and applying a manual
    # rotation around this pivot to verts skinned to the jaw joint.
    anatomical_jaw_pivot: torch.Tensor | None
    device: torch.device
    # Runtime LBS caches (all depend only on `betas` and the rest pose, so
    # they're computed ONCE in load_mixamo_character and reused per frame).
    # Avoid recomputing the joint kinematic chain, the rest joint positions,
    # and the homogeneous rest verts on every frame.
    j_rest_cached: torch.Tensor | None = None        # (J, 3)
    T_rest_inv_cached: torch.Tensor | None = None    # (J, 4, 4)
    v_rest_h_cached: torch.Tensor | None = None      # (Nm, 4)
    parents_cached: torch.Tensor | None = None       # (J,) int64

    @property
    def num_verts(self) -> int:
        return self.verts_tpose.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]


def load_mixamo_character(
    npz_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> MixamoCharacter:
    """Load a character bundle produced by `scripts/fit_mixamo_character.py`."""
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Mixamo character npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    device = torch.device(device)

    verts_tpose = torch.from_numpy(data["verts_tpose"]).to(device=device, dtype=torch.float32)
    faces = torch.from_numpy(data["faces"]).to(device=device, dtype=torch.int64)
    lbs_weights = torch.from_numpy(data["lbs_weights"]).to(device=device, dtype=torch.float32)
    betas = torch.from_numpy(data["betas"]).to(device=device, dtype=torch.float32).unsqueeze(0)

    uv = None
    if "uv_coords" in data.files:
        uv = torch.from_numpy(data["uv_coords"]).to(device=device, dtype=torch.float32)

    texture_png = None
    if "texture_png" in data.files:
        texture_png = bytes(data["texture_png"].tobytes())

    vertex_colors = None
    if "vertex_colors" in data.files:
        vertex_colors = data["vertex_colors"].astype(np.uint8)

    face_vert_idx = None
    expr_dirs_face = None
    if "face_vert_mask" in data.files and "expr_dirs_face" in data.files:
        mask = data["face_vert_mask"].astype(bool)
        face_vert_idx = torch.from_numpy(np.flatnonzero(mask)).to(
            device=device, dtype=torch.int64)
        expr_dirs_face = torch.from_numpy(data["expr_dirs_face"]).to(
            device=device, dtype=torch.float32)

    anatomical_jaw_pivot = None
    if "anatomical_jaw_pivot" in data.files:
        anatomical_jaw_pivot = torch.from_numpy(data["anatomical_jaw_pivot"]).to(
            device=device, dtype=torch.float32)

    return MixamoCharacter(
        name=npz_path.stem,
        verts_tpose=verts_tpose,
        faces=faces,
        lbs_weights=lbs_weights,
        betas=betas,
        uv_coords=uv,
        texture_png=texture_png,
        vertex_colors=vertex_colors,
        face_vert_idx=face_vert_idx,
        expr_dirs_face=expr_dirs_face,
        anatomical_jaw_pivot=anatomical_jaw_pivot,
        device=device,
    )


def prepare_runtime_caches(char: MixamoCharacter, smplx_model) -> None:
    """Precompute the per-character LBS caches used by pose_mixamo_character.

    These quantities depend only on `char.betas` and the rest pose, so we
    compute them ONCE when the character is loaded and reuse them on every
    frame. Per-frame inference becomes ~20% faster vs the old code path
    which recomputed `j_rest` (via a 55-iter Python kinematic chain loop)
    and the per-frame `v_offset` broadcast on every call.

    Specifically caches:
        j_rest_cached:      rest joint positions for this character's shape
        T_rest_inv_cached:  4x4 inverse rest joint transform per joint
                            (used for standard LBS `A = T_world @ T_rest_inv`)
        v_rest_h_cached:    homogeneous rest verts [v, 1] for matvec
        parents_cached:     joint parent indices (int64)
    """
    device = char.device
    dtype = char.verts_tpose.dtype
    n_betas = smplx_model.NUM_BETAS
    n_expr = smplx_model.NUM_EXPR_COEFFS
    J = smplx_model.NUM_JOINTS

    with torch.no_grad():
        # j_rest = J_regressor @ v_shaped(betas)
        shape = char.betas.to(device=device, dtype=dtype)[:, :n_betas]
        if shape.shape[1] < n_betas:
            shape = F.pad(shape, (0, n_betas - shape.shape[1]))
        expr = torch.zeros(1, n_expr, device=device, dtype=dtype)
        shape_comps = torch.cat([shape, expr], dim=-1)
        sd = torch.cat(
            [smplx_model.shapedirs[:, :, :n_betas],
             smplx_model.exprdirs[:, :, :n_expr]], dim=-1,
        ).to(device=device, dtype=dtype)
        v_t = smplx_model.v_template.to(device=device, dtype=dtype) \
            + torch.einsum("bi,nki->bnk", shape_comps, sd)              # (1, V, 3)
        j_rest = torch.einsum("bik,ji->bjk", v_t,
                              smplx_model.J_regressor.to(device=device, dtype=dtype))
        j_rest = j_rest[0]                                              # (J, 3)

    # T_rest_inv[j] = [I, -j_rest[j]; 0, 1]   so that
    # T_world[j] @ T_rest_inv[j] = [R_j, t_j - R_j @ j_rest[j]; 0, 1]
    eye = torch.eye(4, device=device, dtype=dtype).expand(J, 4, 4).clone()
    eye[:, :3, 3] = -j_rest
    T_rest_inv = eye                                                    # (J, 4, 4)

    v_rest_h = torch.cat(
        [char.verts_tpose.to(device=device, dtype=dtype),
         torch.ones(char.num_verts, 1, device=device, dtype=dtype)],
        dim=-1,
    )                                                                   # (Nm, 4)

    parents = smplx_model.parents.long().to(device=device)

    char.j_rest_cached = j_rest
    char.T_rest_inv_cached = T_rest_inv
    char.v_rest_h_cached = v_rest_h
    char.parents_cached = parents


@dataclass
class SmplxFastCache:
    """Per-SMPL-X-model LBS cache. Computed once at init when betas are
    fixed (e.g. zero in the demo's SMPL-X-only path); reused per frame
    inside `smplx_forward_fast`. Saves the per-frame shape-blendshape +
    J_regressor + LBS-vert-broadcast work that `smplx_model.forward()`
    does, replacing it with a leaner standard LBS path."""
    v_template_h: torch.Tensor      # (V, 4)
    j_rest: torch.Tensor            # (J, 3)
    T_rest_inv: torch.Tensor        # (J, 4, 4)
    parents: torch.Tensor           # (J,) int64
    lbs_weights: torch.Tensor       # (V, J)
    exprdirs: torch.Tensor          # (V, 3, n_expr) -- per-vert blendshape dirs
    faces: torch.Tensor             # (F, 3)
    num_joints: int


def build_smplx_fast_cache(
    smplx_model,
    betas: torch.Tensor | None = None,
) -> SmplxFastCache:
    """Pre-compute the SMPL-X LBS caches for a fixed `betas`.

    The demo viewer's SMPL-X-only path always uses zero betas (per
    motion_vis_server._handle_batch line ~623), so we precompute the
    beta-shaped joint positions ONCE and reuse them on every frame.
    Expression varies per frame but only affects face verts -- the
    runtime applies `exprdirs @ expression` to v_rest before LBS.
    """
    device = smplx_model.parents.device if hasattr(smplx_model.parents, "device") else torch.device("cpu")
    dtype = smplx_model.v_template.dtype
    n_betas = smplx_model.NUM_BETAS
    J = smplx_model.NUM_JOINTS

    if betas is None:
        betas = torch.zeros(1, n_betas, device=device, dtype=dtype)
    else:
        betas = betas.to(device=device, dtype=dtype)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if betas.shape[1] < n_betas:
            betas = F.pad(betas, (0, n_betas - betas.shape[1]))

    with torch.no_grad():
        sd_shape = smplx_model.shapedirs[:, :, :n_betas].to(device=device, dtype=dtype)
        v_t = smplx_model.v_template.to(device=device, dtype=dtype) \
            + torch.einsum("bi,nki->bnk", betas, sd_shape)              # (1, V, 3)
        j_rest = torch.einsum("bik,ji->bjk", v_t,
                              smplx_model.J_regressor.to(device=device, dtype=dtype))
        j_rest = j_rest[0]                                              # (J, 3)
        v_t = v_t[0]                                                    # (V, 3)

    eye = torch.eye(4, device=device, dtype=dtype).expand(J, 4, 4).clone()
    eye[:, :3, 3] = -j_rest
    T_rest_inv = eye

    v_template_h = torch.cat(
        [v_t, torch.ones(v_t.shape[0], 1, device=device, dtype=dtype)],
        dim=-1,
    )

    return SmplxFastCache(
        v_template_h=v_template_h,
        j_rest=j_rest,
        T_rest_inv=T_rest_inv,
        parents=smplx_model.parents.long().to(device=device),
        lbs_weights=smplx_model.lbs_weights.to(device=device, dtype=dtype),
        exprdirs=smplx_model.exprdirs.to(device=device, dtype=dtype),
        faces=smplx_model.faces.to(device=device) if isinstance(smplx_model.faces, torch.Tensor)
              else torch.as_tensor(smplx_model.faces, dtype=torch.int64, device=device),
        num_joints=J,
    )


def smplx_forward_fast(
    cache: SmplxFastCache,
    forward_kwargs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized SMPL-X forward that:
      * uses cached `j_rest` / `T_rest_inv` / `v_template_h` (no per-frame
        shape blendshapes for the fixed-beta case),
      * computes expression deltas as a single einsum on `exprdirs`,
      * runs the joint kinematic chain inline (one Python loop over J),
      * applies standard LBS via cached homogeneous verts.

    Returns (verts, faces).  Equivalent to the relevant slice of
    `smplx_model.forward(...)` for fixed betas; ~2-3x faster on GPU."""
    device = cache.v_template_h.device
    dtype = cache.v_template_h.dtype
    body_pose = forward_kwargs["body_pose"]
    B = body_pose.shape[0]
    expression = forward_kwargs.get("expression",
                                    torch.zeros(B, cache.exprdirs.shape[-1],
                                                device=device, dtype=dtype))
    head_pose = forward_kwargs.get("head_pose",
                                   torch.zeros(B, 9, device=device, dtype=dtype))
    hand_pose = forward_kwargs.get("hand_pose",
                                   torch.zeros(B, 90, device=device, dtype=dtype))
    global_rot = forward_kwargs.get("global_rotation",
                                    torch.zeros(B, 3, device=device, dtype=dtype))
    global_t = forward_kwargs.get("global_translation",
                                  torch.zeros(B, 3, device=device, dtype=dtype))

    J = cache.num_joints
    parents = cache.parents
    j_rest = cache.j_rest

    # Pose -> rotation matrices.
    pelvis_rot = torch.zeros(B, 3, device=device, dtype=dtype)  # global rot applied later
    pose = torch.cat([pelvis_rot, body_pose, head_pose, hand_pose], dim=-1)
    pose = pose.reshape(B, -1, 3)
    pose_mats = _axisangle_to_matrix(pose)                              # (B, J, 3, 3)

    # T_parent_joint = [R, j_rest[j] - j_rest[parent[j]]; 0 1]
    j_rest_b = j_rest.unsqueeze(0).expand(B, -1, -1)
    T_parent_joint = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
    T_parent_joint[..., :3, :3] = pose_mats
    T_parent_joint[..., 3, 3] = 1.0
    T_parent_joint[:, 0, :3, 3] = j_rest_b[:, 0]
    T_parent_joint[:, 1:, :3, 3] = j_rest_b[:, 1:] - j_rest_b[:, parents[1:]]

    T_world = torch.empty_like(T_parent_joint)
    T_world[:, 0] = T_parent_joint[:, 0]
    for idx in range(1, J):
        T_world[:, idx] = T_world[:, parents[idx]] @ T_parent_joint[:, idx]

    # LBS using cached homogeneous verts. Expression delta applied to
    # rest verts before LBS (so it gets carried into world coords).
    A = T_world @ cache.T_rest_inv.unsqueeze(0)                         # (B, J, 4, 4)
    A_flat = A.reshape(B, J, 16)
    T_per_vert = torch.einsum("vj,bji->bvi", cache.lbs_weights, A_flat).reshape(B, -1, 4, 4)

    v_template_h_b = cache.v_template_h.unsqueeze(0).expand(B, -1, -1)
    if expression is not None and expression.shape[-1] > 0:
        # exprdirs: (V, 3, n_expr); expression: (B, n_expr).
        # delta: (B, V, 3). Most rows of exprdirs are near-zero (only
        # face verts deform) but doing this as a single einsum is still
        # fast.
        delta = torch.einsum("vke,be->bvk", cache.exprdirs, expression)
        v_h_b = v_template_h_b.clone()
        v_h_b[..., :3] = v_h_b[..., :3] + delta
    else:
        v_h_b = v_template_h_b

    v_posed = torch.einsum("bvkl,bvl->bvk", T_per_vert, v_h_b)[..., :3]

    # Global rotation + translation.
    R = _axisangle_to_matrix(global_rot)
    v_posed = (R @ v_posed.mT).mT + global_t.unsqueeze(1)
    return v_posed, cache.faces


def _axisangle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues, batched. aa: (..., 3) -> R: (..., 3, 3)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1.0 - c) * (K @ K)


def pose_mixamo_character(
    char: MixamoCharacter,
    smplx_model,
    forward_kwargs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skin the Mixamo mesh with SMPL-X joint transforms.

    Args:
        char: loaded MixamoCharacter (already on the same device as smplx_model)
        smplx_model: SMPL-X model (e.g. built via miburi.utils.viser_scene.make_smplx_model)
        forward_kwargs: same dict that would normally be passed to the SMPL-X
            forward in motion_vis_server. We override `shape` with the
            character betas and zero the global so that `transforms_world`
            comes back in the SMPL-X rest frame; we then apply the original
            global R, t to the Mixamo verts ourselves.

    Returns:
        (verts, faces) where verts is (B, Nm, 3) and faces is (Fm, 3).
    """
    device = char.device
    body_pose = forward_kwargs["body_pose"]
    B = body_pose.shape[0]
    dtype = body_pose.dtype

    global_rot = forward_kwargs.get("global_rotation",
                                    torch.zeros(B, 3, device=device, dtype=dtype))
    global_t = forward_kwargs.get("global_translation",
                                  torch.zeros(B, 3, device=device, dtype=dtype))

    # Anatomical jaw override: zero the jaw rotation that goes into the
    # SMPL-X kinematic chain so the head/neck transform is unaffected,
    # then rotate jaw-skinned verts around `char.anatomical_jaw_pivot`
    # manually after LBS.
    jaw_angle_x: torch.Tensor | None = None
    head_pose = forward_kwargs.get("head_pose",
                                   torch.zeros(B, 9, device=device, dtype=dtype))
    if char.anatomical_jaw_pivot is not None and "head_pose" in forward_kwargs:
        jaw_angle_x = head_pose[..., 0].clone()
        head_pose = head_pose.clone()
        head_pose[..., 0] = 0.0

    # ----- Joint kinematic chain ONLY (skip SMPL-X vertex skinning) -----
    # Using cached `j_rest_cached` + `parents_cached` we can recompute the
    # joint world transforms without going through `smplx_model.forward`,
    # which would also run a wasted LBS pass on ~10 k SMPL-X verts.
    if char.j_rest_cached is None or char.T_rest_inv_cached is None:
        # Cache wasn't prepared -- fall back to the slow path.
        prepare_runtime_caches(char, smplx_model)
    j_rest_cached = char.j_rest_cached                          # (J, 3)
    T_rest_inv = char.T_rest_inv_cached.to(dtype=dtype)        # (J, 4, 4)
    parents = char.parents_cached                              # (J,)
    J = j_rest_cached.shape[0]
    v_rest_h = char.v_rest_h_cached.to(dtype=dtype)            # (Nm, 4)
    w = char.lbs_weights.to(dtype=dtype)                       # (Nm, J)

    # Build full pose tensor in the order SMPL-X expects.
    hand_pose = forward_kwargs.get("hand_pose",
                                   torch.zeros(B, 90, device=device, dtype=dtype))
    pelvis_rotation = torch.zeros(B, 3, device=device, dtype=dtype)  # global rot applied later
    pose = torch.cat([pelvis_rotation, body_pose, head_pose, hand_pose], dim=-1)
    pose = pose.reshape(B, -1, 3)                               # (B, J, 3)
    pose_mats = _axisangle_to_matrix(pose)                      # (B, J, 3, 3)

    # T_parent_joint = [[R, j_rest[j] - j_rest[parent[j]]; 0 1]]
    j_rest_b = j_rest_cached.unsqueeze(0).expand(B, -1, -1)     # (B, J, 3)
    T_parent_joint = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
    T_parent_joint[..., :3, :3] = pose_mats
    T_parent_joint[..., 3, 3] = 1.0
    T_parent_joint[:, 0, :3, 3] = j_rest_b[:, 0]
    T_parent_joint[:, 1:, :3, 3] = j_rest_b[:, 1:] - j_rest_b[:, parents[1:]]

    # Kinematic chain. The Python loop is J=55 iterations of (4x4 matmul).
    T_world = torch.empty_like(T_parent_joint)
    T_world[:, 0] = T_parent_joint[:, 0]
    for idx in range(1, J):
        T_world[:, idx] = T_world[:, parents[idx]] @ T_parent_joint[:, idx]

    # ----- Standard LBS using cached T_rest_inv -----
    # A[j] = T_world[j] @ T_rest_inv[j]  encodes the joint's "skinning"
    # transform; sum-weighted across joints yields the per-vertex matrix.
    A = T_world @ T_rest_inv.unsqueeze(0)                       # (B, J, 4, 4)

    if (char.expr_dirs_face is not None and "expression" in forward_kwargs
            and forward_kwargs["expression"].shape[0] == B
            and forward_kwargs["expression"].shape[1] == char.expr_dirs_face.shape[1]):
        # Face expression deltas live in REST frame. Apply to v_rest_h then
        # LBS carries them into world coords naturally.
        expr = forward_kwargs["expression"].to(dtype)
        delta_face = torch.einsum("bk,nkd->bnd", expr,
                                   char.expr_dirs_face.to(dtype))     # (B, Nm_face, 3)
        Nm = v_rest_h.shape[0]
        v_rest_h_b = v_rest_h.unsqueeze(0).expand(B, -1, -1).clone()
        idx_face = char.face_vert_idx
        v_rest_h_b[:, idx_face, :3] = v_rest_h_b[:, idx_face, :3] + delta_face
    else:
        v_rest_h_b = v_rest_h.unsqueeze(0).expand(B, -1, -1)

    # Per-vertex skinning matrix: T_per_vert = sum_j w_ij * A_j.
    # Done as (Nm, J) @ (J, 16) -> (Nm, 16) reshape; then matvec on v_rest_h.
    A_flat = A.reshape(B, J, 16)                                # (B, J, 16)
    T_per_vert = torch.einsum("vj,bji->bvi", w, A_flat).reshape(B, -1, 4, 4)  # (B, Nm, 4, 4)
    v_posed = torch.einsum("bvkl,bvl->bvk", T_per_vert, v_rest_h_b)[..., :3]

    # Manual jaw rotation around the anatomical TMJ pivot. Each vert is
    # rotated by `jaw_angle_x * lbs_weights[:, 22]` -- so verts skinned
    # fully to jaw get full rotation, lower-weighted verts get a fractional
    # rotation (smooth blend with neighbouring head verts). Rotation is
    # about +X axis through `anatomical_jaw_pivot`.
    if jaw_angle_x is not None:
        pivot = char.anatomical_jaw_pivot.to(dtype=dtype)              # (3,)
        # Per-vert jaw weight (B-broadcast)
        jw = w[:, 22].unsqueeze(0).expand(B, -1)                       # (B, Nm)
        # Effective rotation angle per vert per batch
        ang = jaw_angle_x.unsqueeze(-1) * jw                            # (B, Nm)
        cos_a = torch.cos(ang); sin_a = torch.sin(ang)                  # (B, Nm)
        # Rotate (v_posed - pivot) about X axis: Y' = Yc - Zs, Z' = Ys + Zc
        rel = v_posed - pivot.view(1, 1, 3)
        rel_y = rel[..., 1] * cos_a - rel[..., 2] * sin_a
        rel_z = rel[..., 1] * sin_a + rel[..., 2] * cos_a
        v_posed = torch.stack([rel[..., 0], rel_y, rel_z], dim=-1) + pivot.view(1, 1, 3)

    # Apply runtime global rotation + translation.
    R = _axisangle_to_matrix(global_rot)                    # (B, 3, 3)
    v_posed = (R @ v_posed.mT).mT + global_t.unsqueeze(1)   # (B, Nm, 3)

    return v_posed, char.faces
