"""Chamfer-distance fit of SMPL-X (beta + global R, t) to a target mesh.

This is a minimal vendored version of the body-only loop from
bermand/SMPL-X-Fitting. We only fit:
    - global rotation R (axis-angle, 3)
    - global translation t (3)
    - shape parameters beta (NUM_BETAS, optimized but heavily regularized)
    - per-axis scale s (3) -- absorbs unit-of-measure mismatches between the
      Mixamo character and SMPL-X. Mixamo characters often ship slightly
      taller/shorter than SMPL-X's mean human; without scale, beta has to
      stretch unnaturally and the chamfer floor is much higher.

We deliberately do NOT optimize pose theta -- the Mixamo bind pose is T-pose
(or A-pose) and SMPL-X v_template is T-pose. For A-pose characters we add a
fixed shoulder rotation prior, optimized as theta_arms.

Optimization is Adam over a pure-torch chamfer (no pytorch3d dependency):
    chamfer(A, B) = mean_a min_b ||a - b||^2 + mean_b min_a ||a - b||^2

For a target mesh with ~17k verts and SMPL-X's 10475 verts, the all-pairs
distance matrix is ~700MB at float32 -- too big. We use a tile + topk(1)
loop in chunks of 2048 verts to keep peak memory under ~500MB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ChamferFitResult:
    betas: np.ndarray              # (NUM_BETAS,) float32
    body_pose: np.ndarray          # (63,) float32 -- arm rotations baked in
    global_rotation: np.ndarray    # (3,) float32 axis-angle
    global_translation: np.ndarray # (3,) float32
    scale: np.ndarray              # (3,) float32 per-axis scale on target
    final_chamfer: float           # mean of two-sided chamfer in meters^2
    smplx_verts_fit: np.ndarray    # (V_smplx, 3) SMPL-X verts at the fitted params
    smplx_joints_fit: np.ndarray   # (J, 3)


def _chamfer_one_side(
    a: torch.Tensor, b: torch.Tensor, tile: int = 2048,
    return_idx: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """For each row in `a`, find squared distance to nearest row in `b`.

    If `return_idx`, also returns the index of the nearest `b` row for each
    `a` row. The index is needed when the b->a side wants to look up a
    per-source weight under the matched a.
    """
    na = a.shape[0]
    out_d = torch.empty(na, device=a.device, dtype=a.dtype)
    out_i = torch.empty(na, device=a.device, dtype=torch.long) if return_idx else None
    b_sq = (b * b).sum(dim=-1)  # (Nb,)
    for start in range(0, na, tile):
        end = min(start + tile, na)
        chunk = a[start:end]                                  # (T, 3)
        chunk_sq = (chunk * chunk).sum(dim=-1, keepdim=True)  # (T, 1)
        cross = chunk @ b.T                                   # (T, Nb)
        d2 = (chunk_sq + b_sq.unsqueeze(0) - 2.0 * cross).clamp_min(0.0)
        if return_idx:
            d, i = d2.min(dim=-1)
            out_d[start:end] = d
            out_i[start:end] = i
        else:
            out_d[start:end] = d2.min(dim=-1).values
    if return_idx:
        return out_d, out_i
    return out_d


def chamfer(a: torch.Tensor, b: torch.Tensor, tile: int = 2048) -> torch.Tensor:
    """Symmetric chamfer distance (unweighted). mean(a->b) + mean(b->a)."""
    return _chamfer_one_side(a, b, tile).mean() + _chamfer_one_side(b, a, tile).mean()


def chamfer_weighted(
    a: torch.Tensor, b: torch.Tensor,
    w_a: torch.Tensor,
    tile: int = 2048,
) -> torch.Tensor:
    """Region-weighted symmetric chamfer.

    `w_a` is a per-source-vertex weight of shape (a.shape[0],). The a->b
    side weights each a-vert's nearest-neighbour distance by `w_a[i]`. The
    b->a side weights each b-vert's nearest-neighbour distance by the
    matched-a's weight (`w_a[nn_idx_b]`), so the same region weighting
    applies on both directions without needing target-side rigging.
    """
    d_a = _chamfer_one_side(a, b, tile=tile, return_idx=False)
    d_b, idx_b = _chamfer_one_side(b, a, tile=tile, return_idx=True)
    w_total_a = w_a.sum().clamp_min(1e-8)
    loss_ab = (d_a * w_a).sum() / w_total_a
    w_b = w_a[idx_b]
    loss_ba = (d_b * w_b).sum() / w_b.sum().clamp_min(1e-8)
    return loss_ab + loss_ba


def _axisangle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues. aa: (..., 3) -> R: (..., 3, 3)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1] = -z; K[..., 0, 2] = y
    K[..., 1, 0] = z;  K[..., 1, 2] = -x
    K[..., 2, 0] = -y; K[..., 2, 1] = x
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1.0 - c) * (K @ K)


def _arm_pose_apose_init() -> torch.Tensor:
    """Initial body_pose biased toward Mixamo's A-pose: shoulders rotated
    ~50 deg downward from T-pose. SMPL-X body_pose is 21 joints x 3 = 63 dof
    (excludes pelvis which is the global rotation). Joint indices follow
    SMPL-X order. We rotate joint 15 (left shoulder ~clavicle) and 16
    (right shoulder) toward the body sides.
    """
    pose = torch.zeros(21, 3)
    # SMPL-X body joints (1..21): pelvis is joint 0 (=global), then in body_pose:
    #   0  -> 'left_hip'        index 1
    #   1  -> 'right_hip'       2
    #   ...
    #   15 -> 'left_shoulder'   index 16
    #   16 -> 'right_shoulder'  17
    # Rotation about Z brings the arms down from T-pose.
    pose[15, 2] = -1.0  # left shoulder down ~57 deg
    pose[16, 2] = 1.0   # right shoulder down ~57 deg
    return pose.flatten()


def fit_smplx_to_mesh(
    smplx_model,
    target_verts: np.ndarray,
    *,
    device: str | torch.device = "cuda",
    n_iters_align: int = 100,
    n_iters_joint: int = 800,
    lr_align: float = 5e-2,
    lr_joint: float = 1e-2,
    beta_reg: float = 5e-3,
    arm_reg: float = 1e-2,
    scale_reg: float = 10.0,
    t_reg: float = 5.0,
    n_active_betas: int = 16,
    chamfer_tile: int = 2048,
    apose_init: bool = True,
    optimize_body_pose: bool = False,
    hand_loss_weight: float = 4.0,
    verbose: bool = True,
    log_every: int = 50,
) -> ChamferFitResult:
    """Fit SMPL-X beta + global R, t to a target mesh by chamfer optimization.

    The optimization proceeds in two phases:
        A. Coarse align (R, t, scale) with beta=0 frozen.
        B. Joint optimize (beta, R, t, scale, arm-pose).

    Args:
        smplx_model: SMPL-X model (e.g. built via
            miburi.utils.viser_scene.make_smplx_model), already placed on
            `device`. We use its forward to differentiably produce SMPL-X
            vertices.
        target_verts: (N_t, 3) numpy array of the target mesh vertices in a
            Y-up frame.
        device: torch device for optimization.
        n_iters_align: iterations for phase A (global align only).
        n_iters_joint: iterations for phase B (beta + global + arms).
        lr_align / lr_joint: learning rates per phase.
        beta_reg / arm_reg: L2 regularizers.
        chamfer_tile: rows per tile in the chamfer NN loop.
        apose_init: initialize body_pose with A-pose shoulders. Mixamo
            characters ship in A-pose, not T-pose.
        optimize_body_pose: if True, body_pose joins phase-B optimization (gives
            a tighter fit when Mixamo and SMPL-X rest poses differ, e.g. A-pose
            vs T-pose). When False (default), body_pose is frozen at the init
            -- this is the right setting when downstream code expects
            `verts_tpose` to live in SMPL-X's rest frame for runtime LBS.
        hand_loss_weight: extra weight on hand-region verts in the chamfer
            loss. Hand region is defined as SMPL-X verts with dominant LBS
            weight on wrist + finger joints (20, 21, 25..54). With the
            default 4.0, hand verts contribute 5x as much per-vert as body
            verts. Set to 0 to disable region weighting.
        verbose: print loss every `log_every` steps.

    Returns:
        ChamferFitResult with all fitted parameters and the final SMPL-X
        vertices/joints.
    """
    device = torch.device(device)
    # Seed for reproducibility. Without this the chamfer Adam path is
    # non-deterministic (CUDA reductions in the chunked chamfer kernel),
    # which means the same character can fit cleanly on one run and drift
    # to a 30 cm RMS local minimum on another. Fixed seed -> same fit
    # every run, so a known-good β + t result can be reproduced.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    smplx_model = smplx_model.to(device).eval()
    for p in smplx_model.parameters():
        p.requires_grad_(False)

    target = torch.as_tensor(target_verts, dtype=torch.float32, device=device)

    n_betas = smplx_model.NUM_BETAS
    n_expr = smplx_model.NUM_EXPR_COEFFS

    # Per-SMPL-X-vertex weight for region-weighted chamfer. Hand region =
    # verts whose dominant LBS weight is on wrists + finger joints.
    if hand_loss_weight > 0:
        hand_joint_idx = [20, 21] + list(range(25, 55))
        hand_mass = smplx_model.lbs_weights[:, hand_joint_idx].sum(dim=-1)  # (V,)
        # Continuous weight: body verts get 1.0, hand verts get (1 + h * mass).
        vert_weight = 1.0 + hand_loss_weight * hand_mass.clamp(0.0, 1.0)
        if verbose:
            n_hand = (hand_mass > 0.5).sum().item()
            print(f"[chamfer] region-weighted: hand verts={n_hand}/{vert_weight.numel()}, "
                  f"weight ratio body:hand = 1.0 : {1.0 + hand_loss_weight:.2f}")
    else:
        vert_weight = None

    # Only optimize the FIRST n_active_betas principal shape components.
    # High-order components capture training-data noise and the optimizer
    # abuses them to warp SMPL-X into degenerate shapes (e.g. compressing
    # Z so the body becomes a flat pancake from above). n_active_betas=16
    # keeps the meaningful shape modes (height, build, weight, ratios).
    n_active_betas = min(n_active_betas, n_betas)
    betas_active = nn.Parameter(torch.zeros(1, n_active_betas, device=device))
    betas_frozen = torch.zeros(1, n_betas - n_active_betas, device=device)
    expression = torch.zeros(1, n_expr, device=device)
    body_pose = nn.Parameter(
        (_arm_pose_apose_init() if apose_init else torch.zeros(63)).to(device).unsqueeze(0)
    )
    hand_pose = torch.zeros(1, 90, device=device)
    head_pose = torch.zeros(1, 9, device=device)

    global_rot = nn.Parameter(torch.zeros(1, 3, device=device))
    global_t = nn.Parameter(torch.zeros(1, 3, device=device))
    # SCALAR log_scale: a single isotropic scale instead of per-axis. Per-
    # axis was producing degenerate fits where Y compresses (squishes
    # character vertically) and Z stretches (thickens). Scalar scaling can
    # still absorb FBX unit-of-measure mismatches (e.g. Remy at 2x) but
    # prevents anisotropic distortion. exp(log_scale) is broadcast to all
    # three axes.
    log_scale = nn.Parameter(torch.zeros(1, device=device))

    # ---- Smart initialization (no gradient steps): align centroids + extents.
    with torch.no_grad():
        target_centroid = target.mean(dim=0)
        smplx_centroid = smplx_model.v_template.mean(dim=0).to(device)
        global_t.copy_((target_centroid - smplx_centroid).unsqueeze(0))

        # Match the median per-axis extent ratio (robust to one wild axis).
        target_extent = (target.max(dim=0).values - target.min(dim=0).values)
        smplx_extent = (smplx_model.v_template.max(dim=0).values
                        - smplx_model.v_template.min(dim=0).values).to(device)
        ratio_per_axis = (smplx_extent / target_extent.clamp_min(1e-6)).clamp_min(1e-6)
        scalar_ratio = ratio_per_axis.median()
        log_scale.copy_(torch.log(scalar_ratio).unsqueeze(0))
        if verbose:
            print(f"[chamfer] init: global_t={global_t[0].cpu().numpy().round(3)}  "
                  f"init_scale={scalar_ratio.item():.3f} (scalar)  "
                  f"per_axis_ratio={ratio_per_axis.cpu().numpy().round(3)}")
    log_scale_init = log_scale.detach().clone()

    def _smplx_verts():
        # Recompute the full betas tensor each call so betas_active's
        # gradients propagate correctly through the optimizer step.
        full_betas = torch.cat([betas_active, betas_frozen], dim=1)
        out = smplx_model(
            shape=full_betas,
            expression=expression,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            global_rotation=global_rot,
            global_translation=global_t,
        )
        return out["vertices"][0], out["joints"][0]

    def _target_scaled():
        # Scale the target around its centroid; scalar log_scale broadcast to
        # all 3 axes (isotropic scaling).
        centroid = target.mean(dim=0, keepdim=True)
        return centroid + (target - centroid) * torch.exp(log_scale)  # broadcasts (1,) -> (1, 3)

    # ----- phase 0: beta landmark pre-fit -----
    # Optimize beta to match TARGET extent + centroid (after current scale +
    # R + t). Cheap (6 scalars per step) and gives beta a shape-aware prior
    # before the per-vertex chamfer kicks in. Tuned conservatively: lr=5e-2
    # for 50 iters with a 10x stronger beta_reg, so beta lands at L2~5-10
    # instead of 45+. The aggressive original (lr=2e-1, 150 iters, reg=1e-4)
    # pushed beta to extreme values that joint-B then drifted t_fit upward
    # by 0.2-0.3 m to compensate, producing the 30 cm RMS misalignment.
    if n_iters_align > 0:
        opt_p0 = torch.optim.Adam([betas_active], lr=5e-2)
        for step in range(50):
            opt_p0.zero_grad()
            smplx_v, _ = _smplx_verts()
            tgt = _target_scaled()
            sx_extent = smplx_v.max(dim=0).values - smplx_v.min(dim=0).values
            tg_extent = tgt.max(dim=0).values - tgt.min(dim=0).values
            sx_c = smplx_v.mean(dim=0)
            tg_c = tgt.mean(dim=0)
            l0 = ((sx_extent - tg_extent) ** 2).sum() + ((sx_c - tg_c) ** 2).sum()
            l0 = l0 + 1e-3 * (betas_active ** 2).mean()
            l0.backward()
            opt_p0.step()
            if verbose and (step % 10 == 0 or step == 49):
                print(f"[phase 0] step {step:3d}  shape_loss={l0.item():.5f}  "
                      f"beta_l2={betas_active.norm().item():.3f}  "
                      f"sx_extent={sx_extent.detach().cpu().numpy().round(3)}  "
                      f"tg_extent={tg_extent.detach().cpu().numpy().round(3)}")

    # ----- phase A: coarse align (R, t, log_scale only) -----
    opt_a = torch.optim.Adam([global_rot, global_t, log_scale], lr=lr_align)
    for step in range(n_iters_align):
        opt_a.zero_grad()
        smplx_v, _ = _smplx_verts()
        if vert_weight is not None:
            loss = chamfer_weighted(smplx_v, _target_scaled(), vert_weight, tile=chamfer_tile)
        else:
            loss = chamfer(smplx_v, _target_scaled(), tile=chamfer_tile)
        loss = loss + scale_reg * ((log_scale - log_scale_init) ** 2).mean()
        loss.backward()
        opt_a.step()
        if verbose and (step % log_every == 0 or step == n_iters_align - 1):
            print(f"[align A] step {step:4d}  chamfer={loss.item():.5f}  "
                  f"scale={torch.exp(log_scale).detach().cpu().numpy()}")

    # ----- phase B: joint optimize (beta + R + t + scale [+ arms]) -----
    params_b = [betas_active, global_rot, global_t, log_scale]
    if optimize_body_pose:
        params_b.append(body_pose)
    opt_b = torch.optim.Adam(params_b, lr=lr_joint)
    for step in range(n_iters_joint):
        opt_b.zero_grad()
        smplx_v, _ = _smplx_verts()
        if vert_weight is not None:
            chamf = chamfer_weighted(smplx_v, _target_scaled(), vert_weight, tile=chamfer_tile)
        else:
            chamf = chamfer(smplx_v, _target_scaled(), tile=chamfer_tile)
        reg = beta_reg * (betas_active ** 2).mean()
        # Pull log_scale toward 0 (= scale 1.0). Without this the optimizer
        # finds degenerate solutions where scale and beta fight each other
        # (e.g. scale=0.17 + extreme beta both shrink SMPL-X down).
        reg = reg + scale_reg * ((log_scale - log_scale_init) ** 2).mean()
        if optimize_body_pose:
            reg = reg + arm_reg * (body_pose ** 2).mean()
        loss = chamf + reg
        loss.backward()
        opt_b.step()
        if verbose and (step % log_every == 0 or step == n_iters_joint - 1):
            print(f"[joint B] step {step:4d}  chamfer={chamf.item():.5f}  "
                  f"reg={reg.item():.5f}  beta_l2={betas_active.norm().item():.3f}")

    with torch.no_grad():
        smplx_v, smplx_j = _smplx_verts()
        # Report the unweighted chamfer in the result, since downstream code
        # interprets it as a uniform geometric error.
        final = chamfer(smplx_v, _target_scaled(), tile=chamfer_tile).item()

    return ChamferFitResult(
        betas=torch.cat([betas_active, betas_frozen], dim=1).detach().cpu().numpy()[0].astype(np.float32),
        body_pose=body_pose.detach().cpu().numpy()[0].astype(np.float32),
        global_rotation=global_rot.detach().cpu().numpy()[0].astype(np.float32),
        global_translation=global_t.detach().cpu().numpy()[0].astype(np.float32),
        scale=(torch.exp(log_scale).detach().cpu().numpy().astype(np.float32)
               * np.ones(3, dtype=np.float32)),
        final_chamfer=float(final),
        smplx_verts_fit=smplx_v.detach().cpu().numpy().astype(np.float32),
        smplx_joints_fit=smplx_j.detach().cpu().numpy().astype(np.float32),
    )
