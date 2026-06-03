"""Transfer SMPL-X expression blendshapes (FLAME) onto a Mixamo face mesh.

SMPL-X exprdirs has shape (V, 3, 100); the "face region" is implicitly the
set of verts whose exprdirs has nonzero magnitude. The naive transfer
(closest SMPL-X triangle, barycentric over its 3 verts) tends to mix mouth
and nose contributions for Mixamo verts whose closest triangle straddles
both regions -- visible as Remy's nose deforming when SMPL-X's mouth opens.

This version uses K-nearest-neighbour distance-weighted blending restricted
to the HIGH-EXPRESSION-MAGNITUDE region of SMPL-X (the actual moving facial
features: mouth, eyes, jaw, nose tip). The semantic locality is much
better because:
    1. The candidate pool excludes the back-of-head / scalp verts that
       carry little expression but were within the "face mask" before.
    2. K-NN with inverse-distance weighting lets close anchors dominate;
       a Remy nose vert mostly picks SMPL-X nose anchors even if a few
       mouth anchors are within range.
"""

from __future__ import annotations

import numpy as np


def detect_face_landmarks_geometric(
    verts: np.ndarray,
    anchor_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Detect 6 anatomically-invariant face landmarks from raw geometry:
       nose_tip   - forward-most vert in central X band, upper face
       chin_tip   - lowest vert in central X band
       head_top   - highest vert in central X band (wider)
       lear / rear - max / min X verts in mid-face Y band
       mouth_ctr  - face vert closest to midpoint of nose + chin
    Inputs are raw vert positions; the detector works in the verts' own
    frame so it can be applied to both SMPL-X face anchors and Mixamo
    face verts. Use this to bootstrap landmark correspondences for TPS
    or to derive an anatomically-grounded lip-line Y for jaw skinning."""
    v = verts
    if anchor_mask is not None:
        v = v[anchor_mask]
    if v.shape[0] < 10:
        return {}
    face_y_lo = v[:, 1].min()
    face_y_hi = v[:, 1].max()
    face_h = face_y_hi - face_y_lo
    cx_mask = np.abs(v[:, 0]) < (face_h * 0.20)
    upper_mask = v[:, 1] > face_y_lo + 0.4 * face_h
    nose_cand = cx_mask & upper_mask
    if nose_cand.sum() < 1:
        nose_cand = cx_mask
    if nose_cand.sum() < 1:
        return {}
    nose = v[nose_cand][np.argmax(v[nose_cand, 2])]
    chin = (v[cx_mask][np.argmin(v[cx_mask, 1])]
            if cx_mask.sum() else v[np.argmin(v[:, 1])])
    ht_mask = np.abs(v[:, 0]) < (face_h * 0.35)
    head_top = (v[ht_mask][np.argmax(v[ht_mask, 1])]
                if ht_mask.sum() else v[np.argmax(v[:, 1])])
    mid_y_mask = (v[:, 1] > face_y_lo + 0.2 * face_h) & (v[:, 1] < face_y_hi - 0.2 * face_h)
    if mid_y_mask.sum() < 2:
        mid_y_mask = np.ones(v.shape[0], dtype=bool)
    lear = v[mid_y_mask][np.argmax(v[mid_y_mask, 0])]
    rear = v[mid_y_mask][np.argmin(v[mid_y_mask, 0])]
    mouth_target = 0.5 * (nose + chin)
    d2 = ((v - mouth_target[None, :]) ** 2).sum(axis=-1)
    d2[np.abs(v[:, 0]) > face_h * 0.15] = np.inf
    mouth_v = v[np.argmin(d2)]
    return {
        "nose": nose.astype(np.float32),
        "chin": chin.astype(np.float32),
        "head_top": head_top.astype(np.float32),
        "lear": lear.astype(np.float32),
        "rear": rear.astype(np.float32),
        "mouth": mouth_v.astype(np.float32),
    }


def detect_mixamo_lip_line(
    smplx_anchor_verts: np.ndarray,
    smplx_lip_landmarks: np.ndarray,   # (Nlip, 3) FLAME lip landmark world positions
    target_face_verts: np.ndarray,
) -> dict:
    """Find Mixamo's lip line by TPS-warping SMPL-X's FLAME lip landmarks
    (which sit on the actual upper-lip / lower-lip ring of SMPL-X) into
    Mixamo space using a 6-landmark bootstrap warp. Then snap each warped
    lip landmark to its nearest Mixamo vert -- those are the Mixamo lip
    verts. Returns the lip-line Y (mean Y of Mixamo lip verts), the lip
    verts themselves, and the bootstrap landmark dicts for QA logging."""
    sm_geom = detect_face_landmarks_geometric(smplx_anchor_verts)
    mx_geom = detect_face_landmarks_geometric(target_face_verts)
    common = sorted(set(sm_geom.keys()) & set(mx_geom.keys()))
    if len(common) < 4 or smplx_lip_landmarks.shape[0] < 3:
        return {}
    sm_lmk = np.stack([sm_geom[k] for k in common], axis=0).astype(np.float32)
    mx_lmk = np.stack([mx_geom[k] for k in common], axis=0).astype(np.float32)
    # Warp SMPL-X lip landmarks (which are on the actual lip ring) into
    # Mixamo space using the 6 anatomical anchors.
    lip_warped = _tps_3d_warp(sm_lmk, mx_lmk, smplx_lip_landmarks.astype(np.float32))
    # Snap each warped lip landmark to the nearest Mixamo face vert.
    d2 = ((lip_warped[:, None, :] - target_face_verts[None, :, :]) ** 2).sum(axis=-1)
    nearest = np.argmin(d2, axis=-1)
    lip_verts = target_face_verts[nearest]                       # (Nlip, 3)
    lip_line_y = float(lip_verts[:, 1].mean())
    lip_line_z = float(lip_verts[:, 2].mean())
    return {
        "lip_line_y": lip_line_y,
        "lip_line_z": lip_line_z,
        "lip_verts": lip_verts,
        "smplx_geom_lmks": sm_geom,
        "mixamo_geom_lmks": mx_geom,
        "common_landmarks": common,
    }


def _tps_3d_warp(
    src: np.ndarray,       # (N, 3) source landmarks (SMPL-X side)
    dst: np.ndarray,       # (N, 3) corresponding target landmarks (Mixamo side)
    query: np.ndarray,     # (M, 3) points in SMPL-X frame to warp to Mixamo frame
    smoothing: float = 1e-4,
) -> np.ndarray:
    """3D Thin-Plate-Spline warp. Returns query points moved so that each
    src landmark lands at its paired dst landmark, smoothly interpolating
    in between. Solves an (N+4)x(N+4) linear system per call so N should
    stay under a few hundred -- fine for FLAME's 51 landmarks.

    Radial basis: K(r) = r (3D biharmonic), the natural TPS analogue in
    three dimensions. The affine portion (1, x, y, z) absorbs global
    translation + linear scaling + shear; the RBF part adds local
    nonlinear corrections so e.g. a longer Mixamo chin doesn't drag the
    nose with it.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    n = src.shape[0]
    # Build K = pairwise distance matrix
    dij = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    K = dij + smoothing * np.eye(n)
    P = np.concatenate([np.ones((n, 1)), src], axis=-1)        # (N, 4)
    L = np.zeros((n + 4, n + 4))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.zeros((n + 4, 3))
    Y[:n] = dst
    sol, *_ = np.linalg.lstsq(L, Y, rcond=None)
    alpha = sol[:n]                                             # (N, 3)
    beta = sol[n:]                                              # (4, 3)
    # Evaluate at query points
    diq = np.linalg.norm(query[:, None, :] - src[None, :, :], axis=-1)  # (M, N)
    Pq = np.concatenate([np.ones((query.shape[0], 1)), query], axis=-1) # (M, 4)
    warped = diq @ alpha + Pq @ beta
    return warped.astype(np.float32)


def _knn_chunked(target: np.ndarray, source: np.ndarray, k: int,
                 tile: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each row in target, find the K nearest rows in source. Returns
    (idx (Nt, K) int64, dist (Nt, K) float32, sorted ascending)."""
    nt = target.shape[0]
    src_sq = (source ** 2).sum(axis=-1).astype(np.float32)
    out_i = np.empty((nt, k), dtype=np.int64)
    out_d = np.empty((nt, k), dtype=np.float32)
    for s in range(0, nt, tile):
        e = min(s + tile, nt)
        chk = target[s:e].astype(np.float32)
        chk_sq = (chk ** 2).sum(axis=-1, keepdims=True)
        d2 = chk_sq + src_sq[None, :] - 2.0 * (chk @ source.T)
        d2 = np.maximum(d2, 0.0)
        # Top-K smallest, then sort within those K.
        cand = np.argpartition(d2, k, axis=-1)[:, :k]
        cand_d = np.take_along_axis(d2, cand, axis=-1)
        order = np.argsort(cand_d, axis=-1)
        out_i[s:e] = np.take_along_axis(cand, order, axis=-1)
        out_d[s:e] = np.sqrt(np.take_along_axis(cand_d, order, axis=-1))
    return out_i, out_d


def transfer_face_blendshapes(
    target_verts: np.ndarray,        # (Nm, 3) Mixamo verts in same frame as smplx_verts
    smplx_verts: np.ndarray,         # (V_smplx, 3)
    smplx_faces: np.ndarray,         # (F_smplx, 3) -- unused now, kept for API parity
    smplx_exprdirs: np.ndarray,      # (V_smplx, 3, 100)
    mass_threshold_frac: float = 0.20,
    k: int = 6,
    max_dist_m: float = 0.04,
    mouth_center: np.ndarray | None = None,  # (3,) world position of jaw/mouth center
    mouth_radius_m: float = 0.08,    # Gaussian radius for mouth-region falloff
    expr_dir_scale: float = 1.0,     # extra multiplier on transferred displacements
    landmark_world_pos: dict[str, np.ndarray] | None = None,
    flame_landmark_positions: np.ndarray | None = None,  # (51, 3) FLAME landmarks
    eps: float = 1e-4,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Returns (face_vert_mask (Nm,) bool, expr_dirs (N_face, 100, 3) float32).

    Args:
        mass_threshold_frac: fraction of the maximum per-vert expr magnitude
            used as the cutoff for "high-expression" SMPL-X anchors. 0.20
            keeps anchors at >= 20% of the maximum, which roughly selects
            the mouth + eye + nose-tip region while excluding scalp/back-
            of-head verts whose expr contribution is near zero.
        k: K nearest SMPL-X anchors per Mixamo vert.
        max_dist_m: a Mixamo vert is considered "face" only if its nearest
            anchor is within this distance.
    """
    del smplx_faces  # not used anymore -- vert-based not surface-based
    assert smplx_exprdirs.shape[0] == smplx_verts.shape[0]
    n_expr = smplx_exprdirs.shape[-1]

    # ---- 1. High-expression SMPL-X anchors only ----
    expr_mass = np.linalg.norm(
        smplx_exprdirs.reshape(smplx_exprdirs.shape[0], -1), axis=1
    )
    threshold = mass_threshold_frac * expr_mass.max()
    anchor_mask = expr_mass > threshold
    anchor_verts = smplx_verts[anchor_mask].astype(np.float32)
    anchor_exprdirs = smplx_exprdirs[anchor_mask].astype(np.float32)  # (Na, 3, 100)
    if verbose:
        print(f"[face_transfer] SMPL-X anchors: {anchor_mask.sum()}/{anchor_mask.size} "
              f"(top {(1-mass_threshold_frac)*100:.0f}% by expr magnitude)")

    # ---- 1b. Face-bbox normalization for semantic K-NN ----
    # Even after chamfer fit, the Mixamo face and SMPL-X face don't sit at
    # identical anatomical positions: Mixamo characters often have a larger
    # head, a longer/shorter chin, or the lip line at a different Y than
    # SMPL-X. K-NN in raw world coords then matches Mixamo lip verts to
    # SMPL-X upper-lip/philtrum/nose verts, and FLAME's "lip drop"
    # displacement gets applied to the wrong anatomy.
    #
    # Fix: compute the face bbox per character, normalize each to [0,1]^3
    # in its own face bbox, and K-NN in that normalized space. Chin-of-
    # Mixamo (Y_min in its face bbox) maps to chin-of-SMPL-X.
    sm_min = anchor_verts.min(axis=0)
    sm_max = anchor_verts.max(axis=0)
    sm_extent = np.maximum(sm_max - sm_min, 1e-6)
    # Identify Mixamo verts inside SMPL-X face bbox (small slack) and use
    # them to compute the Mixamo face bbox. This excludes scalp / neck /
    # body verts that would otherwise stretch the bbox.
    target_verts_np = target_verts.astype(np.float32)
    slack = 0.03
    within = ((target_verts_np >= sm_min - slack)
              & (target_verts_np <= sm_max + slack)).all(axis=1)
    if within.sum() > 100:
        mx_min = target_verts_np[within].min(axis=0)
        mx_max = target_verts_np[within].max(axis=0)
    else:
        mx_min, mx_max = sm_min.copy(), sm_max.copy()
    mx_extent = np.maximum(mx_max - mx_min, 1e-6)
    if verbose:
        print(f"[face_transfer] SMPL-X face bbox extent (cm): "
              f"{(sm_extent*100).round(1)}")
        print(f"[face_transfer] Mixamo face bbox extent (cm): "
              f"{(mx_extent*100).round(1)} (from {within.sum()} candidate verts)")
    anchor_normed = (anchor_verts - sm_min) / sm_extent
    target_normed = (target_verts_np - mx_min) / mx_extent

    # ---- 1c. Landmark-anchored TPS warp ----
    # The bbox-normalized K-NN alone can't compensate for non-uniform
    # proportion differences (e.g. SMPL-X has 20 FLAME mouth landmarks
    # tightly clustered; remy's mouth is in a different fraction of its
    # face bbox -> the mouth Voronoi cell pulls in remy's nose/chin verts
    # too). The TPS warp uses the known SMPL-X FLAME + ear landmarks AS
    # CONTROL POINTS: we bootstrap their Mixamo correspondences by finding
    # the nearest Mixamo vert (in bbox-normalized space) for each
    # SMPL-X landmark, then warp the SMPL-X anchor verts to align with
    # those Mixamo positions. After warp, K-NN in raw coordinates finds
    # anatomically aligned matches.
    use_tps = flame_landmark_positions is not None and within.sum() > 100
    anchor_warped = anchor_verts.copy()
    if use_tps:
        # Anchor pairs are detected GEOMETRICALLY on each mesh so the
        # correspondence does not depend on bbox-fraction K-NN (which
        # breaks when SMPL-X and Mixamo have different lip-to-chin
        # proportions inside the same bbox -- the previous bootstrap
        # would match SMPL-X lip landmarks to Mixamo chin verts because
        # Mixamo's longer chin shifted the bbox-relative lip position).
        #
        # 6 invariant geometric landmarks (see detect_face_landmarks_geometric).
        # SMPL-X side: detect on anchors (FLAME face region only).
        smplx_lmk = detect_face_landmarks_geometric(anchor_verts)
        # Mixamo side: detect on verts inside the (sm_bbox + slack) so
        # only face-region verts are considered.
        mx_face_pool = target_verts_np[within] if within.sum() > 100 else target_verts_np
        mixamo_lmk = detect_face_landmarks_geometric(mx_face_pool)
        # Prefer the vertex_ids landmarks for ears (more accurate than
        # mid-face X extreme) if provided.
        if landmark_world_pos is not None:
            for name in ("lear", "rear"):
                if name in landmark_world_pos:
                    smplx_lmk[name] = np.asarray(
                        landmark_world_pos[name], dtype=np.float32
                    )
        common = sorted(set(smplx_lmk.keys()) & set(mixamo_lmk.keys()))
        if len(common) >= 4:
            sm_lmk = np.stack([smplx_lmk[k] for k in common], axis=0)
            mx_lmk = np.stack([mixamo_lmk[k] for k in common], axis=0)
            anchor_warped = _tps_3d_warp(sm_lmk, mx_lmk, anchor_verts)
            if verbose:
                mean_shift_m = float(np.linalg.norm(
                    anchor_warped - anchor_verts, axis=-1
                ).mean())
                print(f"[face_transfer] TPS warp: geometric landmarks "
                      f"{common}, mean anchor shift = {mean_shift_m*100:.1f}cm")
                for name in common:
                    print(f"  {name}: sm={smplx_lmk[name].round(3)} "
                          f"-> mx={mixamo_lmk[name].round(3)}")
        else:
            use_tps = False
            if verbose:
                print(f"[face_transfer] TPS skipped: only {len(common)} "
                      f"matching landmarks (need >= 4)")

    # ---- 2. K-NN in warped raw space (or bbox-normalized fallback) ----
    if use_tps:
        idx, _ = _knn_chunked(target_verts_np, anchor_warped, k=k)
    else:
        idx, _ = _knn_chunked(target_normed, anchor_normed, k=k)
    # World-space distance from each target vert to its nearest matched
    # anchor — used for face_vert_mask cutoff and inverse-distance weights.
    # We use the WARPED anchor positions here so the distance reflects
    # the post-TPS geometric proximity (which is what we just used for K-NN).
    anchors_for_dist = anchor_warped if use_tps else anchor_verts
    nearest_anchor_world = anchors_for_dist[idx]             # (Nm, k, 3)
    dist = np.linalg.norm(
        target_verts_np[:, None, :] - nearest_anchor_world, axis=-1
    ).astype(np.float32)
    # ---- 3. Face mask: nearest anchor within max_dist ----
    face_vert_mask = dist[:, 0] < max_dist_m
    if verbose:
        print(f"[face_transfer] Mixamo face verts: {face_vert_mask.sum()}/{face_vert_mask.size} "
              f"(within {max_dist_m*100:.0f}cm of an anchor)")

    # ---- 4. Inverse-distance weighted blend of expr_dirs ----
    inv = 1.0 / (dist + eps)
    inv = inv / inv.sum(axis=1, keepdims=True)         # (Nm, k)
    # Gather anchor exprdirs: (Nm, k, 3, 100)
    e_neighbors = anchor_exprdirs[idx]                  # (Nm, k, 3, 100)
    # Weighted sum
    expr_dirs_target = (e_neighbors * inv[..., None, None]).sum(axis=1)  # (Nm, 3, 100)

    # ---- 4b. Scale displacement magnitudes by face-bbox extent ratio ----
    # FLAME expr_dirs were authored on the SMPL-X face geometry (~15 cm tall).
    # A taller Mixamo face (e.g. Ch08 at ~22 cm) needs proportionally larger
    # displacements to produce the same fraction-of-face motion. We scale
    # per-axis by mx_extent / sm_extent so a "1 cm mouth open" on SMPL-X
    # becomes a 1.5 cm mouth open on the taller face.
    axis_scale = (mx_extent / sm_extent).astype(np.float32)
    # Clamp to a reasonable range so very-different-bbox characters don't
    # blow up the motion.
    axis_scale = np.clip(axis_scale, 0.5, 2.5)
    expr_dirs_target = expr_dirs_target * axis_scale[None, :, None]
    # Extra per-character multiplier (set via CHARACTER_DEFAULTS or CLI).
    if abs(expr_dir_scale - 1.0) > 1e-6:
        expr_dirs_target = expr_dirs_target * float(expr_dir_scale)
    if verbose:
        print(f"[face_transfer] expr_dir axis scale (mx/sm): {axis_scale.round(2)} "
              f"x extra={expr_dir_scale:.2f}")

    # ---- 5. Tighten face_vert_mask to mouth/lip/cheek/jaw region only ----
    # Speech motion should be concentrated below the nose. Including nose,
    # brow, and scalp in the face_vert_mask led to weird FLAME-driven
    # motion on the wrong regions (nose twitch, scalp wobble). Restrict
    # face_vert_mask to verts BELOW (mouth.Y + 3cm) -- that captures lips,
    # chin, lower cheeks, and side jaw, while excluding nose tip / nose
    # ridge / brow / eyes / scalp.
    if mouth_center is not None:
        y_below_upper_lip = target_verts[:, 1] < (mouth_center[1] + 0.03)
        face_vert_mask = face_vert_mask & y_below_upper_lip
        if verbose:
            print(f"[face_transfer] tightened to mouth/jaw region: "
                  f"{face_vert_mask.sum()} verts (Y < mouth.Y+3cm)")

    # Only keep face-region verts; transpose to (N_face, 100, 3) for runtime einsum.
    expr_dirs_face = expr_dirs_target[face_vert_mask].transpose(0, 2, 1).astype(np.float32)

    # ---- 6. Per-vert region labels for diagnostic visualization ----
    # Each face vert is labeled by which anatomical region of the SMPL-X
    # face it K-NN-matched to. Uses landmark Voronoi (nose / leye / reye /
    # lear / rear) plus a synthesized mouth_center -> chin axis. This is
    # what gets visualized to verify the K-NN mapping is semantically
    # correct: same colour on both panels = same body part.
    smplx_region_per_anchor = None
    mixamo_region_per_vert = None
    # FLAME 51-landmark grouping into anatomical regions. Multiple
    # landmarks per region give a much more balanced Voronoi than a few
    # single-vertex landmarks (where the nose tip dominates).
    region_names = ["brow", "nose", "leye", "reye", "mouth", "lear", "rear", "chin"]
    FLAME_GROUPS = {
        "brow":  list(range(0, 10)),     # FLAME 17-26 (eyebrows: 10 lmks)
        "nose":  list(range(10, 19)),    # FLAME 27-35 (nose: 9 lmks)
        "leye":  list(range(19, 25)),    # FLAME 36-41 (left eye: 6 lmks)
        "reye":  list(range(25, 31)),    # FLAME 42-47 (right eye: 6 lmks)
        "mouth": list(range(31, 51)),    # FLAME 48-67 (mouth: 20 lmks)
    }
    if flame_landmark_positions is not None:
        # Combine FLAME 51 landmarks (brow / nose / eyes / mouth) with
        # the 5 single-vertex landmarks from vertex_ids (ears) so the
        # ear/cheek/side-of-face regions are not absorbed by the mouth
        # Voronoi cell (which has 20 landmarks and otherwise dominates).
        # Per-landmark region label: for each of the 51 + extras, which
        # anatomical region it belongs to.
        extra_landmarks = []        # list of (pos, region_name)
        if landmark_world_pos is not None:
            for name in ("lear", "rear"):
                if name in landmark_world_pos:
                    extra_landmarks.append(
                        (np.asarray(landmark_world_pos[name], dtype=np.float32), name)
                    )
        n_flame = flame_landmark_positions.shape[0]
        all_lmk = np.concatenate(
            [flame_landmark_positions]
            + [p[None, :] for p, _ in extra_landmarks], axis=0,
        ) if extra_landmarks else flame_landmark_positions.copy()
        per_lmk_region = np.full(all_lmk.shape[0], -1, dtype=np.int32)
        for region_idx, name in enumerate(region_names):
            if name in FLAME_GROUPS:
                for li in FLAME_GROUPS[name]:
                    if 0 <= li < n_flame:
                        per_lmk_region[li] = region_idx
        # Tag the extra landmarks (ears).
        for i, (_, name) in enumerate(extra_landmarks):
            if name in region_names:
                per_lmk_region[n_flame + i] = region_names.index(name)
        # Voronoi (3D distance) from each anchor to nearest landmark.
        d2_anchor = ((anchor_verts[:, None, :] - all_lmk[None, :, :])
                     ** 2).sum(axis=-1)
        nearest_lmk = np.argmin(d2_anchor, axis=-1)
        smplx_region_per_anchor = per_lmk_region[nearest_lmk]
        # Below-mouth anchors -> chin (cleans up the lower face region).
        if mouth_center is not None:
            is_chin = anchor_verts[:, 1] < (mouth_center[1] - 0.015)
            smplx_region_per_anchor[is_chin] = region_names.index("chin")
    elif landmark_world_pos is not None:
        # Fallback to 5-landmark scheme.
        centers = []
        for name in region_names:
            if name == "mouth" and mouth_center is not None:
                centers.append(np.asarray(mouth_center, dtype=np.float32))
            elif name == "chin" and mouth_center is not None:
                centers.append(np.asarray(mouth_center, dtype=np.float32)
                               + np.array([0, -0.05, 0], dtype=np.float32))
            elif name in landmark_world_pos:
                centers.append(np.asarray(
                    landmark_world_pos[name], dtype=np.float32
                ))
            else:
                centers.append(None)
        valid = [(i, c) for i, c in enumerate(centers) if c is not None]
        if valid:
            valid_idx = np.array([i for i, _ in valid], dtype=np.int32)
            valid_centers = np.stack([c for _, c in valid], axis=0)
            anchor_xy = anchor_verts[:, :2]
            center_xy = valid_centers[:, :2]
            d2_anchor = ((anchor_xy[:, None, :] - center_xy[None, :, :])
                         ** 2).sum(axis=-1)
            nearest = np.argmin(d2_anchor, axis=-1)
            smplx_region_per_anchor = valid_idx[nearest].astype(np.int32)

    if smplx_region_per_anchor is not None:
        # Propagate to mixamo via existing K-NN: take majority label
        # among k neighbors.
        neighbor_labels = smplx_region_per_anchor[idx]           # (Nm, k)
        n_regions = len(region_names)
        counts = np.zeros((target_verts_np.shape[0], n_regions), dtype=np.int32)
        for ki in range(k):
            valid_labels = neighbor_labels[:, ki]
            ok = valid_labels >= 0
            counts[np.where(ok)[0], valid_labels[ok]] += 1
        mixamo_region_per_vert = np.argmax(counts, axis=-1).astype(np.int32)
        # Verts with no label assigned anywhere -> -1 sentinel
        no_label = counts.sum(axis=-1) == 0
        mixamo_region_per_vert[no_label] = -1
        if verbose:
            from collections import Counter
            a_dist = Counter(int(x) for x in smplx_region_per_anchor if x >= 0)
            m_dist = Counter(int(x) for x in mixamo_region_per_vert[face_vert_mask] if x >= 0)
            print(f"[face_transfer] region labels (SMPL-X anchors): "
                  f"{ {region_names[k]: v for k, v in a_dist.items()} }")
            print(f"[face_transfer] region labels (Mixamo face verts): "
                  f"{ {region_names[k]: v for k, v in m_dist.items()} }")

    return face_vert_mask, expr_dirs_face, smplx_region_per_anchor, mixamo_region_per_vert
