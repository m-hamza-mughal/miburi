"""Reduce the streamed character mesh without touching the motion.

The viewer ships every vertex position on every frame. For SMPL-X that is
10475 verts x 3 x float32 = ~128 kB per frame, or ~3.2 MB/s at 25 fps -- more
than an ordinary connection sustains. Worse, Viser's message buffer coalesces
updates to the same mesh (newest wins), so a client that cannot keep up does
not merely lag: frames are dropped and the animation visibly stutters.

This module precomputes, once, a *subset* of the original vertices plus a
retriangulated face list. At runtime the per-frame cost is a single gather,
``verts[keep_idx]`` -- the motion, the codecs and the streaming loop are
untouched. Only the surface gets coarser.

Method: voxel clustering. Vertices are bucketed on a grid and each bucket
elects the member closest to its centroid. Electing a real vertex (rather
than averaging to a new point) is what makes the result indexable, so posed
frames need no geometry work at all.

Density is not uniform. SMPL-X packs vertices densely around the face, which
is exactly where expression lives, so the head gets a finer grid than the
body -- gestures read fine on a coarser torso, a smeared face does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..client_utils import log

# SMPL-X joints that drive the face: head, jaw, and the two eyes.
FACE_JOINTS = (15, 22, 23, 24)


def face_vertex_mask(smplx_model, threshold: float = 0.5) -> np.ndarray:
    """Boolean (V,) mask of vertices skinned mostly to the head/jaw/eyes.

    Uses the model's own LBS weights, so it needs no hand-labelled region
    list and stays correct if the template changes.
    """
    w = smplx_model.lbs_weights.detach().cpu().numpy()  # (V, J)
    idx = [j for j in FACE_JOINTS if j < w.shape[1]]
    return w[:, idx].sum(axis=1) > threshold


def _cluster(vertices: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elect one representative vertex per occupied grid cell.

    Returns ``(reps, cell_of_vertex)`` where ``reps[c]`` is the original
    vertex index representing cell ``c``, and ``cell_of_vertex[v]`` is the
    cell of vertex ``v``. Both are needed: collapsing a triangle means
    sending each of its corners to *its own cell's* representative.

    Resolving a vertex to the geometrically nearest representative instead
    is wrong and looks it -- on a thin surface it sends front-face vertices
    to back-face representatives and bridges the gap between a limb and the
    torso, which shreds the mesh into fold-over triangles and holes.

    `cell` is per-vertex, so different regions can use different grid sizes
    in a single pass.
    """
    keys = np.floor(vertices / cell[:, None]).astype(np.int64)
    # Fold the 3 integer coords into one sortable key.
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]

    # `inverse` maps every vertex to its cell, which is exactly the remap the
    # face rebuild needs once `reps` is indexed by cell.
    _, inverse = np.unique(flat, return_inverse=True)
    inverse = inverse.astype(np.int64)
    n_cells = int(inverse.max()) + 1

    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    groups = np.split(order, starts[1:])

    reps = np.empty(n_cells, dtype=np.int64)
    for g in groups:
        c = inverse[g[0]]
        if len(g) == 1:
            reps[c] = g[0]
        else:
            pts = vertices[g]
            reps[c] = g[np.argmin(((pts - pts.mean(axis=0)) ** 2).sum(axis=1))]
    return reps, inverse


def build_decimation_table(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
    target_verts: int = 3500,
    face_detail: float = 3.0,
) -> dict[str, np.ndarray]:
    """Solve for the grid size that lands nearest `target_verts`.

    Args:
        vertices: (V, 3) template vertices, rest pose.
        faces:    (F, 3) template triangles.
        face_mask: (V,) bool; these vertices use a `face_detail`x finer grid.
        target_verts: desired vertex count after decimation.
        face_detail: how much finer the facial grid is than the body grid.

    Returns dict with `keep_idx` (M,), `faces` (F', 3), and the achieved counts.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if face_mask is None:
        face_mask = np.zeros(len(vertices), dtype=bool)

    extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    lo, hi = extent / 4000.0, extent / 4.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        cell = np.where(face_mask, mid / face_detail, mid)
        reps, inverse = _cluster(vertices, cell)
        if best is None or abs(len(reps) - target_verts) < abs(len(best[0]) - target_verts):
            best = (reps, inverse)
        if len(reps) > target_verts:
            lo = mid          # cells too small -> too many verts
        else:
            hi = mid
        if abs(len(reps) - target_verts) <= max(8, target_verts // 200):
            best = (reps, inverse)
            break
    keep_idx, remap = best

    # `remap` already sends each vertex to its own cell, and `keep_idx` is
    # indexed by cell, so the face rebuild is a straight gather.
    new_faces = remap[faces]
    # Drop triangles that collapsed to a line or a point.
    ok = (new_faces[:, 0] != new_faces[:, 1]) & \
         (new_faces[:, 1] != new_faces[:, 2]) & \
         (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[ok]
    # Drop duplicates (two originals can collapse onto the same triangle).
    # Sort only to build the dedupe KEY -- assigning the sorted array back
    # would reorder each triangle's corners and destroy its winding, leaving
    # roughly half the mesh facing inward. Viser renders side="front", so
    # those become invisible and the character appears shot through with
    # black speckles over an otherwise correct silhouette.
    key = np.sort(new_faces, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    new_faces = new_faces[np.sort(first)]

    return {
        "keep_idx": keep_idx.astype(np.int64),
        "faces": new_faces.astype(np.uint32),
        "n_verts_in": np.int64(len(vertices)),
        "n_faces_in": np.int64(len(faces)),
    }


def load_or_build(
    cache_path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
    target_verts: int = 3500,
    face_detail: float = 3.0,
) -> dict[str, np.ndarray]:
    """Cached `build_decimation_table`. The table depends only on the
    template, so it is computed once and reused for the life of the install."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            d = np.load(cache_path)
            if int(d["n_verts_in"]) == len(vertices) and len(d["keep_idx"]) > 0:
                log("info", f"decimate: table loaded from {cache_path} "
                            f"({len(d['keep_idx'])} verts, {len(d['faces'])} faces)")
                return {k: d[k] for k in d.files}
        except Exception as exc:
            log("warning", f"decimate: cache unreadable ({exc!r}); rebuilding")

    table = build_decimation_table(
        vertices, faces, face_mask=face_mask,
        target_verts=target_verts, face_detail=face_detail,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **table)
    except Exception as exc:
        log("warning", f"decimate: could not cache table ({exc!r})")
    log("info", f"decimate: {len(vertices)} -> {len(table['keep_idx'])} verts, "
                f"{len(faces)} -> {len(table['faces'])} faces "
                f"(target {target_verts}, face detail {face_detail}x)")
    return table
