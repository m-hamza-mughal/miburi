"""Transfer SMPL-X per-vertex LBS weights to a Mixamo character mesh.

Strategy: for each Mixamo vertex, find the closest point on the *fitted*
SMPL-X surface (after applying the chamfer-fit beta, R, t). The closest point
lives on some SMPL-X triangle with three vertex indices and barycentric
coordinates. We interpolate the SMPL-X vertex LBS weights at those bary
coords. The result is a per-Mixamo-vertex LBS weight that speaks the same
joint vocabulary as SMPL-X.

Uses trimesh.proximity.closest_point (R-tree under the hood -- needs the
`rtree` package installed).
"""

from __future__ import annotations

import numpy as np
import trimesh


def transfer_lbs_weights(
    target_verts: np.ndarray,        # (Nm, 3) Mixamo verts, in same frame as smplx_verts
    smplx_verts: np.ndarray,         # (V_smplx, 3) fitted SMPL-X verts
    smplx_faces: np.ndarray,         # (F_smplx, 3) SMPL-X triangle vertex indices
    smplx_lbs_weights: np.ndarray,   # (V_smplx, J) per-vertex LBS weights
) -> np.ndarray:
    """Returns Mixamo-vertex LBS weights of shape (Nm, J), each row summing to 1."""
    assert target_verts.ndim == 2 and target_verts.shape[1] == 3
    assert smplx_verts.shape[0] == smplx_lbs_weights.shape[0], \
        "smplx_verts and smplx_lbs_weights must have matching vertex counts"

    smplx_mesh = trimesh.Trimesh(
        vertices=smplx_verts.astype(np.float64),
        faces=smplx_faces.astype(np.int64),
        process=False,
    )

    closest, _dist, tri_id = trimesh.proximity.closest_point(
        smplx_mesh, target_verts.astype(np.float64)
    )
    tri_id = np.asarray(tri_id, dtype=np.int64)
    closest = np.asarray(closest, dtype=np.float64)

    tri_v_idx = smplx_faces[tri_id]                       # (Nm, 3)
    tri_verts = smplx_verts[tri_v_idx].astype(np.float64) # (Nm, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(
        triangles=tri_verts, points=closest,
    ).astype(np.float32)                                  # (Nm, 3)

    w_tri = smplx_lbs_weights[tri_v_idx]                  # (Nm, 3, J)
    weights = (w_tri * bary[:, :, None]).sum(axis=1)       # (Nm, J)

    s = weights.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return (weights / s).astype(np.float32)
