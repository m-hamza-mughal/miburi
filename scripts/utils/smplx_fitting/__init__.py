"""Offline tooling for fitting SMPL-X to a Mixamo character mesh."""

from .fbx_to_mesh import load_fbx_mesh, MixamoFBXMesh

__all__ = [
    "MixamoFBXMesh",
    "load_fbx_mesh",
]

# Optional re-exports (lazy — submodules import torch/smplx which we want to
# avoid pulling in for lightweight FBX-only callers).
try:
    from .chamfer_fit import fit_smplx_to_mesh, ChamferFitResult  # noqa: F401
    __all__ += ["fit_smplx_to_mesh", "ChamferFitResult"]
except ImportError:
    pass

try:
    from .transfer_lbs_weights import transfer_lbs_weights  # noqa: F401
    __all__ += ["transfer_lbs_weights"]
except ImportError:
    pass
