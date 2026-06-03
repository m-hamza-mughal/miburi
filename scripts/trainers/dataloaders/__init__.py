"""Convenience re-exports of dataset classes.

Imports are tolerant: a dataset module that hasn't been ported into this
release tree (or that fails to import due to a missing dep) only takes that
one symbol out of the registry. The HDF5 builders in
``scripts.trainers.dataloaders.{beatx,embody3d_dyadic,seamlessinteraction}``
do not need any of these classes and remain importable in slim envs.
"""

_DATASET_IMPORTS = (
    (".beatx.dataset", ("BEATXDataset",)),
    (".seamlessinteraction.dataset", ("SEAMLESSINTERACTIONDataset", "SEAMLESSINTERACTIONBEATXDataset")),
    (".embody3d_dyadic.dataset", ("EMBODY3DBEATXDataset",)),
    (".unified_dataset", ("UNIFIEDDataset",)),
)

__all__: list[str] = []
for _module_name, _names in _DATASET_IMPORTS:
    try:
        _mod = __import__(__name__ + _module_name, fromlist=list(_names))
        for _name in _names:
            globals()[_name] = getattr(_mod, _name)
            __all__.append(_name)
    except Exception:
        pass
