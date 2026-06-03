"""Eager trainer registration for ``scripts/train.py``'s
``getattr(trainers, args.trainer + 'Trainer')`` dispatch.

Each trainer is imported in isolation so a missing dep (or a trainer file
that hasn't been ported in this release tree) only takes that one trainer
out of the registry, not the entire ``scripts.trainers`` package.

Import failures are logged via stdlib so they're visible without depending on
loguru being importable (HDF5 build scripts import this package in slim envs).
"""

import traceback as _traceback

_TRAINER_IMPORTS = (
    (".upperbodycausalcodec_trainer", "UpperBodyCausalCodecTrainer"),
    (".lowertranscausalcodec_trainer", "LowerBodyCausalCodecTrainer"),
    (".faceexpcausalcodec_trainer", "FaceExpCausalCodecTrainer"),
    (".uflgtdm3_trainer", "UpperFaceLowerGTDM3Trainer"),
)

for _module_name, _cls_name in _TRAINER_IMPORTS:
    try:
        _mod = __import__(__name__ + _module_name, fromlist=[_cls_name])
        globals()[_cls_name] = getattr(_mod, _cls_name)
    except Exception as _exc:
        # Surface the actual reason this trainer didn't register. Most often a
        # missing dep, a typo introduced during refactoring, or a module-level
        # raise inside the trainer. Without this log, the caller would only see
        # `AttributeError: module 'trainers' has no attribute 'XxxTrainer'`
        # from the downstream getattr dispatch — which is unhelpful.
        print(
            f"[trainers] Failed to register {_cls_name} from {_module_name}: "
            f"{type(_exc).__name__}: {_exc}"
        )
        _traceback.print_exc()
