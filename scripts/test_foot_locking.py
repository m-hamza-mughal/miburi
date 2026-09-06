#!/usr/bin/env python3
"""Checks for the online foot locker (`miburi.utils.foot_locking`).

    python scripts/test_foot_locking.py

The port's reference is the Rust solver in MPI's `body-models-postprocess`.
That is not vendored here, so the parity check is opt-in: build the crate
(`cargo build --release` in a checkout; `[lib] name = "_rust"`) and point
MIBURI_FOOT_LOCK_ORACLE at the directory holding the resulting `_rust.so`.
Without it the remaining checks still run.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from miburi.utils import foot_locking as fl  # noqa: E402
from miburi.utils.viser_scene import make_smplx_model  # noqa: E402

SMPLX_DIR = os.environ.get("MIBURI_SMPLX_DIR", "assets_dep/smplx_2020/")
FPS = 25.0
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def _random_quats(rng, n, spread=0.08):
    aa = rng.normal(0, spread, size=(n, 3)).astype(np.float32)
    ang = np.linalg.norm(aa, axis=1, keepdims=True)
    axis = np.where(ang > 1e-8, aa / np.maximum(ang, 1e-8), np.array([0.0, 0.0, 1.0]))
    return np.concatenate([axis * np.sin(ang / 2), np.cos(ang / 2)], axis=1).astype(np.float32)


def main() -> int:
    model = make_smplx_model(SMPLX_DIR, gender="NEUTRAL_2020")
    full = fl.build_foot_rig(model, world_up_axis="y")
    reduced = fl.reduce_to_legs(full)

    print("\nrig")
    check("both feet form toe-ankle-knee-hip chains", True,
          f"legs {[(l.ankle, l.toe) for l in full.legs]}")
    check("reduction keeps only the leg chains", len(reduced.parents) == 9,
          f"{len(full.parents)} -> {len(reduced.parents)} joints")
    check("rest foot heights are plausible",
          all(0.0 < l.toe_height < l.ankle_height < 0.20 for l in full.legs),
          f"toe {full.legs[0].toe_height:.3f} m, ankle {full.legs[0].ankle_height:.3f} m")

    print("\nreduced rig matches the full rig")
    rng = np.random.default_rng(3)
    lf, lr = fl.OnlineFootLocker(full, fps=FPS), fl.OnlineFootLocker(reduced, fps=FPS)
    worst = 0.0
    for f in range(60):
        q = _random_quats(rng, len(full.parents), 0.10)
        root = np.array([0.002 * f, 0.95, 0.0], dtype=np.float32)
        c = np.array([1.0, 1.0], dtype=np.float32)
        a = lf.step(q.copy(), root, c)[reduced.chain]
        b = lr.step(q[reduced.chain].copy(), root, c)
        worst = max(worst, float(np.minimum(np.abs(a - b).max(1), np.abs(a + b).max(1)).max()))
    check("bit-identical", worst == 0.0, f"max quaternion difference {worst:.1e}")

    print("\nlocking removes slide from a planted foot")
    ident = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(reduced.parents), 1))
    for root_y in (0.0, 1.3):
        locker = fl.OnlineFootLocker(reduced, fps=FPS)
        det = fl.CausalFootContacts(fps=FPS, up=reduced.up,
                                    toe_rest_height=reduced.legs[0].toe_height)
        raw, locked, prev_r, prev_l = [], [], None, None
        for f in range(100):
            root = np.array([0.0016 * f, root_y, 0.0], dtype=np.float32)
            toes = fl.toe_positions(reduced, ident, root)
            contacts = det.step(toes)
            out = locker.step(ident.copy(), root, contacts, ground=det.floor)
            lt = fl.toe_positions(reduced, out, root)
            if prev_r is not None:
                raw.append(np.linalg.norm(toes - prev_r, axis=1).max())
                locked.append(np.linalg.norm(lt - prev_l, axis=1).max())
            prev_r, prev_l = toes, lt
        cut = 1.0 - np.mean(locked) / np.mean(raw)
        check(f"root y={root_y}: slide removed", cut > 0.75, f"{cut * 100:.1f}%")

    print("\ncontact hysteresis suppresses flicker")
    for factor, label in ((1.0, "without"), (2.0, "with")):
        det = fl.CausalFootContacts(fps=FPS, up=reduced.up, release_factor=factor,
                                    ground_height=0.0)
        rng2 = np.random.default_rng(11)
        flips, prev = 0, None
        for _ in range(300):
            jitter = rng2.normal(0.15, 0.03)   # speed hovering at the threshold
            toes = np.array([[0.0, 0.02, 0.0], [0.0, 0.02, 0.0]], dtype=np.float32)
            det._previous = toes - np.array([jitter / FPS, 0.0, 0.0], dtype=np.float32)
            c = det.step(toes)
            if prev is not None and (c[0] > 0.5) != (prev > 0.5):
                flips += 1
            prev = c[0]
        print(f"        {label} hysteresis: {flips} contact flips in 300 frames")
        if factor == 2.0:
            check("hysteresis holds contact steady", flips <= 2, f"{flips} flips")

    print("\nreference parity (opt-in)")
    oracle_dir = os.environ.get("MIBURI_FOOT_LOCK_ORACLE")
    if not oracle_dir:
        print("  SKIP  set MIBURI_FOOT_LOCK_ORACLE to a directory containing _rust.so")
    else:
        sys.path.insert(0, oracle_dir)
        import _rust  # type: ignore
        ref = _rust.OnlineFootLocker(
            full.parents, full.offsets,
            np.array([l.ankle for l in full.legs], dtype=np.int64),
            np.array([l.toe for l in full.legs], dtype=np.int64),
            np.array([l.ankle_height for l in full.legs], dtype=np.float32),
            np.array([l.toe_height for l in full.legs], dtype=np.float32),
            np.stack([l.toe_end for l in full.legs]).astype(np.float32),
            np.array([l.toe_end_height for l in full.legs], dtype=np.float32),
            full.up, np.float32(1 / FPS), np.float32(0.75), np.float32(0.25),
            np.float32(0.01), np.float32(0.5), np.float32(fl.IK_SOFTENING))
        mine = fl.OnlineFootLocker(full, fps=FPS)
        rng3 = np.random.default_rng(0)
        worst_pos = 0.0
        for f in range(120):
            q = _random_quats(rng3, len(full.parents))
            root = np.array([0.004 * f, 1.0 + 0.01 * np.sin(f / 7), 0.002 * f], dtype=np.float32)
            c = np.array([1.0 if (f // 15) % 2 == 0 else 0.0,
                          0.0 if (f // 15) % 2 == 0 else 1.0], dtype=np.float32)
            got = mine.step(q.copy(), root, c)   # ground 0, which the reference assumes
            exp = np.asarray(ref.lock_frame(np.ascontiguousarray(q.copy()),
                                            np.ascontiguousarray(root),
                                            np.ascontiguousarray(c)), dtype=np.float32)
            _, pg = fl._forward(full.parents, got, full.offsets, root)
            _, pe = fl._forward(full.parents, exp, full.offsets, root)
            worst_pos = max(worst_pos, float(np.abs(pg - pe).max()))
        check("matches the Rust reference", worst_pos < 1e-5,
              f"max joint position difference {worst_pos:.2e} m")

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
