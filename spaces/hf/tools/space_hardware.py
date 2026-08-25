#!/usr/bin/env python3
"""Turn the MIBURI Space's GPU on and off, and cap what it can cost.

The Space rests on `cpu-basic` (free, serves the landing page) and is flipped
to a dedicated GPU only for demo windows. That makes cost a function of hours
you deliberately run -- but "deliberately" is doing a lot of work in that
sentence, so `guard` exists to make the ceiling a number instead of a promise.

    # before a demo
    python space_hardware.py up                 # -> L40S, sleeps after 15 min idle
    python space_hardware.py status

    # after
    python space_hardware.py down               # -> cpu-basic, billing stops

    # belt and braces: cron this every 5 minutes on any always-on machine
    */5 * * * * python space_hardware.py guard --budget-usd 10 --once

Requires a token with **write** access to the Space: `HF_TOKEN` in the
environment, or a prior `hf auth login`.

Billing note: a Space is billed while `Starting` *or* `Running`, and is free
while `Sleeping` or `Paused`. `down` (cpu-basic) and `pause` both stop billing;
`down` keeps the landing page up, `pause` takes the URL down entirely.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = os.environ.get("MIBURI_SPACE_ID", "m-hamza-mughal/miburi")
LEDGER = Path(os.environ.get("MIBURI_LEDGER", Path.home() / ".miburi-space-ledger.json"))

# https://huggingface.co/docs/hub/spaces-gpus -- USD/hour.
PRICE_PER_HOUR = {
    "cpu-basic": 0.00,
    "cpu-upgrade": 0.03,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "l40sx1": 1.80,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "a100-large": 2.50,
}

# Stages during which the Space is on the clock. Note that a plain BUILDING
# is NOT billed -- the docs are explicit that there is no cost during build,
# and nothing is running yet. RUNNING_BUILDING is different: the previous
# revision is still up while the new one builds. Counting plain BUILDING
# would let a long, free build trip the budget cap.
BILLED_STAGES = {"RUNNING", "RUNNING_BUILDING", "RUNNING_APP_STARTING", "APP_STARTING"}

DEFAULT_HARDWARE = "l40sx1"
DEFAULT_SLEEP = 900  # 15 min idle -> sleep (free) without a full teardown


def api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    return HfApi(token=token) if token else HfApi()


def _fmt_runtime(rt) -> str:
    hw = getattr(rt, "hardware", None)
    req = getattr(rt, "requested_hardware", None)
    rate = PRICE_PER_HOUR.get(str(hw), None)
    billed = str(rt.stage) in BILLED_STAGES and (rate or 0) > 0
    lines = [
        f"  stage              {rt.stage}",
        f"  hardware           {hw}" + (f"  (${rate:.2f}/hr)" if rate else ""),
        f"  requested          {req}",
        f"  sleep_time         {getattr(rt, 'sleep_time', None)}",
        f"  billing            {'ON THE CLOCK' if billed else 'not billed'}",
    ]
    return "\n".join(lines)


def cmd_status(args) -> int:
    rt = api().get_space_runtime(repo_id=args.space)
    print(f"{args.space}")
    print(_fmt_runtime(rt))
    led = _load_ledger()
    month = _month_key()
    spent = led.get(month, {}).get("usd", 0.0)
    print(f"  ledger {month}     ${spent:.2f} accrued (local estimate)")
    return 0


def cmd_up(args) -> int:
    a = api()
    print(f"{args.space}: requesting {args.hardware} (sleep after {args.sleep}s idle)")
    a.request_space_hardware(repo_id=args.space, hardware=args.hardware, sleep_time=args.sleep)
    rate = PRICE_PER_HOUR.get(args.hardware, 0.0)
    print(f"  billing starts now: ${rate:.2f}/hr while Starting or Running.")
    print(f"  remember: `space_hardware.py down` when the demo window ends.")
    if not args.no_wait:
        _wait_for(a, args.space, {"RUNNING"}, timeout=args.timeout)
    return 0


def cmd_down(args) -> int:
    a = api()
    print(f"{args.space}: downgrading to cpu-basic (landing page stays up, billing stops)")
    a.request_space_hardware(repo_id=args.space, hardware="cpu-basic")
    return 0


def cmd_pause(args) -> int:
    print(f"{args.space}: pausing (URL goes down; only the owner can restart)")
    api().pause_space(repo_id=args.space)
    return 0


def cmd_restart(args) -> int:
    print(f"{args.space}: restarting")
    api().restart_space(repo_id=args.space)
    return 0


def _wait_for(a: HfApi, space: str, stages: set[str], timeout: int = 1800) -> None:
    print(f"  waiting for {'/'.join(sorted(stages))} (timeout {timeout}s)...")
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        rt = a.get_space_runtime(repo_id=space)
        stage = str(rt.stage)
        if stage != last:
            print(f"    [{int(time.time()-t0):>4}s] {stage}  hw={rt.hardware}")
            last = stage
        if stage in stages:
            print(f"  ready after {int(time.time()-t0)}s")
            return
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            print(f"  FAILED in stage {stage} -- check `hf spaces logs {space} --build`")
            return
        time.sleep(10)
    print("  timed out waiting")


# --------------------------------------------------------------------------
# budget guard
# --------------------------------------------------------------------------

def _month_key() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")


def _load_ledger() -> dict:
    if LEDGER.is_file():
        try:
            return json.loads(LEDGER.read_text())
        except Exception:
            pass
    return {}


def _save_ledger(led: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))


def _tick(a: HfApi, space: str, interval: int, budget: float, dry_run: bool) -> bool:
    """One poll. Returns True if the Space was downgraded."""
    rt = a.get_space_runtime(repo_id=space)
    stage, hw = str(rt.stage), str(rt.hardware)
    # HF reports hardware.current as None while a hardware change is in
    # flight, but the GPU can already be attached and billing during that
    # window -- observed 2026-08-24, where the ledger read $0.00 through
    # several minutes of real A100 time. Fall back to the requested tier so
    # the guard errs toward over-counting rather than under-counting; a cost
    # cap that reads low is worse than useless.
    # Only substitute when the platform reports NOTHING. An explicit
    # "cpu-basic" is a real answer and means free -- treating it as unknown
    # billed a GPU rate for genuinely free cpu-basic time and tripped this
    # cap during a build, cancelling a test window that had cost nothing.
    if hw in ("None", "") and rt.requested_hardware:
        hw = str(rt.requested_hardware)
    rate = PRICE_PER_HOUR.get(hw, 0.0)
    on_clock = stage in BILLED_STAGES and rate > 0

    led = _load_ledger()
    month = _month_key()
    entry = led.setdefault(month, {"usd": 0.0, "seconds": 0})
    if on_clock:
        # Attribute the whole interval to this tick. Over-counts a little at
        # the edges, which is the safe direction for a cost cap.
        entry["seconds"] += interval
        entry["usd"] += rate * interval / 3600.0
    _save_ledger(led)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] stage={stage:<20} hw={hw:<12} "
          f"{'BILLED' if on_clock else 'free':<7} "
          f"month=${entry['usd']:.2f}/{budget:.2f}", flush=True)

    if entry["usd"] >= budget and hw != "cpu-basic":
        print(f"  !! monthly budget ${budget:.2f} reached -> forcing cpu-basic", flush=True)
        if dry_run:
            print("     (dry run: no change made)", flush=True)
        else:
            a.request_space_hardware(repo_id=space, hardware="cpu-basic")
        return True
    return False


def cmd_guard(args) -> int:
    a = api()
    print(f"guard: {args.space}, budget ${args.budget_usd:.2f}/month, "
          f"ledger {LEDGER}, interval {args.interval}s"
          f"{' (dry run)' if args.dry_run else ''}")
    if args.once:
        _tick(a, args.space, args.interval, args.budget_usd, args.dry_run)
        return 0
    while True:
        try:
            _tick(a, args.space, args.interval, args.budget_usd, args.dry_run)
        except Exception as exc:  # network blip: keep the loop alive
            print(f"  guard tick failed: {exc!r}", flush=True)
        time.sleep(args.interval)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default=SPACE_ID)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="print stage, hardware, billing state")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("up", help="request the demo GPU")
    s.add_argument("--hardware", default=DEFAULT_HARDWARE, choices=sorted(PRICE_PER_HOUR))
    s.add_argument("--sleep", type=int, default=DEFAULT_SLEEP,
                   help="seconds of inactivity before the Space sleeps (sleeping is free)")
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--timeout", type=int, default=1800)
    s.set_defaults(func=cmd_up)

    s = sub.add_parser("down", help="back to cpu-basic; landing page stays up, billing stops")
    s.set_defaults(func=cmd_down)

    s = sub.add_parser("pause", help="pause the Space entirely (URL goes down)")
    s.set_defaults(func=cmd_pause)

    s = sub.add_parser("restart", help="restart the Space")
    s.set_defaults(func=cmd_restart)

    s = sub.add_parser("guard", help="poll and force cpu-basic once a monthly budget is hit")
    s.add_argument("--budget-usd", type=float, default=10.0)
    s.add_argument("--interval", type=int, default=300)
    s.add_argument("--once", action="store_true", help="single poll, for cron")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_guard)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
