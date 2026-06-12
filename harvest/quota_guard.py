"""Quota guard: protect the Max session window from the bulk labeler.

Polls `claude -p /usage` every POLL_S seconds. If session usage >= CAP_PCT,
kills the labeler (resumable by design), waits until the reported reset time
plus a buffer, relaunches it, and keeps guarding until the run completes.

Usage: nohup .venv/bin/python harvest/quota_guard.py > data/real_train/guard.log 2>&1 &
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/real_train/label_run2.log"
CAP_PCT = 80
POLL_S = 300
BUFFER_S = 240

RELAUNCH = ("nohup .venv/bin/python harvest/pseudo_label.py --limit 3000 --workers 3 "
            f">> {LOG} 2>&1 &")


def usage() -> tuple[int | None, str]:
    out = subprocess.run(["claude", "-p", "/usage", "--output-format", "text"],
                         capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"Current session:\s*(\d+)% used.*?resets ([^(\n]+)", out)
    return (int(m.group(1)), m.group(2).strip()) if m else (None, "")


def seconds_until_reset(reset_str: str) -> int:
    # e.g. "Jun 12 at 8:40pm"
    m = re.search(r"([A-Za-z]{3}) (\d+) at (\d+):(\d+)(am|pm)", reset_str)
    if not m:
        return 3600
    now = dt.datetime.now()
    month = dt.datetime.strptime(m.group(1), "%b").month
    hour = int(m.group(3)) % 12 + (12 if m.group(5) == "pm" else 0)
    target = now.replace(month=month, day=int(m.group(2)), hour=hour,
                         minute=int(m.group(4)), second=0)
    if target < now:
        target += dt.timedelta(days=1)
    return max(60, int((target - now).total_seconds()))


def labeler_pids() -> list[str]:
    out = subprocess.run(["pgrep", "-f", "pseudo_label.py"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


def log(msg: str) -> None:
    print(f"[guard {dt.datetime.now():%H:%M}] {msg}", flush=True)


def main() -> None:
    log(f"guarding: cap {CAP_PCT}%, poll {POLL_S}s")
    while True:
        if "labels ->" in (LOG.read_text()[-2000:] if LOG.exists() else ""):
            log("labeling run complete; guard exiting")
            return
        pct, reset = usage()
        pids = labeler_pids()
        log(f"session {pct}% · labeler pids {pids or 'NONE'}")
        if pct is None:
            time.sleep(POLL_S)
            continue
        if pct >= CAP_PCT and pids:
            log(f"cap hit ({pct}% >= {CAP_PCT}%): stopping labeler, resuming after reset ({reset})")
            subprocess.run(["pkill", "-f", "pseudo_label.py"])
            wait = seconds_until_reset(reset) + BUFFER_S
            log(f"sleeping {wait//60} min until window resets")
            time.sleep(wait)
            subprocess.run(["bash", "-c", RELAUNCH], cwd=ROOT)
            log("labeler relaunched")
        elif not pids and pct < CAP_PCT:
            # labeler died (crash or prior kill) while quota is fine: relaunch
            log("labeler not running and quota OK: relaunching")
            subprocess.run(["bash", "-c", RELAUNCH], cwd=ROOT)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
