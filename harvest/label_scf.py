"""Teacher-label the SeeClickFix harvest with armored Opus (prompt v3).

Reuses the proven labeling path (ask_claude + extract_json + TEACHER_PROMPT_V3).
Self-pacing: a monitor thread polls `claude -p /usage` and pauses all workers at
80% session usage, auto-resuming after the window resets, so an unattended overnight
run never fully locks out the Max plan. Resumable (skips ids already in the output).

Output: data/real_train/labels_scf_v3.jsonl  (same schema as pseudo_labels_v3.jsonl,
so bench/merge.py can load it unchanged).

Usage:
  nohup .venv/bin/python harvest/label_scf.py --workers 4 > data/real_train/label_scf.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harvest"))
from pseudo_label import ask_claude, extract_json  # noqa: E402
from prompt_scf import TEACHER_PROMPT_SCF  # noqa: E402

MANIFEST = ROOT / "data" / "real_train" / "scf_manifest.jsonl"
OUT = ROOT / "data" / "real_train" / "labels_scf_v3.jsonl"
SESSION_CAP = 90
WEEK_CAP = 85

RESUME = threading.Event()
RESUME.set()  # running by default; cleared = paused


def usage_pcts():
    try:
        out = subprocess.run(["claude", "-p", "/usage", "--output-format", "text"],
                             capture_output=True, text=True, timeout=120).stdout
        s = re.search(r"Current session:\s*(\d+)% used", out)
        w = re.search(r"Current week \(all models\):\s*(\d+)% used", out)
        return (int(s.group(1)) if s else None, int(w.group(1)) if w else None)
    except Exception:
        return (None, None)


def monitor():
    while True:
        s, w = usage_pcts()
        over = (s is not None and s >= SESSION_CAP) or (w is not None and w >= WEEK_CAP)
        if s is not None or w is not None:
            if over and RESUME.is_set():
                print(f"[guard] session={s}% week={w}% at/over cap ({SESSION_CAP}/{WEEK_CAP}) -> pausing", flush=True)
                RESUME.clear()
            elif not over and not RESUME.is_set():
                print(f"[guard] session={s}% week={w}% under cap -> resuming", flush=True)
                RESUME.set()
        time.sleep(300)


def label_v3(path: Path):
    attempts = 0
    while True:
        RESUME.wait()  # blocks here while paused by the guard
        try:
            raw = ask_claude(TEACHER_PROMPT_SCF, path)
        except subprocess.TimeoutExpired:
            attempts += 1
            if attempts >= 2:
                return None, False, "TIMEOUT"
            continue
        if any(s in raw for s in ("session limit", "Not logged in", "usage limit")):
            print("  [limit -> sleep 20m]", flush=True)
            time.sleep(1200)
            continue
        break
    m = re.findall(r"```json\s*(.*?)```", raw, re.S)
    pred = extract_json(m[-1] if m else raw)
    ok = isinstance(pred, list)
    if ok:
        pred = [r for r in pred if isinstance(r, dict) and
                (r.get("days") or r.get("start") or r.get("end") or r.get("kind") == "angle_parking")]
    return pred, ok, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    meta = {}
    if MANIFEST.exists():
        for l in MANIFEST.open():
            try:
                r = json.loads(l)
                meta[r["id"]] = r
            except Exception:
                pass
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    items = [r for i, r in meta.items() if i not in done and (ROOT / r["image"]).exists()]
    items = items[:args.limit]
    print(f"to label: {len(items)} images ({len(done)} already done)", flush=True)
    if not items:
        return

    threading.Thread(target=monitor, daemon=True).start()

    def do(r):
        pred, ok, raw = label_v3(ROOT / r["image"])
        hi = [x for x in (pred or []) if x.get("confidence") != "low"]
        return {"id": r["id"], "src": r.get("city"), "url": r.get("url"),
                "image": r["image"], "label": pred if ok else None,
                "label_high_conf": hi if ok else None, "parse_ok": ok,
                "n_signs": len(pred) if ok else 0, "n_high_conf": len(hi),
                "raw": raw, "teacher": {"model": "opus", "prompt": "scf", "armored": True}}

    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(do, items), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  labeled {i}/{len(items)}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
