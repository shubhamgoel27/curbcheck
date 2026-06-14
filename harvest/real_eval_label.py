"""Build the held-out REAL-photo eval set: download the newest-500 reserve 311
cases (never used in training) and label them with armored Opus.

This is the project's true test set: does synthetic + teacher training survive
real, faded, oblique SF poles?

Usage: nohup .venv/bin/python harvest/real_eval_label.py > data/real_eval/run.log 2>&1 &
Resumable; rides out Max session limits.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harvest"))
from full import save_image  # noqa: E402
from pseudo_label import RESERVE_NEWEST, TEACHER_PROMPT, ask_claude, extract_json  # noqa: E402

OUT = ROOT / "data" / "real_eval"
IMG = OUT / "images"


def reserve_cases() -> list[dict]:
    entries = [json.loads(l) for l in (ROOT / "data/raw/sf311_manifest.jsonl").open()]
    entries.sort(key=lambda e: e["date"] or "", reverse=True)
    return entries[:RESERVE_NEWEST]


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    out = OUT / "labels.jsonl"
    done = {json.loads(l)["case"] for l in out.open()} if out.exists() else set()
    cases = [c for c in reserve_cases() if c["case"] not in done]
    print(f"reserve: {RESERVE_NEWEST}, remaining to label: {len(cases)}", flush=True)

    # download images first (skip existing)
    got = 0
    for c in cases:
        dest = IMG / f"311_{c['case']}.jpg"
        if save_image(c["url"], dest):
            got += 1
    print(f"images present: {got} new + cached", flush=True)

    labelable = [c for c in cases if (IMG / f"311_{c['case']}.jpg").exists()]

    def label_one(c):
        path = IMG / f"311_{c['case']}.jpg"
        while True:
            raw = ask_claude(TEACHER_PROMPT, path)
            if any(s in raw for s in ("session limit", "Not logged in", "usage limit")):
                print("  [limit hit -> sleeping 30 min]", flush=True)
                time.sleep(1800)
                continue
            break
        pred = extract_json(raw)
        ok = isinstance(pred, list)
        if ok:
            pred = [r for r in pred if isinstance(r, dict) and (r.get("days") or r.get("start") or r.get("end"))]
        return {"case": c["case"], "image": str(path.relative_to(ROOT)),
                "subtype": c["subtype"], "address": c.get("address"),
                "label": pred if ok else None, "parse_ok": ok,
                "n_signs": len(pred) if ok else 0, "raw": raw,
                "teacher": {"model": "opus", "armored": True, "split": "real_eval"}}

    with out.open("a") as f, ThreadPoolExecutor(max_workers=3) as ex:
        for i, rec in enumerate(ex.map(label_one, labelable), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  labeled {i}/{len(labelable)}", flush=True)
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
