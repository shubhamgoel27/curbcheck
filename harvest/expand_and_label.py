"""Expand the real training set + label with the improved v3 prompt.

New sources (training only; the MTA-Parking newest-500 eval reserve stays separate):
  - DPW permit photos: ALL ~998 unique (public domain, every one is a parking sign)
  - SF 311 "Sign Repair" filtered to parking (service_details ~ parking); mobile311
    photo URLs return real JPEGs. Opus returns [] for any non-parking stragglers.

Labels use TEACHER_PROMPT_V3 (reason -> fenced JSON, per-sign confidence). Resumable,
rides out Max session limits. Output: data/real_train/pseudo_labels_v3.jsonl

Usage: nohup .venv/bin/python harvest/expand_and_label.py --signrepair 2000 > data/real_train/v3_run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harvest"))
from full import save_image  # noqa: E402
from pseudo_label import ask_claude, extract_json  # noqa: E402
from prompt_v3 import TEACHER_PROMPT_V3  # noqa: E402

IMG = ROOT / "data" / "images_v3"
SODA = "https://data.sfgov.org/resource/vw6y-z8j6.json"
DPW = "https://data.sfgov.org/resource/pigs-fac7.json"
UA = {"User-Agent": "curbcheck/0.2 (research; shubhamgoel27@gmail.com)"}


def soda(url, params):
    q = urllib.parse.urlencode(params)
    return json.loads(urllib.request.urlopen(urllib.request.Request(f"{url}?{q}", headers=UA), timeout=60).read())


def worklist(n_signrepair: int) -> list[dict]:
    items = []
    # 1. full DPW (public domain)
    import hashlib
    for r in soda(DPW, {"$limit": 50000, "$order": "dateadded DESC"}):
        u = r.get("filename", "")
        if u.startswith("http"):
            items.append({"src": "dpw", "url": u,
                          "id": "dpw_" + hashlib.md5(u.encode()).hexdigest()[:10]})
    # 2. Sign Repair, parking-filtered, newest first
    got = 0
    offset = 0
    while got < n_signrepair:
        batch = soda(SODA, {"$limit": 2000, "$offset": offset, "$order": "requested_datetime DESC",
                            "$where": "service_name='Sign Repair' AND media_url IS NOT NULL "
                                      "AND lower(service_details) like '%parking%'"})
        if not batch:
            break
        for r in batch:
            m = r.get("media_url") or {}
            u = m.get("url") if isinstance(m, dict) else m
            if u:
                items.append({"src": "signrepair", "url": u, "id": f"sr_{r['service_request_id']}"})
                got += 1
        offset += 2000
    return items



def label_v3(path: Path):
    attempts = 0
    while True:
        try:
            raw = ask_claude(TEACHER_PROMPT_V3, path)
        except subprocess.TimeoutExpired:
            attempts += 1
            if attempts >= 2:
                return None, False, "TIMEOUT"   # skip this image, keep the run alive
            continue
        if any(s in raw for s in ("session limit", "Not logged in", "usage limit")):
            print("  [limit -> sleep 30m]", flush=True)
            time.sleep(1800)
            continue
        break
    m = re.findall(r"```json\s*(.*?)```", raw, re.S)
    pred = extract_json(m[-1] if m else raw)
    ok = isinstance(pred, list)
    if ok:
        pred = [r for r in pred if isinstance(r, dict) and (r.get("days") or r.get("start") or r.get("end"))]
    return pred, ok, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signrepair", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    out = ROOT / "data/real_train/pseudo_labels_v3.jsonl"
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()

    wl = worklist(args.signrepair)
    seen = set(); wl = [i for i in wl if not (i["id"] in seen or seen.add(i["id"]))]
    items = [it for it in wl if it["id"] not in done]
    print(f"worklist: {len(items)} new images ({len(done)} already done)", flush=True)

    # download
    for it in items:
        save_image(it["url"], IMG / f"{it['id']}.jpg")
    items = [it for it in items if (IMG / f"{it['id']}.jpg").exists()]
    print(f"downloaded/cached: {len(items)}", flush=True)

    def do(it):
        pred, ok, raw = label_v3(IMG / f"{it['id']}.jpg")
        hi = [r for r in (pred or []) if r.get("confidence") != "low"]
        return {"id": it["id"], "src": it["src"], "url": it["url"],
                "image": str((IMG / f"{it['id']}.jpg").relative_to(ROOT)),
                "label": pred if ok else None, "label_high_conf": hi if ok else None,
                "parse_ok": ok, "n_signs": len(pred) if ok else 0,
                "n_high_conf": len(hi), "raw": raw,
                "teacher": {"model": "opus", "prompt": "v3", "armored": True}}

    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(do, items), 1):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"  labeled {i}/{len(items)}", flush=True)
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
