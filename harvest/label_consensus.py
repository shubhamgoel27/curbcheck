"""Consensus re-labeling of real parking-sign positives for higher label quality.

Each image is labeled by armored Opus K times (default 3); restrictions are kept
only when a majority of votes agree on the core key (kind, days, start, end, weeks).
This denoises single-pass teacher labels on faded/oblique real signs, which is the
quality lever for the v5 vision-encoder unfreeze (v4 showed quantity does not help).

Same 90/90 session+weekly guard as label_scf.py. Resumable. Output schema keeps a
`label_high_conf` field so bench/merge.py can consume it unchanged.

  nohup .venv/bin/python harvest/label_consensus.py --k 3 --workers 4 \
        >> data/real_train/label_consensus.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harvest"))
from pseudo_label import ask_claude, extract_json  # noqa: E402
from prompt_scf import TEACHER_PROMPT_SCF  # noqa: E402

SOURCES = [ROOT / "data/real_train/labels_scf_v3.jsonl",
           ROOT / "data/real_train/pseudo_labels_v3.jsonl"]
OUT = ROOT / "data" / "real_train" / "consensus_labels.jsonl"
SESSION_CAP = 90
WEEK_CAP = 85

RESUME = threading.Event()
RESUME.set()


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
                print(f"[guard] session={s}% week={w}% at/over cap -> pausing", flush=True)
                RESUME.clear()
            elif not over and not RESUME.is_set():
                print(f"[guard] session={s}% week={w}% under cap -> resuming", flush=True)
                RESUME.set()
        time.sleep(300)


def one_vote(path: Path):
    attempts = 0
    while True:
        RESUME.wait()
        try:
            raw = ask_claude(TEACHER_PROMPT_SCF, path)
        except subprocess.TimeoutExpired:
            attempts += 1
            if attempts >= 2:
                return None
            continue
        if any(s in raw for s in ("session limit", "Not logged in", "usage limit")):
            time.sleep(1200)
            continue
        break
    m = re.findall(r"```json\s*(.*?)```", raw, re.S)
    pred = extract_json(m[-1] if m else raw)
    if not isinstance(pred, list):
        return None
    return [r for r in pred if isinstance(r, dict) and
            (r.get("days") or r.get("start") or r.get("end") or r.get("kind") == "angle_parking")]


def nkey(r):
    return (str(r.get("kind", "")).lower().replace("-", "_"),
            tuple(sorted(str(d)[:3].upper() for d in (r.get("days") or []))),
            str(r.get("start") or ""), str(r.get("end") or ""),
            tuple(sorted(int(x) for x in (r.get("weeks") or []))))


def consensus(path: Path, k: int):
    votes = [v for v in (one_vote(path) for _ in range(k)) if v is not None]
    if not votes:
        return None
    need = len(votes) // 2 + 1  # majority of successful votes
    counts = Counter()
    examples = {}
    for v in votes:
        for key in {nkey(r) for r in v}:  # one count per vote per distinct key
            counts[key] += 1
        for r in v:
            examples.setdefault(nkey(r), r)
    kept = [examples[key] for key, c in counts.items() if c >= need]
    for r in kept:
        r["confidence"] = "high"
    # agreement signal
    sizes = [len(v) for v in votes]
    if all(c == len(votes) for c in counts.values()) and len(set(sizes)) == 1:
        agreement = "unanimous"
    elif kept:
        agreement = "majority"
    else:
        agreement = "none"
    return {"votes_ok": len(votes), "k": k, "consensus": kept,
            "agreement": agreement, "vote_sizes": sizes}


def worklist():
    items = {}
    for src in SOURCES:
        if not src.exists():
            continue
        for l in src.open():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not (r.get("label_high_conf") or []):
                continue  # positives only
            img = r["image"]
            if (ROOT / img).exists():
                items[r["id"]] = img
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    wl = [(i, img) for i, img in worklist().items() if i not in done]
    print(f"consensus worklist: {len(wl)} positives, k={args.k} ({len(done)} done)", flush=True)
    if not wl:
        return
    threading.Thread(target=monitor, daemon=True).start()

    def do(it):
        iid, img = it
        c = consensus(ROOT / img, args.k)
        if c is None:
            return {"id": iid, "image": img, "ok": False}
        return {"id": iid, "image": img, "ok": True,
                "label_high_conf": c["consensus"], "agreement": c["agreement"],
                "n_consensus": len(c["consensus"]), "votes_ok": c["votes_ok"],
                "vote_sizes": c["vote_sizes"],
                "teacher": {"model": "opus", "prompt": "scf", "method": "consensus", "k": args.k}}

    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(do, wl), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  consensus {i}/{len(wl)}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
