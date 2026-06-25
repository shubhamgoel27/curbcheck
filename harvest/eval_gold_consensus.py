"""Build a trustworthy CONSENSUS gold for the real eval set.

The current real-eval gold is a single Opus pass, noisiest on exactly the hard
images. This re-labels every eval image with 3-vote armored-Opus consensus and
records where the consensus disagrees with the old single-pass gold, so we can
(a) measure how noisy the old gold was and (b) re-score v5 against a cleaner gold.

Reuses label_consensus's voting + 90/85 session/weekly guard. Resumable.

  nohup .venv/bin/python harvest/eval_gold_consensus.py --k 3 --workers 4 \
        >> data/real_train/eval_gold.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harvest"))
import label_consensus as lc  # noqa: E402  (one_vote, consensus, nkey, monitor, RESUME, caps)

SAMPLES = ROOT / "data" / "real_eval_samples.jsonl"
IMGDIR = ROOT / "data" / "real_eval_images"
OUT = ROOT / "data" / "real_train" / "eval_gold_consensus.jsonl"


def read_gold(sample):
    rq = next(q for q in sample["questions"] if q["type"] == "read")
    return rq["gold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    samples = [json.loads(l) for l in SAMPLES.open()]
    done = {json.loads(l)["image"] for l in OUT.open()} if OUT.exists() else set()
    items = []
    for s in samples:
        name = s["image"].split("/")[-1]
        img = IMGDIR / name
        if img.exists() and s["image"] not in done:
            items.append((s["image"], img, read_gold(s)))
    print(f"eval-gold consensus: {len(items)} images, k={args.k} ({len(done)} done)", flush=True)
    if not items:
        return

    threading.Thread(target=lc.monitor, daemon=True).start()

    def do(it):
        sid, img, old_gold = it
        c = lc.consensus(img, args.k)
        if c is None:
            return {"image": sid, "ok": False}
        new_set = {lc.nkey(x) for x in c["consensus"]}
        old_set = {lc.nkey(x) for x in old_gold}
        return {"image": sid, "ok": True, "consensus": c["consensus"],
                "agreement": c["agreement"], "votes_ok": c["votes_ok"],
                "old_gold": old_gold, "matches_old": new_set == old_set,
                "old_n": len(old_gold), "new_n": len(c["consensus"])}

    with OUT.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(do, items), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
