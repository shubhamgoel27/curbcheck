"""Merge synthetic + real-labeled supervision into one training manifest.

Real images contribute READ examples only (teacher gave us restriction labels,
not verdict probes). Synthetic images contribute read + reason. We cap the real
fraction so the model still learns reasoning from synthetic verdicts while
grounding perception on real photos.

Output: data/merged/train.jsonl with records:
  {"image": "<abs-or-rel path>", "prompt": "...", "response": "...",
   "task": "read|reason", "domain": "synth|real"}

Usage: .venv/bin/python bench/merge.py [--real-frac 0.25]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.generate import READ_PROMPT  # noqa: E402  (READ_PROMPT identical to teacher base)


def load_synth() -> list[dict]:
    out = []
    for r in map(json.loads, (ROOT / "data/train/train.jsonl").open()):
        r = dict(r)
        r["image"] = str(ROOT / "data/train" / r["image"])
        r["domain"] = "synth"
        out.append(r)
    return out


def load_real() -> list[dict]:
    out = []
    # v1 labels (no confidence flag; treat all as usable)
    for r in map(json.loads, (ROOT / "data/real_train/pseudo_labels.jsonl").open()):
        if r["parse_ok"] and r["n_signs"] > 0:
            out.append({"image": str(ROOT / r["image"]), "task": "read",
                        "prompt": READ_PROMPT, "response": json.dumps(r["label"]),
                        "domain": "real"})
    # v3 labels: use ONLY the high-confidence restrictions (cleaner truth)
    v3 = ROOT / "data/real_train/pseudo_labels_v3.jsonl"
    if v3.exists():
        for r in map(json.loads, v3.open()):
            hc = r.get("label_high_conf")
            if r["parse_ok"] and hc:
                # drop the confidence key from the training target
                tgt = [{k: v for k, v in x.items() if k != "confidence"} for x in hc]
                out.append({"image": str(ROOT / r["image"]), "task": "read",
                            "prompt": READ_PROMPT, "response": json.dumps(tgt),
                            "domain": "real"})
    return out


def load_real_negatives(limit: int) -> list[dict]:
    """Zero-sign photos teach abstention: the correct read is []."""
    out = []
    for r in map(json.loads, (ROOT / "data/real_train/pseudo_labels.jsonl").open()):
        if r["parse_ok"] and r["n_signs"] == 0:
            out.append({"image": str(ROOT / r["image"]), "task": "read",
                        "prompt": READ_PROMPT, "response": "[]", "domain": "real"})
    random.Random(0).shuffle(out)
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-frac", type=float, default=0.25,
                    help="target real share of total examples")
    ap.add_argument("--neg-frac", type=float, default=0.10,
                    help="share of real examples that are zero-sign negatives")
    args = ap.parse_args()

    synth = load_synth()
    real_pos = load_real()
    # solve for counts: real = frac * (synth + real)  ->  real = frac/(1-frac) * synth
    target_real = int(args.real_frac / (1 - args.real_frac) * len(synth))
    n_neg = min(int(target_real * args.neg_frac), len(load_real_negatives(10_000)))
    n_pos = min(target_real - n_neg, len(real_pos))
    real = random.Random(1).sample(real_pos, n_pos) + load_real_negatives(n_neg)

    merged = synth + real
    random.Random(2).shuffle(merged)
    out_dir = ROOT / "data" / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    print(f"synthetic examples: {len(synth)}")
    print(f"real positive:      {n_pos}   negative: {n_neg}")
    print(f"merged total:       {len(merged)}  (real {100*len(real)/len(merged):.0f}%)")
    print(f"-> {out_dir/'train.jsonl'}")


if __name__ == "__main__":
    main()
