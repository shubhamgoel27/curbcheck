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
    # v3 + cross-city SCF labels: use ONLY the high-confidence restrictions (cleaner truth)
    for name in ("pseudo_labels_v3.jsonl", "labels_scf_v3.jsonl"):
        f = ROOT / "data/real_train" / name
        if not f.exists():
            continue
        for r in map(json.loads, f.open()):
            hc = r.get("label_high_conf")
            if r["parse_ok"] and hc:
                # drop the confidence key from the training target
                tgt = [{k: v for k, v in x.items() if k != "confidence"} for x in hc]
                out.append({"image": str(ROOT / r["image"]), "task": "read",
                            "prompt": READ_PROMPT, "response": json.dumps(tgt),
                            "domain": "real"})
    return out


def load_consensus() -> dict[str, list]:
    """3-vote consensus labels (majority-agreed) keyed by abs image path. Higher
    quality than single-pass Opus, so they OVERRIDE it. Empty consensus (votes
    disagreed or all dropped) -> excluded from training entirely (too uncertain)."""
    out = {}
    f = ROOT / "data/real_train/consensus_labels.jsonl"
    if not f.exists():
        return out
    for r in map(json.loads, f.open()):
        if not r.get("ok"):
            continue
        lab = [{k: v for k, v in x.items() if k != "confidence"}
               for x in (r.get("label_high_conf") or [])]
        out[str(ROOT / r["image"])] = lab  # may be [] -> exclude (uncertain)
    return out


def load_real_negatives(limit: int) -> list[dict]:
    """Zero-sign photos teach abstention: the correct read is []. The cross-city
    SCF harvest (wide street shots, vehicle photos) is a rich, diverse negative pool."""
    out = []
    for name in ("pseudo_labels.jsonl", "pseudo_labels_v3.jsonl", "labels_scf_v3.jsonl"):
        f = ROOT / "data/real_train" / name
        if not f.exists():
            continue
        for r in map(json.loads, f.open()):
            if r.get("parse_ok") and r.get("n_signs", 0) == 0:  # teacher emitted [] = truly no sign
                out.append({"image": str(ROOT / r["image"]), "task": "read",
                            "prompt": READ_PROMPT, "response": "[]", "domain": "real"})
    random.Random(0).shuffle(out)
    return out[:limit]


def load_human() -> dict[str, dict]:
    """Human labels (from the labeler tool) keyed by image path. Highest quality:
    they OVERRIDE Opus for the same image. parse_ok=False (illegible) is dropped."""
    out = {}
    f = ROOT / "data/real_train/human_labels.jsonl"
    if not f.exists():
        return out
    for r in map(json.loads, f.open()):
        if not r.get("parse_ok"):
            continue  # illegible -> exclude from training entirely
        out[r["image"]] = r["label"]  # may be [] (a clean negative)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-frac", type=float, default=0.25,
                    help="target real share of total examples")
    ap.add_argument("--neg-frac", type=float, default=0.10,
                    help="share of real examples that are zero-sign negatives")
    args = ap.parse_args()

    synth = load_synth()
    real_pos = load_real()
    # consensus (3-vote) overrides single-pass Opus for the same image; empty -> exclude
    consensus = load_consensus()
    if consensus:
        real_pos = [r for r in real_pos if r["image"] not in consensus]
        added = 0
        for img, lab in consensus.items():
            if lab:
                real_pos.append({"image": img, "task": "read", "prompt": READ_PROMPT,
                                 "response": json.dumps(lab), "domain": "real"})
                added += 1
        print(f"consensus override: {len(consensus)} images ({added} positive, "
              f"{len(consensus)-added} excluded as uncertain)")
    human = load_human()
    # human labels override everything for the same image (and add new ones)
    if human:
        real_pos = [r for r in real_pos if r["image"] not in
                    {str(ROOT / p) for p in human}]
        for img, lab in human.items():
            if lab:  # non-empty human label -> training example
                real_pos.append({"image": str(ROOT / img), "task": "read",
                                 "prompt": READ_PROMPT, "response": json.dumps(lab),
                                 "domain": "real"})
        print(f"human labels applied: {len(human)} (override Opus)")
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
