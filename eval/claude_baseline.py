"""Frontier baseline: run Claude (via claude CLI headless, uses Max subscription)
over the synthetic benchmark and score it.

Usage: .venv/bin/python eval/claude_baseline.py [--limit-images 12]
Writes data/results/claude_baseline.jsonl and prints a scorecard.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "data" / "synth"
RESULTS = ROOT / "data" / "results"


def ask_claude(prompt: str, image_path: Path) -> str:
    full = f"{prompt}\n\nThe image file is at: {image_path}\nRead it, then answer."
    out = subprocess.run(
        ["claude", "-p", full, "--allowedTools", "Read", "--output-format", "text"],
        capture_output=True, text=True, timeout=240,
    )
    return out.stdout.strip()


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    dec = json.JSONDecoder()
    # parse a valid JSON prefix starting at each [ or { (models often append prose)
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def norm_restriction(r: dict) -> tuple:
    kind = str(r.get("kind", "")).lower().replace("-", "_")
    # a time-limit sign with a permit exemption is the same sign under either label
    if kind == "time_limit" and r.get("permit_area"):
        kind = "permit_limit"
    return (
        kind,
        frozenset(str(d)[:3].upper() for d in (r.get("days") or [])),
        str(r.get("start", "")), str(r.get("end", "")),
    )


def score_read(pred, gold) -> dict:
    if not isinstance(pred, list):
        return {"f1": 0.0, "parsed": pred is not None}
    p = {norm_restriction(r) for r in pred if isinstance(r, dict)}
    g = {norm_restriction(r) for r in gold}
    tp = len(p & g)
    prec = tp / len(p) if p else 0
    rec = tp / len(g) if g else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"f1": round(f1, 3), "parsed": True, "tp": tp, "n_pred": len(p), "n_gold": len(g)}


def score_reason(pred, gold) -> dict:
    if not isinstance(pred, dict):
        return {"correct": False, "parsed": pred is not None}
    v_ok = str(pred.get("verdict", "")).lower() == gold["verdict"]
    l_ok = True
    if gold["verdict"] == "limited":
        l_ok = pred.get("limit_minutes") == gold["limit_minutes"]
    return {"correct": bool(v_ok and l_ok), "verdict_only": bool(v_ok), "parsed": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=12)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_f = RESULTS / "claude_baseline.jsonl"

    samples = [json.loads(l) for l in (SYNTH / "samples.jsonl").open()][: args.limit_images]
    n_done = 0
    read_f1s, reason_correct, reason_total = [], 0, 0

    with out_f.open("w") as f:
        for s in samples:
            img = SYNTH / s["image"]
            for q in s["questions"]:
                raw = ask_claude(q["prompt"], img)
                pred = extract_json(raw)
                if q["type"] == "read":
                    sc = score_read(pred, q["gold"])
                    read_f1s.append(sc["f1"])
                else:
                    sc = score_reason(pred, q["gold"])
                    reason_total += 1
                    reason_correct += int(sc["correct"])
                rec = {"image": s["image"], "n_signs": s["n_signs"], "qid": q["qid"],
                       "type": q["type"], "gold": q["gold"], "pred": pred,
                       "raw": raw, "score": sc}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}] {q['qid']} {q['type']} -> {sc}", flush=True)

    print("\n===== SCORECARD: Claude (frontier reference) =====")
    if read_f1s:
        print(f"READ   mean restriction-F1: {sum(read_f1s)/len(read_f1s):.3f}  (n={len(read_f1s)})")
    if reason_total:
        print(f"REASON verdict accuracy:   {reason_correct}/{reason_total} = {reason_correct/reason_total:.1%}")
    print(f"results -> {out_f}")


if __name__ == "__main__":
    main()
