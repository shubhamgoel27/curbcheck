"""Generate benchmark samples: rendered stack image + ground-truth JSON + questions.

Each sample gets:
  - read question: extract all restrictions as JSON
  - reason questions: can_park() verdicts at probe times (computed ground truth)

Usage: python bench/generate.py --n 12 --seed 7
Writes data/synth/{images/*.png, samples.jsonl}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from render.signs import render_stack, sample_stack  # noqa: E402
from schema.rules import Day, Kind, SignStack, can_park  # noqa: E402

# probe times: a spread of weekday/weekend, morning/afternoon/evening
PROBES = [
    datetime(2026, 6, 16, 8, 30),   # Tue morning
    datetime(2026, 6, 16, 17, 30),  # Tue evening
    datetime(2026, 6, 20, 14, 0),   # Sat afternoon
]


def restriction_to_json(r) -> dict:
    return {
        "kind": r.kind.value,
        "days": sorted(d.name for d in r.window.days),
        "start": r.window.start.strftime("%H:%M"),
        "end": r.window.end.strftime("%H:%M"),
        "limit_minutes": r.limit_minutes,
        "permit_area": r.permit_area,
        "tow": r.tow,
    }


READ_PROMPT = """Look at the parking sign stack in this image. Extract EVERY sign as a JSON array.
Each element: {"kind": one of [no_parking, no_stopping, tow_away, time_limit, permit_limit, street_cleaning, loading_only] (use permit_limit when the sign has a permit exemption like EXCEPT AREA X PERMIT, time_limit otherwise),
"days": list like ["MON","TUE"...] (the days the restriction applies),
"start": "HH:MM" 24h, "end": "HH:MM" 24h,
"limit_minutes": int or null, "permit_area": letter or null, "tow": true/false}.
Respond with ONLY the JSON array, nothing else."""

REASON_PROMPT = """Look at the parking sign stack in this image. It is {when}. You have no parking permits.
Considering ALL signs, can you legally park here right now, and what is the result?
Respond with ONLY a JSON object: {{"verdict": one of ["tow_risk","no","limited","ok"],
"limit_minutes": int or null}}.
tow_risk = parking now risks a tow. no = parking prohibited (citation). limited = allowed
up to limit_minutes. ok = no restriction applies at this moment."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_img = ROOT / "data" / "synth" / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    samples = []

    for i in range(args.n):
        stack = sample_stack(rng)
        img = render_stack(stack, rng)
        name = f"synth_{args.seed}_{i:03d}.png"
        img.save(out_img / name)

        questions = [{
            "qid": f"{i:03d}_read", "type": "read", "prompt": READ_PROMPT,
            "gold": [restriction_to_json(r) for r in stack.restrictions],
        }]
        for j, probe in enumerate(PROBES):
            ans = can_park(stack, probe)
            when = probe.strftime("%A %I:%M %p")
            questions.append({
                "qid": f"{i:03d}_reason{j}", "type": "reason",
                "prompt": REASON_PROMPT.format(when=when),
                "probe": probe.isoformat(),
                "gold": {"verdict": ans.verdict.value, "limit_minutes": ans.limit_minutes},
            })
        samples.append({"image": f"images/{name}", "n_signs": len(stack.restrictions),
                        "questions": questions})

    out = ROOT / "data" / "synth" / "samples.jsonl"
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    n_q = sum(len(s["questions"]) for s in samples)
    print(f"wrote {len(samples)} images, {n_q} questions -> {out}")


if __name__ == "__main__":
    main()
