"""Turn the Opus-labeled real-eval reserve into a benchmark in the synth-bench
format: read gold = Opus label, reason gold = can_park(Opus label, probe time).

This is a teacher-referenced real eval (gold = frontier model, not human-verified).
Honest framing: it measures how well a student matches the teacher on real photos.

Usage: .venv/bin/python bench/build_real_bench.py
Writes data/real_eval/samples.jsonl (+ images already in data/real_eval/images).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.generate import READ_PROMPT, REASON_PROMPT, PROBES  # noqa: E402
from schema.rules import Day, Kind, Restriction, SignStack, Window, can_park  # noqa: E402


def build_stack(label) -> SignStack:
    out = []
    for r in label or []:
        if not isinstance(r, dict):
            continue
        try:
            kind = Kind(str(r["kind"]).lower().replace("-", "_"))
            days = frozenset(Day[str(d)[:3].upper()] for d in (r.get("days") or []))
            sh, sm = map(int, str(r["start"]).split(":"))
            eh, em = map(int, str(r["end"]).split(":"))
            wk = frozenset(int(x) for x in (r.get("weeks") or []))
            out.append(Restriction(kind, Window(days, time(sh, sm), time(eh, em), weeks=wk),
                                   limit_minutes=r.get("limit_minutes"),
                                   permit_area=r.get("permit_area"), tow=bool(r.get("tow"))))
        except Exception:
            continue
    return SignStack(out)


def main() -> None:
    labels = [json.loads(l) for l in (ROOT / "data/real_eval/labels.jsonl").open()]
    labels = [r for r in labels if r["parse_ok"]]  # keep 0-sign (abstention) too
    samples = []
    for r in labels:
        stack = build_stack(r["label"])
        qs = [{"qid": f"{r['case']}_read", "type": "read", "prompt": READ_PROMPT,
               "gold": r["label"]}]
        for j, probe in enumerate(PROBES):
            ans = can_park(stack, probe)
            qs.append({"qid": f"{r['case']}_reason{j}", "type": "reason",
                       "prompt": REASON_PROMPT.format(when=probe.strftime("%A %I:%M %p")),
                       "probe": probe.isoformat(),
                       "gold": {"verdict": ans.verdict.value, "limit_minutes": ans.limit_minutes}})
        samples.append({"image": r["image"].split("/")[-1], "n_signs": r["n_signs"],
                        "questions": qs})
    out = ROOT / "data/real_eval/samples.jsonl"
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"built real bench: {len(samples)} images, "
          f"{sum(len(s['questions']) for s in samples)} questions -> {out}")


if __name__ == "__main__":
    main()
