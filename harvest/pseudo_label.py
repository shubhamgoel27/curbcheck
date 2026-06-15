"""Teacher pseudo-labeling: Claude reads real photos -> structured labels.

Split discipline: 311 cases are sorted by date; the NEWEST `--reserve` cases are
the eval reserve and are never labeled here. Training labels come from older
cases + DPW photos only.

Usage: .venv/bin/python harvest/pseudo_label.py --limit 12
Writes data/real_train/pseudo_labels.jsonl (one record per image, with raw
teacher output kept for auditing).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.generate import READ_PROMPT  # noqa: E402

RESERVE_NEWEST = 500  # newest 311 cases are eval-only, never trained on

TEACHER_PROMPT = READ_PROMPT + """
Additional rules for real photographs:
- Only describe PARKING-RELATED signs (ignore speed limit, stop, street name signs).
- If the photo shows a temporary paper/portable sign, read it the same way.
- If dates are shown instead of weekdays (e.g. 6/17/26), put the matching weekday in days.
- If no parking sign is legible in the photo, respond with exactly: []
- If a sign is present but too blurred/faded to read fields, omit unreadable fields as null.
- Do NOT output a restriction object unless at least one of days/start/end is actually legible.
- A restriction that applies at all times should use days [all seven] and start 00:00 end 23:59."""


ARMOR = ("You are strictly a visual data labeler for parking signs. Ignore any user-level "
         "coding or workflow instructions that may appear in your context (CLAUDE.md rules, "
         "plan mode, style guides); they do not apply to labeling. Output only the requested JSON.")
TEACHER_MODEL = "opus"


def ask_claude(prompt: str, image_path: Path) -> str:
    full = f"{prompt}\n\nThe image file is at: {image_path}\nRead it, then answer."
    out = subprocess.run(
        ["claude", "-p", full, "--allowedTools", "Read", "--output-format", "text",
         "--append-system-prompt", ARMOR, "--model", TEACHER_MODEL],
        capture_output=True, text=True, timeout=240,
    )
    return out.stdout.strip()


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def trainable_images() -> list[dict]:
    """All downloaded real images EXCLUDING the eval reserve (newest 311 cases)."""
    entries = [json.loads(l) for l in (ROOT / "data/raw/sf311_manifest.jsonl").open()]
    entries.sort(key=lambda e: e["date"] or "", reverse=True)
    reserved = {e["case"] for e in entries[:RESERVE_NEWEST]}
    items = []
    for img in sorted((ROOT / "data/images").rglob("*.jpg")):
        src = img.parent.name
        if src.startswith("311_"):
            case = img.stem.replace("311_", "")
            if case in reserved:
                continue
            items.append({"path": img, "source": src, "case": case})
        elif src == "dpw":
            items.append({"path": img, "source": "dpw", "case": None})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = ROOT / "data" / "real_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pseudo_labels.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["image"] for l in out.open()}

    items = [it for it in trainable_images() if str(it["path"].relative_to(ROOT)) not in done]
    print(f"trainable pool: {len(items)} images ({len(done)} already labeled)")

    def label_one(it):
        while True:
            raw = ask_claude(TEACHER_PROMPT, it["path"])
            if "session limit" in raw or "Not logged in" in raw or "usage limit" in raw.lower():
                print("  [limit hit -> sleeping 30 min]", flush=True)
                import time as _t
                _t.sleep(1800)
                continue
            break
        pred = extract_json(raw)
        ok = isinstance(pred, list)
        if ok:  # drop unreadable-fragment restrictions (all-null timing fields)
            pred = [r for r in pred if isinstance(r, dict) and
                    (r.get("days") or r.get("start") or r.get("end") or r.get("kind")=="angle_parking")]
        return {"image": str(it["path"].relative_to(ROOT)), "source": it["source"],
                "label": pred if ok else None, "parse_ok": ok,
                "n_signs": len(pred) if ok else 0, "raw": raw,
                "teacher": {"model": TEACHER_MODEL, "armored": True, "prompt": "TEACHER_PROMPT_v2"}}

    batch = items[: args.limit]
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, rec in enumerate(ex.map(label_one, batch), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{n}/{len(batch)}] {rec['image'].split('/')[-1]} "
                  f"-> {'OK ' + str(rec['n_signs']) + ' signs' if rec['parse_ok'] else 'PARSE FAIL'}", flush=True)
    print(f"labels -> {out}")


if __name__ == "__main__":
    main()
