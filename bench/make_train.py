"""Generate the VLM training set: rendered sign stacks + supervision pairs.

Split policy (leakage guards):
  - eval set: seed 7 (bench/generate.py), FIXED probe times -> never reuse
  - train set: seeds >= 1000, RANDOM probe times
  - val set:   seeds 500-999
Per image, three supervision examples:
  1. READ   -> extract all restrictions as JSON (the structured target)
  2. REASON -> verdict at a random probe time (computed by can_park)
  3. REASON -> verdict at a second probe time, with permits sometimes held
Photometric jitter (brightness/contrast/blur/downscale) applied to train images
only, so the model sees imperfect renders.

Usage:
  .venv/bin/python bench/make_train.py --split train --n 2000
  .venv/bin/python bench/make_train.py --split val   --n 200
Output: data/{split}/images/*.jpg + data/{split}/{split}.jsonl
Record format (model-agnostic, mapped to chat templates at train time):
  {"image": "images/xxx.jpg", "prompt": "...", "response": "...", "task": "read|reason"}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402

from bench.generate import READ_PROMPT, REASON_PROMPT, restriction_to_json  # noqa: E402
from render.signs import render_stack, sample_stack  # noqa: E402
from schema.rules import Kind  # noqa: E402
from schema.rules import can_park  # noqa: E402

SEED_BASE = {"train": 1000, "val": 500}


def jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.15))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
    if rng.random() < 0.3:  # downscale-upscale: cheap camera softness
        w, h = img.size
        f = rng.uniform(0.5, 0.8)
        img = img.resize((int(w * f), int(h * f))).resize((w, h))
    return img


def random_probe(rng: random.Random) -> datetime:
    base = datetime(2026, 6, 15)  # a Monday
    return base + timedelta(days=rng.randint(0, 6),
                            hours=rng.randint(6, 22),
                            minutes=rng.choice([0, 15, 30, 45]))


def reason_prompt_with_permit(when: datetime, permit: str | None) -> str:
    base = REASON_PROMPT.format(when=when.strftime("%A %I:%M %p"))
    if permit:
        base = base.replace("You have no parking permits.",
                            f"You hold a residential parking permit for area {permit}.")
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--rebalance", action="store_true",
                    help="oversample 3/4-sign stacks and tow_away/no_stopping signs")
    args = ap.parse_args()

    # rebalanced distribution targets the measured weaknesses (4-sign + tow_risk recall)
    REBAL_SIZE = (0.1, 0.2, 0.35, 0.35)
    REBAL_KINDS = [
        (Kind.PERMIT_EXEMPT_LIMIT, 0.22), (Kind.STREET_CLEANING, 0.20),
        (Kind.NO_STOPPING, 0.20), (Kind.TOW_AWAY, 0.13), (Kind.NO_PARKING, 0.10),
        (Kind.TIME_LIMIT, 0.10), (Kind.LOADING_ONLY, 0.05),
    ]

    out_dir = ROOT / "data" / args.split
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i in range(args.n):
        rng = random.Random(SEED_BASE[args.split] * 100_000 + i)
        stack = (sample_stack(rng, size_weights=REBAL_SIZE, kind_weights=REBAL_KINDS)
                 if args.rebalance else sample_stack(rng))
        img = render_stack(stack, rng)
        if args.split == "train":
            img = jitter(img, rng)
        name = f"{args.split}_{i:05d}.jpg"
        img.convert("RGB").save(img_dir / name, quality=88)

        gold_read = json.dumps([restriction_to_json(r) for r in stack.restrictions])
        records.append({"image": f"images/{name}", "task": "read",
                        "prompt": READ_PROMPT, "response": gold_read})

        for k in range(2):
            probe = random_probe(rng)
            # sometimes the driver holds the permit shown on the pole (teaches exemptions)
            permit = None
            areas = [r.permit_area for r in stack.restrictions if r.permit_area]
            if k == 1 and areas and rng.random() < 0.5:
                permit = areas[0]
            ans = can_park(stack, probe, frozenset({permit} if permit else ()))
            records.append({
                "image": f"images/{name}", "task": "reason",
                "prompt": reason_prompt_with_permit(probe, permit),
                "response": json.dumps({"verdict": ans.verdict.value,
                                        "limit_minutes": ans.limit_minutes}),
            })
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.n} images", flush=True)

    out = out_dir / f"{args.split}.jsonl"
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.n} images, {len(records)} examples -> {out}")


if __name__ == "__main__":
    main()
