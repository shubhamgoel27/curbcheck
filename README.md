# curbcheck 🅿️

**Can a small VLM tell you if you can legally park here?**

A benchmark, a leaderboard, a fine-tuned model, and a demo, all built around the most
quietly stressful object in San Francisco: the parking sign pole.

---

## Why this exists (a $160 origin story)

I spent a week in San Francisco in April 2026. I came back with good memories and **two
parking tickets**, both for the same reason: I stood in front of a pole holding four signs,
read all four, and still could not work out whether I was allowed to leave my car there.

You know the pole. A 2-hour limit. *Except* with an Area S permit. *Except* it's also a
street-cleaning zone Tuesday mornings. *Also* tow-away during evening rush. Each sign is
legible on its own. Stacked, they form a little logic puzzle with a time variable, and my
brain, jet-lagged and double-parked, did not solve it. Twice.

The frustrating part is that this is a *solvable* problem. Every fact you need is printed
right there. It is pure perception plus rule-logic plus a clock. So the question became:
**can a small, cheap, runs-on-a-phone vision-language model do the thing my brain failed to
do?** And if it can, can it explain *why*, so the next person standing at that pole walks
away understanding the sign instead of just trusting a green checkmark?

That is curbcheck.

---

## What it actually is

Four artifacts, each useful on its own:

1. **The exam** (a benchmark). Parking-sign images, each paired with exact ground truth and
   three layers of questions:
   - **Read**: extract every restriction on the pole into structured JSON.
   - **Reason**: "It is Tuesday 5:30 PM, you have no permit. Can you park, and for how long?"
   - **Abstain**: when the signs alone cannot answer (curb paint out of frame, occluded
     plate), the correct answer is "can't tell." Knowing what you *don't* know is scored.

2. **The report card** (a leaderboard). Small open VLMs vs a frontier reference, with the
   chart that matters: accuracy as a function of how many signs are on the pole. Reading one
   sign is OCR. Reading four and combining them is the actual job.

3. **The student** (a fine-tuned model). A 3B VLM, QLoRA-tuned on rendered signs plus real
   SF photos, then measured on held-out real photos. The headline question: does training on
   clean synthetic signs survive contact with a faded, sticker-covered Mission Street pole?

4. **The demo**. Point your phone at a pole, pick a day and time, get an answer. But not
   *only* an answer (see below).

---

## The design choice that makes it more than a magic box

A lazy version of this takes a photo and prints "✅ you can park." curbcheck deliberately
does not stop there. The system is split in two:

```
  photo ──▶  VLM (perception)  ──▶  structured restrictions  ──▶  resolver (logic)  ──▶  verdict
                                          │                                              │
                                          ▼                                              ▼
                                   shown to the user                            shown to the user
                                 "here is what each                          "street cleaning does
                                  sign actually says"                         not apply: today is Wed"
```

The VLM only does **perception**: turn the pole into a list of restrictions. A small,
deterministic resolver (`schema/rules.py`) does the **logic**: apply those restrictions to
the current time and produce the verdict. Both halves are surfaced to the user.

This matters for three reasons:

- **Auditable.** If the model misreads "8 AM" as "6 AM," you see it on the parsed card and
  can catch it. Errors are visible, not buried inside a confident sentence.
- **Educational.** A user who sees "street cleaning ✗ (today is Wednesday)" a few times
  learns to read the pole themselves. The tool makes you *more informed over time* instead
  of more dependent. The goal is for you to eventually not need it.
- **Honest evals.** The demo's "here is what each sign says" view and the benchmark's Read
  metric are the *same thing*. The demo is the benchmark, wearing a friendlier coat.

---

## Status

| Piece | State |
|---|---|
| Rule schema + resolver | done (`schema/rules.py`) |
| Synthetic renderer | done (`render/signs.py`), CA R-series styling |
| Real-photo harvest | done: 2,698 real SF photos, 144k-sign inventory, licensing mapped |
| Teacher labeling | done: 2,462 photos labeled by Opus, provenance tracked |
| Training set | done: 7,803 examples (77% synthetic, 23% real Opus-labeled) |
| QLoRA training (Modal) | working; smoke test passed, full run in progress |
| Leaderboard sweep | built (`modal_app/eval_sweep.py`) |
| Demo | not started |

See **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** for the full architecture: data
pipeline, the synthetic-vs-real split, what a training example looks like, the loss, and the
licensing story.

## Repo layout

```
schema/rules.py          the restriction model + can_park() resolver (ground truth engine)
render/signs.py          synthetic CA-style sign-stack renderer
bench/generate.py        builds the eval benchmark (read + reason + abstain questions)
bench/make_train.py      builds the synthetic training set (with leakage guards)
bench/merge.py           folds synthetic + real Opus-labeled into one training manifest
harvest/                 real-photo collection + teacher labeling (see notes/sources.md)
eval/claude_baseline.py  frontier reference scorer
modal_app/               train.py (QLoRA), eval_sweep.py (leaderboard), upload.py
notes/sources.md         the data licensing map
```

## Running it

```bash
uv venv && uv pip install pillow                 # local: renderer + bench
.venv/bin/python bench/generate.py --n 12        # build a small eval set
.venv/bin/python bench/make_train.py --split train --n 2000
.venv/bin/python bench/merge.py                  # merge synthetic + real

# Modal (training + eval on rented GPUs):
.venv/bin/python -m modal setup
.venv/bin/python modal_app/upload.py             # push data to a volume
.venv/bin/python -m modal run modal_app/train.py --smoke   # 30-step sanity check
.venv/bin/python -m modal run modal_app/train.py           # full QLoRA run (~$4)
.venv/bin/python -m modal run modal_app/eval_sweep.py      # leaderboard
```

## License & data ethics

Code: MIT. The benchmark ships images only where licensing allows it (city public-domain
photos, CC-licensed sources); citizen-reported photos are referenced by URL, not rehosted.
Google Street View is deliberately not used. Full map in `notes/sources.md`.
