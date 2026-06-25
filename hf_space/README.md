---
title: curbcheck
emoji: 🅿️
colorFrom: red
colorTo: gray
sdk: gradio
app_file: app.py
pinned: true
license: mit
tags:
  - build-small-hackathon
  - backyard-ai
  - vision-language-model
  - qlora
  - qwen2.5-vl
short_description: Can a small VLM tell you if you can legally park in SF?
---

# 🅿️ curbcheck

**Can a small vision-language model tell you if you can legally park in San Francisco?**

Upload a photo of a parking-sign pole, pick a day and time, and a QLoRA-tuned
**Qwen2.5-VL-3B** reads each sign into structured rules. A tiny deterministic resolver
then applies them to that moment and returns the verdict, showing you *both* what the
model read and why. Read-then-resolve: the VLM only perceives; the logic is exact.

I came to SF for a week in April 2026 and left with two parking tickets, both because I
couldn't parse a pole holding four stacked signs. curbcheck is the model that gets it.

## Demo video

<video controls src="https://huggingface.co/spaces/build-small-hackathon/curbcheck/resolve/main/curbcheck_demo.mp4"></video>

[▶ Watch the demo](https://huggingface.co/spaces/build-small-hackathon/curbcheck/resolve/main/curbcheck_demo.mp4)

## Track & badges

- **Track:** Backyard AI (a real, local, personal problem solved with a small model)
- A 3B model, well under the 32B cap, runs on ZeroGPU.

## How it works

```
photo ─▶ VLM (reads each sign to JSON) ─▶ deterministic resolver ─▶ verdict + reason
```

- The model only **reads** the pole into structured restrictions (kind, days, hours, limits, permits).
- A deterministic resolver (`rules.py`, no model in the loop) applies them to the current time.
- Both halves are shown, so misreads are visible, not buried in a confident sentence.

## The result

A stock Qwen2.5-VL-3B scores 0.16 on "can I park here right now", below random. The tuned
model (v5: cross-city real data + vision encoder unfrozen) reasons about real SF photos at
**0.90 (pipeline)** and **0.82 (end to end)**, up from the base model's 0.78 / 0.09.

Full benchmark, training, and honest results: **https://github.com/shubhamgoel27/curbcheck**

## Results in brief (model: v5)

| | base Qwen2.5-VL-3B | tuned (v5) |
|---|:---:|:---:|
| Reasoning, real photos (pipeline) | 0.78 | **0.90** |
| Reasoning, real photos (end to end) | 0.09 | **0.82** |
| Read F1, real sign-bearing photos | 0.08 | **0.62** |
| Abstains correctly on no-sign photos | 0.57 | **0.83** |
| Read F1 (synthetic) | 0.34 | **0.93** |

The honest part: the model reads real sign-bearing photos at ~0.62 F1 (~0.52 against a stricter
3-vote consensus gold) and correctly says "no readable sign here" 83% of the time. Reading is
decent on 1 to 2 sign poles and weakest on cluttered 4-sign ones. (An earlier version of this
card reported ~0.33 real read F1; that was a scoring bug that counted correct abstentions as
failures, since ~46% of the eval set is downed/missing poles. Corrected here.) The read-then-resolve split is what keeps
the verdict reliable (0.90) even when a read is imperfect. Full, honest results in the repo.

## Links

- **Live demo (this Space):** https://huggingface.co/spaces/build-small-hackathon/curbcheck
- **Demo video:** https://huggingface.co/spaces/build-small-hackathon/curbcheck/resolve/main/curbcheck_demo.mp4
- **Code + benchmark (GitHub):** https://github.com/shubhamgoel27/curbcheck
- **Fine-tuned model (QLoRA adapter):** https://huggingface.co/shubhamgoel27/curbcheck-qwen25vl3b-v5-lora
- **Base model:** https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **Writeup:** https://shubham.gg/artifold-share/50558ca9.html
