# Results

## Leaderboard v1 (synthetic eval, 100 images / 400 questions)

| Model | Read F1 | Reason acc | Notes |
|---|---:|---:|---|
| Qwen2.5-VL-3B (base) | 0.35 | 0.15 | below the 0.25 random line on 4-class verdicts |
| **Qwen2.5-VL-3B (tuned, QLoRA)** | **0.96** | **0.76** | ~$4 of training |

Reasoning accuracy by number of signs on the pole (the difficulty axis):

| signs | base | tuned |
|---|---:|---:|
| 1 | 0.07 | 0.87 |
| 2 | 0.15 | 0.77 |
| 3 | 0.38 | 0.60 |
| 4 | 0.17 | 0.39 |

**Takeaways**
- Out of the box, a small VLM cannot read SF parking signs: 0.15 verdict accuracy is worse
  than guessing. It reads individual fields poorly and combines them even worse.
- A single ~$4 QLoRA run on synthetic + teacher-labeled real data lifts read-F1 from 0.35 to
  0.96 and reasoning from 0.15 to 0.76.
- Difficulty scales with stacking. Even the tuned model is weakest at 4-sign poles (0.39),
  which is exactly the case that is hardest for humans too.

**Still open**
- Frontier reference row (Claude) on the same 100-image set, for the top of the board.
- The real test: tuned vs base on the held-out real SF photos (seed reserve), not synthetic.
- SmolVLM2 loader (needs AutoModelForImageTextToText, not AutoModelForCausalLM).

Measured June 2026. Synthetic eval is disjoint from training (seed 7, fixed probe times).
