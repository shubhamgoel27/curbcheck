# Results

## Leaderboard (synthetic eval, 100 images / 400 questions)

Two ways to score "can I park": **end-to-end** (the VLM does perception AND time-logic itself)
and **pipeline** (the VLM only reads the pole; the deterministic resolver does the logic). The
pipeline is the actual product path. Reported as **balanced (macro) accuracy** because verdict
classes are imbalanced (limited 39%, ok 37%, no 15%, tow_risk 8%; majority-class baseline 39%).

| Metric | base Qwen2.5-VL-3B | tuned (QLoRA, ~$4) |
|---|:---:|:---:|
| Read F1 | 0.34 | **0.96** |
| Reason, end-to-end (raw) | 0.16 | 0.76 |
| Reason, end-to-end (balanced) | 0.37 | **0.63** |
| Reason, pipeline (raw) | 0.83 | 0.97 |
| Reason, pipeline (balanced) | 0.74 | **0.92** |
| Verdict parse rate | 1.00 | 1.00 |

## Is the baseline real? Yes.

`parse_rate = 1.00`: the base model always emits a valid, parseable verdict. The 0.16 e2e
score is genuine failure, not a format artifact. Per-class recall shows why:

```
BASE end-to-end recall:  ok 0.01   limited 0.00   no 0.51   tow_risk 0.96
```

The base model is a degenerate pessimist: it answers "no" or "tow_risk" to almost everything,
never "ok" or "limited". It is not reasoning. That is an honest floor.

## Pipeline scoring erases the stacking cliff

End-to-end accuracy collapses as signs stack (juggling 4 windows + permits + severity in one
shot is hard for a 3B model). Routing the model's READ through the resolver fixes it:

| signs | tuned e2e | tuned pipeline |
|---|:---:|:---:|
| 1 | 0.87 | 0.98 |
| 2 | 0.77 | 0.97 |
| 3 | 0.60 | 1.00 |
| 4 | 0.44 | 0.89 |

The model reads 4-sign poles at 0.92 F1; the resolver never makes a logic error; so the real
product accuracy at 4 signs is 0.89, not 0.44. This validates the perception/logic split.

## The real weakness: tow_risk recall

```
tow_risk recall (pipeline):  base 0.52   tuned 0.68
```

Even the tuned model misses ~32% of tow-away situations, because its read occasionally misses
the tow-away / no-stopping plate in a busy stack. This is the safety-critical error (missing a
tow is the expensive mistake) and the rarest class (8%). Fix in progress: rebalance training
to oversample tow_away signs and 4-sign stacks.

## Still open
- Real-photo eval: base vs tuned vs frontier on held-out real SF photos (in progress).
- Frontier reference row (Claude) on the synthetic bench.
- Rebalanced retrain targeting tow_risk recall + 4-sign stacks.

Measured June 2026. Synthetic eval disjoint from training (seed 7, fixed probes). Per-sample
audit logs in data/results/audit/.

---

## Real-photo eval (500 held-out SF photos, Opus-referenced)

The true test: 500 newest 311 reserve photos, never seen in training, labeled by Opus as gold.
This is teacher-referenced (gold = frontier model, not human-verified), so it measures how well
the student matches a strong teacher on real, faded, oblique poles.

Important: real photos are mostly simple (231/500 are downed/missing poles, verdict dist is
75% "ok"), so the majority-class baseline is 75% and raw verdict accuracy is misleading. Use
balanced (macro) accuracy.

| Metric | base | tuned |
|---|:---:|:---:|
| Read F1 | 0.04 | **0.33** |
| Pipeline reasoning, raw | 0.77 (= majority baseline) | 0.88 |
| Pipeline reasoning, **balanced** | 0.45 | **0.82** |
| tow_risk recall | 0.26 | 0.56 |

**What this proves**
- Synthetic + teacher training transfers to real photos: read F1 0.04 -> 0.33 (8x), pipeline
  balanced accuracy 0.45 -> 0.82. The base model genuinely cannot read real signs (0.04 F1).

**The honest gaps**
- Sim-to-real is large on reading: 0.97 F1 on clean renders vs 0.33 on real photos. Clean
  synthetic signs are not faded, sticker-covered, sun-bleached Mission Street poles. This is
  the motivation for more real training data and renderer realism (fading, occlusion).
- tow_risk recall remains the weakest class (0.56 even tuned). Missing a tow is the expensive
  error; it is rare (5% of real cases) and hard (the binding sign is often the faded one).
- Gold is Opus, which itself struggles on the faded signs, so part of the 0.33 read gap is
  student-teacher disagreement on genuinely illegible images. Human-verifying a sample is the
  gold-standard follow-up.

## Rebalanced v2: helped synthetic, mixed on real

The v2 model (training oversampled 3/4-sign stacks + tow signs) clearly beat v1 on synthetic
hard cases (4-sign e2e 0.39 -> 0.67, 4-sign pipeline 0.89 -> 1.0). On real photos v2 was a wash
on the deployment metric (pipeline raw 0.895 vs v1 0.877) but worse end-to-end (0.43 vs 0.63):
trained to expect complex stacks, it over-calls restrictions on the mostly-simple real poles.
A clean distribution-shift lesson: optimize for the eval distribution you will actually deploy
on. For SF, that is mostly 0-2 sign poles, so v1's distribution generalized better end-to-end.
