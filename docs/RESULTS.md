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
