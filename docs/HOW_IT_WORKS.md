# How curbcheck works

The deep version. If the README is the pitch, this is the engineering.

The whole project rests on one decision: **separate perception from logic.** The VLM reads
the pole into structured data; a deterministic resolver turns that data into a verdict. Every
downstream choice follows from that split.

---

## 1. The rule schema (the ground-truth engine)

`schema/rules.py` defines what a San Francisco parking pole can *say*, as data:

- A `SignStack` is a list of `Restriction`s (one pole, top to bottom).
- A `Restriction` has a `kind` (street_cleaning, no_parking, no_stopping, tow_away,
  time_limit, permit_limit, loading_only), a recurring weekly `Window` (days + start/end),
  and optional `limit_minutes`, `permit_area`, `tow`.
- `can_park(stack, when, permits)` is the **resolver**: apply every restriction to a moment
  in time, rank by severity (`TOW_RISK > NO > LIMITED > OK`), and return the verdict plus the
  binding reason. Permit exemptions and strictest-limit-wins are handled here.

This function is the spine of the project. It generates training labels, it scores the
benchmark, and at inference time it produces the verdict from the model's reading. There is
exactly one source of parking-logic truth, and it is 120 lines of pure Python with no model
in the loop.

---

## 2. Two sources of ground truth

The training data is **mixed**, and the two halves get their labels in completely different
ways. This is the most important thing to understand about the dataset.

### Synthetic (77%, 6,000 examples from 2,000 rendered images)

`render/signs.py` draws CA-style sign plates with PIL, stacks 1 to 4 on a pole, and applies
light photometric jitter. Crucially, the renderer draws each sign *from* a `SignStack`
object, so **the label exists before the pixels do.** There is no annotation step and no
possibility of label error:

- **Read** target: serialize the `SignStack` the image was drawn from.
- **Reason** target: run `can_park()` on that stack at a probe time. Exact, by construction.

Synthetic data is the only source of **reasoning** supervision, because only the resolver can
compute a correct verdict at an arbitrary time.

### Real (23%, 1,803 examples from real SF photos)

Real photos have no labels, so we *make* them, the same way the Meta ads pipeline did:
**teacher distillation.** An armored Opus reads each real photo into the same restriction
JSON schema (the 2,462-image labeling run in `harvest/pseudo_label.py`). These are
frontier-quality pseudo-labels.

Real photos contribute **Read** examples only (Opus gave us restrictions, not time-probe
verdicts). So the skills divide cleanly:

| Skill | Learned from |
|---|---|
| Perception / structure (pole -> restrictions) | synthetic **and** real Opus-labeled photos |
| Time-logic reasoning (restrictions + clock -> verdict) | synthetic only (resolver-computed) |

The real data also includes **859 zero-sign negatives** (photos of downed or missing poles,
which is what people actually report to 311). A sample of these trains abstention: the
correct Read of a bare pole is `[]`.

### Why "armored" Opus

A context audit found that headless `claude -p` calls load the global CLAUDE.md every time.
An A/B test on 12 photos showed it did not change sign readings (10/12 exact agreement), but
the labeler appends an explicit instruction de-scoping any workflow rules, pins the model,
and records `{model, armored, prompt_version}` provenance per label anyway. Rigor is cheap.

---

## 3. What one training example looks like

Every example is a triple `(image, prompt, target)`:

**Synthetic, Read:**
```
image:  a rendered 2-sign pole
prompt: "Extract EVERY sign as a JSON array. Each element: {kind, days, start, end, ...}"
target: [{"kind":"time_limit","days":["MON",...,"FRI"],"start":"09:00","end":"21:00",
          "limit_minutes":60,...}, {"kind":"permit_limit",...}]
```

**Synthetic, Reason:**
```
prompt: "It is Tuesday 07:00 PM. You have no parking permits. Can you park?"
target: {"verdict":"limited","limit_minutes":60}
```

**Real, Read** (Opus-labeled):
```
image:  an actual SF 311 photo
target: [{"kind":"street_cleaning","days":["TUE"],"start":"07:00","end":"08:00",...},
         {"kind":"permit_limit",...}]
```

`bench/merge.py` shuffles these into one manifest at a controlled real fraction (default
25%, actual 23%).

---

## 4. Training: QLoRA on a 3B VLM

Base model: **Qwen2.5-VL-3B-Instruct**, quantized to 4-bit and frozen. We inject rank-16
LoRA adapters into the language layers (attention + MLP) and train only those. The vision
encoder stays frozen: we are teaching the model to *reason about* sign content, not to see
differently.

**How a row becomes input:**
1. Wrapped in Qwen's chat template: a `user` turn (image + prompt), an `assistant` turn (the
   target JSON).
2. The frozen ViT encodes the image to vision tokens (dynamic resolution: a big 311 photo
   becomes more tokens than a small render, which is why peak VRAM tracks image size).
3. Tokenized into one sequence: `[image tokens][prompt tokens][response tokens]`.

**The loss:** standard next-token cross-entropy, **masked to the response tokens only**
(via `UnslothVisionDataCollator`). The model is not graded on the image or the prompt, only
on autoregressively producing the gold JSON. There is no RL and no separate accuracy loss;
training is pure imitation of the gold strings. Verdict accuracy is measured later, by the
benchmark.

**Why these numbers** (`modal_app/train.py`):

| Setting | Value | Reason |
|---|---|---|
| trainable | LoRA r=16, LM attn+MLP | tiny, enough capacity for a narrow task, can't forget the base |
| effective batch | 16 | the binding constraint is optimizer *steps*, not VRAM (see below) |
| steps | ~975 (2 epochs) | healthy for a 3B QLoRA on ~8k examples |
| LR / schedule | 1e-4 cosine, 10 warmup | standard QLoRA |
| optimizer | adamw_8bit | quantized optimizer states |

**The batch-size logic.** VRAM probes on the A100-40GB measured 8GB at batch 8 and 28GB at
batch 32. VRAM is not the bottleneck; we have plenty. The real constraint is that total
samples processed (dataset x epochs) is fixed, so a bigger batch just means *fewer weight
updates*. Batch 16 (~975 steps, ~15GB, 36% of the card) trades a little idle VRAM for enough
optimizer steps. Filling the GPU with batch 32 would only update the model 488 times and
likely undertrain.

---

## 5. Compute (Modal)

- **Training**: A100-40GB ($2.10/hr). A 3B QLoRA is not memory-bound, so the 40GB tier runs
  at the same speed as 80GB for ~16% less. Full run ~$4.
- **Eval sweep**: A10G ($1.10/hr), one container per model, run in parallel via `.map()`.
- **GPU choice rule of thumb learned here**: cheapest-per-hour is not cheapest-per-run.
  Slower cards are disproportionately slower per step, so for a small workload the fast-but-
  pricier A100-40 often wins on total cost. Measure, don't assume.

Why not full fine-tuning? On ~8k examples, full FT of 3B params risks overfitting and
catastrophic forgetting, costs ~2-3x more, and usually gains little on a task this narrow.
It becomes worth it only with far more synthetic data (the faucet is free). Kept as a future
ablation: frozen baseline vs QLoRA vs full-FT, same eval.

---

## 6. Leakage discipline

- **Eval reserve**: the newest 500 real 311 cases are never labeled for training and never
  rendered. They are the held-out real test set, awaiting human verification.
- **Split seeds**: synthetic eval uses seed 7 with fixed probe times; training uses seeds
  >= 1000 with random probe times; val uses 500-999. Disjoint by construction.
- **Jitter on train only**: val and eval renders stay clean, so they measure the task, not
  the noise.

---

## 7. Evaluation

`bench/generate.py` produces the benchmark; `eval/claude_baseline.py` (frontier reference)
and `modal_app/eval_sweep.py` (small VLMs) score it with the same resolver-based scorer:

- **Read**: per-restriction F1 (a restriction matches on kind + days + start + end; a
  time-limit with a permit exemption is normalized to permit_limit).
- **Reason**: verdict accuracy, with limit-minutes checked when the verdict is "limited."
- Both broken down by **number of signs on the pole**, which is the difficulty axis.

The frontier reference reads clean synthetic renders essentially perfectly (the correct
day-one result: it confirms the task is well-posed). The interesting headroom is in the small
models, the corrupted renders, and the faded real poles, which is exactly where a fine-tuned
student earns its keep.
