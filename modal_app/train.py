"""QLoRA fine-tune of Qwen2.5-VL-3B on the curbcheck training set, on Modal.

Setup (one time):
  modal setup
  # store tokens as secrets:
  modal secret create huggingface HF_TOKEN=hf_xxx
  modal secret create wandb WANDB_API_KEY=xxx        # optional
  # push the dataset to a volume:
  .venv/bin/python -m modal volume create curbcheck-data
  .venv/bin/python -m modal volume put curbcheck-data data/merged/train.jsonl /train.jsonl
  .venv/bin/python -m modal volume put curbcheck-data data/train/images        /synth_images
  .venv/bin/python -m modal volume put curbcheck-data data/images              /real_images
  .venv/bin/python -m modal volume put curbcheck-data data/val                 /val

Run:
  .venv/bin/python -m modal run modal_app/train.py            # full run
  .venv/bin/python -m modal run modal_app/train.py --smoke    # 30-step sanity check
"""

import modal

app = modal.App("curbcheck-train")
vol = modal.Volume.from_name("curbcheck-data", create_if_missing=True)

MODEL = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"


def _prefetch_base():
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(MODEL)
        print("prefetched", MODEL)
    except Exception as e:
        print("prefetch skipped", e)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth",                     # latest; pin once a known-good build is found
        "trl>=0.13.0",
        "transformers>=4.49.0",
        "datasets>=3.2.0",
        "pillow>=10.0.0",
        "wandb>=0.18.0",
        "accelerate>=1.2.0",
        "huggingface_hub",
    )
    .run_function(_prefetch_base, secrets=[modal.Secret.from_name("huggingface")])
)


@app.function(
    image=image,
    gpu="A100-40GB",  # 3B QLoRA peaks ~12-18GB; 40GB tier = same speed, ~16% cheaper than 80GB
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("huggingface"),
        # wandb optional: create the secret and add it back to enable curve logging
    ],
    timeout=6 * 60 * 60,  # xformers fell back to PyTorch attn (~9s/it); 6h covers 1438 steps + load with margin
)
def train(smoke: bool = False, batch: int = 16, accum: int = 1, probe: bool = False,
          manifest: str = "/data/train.jsonl", run: str = "qwen25vl3b"):
    import json
    import os
    from pathlib import Path

    from datasets import Dataset
    from pathlib import Path
    from PIL import Image as PImage
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    os.environ.setdefault("WANDB_PROJECT", "curbcheck")

    # ---- resolve Mac-side image paths to volume paths ----
    # build a name->path index once over the real-image subdirs
    real_index = {}
    for root in (Path("/data/real_images"), Path("/data/real_images_v3")):
        if not root.exists():
            continue
        # v1 reals are nested in subdirs; v3 reals are flat
        for f in root.rglob("*.jpg"):
            real_index[f.name] = str(f)

    def fix(p: str) -> str:
        name = p.rsplit("/", 1)[-1]
        if name.startswith(("train_", "val_")):
            return f"/data/synth_images/{name}"
        return real_index.get(name, p)

    rows = [json.loads(l) for l in open(manifest)]
    if smoke or probe:
        # probe uses the LARGEST real images so peak VRAM is worst-case, not average
        rows = sorted(rows, key=lambda r: 0 if r["domain"] == "real" else 1)[:64]

    # plain list of message dicts with PIL images embedded (the Unsloth-vision contract)
    def to_msg(r):
        img = PImage.open(fix(r["image"])).convert("RGB")
        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": r["prompt"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": r["response"]}]},
        ]}

    dataset = [to_msg(r) for r in rows]

    model, processor = FastVisionModel.from_pretrained(
        MODEL, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=7,
    )
    FastVisionModel.for_training(model)

    import torch
    cfg = SFTConfig(
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_steps=5 if (smoke or probe) else 10,
        max_steps=8 if probe else (30 if smoke else -1),
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=7,
        output_dir=f"/data/runs/{run}",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model, tokenizer=processor, train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, processor), args=cfg,
    )
    trainer.train()

    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n>>> PEAK VRAM: {peak:.1f} GB / {total:.0f} GB  "
          f"(batch={batch}, accum={accum}, eff={batch*accum}, {peak/total:.0%} used)")

    if probe:
        print(">>> probe only, not saving")
        return
    out = f"/data/runs/{run}/final"
    model.save_pretrained(out)
    processor.save_pretrained(out)
    vol.commit()
    print(f"saved adapter -> {out}")


@app.local_entrypoint()
def main(smoke: bool = False, batch: int = 16, accum: int = 1, probe: bool = False,
         manifest: str = "/data/train.jsonl", run: str = "qwen25vl3b"):
    train.remote(smoke=smoke, batch=batch, accum=accum, probe=probe, manifest=manifest, run=run)
