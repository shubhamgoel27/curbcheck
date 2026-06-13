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
    )
)

MODEL = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    timeout=4 * 60 * 60,
)
def train(smoke: bool = False):
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
    real_root = Path("/data/real_images")
    if real_root.exists():
        for sub in real_root.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.jpg"):
                    real_index[f.name] = str(f)

    def fix(p: str) -> str:
        name = p.rsplit("/", 1)[-1]
        if name.startswith(("train_", "val_")):
            return f"/data/synth_images/{name}"
        return real_index.get(name, p)

    rows = [json.loads(l) for l in open("/data/train.jsonl")]
    if smoke:
        rows = rows[:64]

    def to_msg(r):
        img_path = fix(r["image"])
        return {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": r["prompt"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": r["response"]}]},
        ]}

    dataset = Dataset.from_list([to_msg(r) for r in rows])

    model, processor = FastVisionModel.from_pretrained(
        MODEL, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=7,
    )
    FastVisionModel.for_training(model)

    def load_images(example):
        for m in example["messages"]:
            for c in m["content"]:
                if c.get("type") == "image":
                    c["image"] = PImage.open(c["image"]).convert("RGB")
        return example

    dataset = dataset.map(load_images)

    cfg = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=30 if smoke else 0,
        num_train_epochs=0 if smoke else 2,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=7,
        output_dir="/data/runs/qwen25vl3b",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model, processor=processor, train_dataset=dataset,
        data_collator=UnslothVisionDataCollator(model, processor), args=cfg,
    )
    trainer.train()

    out = "/data/runs/qwen25vl3b/final"
    model.save_pretrained(out)
    processor.save_pretrained(out)
    vol.commit()
    print(f"saved adapter -> {out}")


@app.local_entrypoint()
def main(smoke: bool = False):
    train.remote(smoke=smoke)
