"""Dump tuned_v5's per-image READ predictions on the real eval set, so we can
re-score read-F1 against the cleaner consensus gold locally.

  .venv/bin/python -m modal run --detach modal_app/dump_reads.py
"""
import modal

app = modal.App("curbcheck-dumpreads")
vol = modal.Volume.from_name("curbcheck-data", create_if_missing=True)


def _prefetch():
    from huggingface_hub import snapshot_download
    try:
        snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", ignore_patterns=["*.pth", "*.onnx"])
    except Exception as e:
        print("prefetch skipped", e)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.49.0", "accelerate>=1.2.0", "pillow>=10.0.0",
                 "torch>=2.5.0", "torchvision", "qwen-vl-utils", "einops", "peft>=0.14.0",
                 "bitsandbytes>=0.45.0", "huggingface_hub")
    .run_function(_prefetch, secrets=[modal.Secret.from_name("huggingface")])
)

ADAPTER = "/data/runs/qwen25vl3b_v5/final"
BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@app.function(image=image, gpu="A10G", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=2 * 60 * 60)
def dump(dataset: str = "real"):
    import json
    import re
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    samp = "/data/real_eval_samples.jsonl" if dataset == "real" else "/data/eval_samples.jsonl"
    img_dir = "/data/real_eval_images" if dataset == "real" else "/data/eval_images"
    samples = [json.loads(l) for l in open(samp)]

    proc = AutoProcessor.from_pretrained(BASE_ID)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    dec = json.JSONDecoder()

    def extract(t):
        t = re.sub(r"```(?:json)?", "", t).strip("` \n")
        for i, ch in enumerate(t):
            if ch in "[{":
                try:
                    return dec.raw_decode(t[i:])[0]
                except json.JSONDecodeError:
                    continue
        return None

    out = []
    for s in samples:
        img = f"{img_dir}/{s['image'].split('/')[-1]}"
        rq = next(q for q in s["questions"] if q["type"] == "read")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": rq["prompt"]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=400, do_sample=False)
        raw = proc.decode(o[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract(raw)
        out.append({"image": s["image"], "n_signs": s["n_signs"],
                    "read": pred if isinstance(pred, list) else []})
    return out


@app.local_entrypoint()
def main(dataset: str = "real"):
    import json
    from pathlib import Path
    res = dump.remote(dataset=dataset)
    Path(f"/tmp/v5_reads_{dataset}.json").write_text(json.dumps(res))
    print(f"dumped {len(res)} reads -> /tmp/v5_reads_{dataset}.json")
