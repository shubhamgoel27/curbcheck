"""Leaderboard sweep: run several small VLMs over the curbcheck eval bench on
Modal, score with the shared resolver-based scorer, emit a leaderboard.

Each model runs in its own container (parallel). The fine-tuned adapter is
included once it exists on the volume at /runs/qwen25vl3b/final.

  .venv/bin/python -m modal run modal_app/eval_sweep.py
  .venv/bin/python -m modal run modal_app/eval_sweep.py --models qwen3b,smolvlm
"""

import modal

app = modal.App("curbcheck-eval")
vol = modal.Volume.from_name("curbcheck-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.49.0", "accelerate>=1.2.0", "pillow>=10.0.0",
        "torch>=2.5.0", "qwen-vl-utils", "einops", "torchvision",
        "peft>=0.14.0", "bitsandbytes>=0.45.0", "num2words",
    )
)

# model_key -> (hf_id, loader_kind)
MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen"),
    "qwen7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen"),
    "smolvlm": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "auto"),
    "moondream": ("vikhyatk/moondream2", "auto"),
    "tuned": ("/data/runs/qwen25vl3b/final", "qwen"),  # the student, if present
}


@app.function(image=image, gpu="A10G", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=60 * 60)
def run_model(model_key: str):
    import json
    import re
    from pathlib import Path

    import torch
    from PIL import Image as PImage

    hf_id, kind = MODELS[model_key]
    if model_key == "tuned" and not Path(hf_id).exists():
        return {"model": model_key, "skipped": "adapter not on volume yet"}

    samples = [json.loads(l) for l in open("/data/eval_samples.jsonl")]

    if kind == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        is_adapter = Path(hf_id).exists() and (Path(hf_id) / "adapter_config.json").exists()
        if is_adapter:
            from peft import PeftModel
            # load the official (non-quantized) base, apply the LoRA adapter, no merge
            # (the adapter was trained on the unsloth 4-bit base; merging a quantized
            #  base asserts, so we keep the PeftModel wrapper for inference)
            base_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            proc = AutoProcessor.from_pretrained(base_id)
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_id, torch_dtype=torch.bfloat16, device_map="cuda")
            model = PeftModel.from_pretrained(base, hf_id)
            model.eval()
        else:
            proc = AutoProcessor.from_pretrained(hf_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_id, torch_dtype=torch.bfloat16, device_map="cuda")

        def infer(prompt, img_path):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=400)
            trim = out[0][inp.input_ids.shape[1]:]
            return proc.decode(trim, skip_special_tokens=True)
    else:
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)

        def infer(prompt, img_path):
            img = PImage.open(img_path).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=text, images=[img], return_tensors="pt").to("cuda")
            out = model.generate(**inp, max_new_tokens=400)
            return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    # ---- scoring (mirrors eval/claude_baseline.py) ----
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

    def nkey(r):
        k = str(r.get("kind", "")).lower().replace("-", "_")
        if k == "time_limit" and r.get("permit_area"):
            k = "permit_limit"
        return (k, frozenset(str(d)[:3].upper() for d in (r.get("days") or [])),
                str(r.get("start", "")), str(r.get("end", "")))

    read_f1, reason_hit, reason_tot = [], 0, 0
    by_n = {}
    for s in samples:
        img = f"/data/eval_images/{s['image'].split('/')[-1]}"
        for q in s["questions"]:
            pred = extract(infer(q["prompt"], img))
            n = s["n_signs"]
            by_n.setdefault(n, {"r": [], "v": []})
            if q["type"] == "read":
                if isinstance(pred, list):
                    p = {nkey(x) for x in pred if isinstance(x, dict)}
                    g = {nkey(x) for x in q["gold"]}
                    tp = len(p & g)
                    pr = tp / len(p) if p else 0
                    rc = tp / len(g) if g else 0
                    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
                else:
                    f1 = 0.0
                read_f1.append(f1); by_n[n]["r"].append(f1)
            else:
                ok = isinstance(pred, dict) and str(pred.get("verdict", "")).lower() == q["gold"]["verdict"]
                reason_tot += 1; reason_hit += int(ok); by_n[n]["v"].append(int(ok))

    return {
        "model": model_key, "hf_id": hf_id,
        "read_f1": round(sum(read_f1) / len(read_f1), 3) if read_f1 else None,
        "reason_acc": round(reason_hit / reason_tot, 3) if reason_tot else None,
        "by_signs": {n: {"read_f1": round(sum(d["r"]) / len(d["r"]), 3) if d["r"] else None,
                         "reason_acc": round(sum(d["v"]) / len(d["v"]), 3) if d["v"] else None}
                     for n, d in sorted(by_n.items())},
    }


@app.local_entrypoint()
def main(models: str = "qwen3b,smolvlm,tuned"):
    keys = [m.strip() for m in models.split(",")]
    raw = list(run_model.map(keys, return_exceptions=True))
    results = []
    for k, r in zip(keys, raw):
        if isinstance(r, Exception):
            results.append({"model": k, "error": f"{type(r).__name__}: {r}"[:160]})
        else:
            results.append(r)
    print("\n=== curbcheck leaderboard ===")
    print(f"{'model':<12} {'read F1':>8} {'reason':>8}")
    for r in results:
        if r.get("error"):
            print(f"{r['model']:<12}  ERROR: {r['error']}")
        elif r.get("skipped"):
            print(f"{r['model']:<12}  skipped: {r['skipped']}")
        else:
            print(f"{r['model']:<12} {r['read_f1']!s:>8} {r['reason_acc']!s:>8}")
    import json
    from pathlib import Path
    Path("/tmp/curbcheck_leaderboard.json").write_text(json.dumps(results, indent=1))
    print("\nfull:", json.dumps(results, indent=1))
