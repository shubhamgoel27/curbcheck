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

# base weights to bake into the image at build (downloaded once, then instant per container)
PREFETCH = ["Qwen/Qwen2.5-VL-3B-Instruct", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"]


def _prefetch_weights():
    from huggingface_hub import snapshot_download
    for m in PREFETCH:
        try:
            snapshot_download(m, ignore_patterns=["*.pth", "*.onnx"])
            print("prefetched", m)
        except Exception as e:
            print("prefetch skipped", m, e)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.49.0", "accelerate>=1.2.0", "pillow>=10.0.0",
        "torch>=2.5.0", "qwen-vl-utils", "einops", "torchvision",
        "peft>=0.14.0", "bitsandbytes>=0.45.0", "num2words", "huggingface_hub",
    )
    .run_function(_prefetch_weights, secrets=[modal.Secret.from_name("huggingface")])
    .add_local_file("schema/rules.py", "/root/rules.py", copy=True)
)

# model_key -> (hf_id, loader_kind)
MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen"),
    "qwen7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen"),
    "smolvlm": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "auto"),
    "moondream": ("vikhyatk/moondream2", "auto"),
    "tuned": ("/data/runs/qwen25vl3b/final", "qwen"),  # v1 student
    "tuned_v2": ("/data/runs/qwen25vl3b_v2/final", "qwen"),  # rebalanced student
    "tuned_v3": ("/data/runs/qwen25vl3b_v3/final", "qwen"),  # rebalanced + realism + v3-labels
    "tuned_v4": ("/data/runs/qwen25vl3b_v4/final", "qwen"),  # + human labels + angle/weeks
}


@app.function(image=image, gpu="A10G", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3 * 60 * 60)
def run_model(model_key: str, dataset: str = "synth"):
    import json
    import re
    from pathlib import Path

    import torch
    from PIL import Image as PImage

    hf_id, kind = MODELS[model_key]
    if model_key == "tuned" and not Path(hf_id).exists():
        return {"model": model_key, "skipped": "adapter not on volume yet"}

    samp_file = "/data/eval_samples.jsonl" if dataset == "synth" else "/data/real_eval_samples.jsonl"
    img_dir = "/data/eval_images" if dataset == "synth" else "/data/real_eval_images"
    samples = [json.loads(l) for l in open(samp_file)]

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

    # ---- the resolver (the deterministic logic engine), imported from the repo ----
    import sys
    sys.path.insert(0, "/root")
    from datetime import datetime, time as dtime
    from rules import Restriction, Window, SignStack, Kind, Day, can_park

    def build_stack(read_json):
        """Reconstruct a SignStack from a model's READ output, skipping malformed rows."""
        out = []
        for r in read_json if isinstance(read_json, list) else []:
            if not isinstance(r, dict):
                continue
            try:
                kind = Kind(str(r["kind"]).lower().replace("-", "_"))
                days = frozenset(Day[str(d)[:3].upper()] for d in (r.get("days") or []))
                sh, sm = map(int, str(r["start"]).split(":"))
                eh, em = map(int, str(r["end"]).split(":"))
                wk = frozenset(int(x) for x in (r.get("weeks") or []))
                out.append(Restriction(kind, Window(days, dtime(sh, sm), dtime(eh, em), weeks=wk),
                                       limit_minutes=r.get("limit_minutes"),
                                       permit_area=r.get("permit_area"), tow=bool(r.get("tow"))))
            except Exception:
                continue
        return SignStack(out)

    # ---- scoring ----
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

    VERDICTS = ("tow_risk", "limited", "ok", "no")

    def extract_verdict(t):
        """Lenient: JSON verdict first, then prose keywords. Returns (verdict|None, parsed_ok)."""
        obj = extract(t)
        if isinstance(obj, dict) and obj.get("verdict"):
            v = str(obj["verdict"]).lower().replace("-", "_").replace(" ", "_")
            if v in VERDICTS:
                return v, True
        low = t.lower()
        # generous prose mapping, in severity order so "tow" wins over "no"
        if "tow" in low: return "tow_risk", False
        if "limited" in low or "hour" in low or "minute" in low: return "limited", False
        if any(p in low for p in ("cannot park", "can't park", "not allowed", "no parking", "prohibited")): return "no", False
        if any(p in low for p in ("you can park", "allowed to park", "free to park", '"ok"', "verdict: ok")): return "ok", False
        return None, False

    def nkey(r):
        k = str(r.get("kind", "")).lower().replace("-", "_")
        if k == "time_limit" and r.get("permit_area"):
            k = "permit_limit"
        return (k, frozenset(str(d)[:3].upper() for d in (r.get("days") or [])),
                str(r.get("start", "")), str(r.get("end", "")),
                frozenset(int(x) for x in (r.get("weeks") or [])))

    read_f1, by_n = [], {}
    e2e_hit = e2e_tot = pipe_hit = pipe_tot = parse_ok = parse_tot = 0
    records = []

    for s in samples:
        img = f"{img_dir}/{s['image'].split('/')[-1]}"
        n = s["n_signs"]
        by_n.setdefault(n, {"r": [], "e2e": [], "pipe": []})
        # READ once, reuse for the pipeline verdict
        read_q = next(q for q in s["questions"] if q["type"] == "read")
        read_raw = infer(read_q["prompt"], img)
        read_pred = extract(read_raw)
        if isinstance(read_pred, list):
            p = {nkey(x) for x in read_pred if isinstance(x, dict)}
            g = {nkey(x) for x in read_q["gold"]}
            tp = len(p & g)
            pr = tp / len(p) if p else 0
            rc = tp / len(g) if g else 0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        else:
            f1 = 0.0
        read_f1.append(f1); by_n[n]["r"].append(f1)
        stack = build_stack(read_pred)

        for q in s["questions"]:
            if q["type"] != "reason":
                continue
            raw = infer(q["prompt"], img)
            v, ok = extract_verdict(raw)
            gold = q["gold"]["verdict"]
            # end-to-end: the model does perception + logic itself
            e2e_tot += 1; e2e_hit += int(v == gold)
            parse_tot += 1; parse_ok += int(ok)
            by_n[n]["e2e"].append(int(v == gold))
            # pipeline: the model's READ -> deterministic resolver
            probe = datetime.fromisoformat(q["probe"])
            pv = can_park(stack, probe).verdict.value
            pipe_tot += 1; pipe_hit += int(pv == gold)
            by_n[n]["pipe"].append(int(pv == gold))
            if model_key in ("qwen3b", "tuned", "tuned_v4"):  # keep audit trail for the headline models
                records.append({"image": s["image"].split("/")[-1], "n_signs": n,
                                "probe": q["probe"], "gold": gold,
                                "e2e_verdict": v, "e2e_parsed_json": ok,
                                "pipeline_verdict": pv, "e2e_raw": raw[:200]})

    # persist the audit trail to the volume
    if records:
        import pathlib
        pathlib.Path("/data/results").mkdir(parents=True, exist_ok=True)
        with open(f"/data/results/{model_key}_records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        vol.commit()

    def rate(h, t):
        return round(h / t, 3) if t else None

    return {
        "model": model_key, "hf_id": hf_id,
        "read_f1": rate(sum(read_f1) * 1, len(read_f1)) if read_f1 else None,
        "reason_e2e": rate(e2e_hit, e2e_tot),
        "reason_pipeline": rate(pipe_hit, pipe_tot),
        "verdict_parse_rate": rate(parse_ok, parse_tot),
        "by_signs": {n: {"read_f1": round(sum(d["r"]) / len(d["r"]), 3) if d["r"] else None,
                         "e2e": round(sum(d["e2e"]) / len(d["e2e"]), 3) if d["e2e"] else None,
                         "pipe": round(sum(d["pipe"]) / len(d["pipe"]), 3) if d["pipe"] else None}
                     for n, d in sorted(by_n.items())},
    }


@app.local_entrypoint()
def main(models: str = "qwen3b,smolvlm,tuned", dataset: str = "synth"):
    keys = [m.strip() for m in models.split(",")]
    raw = list(run_model.starmap([(k, dataset) for k in keys], return_exceptions=True))
    results = []
    for k, r in zip(keys, raw):
        if isinstance(r, Exception):
            results.append({"model": k, "error": f"{type(r).__name__}: {r}"[:160]})
        else:
            results.append(r)
    print("\n=== curbcheck leaderboard ===")
    print(f"{'model':<12} {'read F1':>8} {'reason(e2e)':>12} {'reason(pipe)':>13} {'parse%':>8}")
    for r in results:
        if r.get("error"):
            print(f"{r['model']:<12}  ERROR: {r['error']}")
        elif r.get("skipped"):
            print(f"{r['model']:<12}  skipped: {r['skipped']}")
        else:
            print(f"{r['model']:<12} {r['read_f1']!s:>8} {r['reason_e2e']!s:>12} "
                  f"{r['reason_pipeline']!s:>13} {r['verdict_parse_rate']!s:>8}")
    import json
    from pathlib import Path
    Path("/tmp/curbcheck_leaderboard.json").write_text(json.dumps(results, indent=1))
    print("\nfull:", json.dumps(results, indent=1))
