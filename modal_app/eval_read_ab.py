"""A/B the reading pipeline on the FULL eval set, regression-safely.

Tests whether OCR-assist and a CLAHE transform actually help READ-F1 in the
general case (not handpicked). Read-F1 is scored identically to eval_sweep.py so
numbers are comparable. Four variants on tuned_v5:

  baseline        VLM reads the original (= v5 today; the control)
  ocr_assist      always append portable-OCR text to the read prompt as a hint
  clahe_fallback  read original; if read is EMPTY, re-read a CLAHE-enhanced image
  cascade         read original; escalate (CLAHE + OCR hint) only if empty or if
                  OCR sees lots of text but the VLM read <2 signs; flag if still empty

The cascade leaves the easy path untouched, so cases that already work cannot
regress; the numbers below decide what (if anything) ships.

  .venv/bin/python -m modal run --detach modal_app/eval_read_ab.py --dataset real
"""
import modal

app = modal.App("curbcheck-read-ab")
vol = modal.Volume.from_name("curbcheck-data", create_if_missing=True)


def _prefetch():
    from huggingface_hub import snapshot_download
    try:
        snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", ignore_patterns=["*.pth", "*.onnx"])
    except Exception as e:
        print("prefetch skipped", e)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # opencv runtime deps
    .pip_install(
        "transformers>=4.49.0", "accelerate>=1.2.0", "pillow>=10.0.0",
        "torch>=2.5.0", "qwen-vl-utils", "einops", "peft>=0.14.0",
        "bitsandbytes>=0.45.0", "huggingface_hub",
        "opencv-python-headless", "numpy", "easyocr",
    )
    .run_function(_prefetch, secrets=[modal.Secret.from_name("huggingface")])
    .add_local_file("schema/rules.py", "/root/rules.py", copy=True)
)

ADAPTER = "/data/runs/qwen25vl3b_v5/final"
BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OCR_HINT = ("\n\nOCR detected this text on the sign(s) (may be noisy, garbled, or "
            "incomplete; use only as a hint, trust the image over it):\n")


@app.function(image=image, gpu="A10G", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3 * 60 * 60)
def run(dataset: str = "real"):
    import json
    import re
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np
    import torch
    from PIL import Image as PImage
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    import easyocr

    samp_file = "/data/eval_samples.jsonl" if dataset == "synth" else "/data/real_eval_samples.jsonl"
    img_dir = "/data/eval_images" if dataset == "synth" else "/data/real_eval_images"
    samples = [json.loads(l) for l in open(samp_file)]

    proc = AutoProcessor.from_pretrained(BASE_ID)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    def infer(prompt, img_path):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img_path}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=400, do_sample=False)
        return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    def ocr_text(img_path):
        try:
            res = ocr_reader.readtext(img_path, detail=1)
            return [t for _b, t, c in res if c > 0.3]
        except Exception:
            return []

    def clahe_path(img_path):
        im = cv2.imread(img_path)
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        big = cv2.resize(out, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(f.name, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        return f.name

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
                str(r.get("start", "")), str(r.get("end", "")),
                frozenset(int(x) for x in (r.get("weeks") or [])))

    def f1(pred, gold):
        if not isinstance(pred, list):
            return 0.0
        p = {nkey(x) for x in pred if isinstance(x, dict)}
        g = {nkey(x) for x in gold}
        tp = len(p & g)
        pr = tp / len(p) if p else 0
        rc = tp / len(g) if g else 0
        return 2 * pr * rc / (pr + rc) if pr + rc else 0.0

    VARIANTS = ["baseline", "ocr_assist", "clahe_fallback", "cascade"]
    scores = {v: [] for v in VARIANTS}
    by_n = {v: {} for v in VARIANTS}
    escalated = flagged = 0

    for s in samples:
        img = f"{img_dir}/{s['image'].split('/')[-1]}"
        n = s["n_signs"]
        rq = next(q for q in s["questions"] if q["type"] == "read")
        prompt, gold = rq["prompt"], rq["gold"]

        base_raw = infer(prompt, img)
        base_pred = extract(base_raw)
        base_signs = [x for x in base_pred if isinstance(x, dict)] if isinstance(base_pred, list) else []
        ocr = ocr_text(img)

        # ocr_assist: always hint
        oa_pred = extract(infer(prompt + OCR_HINT + " | ".join(ocr), img)) if ocr else base_pred

        # clahe_fallback: only re-read if baseline produced nothing
        if not base_signs:
            cf_pred = extract(infer(prompt, clahe_path(img)))
        else:
            cf_pred = base_pred

        # cascade: escalate only when base empty OR (lots of OCR text but <2 signs read)
        need = (not base_signs) or (len(ocr) >= 8 and len(base_signs) < 2)
        if need:
            escalated += 1
            casc_pred = extract(infer(prompt + OCR_HINT + " | ".join(ocr), clahe_path(img)))
            casc_signs = [x for x in casc_pred if isinstance(x, dict)] if isinstance(casc_pred, list) else []
            if not casc_signs:
                flagged += 1
        else:
            casc_pred = base_pred

        for v, pred in (("baseline", base_pred), ("ocr_assist", oa_pred),
                        ("clahe_fallback", cf_pred), ("cascade", casc_pred)):
            sc = f1(pred, gold)
            scores[v].append(sc)
            by_n[v].setdefault(n, []).append(sc)

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "dataset": dataset, "n_samples": len(samples),
        "read_f1": {v: avg(scores[v]) for v in VARIANTS},
        "by_signs": {v: {n: avg(by_n[v][n]) for n in sorted(by_n[v])} for v in VARIANTS},
        "cascade_escalated": escalated, "cascade_flagged": flagged,
    }


@app.local_entrypoint()
def main(dataset: str = "real"):
    import json
    from pathlib import Path
    r = run.remote(dataset=dataset)
    print(f"\n=== read-F1 A/B ({r['dataset']}, n={r['n_samples']}) ===")
    for v, f in r["read_f1"].items():
        print(f"  {v:16s} {f}")
    print(f"cascade escalated {r['cascade_escalated']}/{r['n_samples']}, "
          f"flagged low-legibility {r['cascade_flagged']}")
    print("by_signs:", json.dumps(r["by_signs"], indent=1))
    Path(f"/tmp/read_ab_{dataset}.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
