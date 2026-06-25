"""curbcheck demo: can a small VLM tell you if you can legally park in SF?

Upload a photo of a parking-sign pole, pick a day/time, and the model reads each
sign into structured rules, then a deterministic resolver applies them to that
moment and returns a verdict, with its reasoning shown. Read-then-resolve: the
VLM only perceives, the logic is exact.

Qwen2.5-VL-3B + a QLoRA adapter (curbcheck v4), on ZeroGPU.
"""

import json
import os
import re
from datetime import datetime, time as dtime

import gradio as gr
import spaces
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from rules import Kind, Day, Restriction, Window, SignStack, can_park, Verdict

BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "shubhamgoel27/curbcheck-qwen25vl3b-v5-lora")

READ_PROMPT = """Look at the parking sign stack in this image. Extract EVERY sign as a JSON array.
Each element: {"kind": one of [no_parking, no_stopping, tow_away, time_limit, permit_limit, street_cleaning, loading_only, angle_parking] (use permit_limit when the sign has a permit exemption like EXCEPT AREA X PERMIT, time_limit otherwise; use angle_parking for orientation signs like "PARK AT 90 DEGREES" which do not restrict parking),
"days": list like ["MON","TUE"...] (the days the restriction applies, null for angle_parking),
"start": "HH:MM" 24h, "end": "HH:MM" 24h,
"limit_minutes": int or null, "permit_area": letter or null, "tow": true/false,
"weeks": list of which weeks of the month it applies like [2,4] for "2nd & 4th MONDAY" (works on ANY sign type, not just cleaning), or null for every week}.
Respond with ONLY the JSON array, nothing else."""

# ---- ZeroGPU: prefetch weights at import (no CUDA), load lazily inside @spaces.GPU ----
from huggingface_hub import snapshot_download
print("prefetching weights...")
snapshot_download(BASE_ID)
snapshot_download(ADAPTER_REPO)
print("weights cached")

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(BASE_ID)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(base, ADAPTER_REPO)
        m.eval()
        _model = m.to("cuda")
    return _model, _processor

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


def build_stack(read_json):
    out = []
    for r in read_json if isinstance(read_json, list) else []:
        if not isinstance(r, dict):
            continue
        try:
            kind = Kind(str(r["kind"]).lower().replace("-", "_"))
            if kind is Kind.ANGLE_PARKING:
                out.append(Restriction(kind, Window(frozenset(), dtime(0), dtime(0))))
                continue
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


VERDICT_UI = {
    Verdict.OK: ("✅ You can park", "#0a7d3c"),
    Verdict.LIMITED: ("⏱️ Limited parking", "#b8860b"),
    Verdict.NO: ("🚫 No parking", "#c1452a"),
    Verdict.TOW_RISK: ("🚨 Tow risk", "#8b0000"),
    Verdict.ABSTAIN: ("🤔 Can't tell from the sign", "#555"),
}


@spaces.GPU(duration=120)
def read_signs(image):
    model, processor = _load()
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": READ_PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=400, do_sample=False)
    trim = out[0][inp.input_ids.shape[1]:]
    return processor.decode(trim, skip_special_tokens=True)


DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
KIND_LABEL = {
    Kind.NO_PARKING: "No parking", Kind.NO_STOPPING: "No stopping",
    Kind.TOW_AWAY: "Tow away", Kind.TIME_LIMIT: "Time limit",
    Kind.STREET_CLEANING: "Street cleaning", Kind.PERMIT_EXEMPT_LIMIT: "Permit / time limit",
    Kind.LOADING_ONLY: "Loading only", Kind.ANGLE_PARKING: "Angle parking (info)",
}


def fmt_restriction(r):
    if r.kind is Kind.ANGLE_PARKING:
        return f"- **{KIND_LABEL[r.kind]}** (does not restrict parking)"
    days = ", ".join(d.name.title() for d in sorted(r.window.days, key=lambda d: d.value)) or "every day"
    span = f"{r.window.start.strftime('%-I:%M%p').lower()}–{r.window.end.strftime('%-I:%M%p').lower()}"
    bits = [f"**{KIND_LABEL.get(r.kind, r.kind.value)}**", span, days]
    if r.window.weeks:
        bits.append("weeks " + "/".join(str(w) for w in sorted(r.window.weeks)) + " of month")
    if r.limit_minutes:
        bits.append(f"{r.limit_minutes}min limit")
    if r.permit_area:
        bits.append(f"except Area {r.permit_area} permit")
    if r.tow:
        bits.append("TOW")
    return "- " + " · ".join(bits)


def predict(image, day, hour, minute, ampm, permit):
    if image is None:
        return "### Upload a photo of a parking sign first.", "", ""
    raw = read_signs(image)
    parsed = extract(raw)
    stack = build_stack(parsed)

    # build the "when"
    h = int(hour) % 12 + (12 if ampm == "PM" else 0)
    dow_idx = DOW.index(day)
    # next date matching that weekday (anchored to 2026-06-15, a Monday)
    base = datetime(2026, 6, 15, h, int(minute))
    when = base.replace(day=15 + ((dow_idx - 0) % 7))

    permit_set = frozenset(p.strip().upper() for p in permit.split(",") if p.strip())
    ans = can_park(stack, when, permit_areas=permit_set)

    label, color = VERDICT_UI.get(ans.verdict, ("?", "#555"))
    detail = ""
    if ans.verdict is Verdict.LIMITED and ans.limit_minutes:
        detail = f" — up to {ans.limit_minutes} minutes"
    verdict_md = (
        f"<div style='font-size:1.6em;font-weight:700;color:{color}'>{label}{detail}</div>"
        f"<div style='color:#666;margin-top:6px'>on {day} at {int(hour)}:{int(minute):02d} {ampm}"
        + (f", with permit {','.join(permit_set)}" if permit_set else ", no permit") + "</div>"
        f"<div style='margin-top:8px'>{ans.reason}</div>"
    )

    if stack.restrictions:
        signs_md = "### What the model read on the pole\n" + "\n".join(
            fmt_restriction(r) for r in stack.restrictions)
    else:
        signs_md = "### What the model read on the pole\n_No structured signs parsed._"

    return verdict_md, signs_md, raw.strip()


THEME = gr.themes.Soft(primary_hue="red", neutral_hue="stone")

with gr.Blocks(theme=THEME, title="curbcheck") as demo:
    gr.Markdown(
        "# 🅿️ curbcheck\n"
        "**Can a small VLM tell you if you can legally park in San Francisco?** "
        "Upload a photo of a sign pole, pick a day and time, and a QLoRA-tuned "
        "Qwen2.5-VL-3B reads each sign into structured rules. A deterministic resolver "
        "then decides the verdict, so you see *both* what it read and why. "
        "[Project + benchmark on GitHub](https://github.com/shubhamgoel27/curbcheck)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="pil", label="Parking sign photo", height=360)
            with gr.Row():
                day = gr.Dropdown(DOW, value="Tuesday", label="Day")
                hour = gr.Dropdown([str(i) for i in range(1, 13)], value="5", label="Hour")
                minute = gr.Dropdown(["00", "15", "30", "45"], value="30", label="Min")
                ampm = gr.Dropdown(["AM", "PM"], value="PM", label="")
            permit = gr.Textbox(label="Your permit area(s), if any", placeholder="e.g. S")
            btn = gr.Button("Can I park here?", variant="primary")
        with gr.Column(scale=1):
            verdict_out = gr.Markdown()
            signs_out = gr.Markdown()
            with gr.Accordion("Raw model output (JSON)", open=False):
                raw_out = gr.Code(language="json")

    btn.click(predict, [img, day, hour, minute, ampm, permit],
              [verdict_out, signs_out, raw_out])

    import glob
    ex = sorted(glob.glob("examples/*.jpg"))[:4]
    if ex:
        gr.Examples([[e, "Tuesday", "5", "30", "PM", ""] for e in ex],
                    [img, day, hour, minute, ampm, permit], label="Try a real SF photo")

if __name__ == "__main__":
    demo.launch()
