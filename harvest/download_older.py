"""Download the trainable (non-reserved) 311 case images: everything OLDER than
the newest RESERVE_NEWEST cases. Resumable."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from full import save_image, IMG, RAW

RESERVE_NEWEST = 500
entries = [json.loads(l) for l in (RAW / "sf311_manifest.jsonl").open()]
entries.sort(key=lambda e: e["date"] or "", reverse=True)
trainable = entries[RESERVE_NEWEST:]
print(f"downloading {len(trainable)} older-case images", flush=True)
got = skip = 0
for i, e in enumerate(trainable):
    out = IMG / f"311_{e['subtype']}" / f"311_{e['case']}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        skip += 1
        continue
    if save_image(e["url"], out):
        got += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(trainable)} (new {got}, had {skip})", flush=True)
print(f"DONE: downloaded {got} new, {skip} already present", flush=True)
