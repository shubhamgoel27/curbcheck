"""Push the dataset to the Modal volume. Run after `modal setup`.

  .venv/bin/python modal_app/upload.py

Idempotent-ish: re-running overwrites the same volume paths.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOL = "curbcheck-data"
PY = [sys.executable, "-m", "modal"]

PUTS = [
    ("data/merged/train.jsonl", "/train.jsonl"),
    ("data/train/images", "/synth_images"),
    ("data/images", "/real_images"),
    ("data/val", "/val"),
    ("data/synth/samples.jsonl", "/eval_samples.jsonl"),
    ("data/synth/images", "/eval_images"),
]


def main() -> None:
    subprocess.run(PY + ["volume", "create", VOL], cwd=ROOT)
    for local, remote in PUTS:
        src = ROOT / local
        if not src.exists():
            print(f"skip (missing): {local}")
            continue
        print(f"put {local} -> {remote}")
        subprocess.run(PY + ["volume", "put", "--force", VOL, str(src), remote], cwd=ROOT)
    print("done")


if __name__ == "__main__":
    main()
