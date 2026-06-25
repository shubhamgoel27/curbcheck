"""Unattended overnight controller.

Sequences the fresh-window work so the high-yield SCF harvest (Syracuse + Bloomington)
gets labeled FIRST, then consensus re-labeling runs on all real positives. Both phases
are paced by each labeler's own 90/90 session+weekly guard.

  nohup .venv/bin/python harvest/overnight_controller.py > /dev/null 2>&1 &
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAN = ROOT / "data/real_train/scf_manifest.jsonl"
LAB = ROOT / "data/real_train/labels_scf_v3.jsonl"
LOG = ROOT / "data/real_train/overnight_controller.log"


def log(m):
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")


def scf_pending():
    man = set()
    for l in MAN.open():
        try:
            d = json.loads(l)
            if (ROOT / d["image"]).exists():
                man.add(d["id"])
        except Exception:
            pass
    lab = set()
    if LAB.exists():
        for l in LAB.open():
            try:
                lab.add(json.loads(l)["id"])
            except Exception:
                pass
    return len(man - lab)


def main():
    log("controller start; phase 1 = finish SCF labeling (Syracuse/Bloomington high-yield)")
    stable = 0
    while True:
        try:
            p = scf_pending()
        except Exception as e:
            log(f"pending check error: {e}")
            time.sleep(300)
            continue
        log(f"scf pending={p}")
        stable = stable + 1 if p <= 3 else 0
        if stable >= 2:  # two consecutive clean checks (~5 min apart)
            break
        time.sleep(300)
    log("phase 1 complete -> stopping SCF labeler, starting phase 2 consensus")
    subprocess.run(["pkill", "-f", "label_scf.py"])
    time.sleep(3)
    subprocess.Popen(
        ["bash", "-c",
         "while true; do .venv/bin/python harvest/label_consensus.py --k 3 --workers 4 "
         ">> data/real_train/label_consensus.log 2>&1; sleep 120; done"],
        cwd=str(ROOT), start_new_session=True)
    log("phase 2 consensus loop launched; controller exiting")


if __name__ == "__main__":
    main()
