"""Harvest parking/sign photos from Bloomington IN via Open311 GeoReport v2.

Open311 exposes a media_url field; Bloomington populates it (~20-36% of records).
Parking-relevant service_codes (verified by recon):
  44 Parking Meters and Citations, 43 Parking Permits,
  73 Inaccessible Parking, 86 Street & Traffic Signs.

Writes images to data/images_scf/bloom/ and APPENDS rows to the shared
scf_manifest.jsonl (same schema as seeclickfix.py) so harvest/label_scf.py
picks them up automatically.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images_scf" / "bloom"
MANIFEST = ROOT / "data" / "real_train" / "scf_manifest.jsonl"
BASE = "https://bloomington.in.gov/crm/open311/v2/requests.json"
CODES = {"44": "Parking Meters and Citations", "43": "Parking Permits",
         "73": "Inaccessible Parking", "86": "Street & Traffic Signs"}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}


def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
            return json.load(r)
    except Exception as e:
        print(f"   api fail: {str(e)[:80]}", flush=True)
        return None


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
            b = r.read()
        if b[:3] == b"\xff\xd8\xff" or b[:8] == b"\x89PNG\r\n\x1a\n":
            dest.write_bytes(b)
            return True
    except Exception:
        return False
    return False


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    seen = set()
    if MANIFEST.exists():
        for l in MANIFEST.open():
            try:
                seen.add(json.loads(l)["id"])
            except Exception:
                pass
    got = 0
    with MANIFEST.open("a") as mf:
        for code, name in CODES.items():
            page = 1
            while True:
                d = get(f"{BASE}?service_code={code}&page={page}")
                if not d:
                    break
                reqs = d if isinstance(d, list) else d.get("requests") or d.get("service_requests") or []
                if not reqs:
                    break
                batch = [r for r in reqs if r.get("media_url")
                         and f"bloom_{r.get('service_request_id')}" not in seen]
                for r in batch:
                    seen.add(f"bloom_{r.get('service_request_id')}")

                def fetch(r):
                    iid = f"bloom_{r.get('service_request_id')}"
                    dest = IMG / f"{iid}.jpg"
                    if download(r["media_url"], dest):
                        return {"id": iid, "city": "bloomington", "is_ca": False,
                                "request_type": r.get("service_name") or name,
                                "url": r["media_url"], "address": r.get("address"),
                                "image": str(dest.relative_to(ROOT))}
                    return None

                with ThreadPoolExecutor(max_workers=8) as ex:
                    for rec in ex.map(fetch, batch):
                        if rec:
                            mf.write(json.dumps(rec) + "\n")
                            got += 1
                mf.flush()
                print(f"   bloom code {code} ({name}) p{page}: +{got}", flush=True)
                page += 1
                time.sleep(0.3)
                if page > 60:
                    break
    print(f"bloomington: {got} photos -> {IMG}", flush=True)


if __name__ == "__main__":
    main()
