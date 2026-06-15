"""Harvest parking-sign street imagery from Mapillary across California regions
where Caltrans sign specs + the CA parking regime apply (so read AND reason stay
valid, unlike out-of-state).

Mapillary imagery is CC-BY-SA 4.0: redistribution allowed with attribution +
share-alike. Each record stores the image id + creator for attribution.

Needs a free Mapillary token: https://www.mapillary.com/dashboard/developers
  export MAPILLARY_TOKEN=MLY|...    (or put it in data/.mapillary_token)

Usage: python harvest/mapillary_ca.py --per-region 400
Output: data/images_ca/*.jpg + data/real_train/mapillary_ca.jsonl (manifest)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images_ca"
GRAPH = "https://graph.mapillary.com"

# CA regions with SF-similar signs/rules. (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "oakland_berkeley": (-122.30, 37.79, -122.23, 37.88),
    "san_jose":         (-121.92, 37.31, -121.84, 37.37),
    "sacramento":       (-121.50, 38.55, -121.46, 38.59),
    "peninsula":        (-122.25, 37.45, -122.13, 37.56),  # Palo Alto / Redwood City
    "santa_cruz":       (-122.04, 36.96, -121.99, 37.00),
    "marin":            (-122.53, 37.93, -122.49, 37.97),  # San Rafael
}

# Mapillary traffic-sign object classes that are parking-relevant
PARKING_CLASSES = [
    "regulatory--no-parking--g1", "regulatory--no-parking--g2",
    "regulatory--no-parking-or-no-stopping--g1",
    "regulatory--no-stopping--g1", "regulatory--no-stopping--g2",
    "information--parking--g1", "information--parking--g2",
    "regulatory--no-parking--g6",
]


def token() -> str:
    t = os.environ.get("MAPILLARY_TOKEN")
    if not t and (ROOT / "data/.mapillary_token").exists():
        t = (ROOT / "data/.mapillary_token").read_text().strip()
    if not t:
        raise SystemExit("Set MAPILLARY_TOKEN env or data/.mapillary_token "
                         "(get one at mapillary.com/dashboard/developers)")
    return t


def get_json(url: str, params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}")
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def tiles(bbox, step=0.02):
    lon0, lat0, lon1, lat1 = bbox
    lon = lon0
    while lon < lon1:
        lat = lat0
        while lat < lat1:
            yield (lon, lat, min(lon + step, lon1), min(lat + step, lat1))
            lat += step
        lon += step


def harvest_region(name, bbox, tok, cap, manifest_f):
    seen_imgs = set()
    got = 0
    for tile in tiles(bbox):
        if got >= cap:
            break
        try:
            feats = get_json(f"{GRAPH}/map_features", {
                "access_token": tok, "bbox": ",".join(map(str, tile)),
                "object_values": ",".join(PARKING_CLASSES),
                "fields": "id,object_value,images",
            }).get("data", [])
        except Exception as e:
            print(f"   {name} tile fail: {e}", flush=True)
            continue
        for ft in feats:
            if got >= cap:
                break
            for im in (ft.get("images", {}).get("data", []) or [])[:1]:
                iid = im.get("id")
                if not iid or iid in seen_imgs:
                    continue
                seen_imgs.add(iid)
                try:
                    meta = get_json(f"{GRAPH}/{iid}", {
                        "access_token": tok, "fields": "thumb_1024_url,creator"})
                except Exception:
                    continue
                url = meta.get("thumb_1024_url")
                if not url:
                    continue
                manifest_f.write(json.dumps({
                    "id": f"ca_{name}_{iid}", "region": name, "mapillary_image_id": iid,
                    "object_value": ft.get("object_value"), "url": url,
                    "creator": (meta.get("creator") or {}).get("username"),
                    "license": "CC-BY-SA-4.0"}) + "\n")
                manifest_f.flush()
                got += 1
                time.sleep(0.05)
    print(f"   {name}: {got} sign images", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-region", type=int, default=400)
    ap.add_argument("--download-workers", type=int, default=10)
    args = ap.parse_args()
    tok = token()
    IMG.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/real_train/mapillary_ca.jsonl"

    with manifest.open("w") as f:
        total = sum(harvest_region(n, b, tok, args.per_region, f) for n, b in REGIONS.items())
    print(f"manifest: {total} images across {len(REGIONS)} CA regions", flush=True)

    rows = [json.loads(l) for l in manifest.open()]

    def dl(r):
        dest = IMG / f"{r['id']}.jpg"
        if dest.exists():
            return
        try:
            data = urllib.request.urlopen(r["url"], timeout=20).read()
            if data[:3] == b"\xff\xd8\xff":
                dest.write_bytes(data)
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=args.download_workers) as ex:
        list(ex.map(dl, rows))
    have = len(list(IMG.glob("*.jpg")))
    print(f"downloaded {have} images -> {IMG}", flush=True)


if __name__ == "__main__":
    main()
