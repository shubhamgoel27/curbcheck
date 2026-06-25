"""Harvest close-up parking/traffic-sign photos from the SeeClickFix native API.

SF's DataSF/DPW pipeline gave us SF close-ups. SeeClickFix (the platform behind
many city 311 systems) exposes user-submitted report photos via media.image_full,
no auth (browser UA required; the URL 302-redirects to a single-use signed S3 blob,
so we download inline as we page).

We only ever train READING on real photos (reasoning stays synthetic), so non-CA
cities are fine too: their signs still teach the vision encoder real-world glyphs.

Output:
  data/images_scf/<city>/<prefix>_<id>.jpg
  data/real_train/scf_manifest.jsonl   (one row per downloaded image)

Usage:
  .venv/bin/python harvest/seeclickfix.py --max-per-city 4000
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images_scf"
MANIFEST = ROOT / "data" / "real_train" / "scf_manifest.jsonl"
API = "https://seeclickfix.com/api/v2/issues"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

# (place_url, prefix, [request_type ids] or None to keyword-filter, is_california)
# ids verified during recon; None = page and keep titles matching SIGN_KW.
CITIES = [
    ("us-ca-oakland",  "oak",  ["20982", "20974", "26566", "26565"], True),   # CA, ~4.7k sign issues
    ("san-francisco",  "sf",   ["4053"], True),                                # CA, tight sign-repair close-ups
    ("chicago",        "chi",  ["24069", "26385", "27541"], False),           # read-only augmentation
    ("berkeley",       "berk", None, True),                                    # small; keyword-filter
    ("san-jose",       "sj",   None, True),
    ("syracuse",       "syr",  ["34423"], False),                              # "Traffic & Parking Signs" ~1.9k
    ("detroit",        "det",  ["9123", "25151"], False),                      # Traffic Sign Issue / SIGN
    # ---- expansion batch (verified by recon 2026-06-24): sign/parking request types ----
    ("minneapolis",    "mpls", ["2471", "2468", "2467"], False),              # Traffic Sign Repair, Parking Violation, Meter
    ("houston",        "hou",  ["45162", "45156", "69758"], False),           # Traffic Signs, Parking Violation, Bandit Sign
    ("new-haven",      "nhv",  ["373", "121", "372"], False),                  # Signs, Parking Violation, Meter
    ("albuquerque",    "abq",  ["50464"], False),                              # Parking
    ("tucson",         "tuc",  ["26526", "26531"], False),                     # Signs, Traffic Markings
    ("bridgeport",     "bpt",  ["16874"], False),                              # Street Sign Issue
    ("toledo",         "tol",  ["17751"], False),                              # Street Sign Down/Damaged
    ("albany_2",       "alb",  ["3408", "3410", "3401"], False),              # Street Sign, Street Cleaning, Parking Enf
    ("jersey-city",    "jc",   ["36777", "34345"], False),                     # Street Signage, Handicap Parking
    ("newark",         "nwk",  ["27345"], False),                              # Signal/Signage/Striping
    ("tuscaloosa",     "tsc",  ["45535", "45917", "45915"], False),           # Street Signs, Illegal/Code Parking
    ("schenectady",    "sch",  ["35452"], False),                              # Traffic/Street Sign Issue
    ("abilene",        "abi",  ["24511"], False),                              # Street Sign
    ("macon_bibb_county_ga", "mac", ["2475"], False),                         # Traffic Signals & Signs
    ("peoria_2",       "peo",  ["45477", "45485", "45496"], False),           # Non-Code/Temp Signs
]
SIGN_KW = ("sign", "parking", "curb", "meter", "tow", "loading")
STATUS = "open,acknowledged,closed,archived"


def get_json(url: str, tries: int = 3) -> dict:
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as e:
            if t == tries - 1:
                print(f"   api fail: {str(e)[:80]}", flush=True)
                return {}
            time.sleep(2 * (t + 1))
    return {}


def download(url: str, dest: Path) -> bool:
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


def want(issue: dict, req_ids: list[str] | None) -> bool:
    if not (issue.get("media") or {}).get("image_full"):
        return False
    if req_ids is None:
        title = ((issue.get("request_type") or {}).get("title") or "").lower()
        return any(k in title for k in SIGN_KW)
    return True


def harvest_city(place: str, prefix: str, req_ids, is_ca: bool, cap: int, seen: set, mf) -> int:
    out = IMG / prefix
    out.mkdir(parents=True, exist_ok=True)
    params = {"place_url": place, "status": STATUS, "per_page": "200"}
    if req_ids:
        params["request_types"] = ",".join(req_ids)
    url = f"{API}?{urllib.parse.urlencode(params)}"
    got = 0
    page = 0
    while url and got < cap:
        page += 1
        d = get_json(url)
        issues = d.get("issues") or []
        if not issues:
            break
        batch = []
        for i in issues:
            iid = str(i.get("id"))
            if iid in seen or not want(i, req_ids):
                continue
            seen.add(iid)
            batch.append(i)
        # download this page's images inline (signed URLs are single-use)
        def fetch(it):
            iid = str(it["id"])
            dest = out / f"{prefix}_{iid}.jpg"
            ok = download(it["media"]["image_full"], dest)
            if ok:
                return {"id": f"{prefix}_{iid}", "city": place, "is_ca": is_ca,
                        "request_type": (it.get("request_type") or {}).get("title"),
                        "url": it["media"]["image_full"].split("?")[0],
                        "address": it.get("address"),
                        "image": str(dest.relative_to(ROOT))}
            return None
        with ThreadPoolExecutor(max_workers=8) as ex:
            for rec in ex.map(fetch, batch):
                if rec:
                    mf.write(json.dumps(rec) + "\n")
                    got += 1
        mf.flush()
        url = (d.get("metadata", {}).get("pagination", {}) or {}).get("next_page_url")
        print(f"   {place} p{page}: +{got} images", flush=True)
        time.sleep(1.2)  # SCF rate-limits aggressively (429 -> 403); throttle listing calls
    print(f"  {place}: {got} sign photos", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-city", type=int, default=4000)
    ap.add_argument("--cities", default="", help="comma-separated prefixes to limit (e.g. oak,sf)")
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if MANIFEST.exists():
        for l in MANIFEST.open():
            try:
                seen.add(json.loads(l)["id"].split("_", 1)[1])
            except Exception:
                pass
    only = set(args.cities.split(",")) if args.cities else None
    total = 0
    with MANIFEST.open("a") as mf:
        for place, prefix, req_ids, is_ca in CITIES:
            if only and prefix not in only:
                continue
            print(f"== {place} ({'CA' if is_ca else 'non-CA'}) ==", flush=True)
            total += harvest_city(place, prefix, req_ids, is_ca, args.max_per_city, seen, mf)
    print(f"TOTAL downloaded: {total} -> {IMG}", flush=True)
    print(f"manifest: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
