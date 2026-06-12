"""Full harvest: manifests for every source + capped image downloads.

Strategy per the licensing map (notes/sources.md):
  - SFMTA sign inventory: full tabular pull (PDDL, drives the synthetic generator)
  - DPW permit photos: full manifest + capped download (PDDL, Tier 1)
  - SF 311 parking cases: full manifest (URL list, Tier 2) + capped per-subtype download
  - Wikimedia Commons: CA/US parking sign categories with per-file license (Tier 1)

Downloads are resumable: existing valid files are skipped.

Usage: python harvest/full.py [--dpw 200] [--per-subtype 50] [--commons 120]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMG = ROOT / "data" / "images"

UA = {"User-Agent": "curbcheck-harvest/0.1 (research; shubhamgoel27@gmail.com)"}
SODA_311 = "https://data.sfgov.org/resource/vw6y-z8j6.json"
SODA_DPW = "https://data.sfgov.org/resource/pigs-fac7.json"
SODA_SIGNS = "https://data.sfgov.org/resource/m48z-6ji4.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

MAGIC = (b"\xff\xd8\xff", b"\x89PNG")
PARKING_SUBTYPES = ["street_cleaning", "no_parking", "permit_parking", "tow_away",
                    "other_parking_type"]


def get(url: str, params: dict | None = None) -> bytes:
    q = f"?{urllib.parse.urlencode(params)}" if params else ""
    req = urllib.request.Request(f"{url}{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def get_json(url: str, params: dict) -> list | dict:
    return json.loads(get(url, params))


def save_image(url: str, dest: pathlib.Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True  # resumable
    if "verintcloudservices.com" in url:
        return False
    try:
        data = get(url)
        if not data.startswith(MAGIC[0]) and not data.startswith(MAGIC[1]):
            return False
        dest.write_bytes(data)
        time.sleep(0.15)  # be polite
        return True
    except Exception:
        return False


def harvest_sfmta_inventory() -> None:
    """Full sign inventory, paginated. PDDL."""
    out = RAW / "sfmta_inventory.jsonl"
    print("== SFMTA sign inventory ==", flush=True)
    rows_written = 0
    with out.open("w") as f:
        offset = 0
        while True:
            batch = get_json(SODA_SIGNS, {"$limit": 50000, "$offset": offset})
            if not batch:
                break
            for row in batch:
                f.write(json.dumps(row) + "\n")
            rows_written += len(batch)
            offset += 50000
            print(f"   {rows_written} rows", flush=True)
    print(f"   done: {rows_written} signs -> {out.name}", flush=True)


def harvest_dpw(limit: int) -> None:
    print("== DPW permit photos ==", flush=True)
    manifest = RAW / "dpw_manifest.jsonl"
    rows = get_json(SODA_DPW, {"$limit": 50000, "$order": "dateadded DESC"})
    with manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"   manifest: {len(rows)} rows", flush=True)
    out = IMG / "dpw"
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for row in rows:
        if got >= limit:
            break
        url = row.get("filename", "")
        if not url.startswith("http"):
            continue
        # permits have multiple photos; use the upload filename to avoid collisions
        uniq = url.rsplit("/", 1)[-1].replace("/", "-")[:90]
        if save_image(url, out / f"dpw_{uniq}"):
            got += 1
            if got % 25 == 0:
                print(f"   {got}/{limit} images", flush=True)
    print(f"   done: {got} images", flush=True)


def harvest_311(per_subtype: int) -> None:
    print("== SF 311 parking sign cases ==", flush=True)
    manifest = RAW / "sf311_manifest.jsonl"
    total = 0
    with manifest.open("w") as f:
        for st in PARKING_SUBTYPES:
            offset = 0
            while True:
                batch = get_json(SODA_311, {
                    "$limit": 5000, "$offset": offset,
                    "$order": "requested_datetime DESC",
                    "$where": ("service_name like 'MTA Parking Traffic Signs%' "
                               f"AND service_subtype='{st}' AND media_url IS NOT NULL"),
                })
                if not batch:
                    break
                for row in batch:
                    media = row.get("media_url") or {}
                    url = media.get("url") if isinstance(media, dict) else media
                    if url and "cloudinary" in url:
                        f.write(json.dumps({
                            "case": row.get("service_request_id"), "subtype": st,
                            "url": url, "address": row.get("address"),
                            "lat": row.get("lat"), "long": row.get("long"),
                            "date": row.get("requested_datetime"),
                            "status_notes": row.get("status_notes"),
                        }) + "\n")
                        total += 1
                offset += 5000
            print(f"   {st}: manifest at {total} cumulative", flush=True)
    print(f"   manifest: {total} live-URL cases", flush=True)

    # capped downloads per subtype, newest first
    entries = [json.loads(l) for l in manifest.open()]
    for st in PARKING_SUBTYPES:
        out = IMG / f"311_{st}"
        out.mkdir(parents=True, exist_ok=True)
        got = 0
        for e in (x for x in entries if x["subtype"] == st):
            if got >= per_subtype:
                break
            if save_image(e["url"], out / f"311_{e['case']}.jpg"):
                got += 1
        print(f"   {st}: {got} images", flush=True)


def harvest_commons(limit: int) -> None:
    print("== Wikimedia Commons ==", flush=True)
    cats = ["Category:Parking signs in California",
            "Category:No parking signs in California",
            "Category:Parking signs in the United States",
            "Category:No parking signs in the United States"]
    manifest = RAW / "commons_manifest.jsonl"
    out = IMG / "commons"
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    with manifest.open("w") as f:
        for cat in cats:
            params = {
                "action": "query", "format": "json",
                "generator": "categorymembers", "gcmtitle": cat,
                "gcmtype": "file", "gcmlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata", "iiurlwidth": 1280,
            }
            try:
                data = get_json(COMMONS_API, params)
            except Exception as e:
                print(f"   {cat}: fetch failed ({e})", flush=True)
                continue
            pages = (data.get("query") or {}).get("pages", {})
            for p in pages.values():
                if got >= limit:
                    break
                ii = (p.get("imageinfo") or [{}])[0]
                meta = ii.get("extmetadata", {})
                lic = (meta.get("LicenseShortName") or {}).get("value", "unknown")
                if "SVG" in p.get("title", "") or p["title"].lower().endswith(".svg"):
                    continue  # diagrams, not photos
                url = ii.get("thumburl") or ii.get("url")
                if not url:
                    continue
                name = p["title"].replace("File:", "").replace("/", "-")[:80]
                if save_image(url, out / name):
                    got += 1
                    f.write(json.dumps({"title": p["title"], "url": ii.get("url"),
                                        "license": lic, "category": cat}) + "\n")
            print(f"   {cat}: cumulative {got}", flush=True)
    print(f"   done: {got} images", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpw", type=int, default=200)
    ap.add_argument("--per-subtype", type=int, default=50)
    ap.add_argument("--commons", type=int, default=120)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    harvest_sfmta_inventory()
    harvest_dpw(args.dpw)
    harvest_311(args.per_subtype)
    harvest_commons(args.commons)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
