"""Pilot harvest: pull a small sample of real SF parking-sign photos from
the two cleanest sources, so we can eyeball quality before building anything.

Sources:
  1. DPW Street Space Permit Photos (PDDL, city-taken):
     https://data.sfgov.org/resource/pigs-fac7.json
  2. SF 311 "MTA Parking Traffic Signs" cases (citizen photos, URL-list tier):
     https://data.sfgov.org/resource/vw6y-z8j6.json

Usage: python harvest/pilot.py [--n 6]
Writes to data/pilot/{dpw,311_<subtype>}/ and a manifest.json with provenance.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pilot"

SODA_311 = "https://data.sfgov.org/resource/vw6y-z8j6.json"
SODA_DPW = "https://data.sfgov.org/resource/pigs-fac7.json"
UA = {"User-Agent": "curbcheck-pilot/0.1 (research; shubhamgoel27@gmail.com)"}

PARKING_SUBTYPES = ["street_cleaning", "no_parking", "permit_parking", "tow_away"]


def fetch_json(url: str, params: dict) -> list[dict]:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


MAGIC = (b"\xff\xd8\xff", b"\x89PNG")  # jpeg, png

# Hosts known to serve actual images vs auth-walled form attachments.
BAD_HOSTS = ("verintcloudservices.com",)


def download(url: str, dest: pathlib.Path) -> bool:
    if any(h in url for h in BAD_HOSTS):
        return False
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if not data.startswith(MAGIC[0]) and not data.startswith(MAGIC[1]):
            return False  # html error page or other non-image
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    FAIL {url[:80]} ({e})")
        return False


def harvest_dpw(n: int, manifest: list[dict]) -> None:
    print(f"== DPW permit photos (PDDL), n={n} ==")
    rows = fetch_json(SODA_DPW, {"$limit": n * 2, "$order": "dateadded DESC"})
    out = OUT / "dpw"
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for row in rows:
        if got >= n:
            break
        url = row.get("filename", "")
        if not url.startswith("http"):
            continue
        dest = out / f"dpw_{row['permit'].replace('/', '-')}.jpg"
        if download(url, dest):
            got += 1
            manifest.append({"file": str(dest.relative_to(ROOT)), "source": "dpw_permit",
                             "license": "PDDL", "url": url, "permit": row["permit"],
                             "date": row.get("dateadded")})
    print(f"   saved {got}")


def harvest_311(subtype: str, n: int, manifest: list[dict]) -> None:
    print(f"== 311 {subtype}, n={n} ==")
    rows = fetch_json(SODA_311, {
        "$limit": n * 3,
        "$order": "requested_datetime DESC",
        "$where": ("service_name like 'MTA Parking Traffic Signs%' "
                   f"AND service_subtype='{subtype}' AND media_url IS NOT NULL"),
    })
    out = OUT / f"311_{subtype}"
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for row in rows:
        if got >= n:
            break
        media = row.get("media_url") or {}
        url = media.get("url") if isinstance(media, dict) else media
        if not url:
            continue
        dest = out / f"311_{row['service_request_id']}.jpg"
        if download(url, dest):
            got += 1
            manifest.append({"file": str(dest.relative_to(ROOT)), "source": "sf311",
                             "license": "url-reference-only", "url": url,
                             "case": row["service_request_id"], "subtype": subtype,
                             "address": row.get("address"),
                             "date": row.get("requested_datetime")})
    print(f"   saved {got}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="images per source/subtype")
    args = ap.parse_args()
    manifest: list[dict] = []
    harvest_dpw(args.n, manifest)
    for st in PARKING_SUBTYPES:
        harvest_311(st, args.n, manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ntotal: {len(manifest)} images, manifest at data/pilot/manifest.json")


if __name__ == "__main__":
    main()
