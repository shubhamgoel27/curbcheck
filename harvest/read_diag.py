"""Component-level reading diagnostic on hard sign samples.

For each difficult image, render a few image transforms (the cheap preprocessing
that often rescues faded/low-contrast text), run Apple Vision OCR on each, and
record what text each variant recovers. Saves variants + a results.json for a
/craft report. The VLM read + human-legibility ground truth are added by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from ocrmac import ocrmac

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "diag"
SAMPLES = [
    "data/images_scf/sf/sf_13900373.jpg",   # known faded 2HR permit
    "data/images_scf/sf/sf_12563307.jpg",
    "data/images_scf/oak/oak_17919312.jpg",
    "data/images_v3/dpw_872740a3cc.jpg",     # dense, votes disagreed on count
    "data/images_v3/dpw_0c5a3ba610.jpg",
    "data/images_scf/oak/oak_17678584.jpg",
    "data/images_v3/dpw_92b77f9bec.jpg",
    "data/images_v3/dpw_8e2ece0276.jpg",
]


def upscale(im: Image.Image, f=2.0) -> Image.Image:
    return im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)


def gray_autocontrast(im: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(im.convert("L"), cutoff=2).convert("RGB")


def clahe(im: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(arr)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    merged = cv2.merge((cl, a, b))
    return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))


def sharpen(im: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(ImageEnhance.Contrast(im).enhance(1.4)).enhance(2.0)


TRANSFORMS = {
    "original": lambda im: im,
    "upscale2x": upscale,
    "gray_autocontrast": lambda im: upscale(gray_autocontrast(im)),
    "clahe": lambda im: upscale(clahe(im)),
    "sharpen": lambda im: upscale(sharpen(im)),
}


def ocr_text(path: Path) -> str:
    try:
        res = ocrmac.OCR(str(path), recognition_level="accurate").recognize()
        toks = [t for t, c, _ in res if c > 0.3]
        return " | ".join(toks)
    except Exception as e:
        return f"<ocr error: {str(e)[:60]}>"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for rel in SAMPLES:
        src = ROOT / rel
        if not src.exists():
            print("missing", rel)
            continue
        iid = src.stem
        d = OUT / iid
        d.mkdir(exist_ok=True)
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        variants = {}
        for name, fn in TRANSFORMS.items():
            out_im = fn(im)
            out_im.thumbnail((1200, 1200))
            vp = d / f"{name}.jpg"
            out_im.save(vp, quality=88)
            variants[name] = {"image": str(vp.relative_to(ROOT)), "ocr": ocr_text(vp)}
            print(f"{iid} [{name}]: {variants[name]['ocr'][:90]}")
        results.append({"id": iid, "src": rel, "variants": variants})
        print()
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"-> {OUT/'results.json'} ({len(results)} samples)")


if __name__ == "__main__":
    main()
