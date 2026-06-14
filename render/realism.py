"""Realism augmentations: turn a clean rendered sign stack into something that
looks like a real SF 311 photo. Attacks the sim-to-real gap (0.97 synth read F1
vs 0.33 real).

Pipeline (each step probabilistic):
  street background (sky gradient + building silhouettes)  ->  perspective skew
  ->  weathering (fade, color cast, rust, scratches)  ->  occlusion (sticker,
  branch, glare)  ->  camera (blur, noise, JPEG, vignette, exposure).

Ground truth is unchanged: we only degrade pixels, never the SignStack.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _sky_bg(w: int, h: int, rng: random.Random) -> Image.Image:
    top = (rng.randint(70, 130), rng.randint(140, 190), rng.randint(200, 240))
    bot = (rng.randint(200, 235), rng.randint(215, 240), rng.randint(225, 248))
    bg = Image.new("RGB", (w, h))
    px = bg.load()
    for y in range(h):
        t = y / h
        px_row = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = px_row
    d = ImageDraw.Draw(bg)
    # building silhouettes along the bottom third
    bx = 0
    while bx < w:
        bw = rng.randint(w // 8, w // 3)
        bh = rng.randint(h // 5, h // 2)
        shade = rng.randint(150, 200)
        d.rectangle([bx, h - bh, bx + bw, h], fill=(shade, shade - 8, shade - 16))
        bx += bw + rng.randint(0, 12)
    return bg


def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    m = rng.uniform(0.04, 0.13)  # skew strength
    dx, dy = w * m, h * m
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, dx), rng.uniform(0, dy)),
           (w - rng.uniform(0, dx), rng.uniform(0, dy)),
           (w - rng.uniform(0, dx), h - rng.uniform(0, dy)),
           (rng.uniform(0, dx), h - rng.uniform(0, dy))]
    # solve the 8 coeffs for PIL's PERSPECTIVE transform
    import numpy as np
    A = []
    for (xs, ys), (xd, yd) in zip(dst, src):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
    B = [c for pt in src for c in pt]
    coeffs = np.linalg.solve(np.array(A), np.array(B))
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                         fillcolor=(128, 132, 135))


def _weather(plate_region: Image.Image, rng: random.Random) -> Image.Image:
    img = plate_region
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.55, 0.95))   # fade
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 0.95))       # desaturate
    cast = Image.new("RGB", img.size, (rng.randint(150, 210), rng.randint(140, 200), rng.randint(110, 170)))
    img = Image.blend(img, cast, rng.uniform(0.04, 0.16))              # sun/age color cast
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.randint(0, 40)):  # rust / dirt specks
        x, y = rng.randint(0, img.width), rng.randint(0, img.height)
        r = rng.randint(1, 3)
        d.ellipse([x, y, x + r, y + r], fill=(rng.randint(120, 160), 70, 40, rng.randint(40, 120)))
    for _ in range(rng.randint(0, 3)):   # scratches
        x0, y0 = rng.randint(0, img.width), rng.randint(0, img.height)
        d.line([x0, y0, x0 + rng.randint(-40, 40), y0 + rng.randint(-40, 40)],
               fill=(200, 200, 195, rng.randint(60, 140)), width=1)
    return img


def _occlude(img: Image.Image, rng: random.Random) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    if rng.random() < 0.4:  # sticker
        x, y = rng.randint(0, img.width), rng.randint(int(img.height * 0.2), img.height)
        s = rng.randint(20, 55)
        col = rng.choice([(230, 230, 230), (40, 40, 40), (200, 60, 60), (60, 90, 200)])
        d.rectangle([x, y, x + s, y + int(s * 0.7)], fill=col + (rng.randint(180, 240),))
    if rng.random() < 0.3:  # branch / shadow streak
        x = rng.randint(0, img.width)
        d.line([x, 0, x + rng.randint(-60, 60), img.height], fill=(20, 30, 15, rng.randint(60, 130)),
               width=rng.randint(6, 18))
    if rng.random() < 0.25:  # glare blob
        x, y = rng.randint(0, img.width), rng.randint(0, img.height // 2)
        r = rng.randint(30, 80)
        d.ellipse([x, y, x + r, y + r], fill=(255, 255, 250, rng.randint(40, 90)))


def _camera(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.25))
    # vignette
    v = Image.new("L", img.size, 0)
    dv = ImageDraw.Draw(v)
    dv.ellipse([-img.width * 0.2, -img.height * 0.2, img.width * 1.2, img.height * 1.2], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(img.width // 8))
    black = Image.new("RGB", img.size, (0, 0, 0))
    img = Image.composite(img, black, v.point(lambda p: int(p * 0.5 + 128)))
    if rng.random() < 0.5:  # sensor noise
        import numpy as np
        arr = np.array(img).astype("int16")
        arr += rng.randint(4, 14) * np.random.randn(*arr.shape).astype("int16")
        img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    return img


def augment(clean: Image.Image, rng: random.Random, plate_box=None) -> Image.Image:
    """Apply the full realism pipeline to a clean rendered stack.

    Composites the rendered sign+pole over a sky/building background using a tight
    mask, then weathers, skews, occludes, and applies camera effects.
    """
    w, h = clean.size
    bg = _sky_bg(w, h, rng)
    mask = _plate_mask(clean)
    out = Image.composite(clean, bg, mask)   # signs on a real-ish sky first
    out = _weather(out, rng)                  # weather the whole scene together
    _occlude(out, rng)
    out = _perspective(out, rng)
    out = _camera(out, rng)
    return out


def _plate_mask(clean: Image.Image) -> Image.Image:
    """Mask of the non-background (sign + pole) pixels from the clean render."""
    import numpy as np
    arr = np.array(clean.convert("RGB"))
    # clean bg is a flat-ish pastel; signs are white plates + gray pole + red/green text
    # treat near-white plates and dark pole as foreground
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    white = (r > 235) & (g > 235) & (b > 225)
    pole = (abs(r.astype(int) - g) < 18) & (abs(g.astype(int) - b) < 18) & (r > 90) & (r < 170)
    # also capture the dark plate borders / colored text (anything not the pastel bg):
    # the clean bg is a flat pastel; foreground deviates from its own corner color
    corner = arr[0, 0].astype(int)
    dist = (abs(r.astype(int) - corner[0]) + abs(g.astype(int) - corner[1]) +
            abs(b.astype(int) - corner[2]))
    nonbg = dist > 40
    fg = (white | pole | nonbg).astype("uint8") * 255
    img = Image.fromarray(fg)
    # close holes so plate interiors are solid, then tighten
    return img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
