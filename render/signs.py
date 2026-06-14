"""Synthetic SF parking sign stack renderer, v1.

Draws CA-style sign plates (R26 no-parking/no-stopping, R30 time-limit with RPP
exemption, R32 street cleaning, tow-away) with PIL, stacks 1-4 on a pole, applies
light photometric/geometric jitter. Returns (PIL.Image, SignStack) so every pixel
ships with exact ground truth.

v1 is deliberately clean-looking; realism knobs (backgrounds, fading, stickers)
come after the pipeline proves out.
"""

from __future__ import annotations

import random
import sys
from datetime import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema.rules import Day, Kind, Restriction, SignStack, Window, WEEKDAYS

RED = (190, 30, 45)
GREEN = (0, 105, 62)
BLACK = (25, 25, 25)
WHITE = (252, 252, 250)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


DAY_ABBR = {Day.MON: "MON", Day.TUE: "TUE", Day.WED: "WED", Day.THU: "THU",
            Day.FRI: "FRI", Day.SAT: "SAT", Day.SUN: "SUN"}


def fmt_time(t: time) -> str:
    h = t.hour % 12 or 12
    ampm = "AM" if t.hour < 12 else "PM"
    return f"{h}{ampm}" if t.minute == 0 else f"{h}:{t.minute:02d}{ampm}"


def fmt_days(days: frozenset[Day]) -> str:
    if days == WEEKDAYS:
        return "MON THRU FRI"
    if len(days) == 7:
        return "EVERY DAY"
    ordered = sorted(days, key=lambda d: d.value)
    if len(ordered) == 1:
        return DAY_ABBR[ordered[0]]
    vals = [d.value for d in ordered]
    if vals == list(range(vals[0], vals[-1] + 1)):  # contiguous
        return f"{DAY_ABBR[ordered[0]]} THRU {DAY_ABBR[ordered[-1]]}"
    return " & ".join(DAY_ABBR[d] for d in ordered)


def plate_lines(r: Restriction) -> list[tuple[str, tuple, float]]:
    """(text, color, relative size) lines for one sign plate."""
    w = r.window
    span = f"{fmt_time(w.start)} TO {fmt_time(w.end)}"
    days = fmt_days(w.days)
    if r.kind is Kind.STREET_CLEANING:
        return [("NO PARKING", RED, 1.0), (span, BLACK, 0.72), (days, BLACK, 0.72),
                ("STREET CLEANING", RED, 0.55)]
    if r.kind is Kind.NO_PARKING:
        if len(w.days) == 7 and w.start == time(0) and w.end == time(23, 59):
            return [("NO", RED, 1.2), ("PARKING", RED, 1.2), ("ANY", RED, 0.9), ("TIME", RED, 0.9)]
        return [("NO PARKING", RED, 1.0), (span, BLACK, 0.72), (days, BLACK, 0.72)]
    if r.kind is Kind.NO_STOPPING:
        head = [("TOW-AWAY", RED, 0.6)] if r.tow else []
        return head + [("NO STOPPING", RED, 0.95), (span, BLACK, 0.72), (days, BLACK, 0.72)]
    if r.kind is Kind.TOW_AWAY:
        return [("TOW-AWAY", RED, 0.8), ("NO STOPPING", RED, 0.8), (span, BLACK, 0.72),
                (days, BLACK, 0.72)]
    if r.kind is Kind.PERMIT_EXEMPT_LIMIT:
        hrs = r.limit_minutes // 60
        return [(f"{hrs} HOUR", GREEN, 1.0), ("PARKING", GREEN, 1.0),
                (span, BLACK, 0.62), (days, BLACK, 0.62),
                ("EXCEPT VEHICLES", BLACK, 0.45),
                (f"WITH AREA {r.permit_area} PERMIT", BLACK, 0.45)]
    if r.kind is Kind.TIME_LIMIT:
        if r.limit_minutes < 60:
            head = [(f"{r.limit_minutes} MINUTE", GREEN, 0.9)]
        else:
            head = [(f"{r.limit_minutes // 60} HOUR", GREEN, 1.0)]
        return head + [("PARKING", GREEN, 1.0), (span, BLACK, 0.62), (days, BLACK, 0.62)]
    if r.kind is Kind.LOADING_ONLY:
        return [("PASSENGER", BLACK, 0.8), ("LOADING ONLY", BLACK, 0.8),
                (span, BLACK, 0.62), (days, BLACK, 0.62)]
    raise ValueError(r.kind)


def draw_plate(r: Restriction, width: int = 360) -> Image.Image:
    lines = plate_lines(r)
    pad, gap = 26, 12
    base = 52
    sizes = [int(base * s) for _, _, s in lines]
    heights = [int(sz * 1.15) for sz in sizes]
    h = pad * 2 + sum(heights) + gap * (len(lines) - 1)
    h = max(h, int(width * 1.35))  # CA parking plates are portrait (12x18 ratio)
    img = Image.new("RGB", (width, h), WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, width - 4, h - 4], outline=(120, 120, 120), width=2)
    content = sum(heights) + gap * (len(lines) - 1)
    y = (h - content) // 2  # vertically center on the portrait plate
    for (text, color, _), sz, lh in zip(lines, sizes, heights):
        f = font(sz)
        # shrink to fit
        while d.textlength(text, font=f) > width - 2 * pad and sz > 12:
            sz -= 2
            f = font(sz)
        x = (width - d.textlength(text, font=f)) / 2
        d.text((x, y), text, fill=color, font=f)
        y += lh + gap
    return img


# ---------------- stack sampling ----------------

def rand_window(rng: random.Random, kind: Kind) -> Window:
    if kind is Kind.STREET_CLEANING:
        start_h = rng.choice([6, 7, 8, 9, 10, 12])
        day_opts = [frozenset({d}) for d in Day] + [frozenset({Day.TUE, Day.THU}),
                                                    frozenset({Day.MON, Day.WED, Day.FRI})]
        return Window(rng.choice(day_opts), time(start_h), time(start_h + 2))
    if kind in (Kind.NO_STOPPING, Kind.TOW_AWAY):
        start_h, dur = rng.choice([(7, 2), (15, 3), (16, 2), (16, 3)])
        return Window(WEEKDAYS, time(start_h), time(start_h + dur))
    if kind is Kind.NO_PARKING and rng.random() < 0.3:
        return Window(frozenset(Day), time(0), time(23, 59))  # any time
    start_h = rng.choice([8, 9])
    end_h = rng.choice([18, 21, 22])
    days = rng.choice([WEEKDAYS, frozenset(set(WEEKDAYS) | {Day.SAT}), frozenset(Day)])
    return Window(days, time(start_h), time(end_h))


def sample_restriction(rng: random.Random, kind: Kind) -> Restriction:
    w = rand_window(rng, kind)
    if kind is Kind.PERMIT_EXEMPT_LIMIT:
        return Restriction(kind, w, limit_minutes=rng.choice([60, 120, 240]),
                           permit_area=rng.choice("ABCDEFGHIJKLMNOPQRSUVWXYZ"))
    if kind is Kind.TIME_LIMIT:
        return Restriction(kind, w, limit_minutes=rng.choice([30, 60, 120]))
    if kind is Kind.TOW_AWAY:
        return Restriction(kind, w, tow=True)
    if kind is Kind.NO_STOPPING:
        return Restriction(kind, w, tow=rng.random() < 0.5)
    return Restriction(kind, w)


# rough SF prevalence (SFMTA inventory: R30 time-limit/permit dominant, R32 cleaning next)
KIND_WEIGHTS = [
    (Kind.PERMIT_EXEMPT_LIMIT, 0.30), (Kind.STREET_CLEANING, 0.25),
    (Kind.TIME_LIMIT, 0.15), (Kind.NO_STOPPING, 0.12), (Kind.NO_PARKING, 0.08),
    (Kind.TOW_AWAY, 0.06), (Kind.LOADING_ONLY, 0.04),
]


def sample_stack(rng: random.Random, n_signs: int | None = None,
                 size_weights: tuple = (0.3, 0.4, 0.2, 0.1),
                 kind_weights: list | None = None) -> SignStack:
    n = n_signs or rng.choices([1, 2, 3, 4], weights=list(size_weights))[0]
    kinds = []
    kw = kind_weights or KIND_WEIGHTS
    pool = [k for k, _ in kw]
    weights = [w for _, w in kw]
    while len(kinds) < n:
        k = rng.choices(pool, weights=weights)[0]
        if k not in kinds:  # one of each kind per pole, like real poles mostly
            kinds.append(k)
    return SignStack(restrictions=[sample_restriction(rng, k) for k in kinds])


def render_stack(stack: SignStack, rng: random.Random) -> Image.Image:
    plates = [draw_plate(r) for r in stack.restrictions]
    gap, pole_w = 18, 26
    w = max(p.width for p in plates) + 120
    h = sum(p.height for p in plates) + gap * (len(plates) + 3) + 80
    bg = tuple(rng.randint(168, 215) for _ in range(3))
    canvas = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(canvas)
    pole_x = w // 2
    d.rectangle([pole_x - pole_w // 2, 0, pole_x + pole_w // 2, h], fill=(110, 112, 115))
    y = gap * 2
    for p in plates:
        canvas.paste(p, (pole_x - p.width // 2, y))
        y += p.height + gap
    angle = rng.uniform(-5, 5)
    canvas = canvas.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)
    return canvas
