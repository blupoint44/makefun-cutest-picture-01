"""Cutest picture v3: EXTREME close-up baby red panda face (~80% of frame).

Direct gaze, huge glossy amber eyes, tiny pink nose, small open smile,
one paw reaching toward the viewer at the bottom-right corner, a green
leaf resting on its head, soft warm pastel bokeh background.
Deterministic, pure PIL + numpy, renders at 2048 -> 1024.
Output: entry.png beside this file.
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2
W = 1024 * S
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "entry.png")
rng = random.Random(20261113)
nprng = np.random.default_rng(20261113)

YN, XN = np.mgrid[0:W, 0:W].astype(np.float32) / W

ORANGE_TOP = (243, 150, 78)
ORANGE_BOT = (198, 98, 44)
FACE_TOP = (255, 251, 240)
FACE_BOT = (248, 230, 200)
BROWN = (74, 40, 22)


def sc(box):
    return [v * S for v in box]


def ellipse(d, box, **kw):
    d.ellipse(sc(box), **kw)


def soft_layer(draw_fn, blur):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    draw_fn(d)
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur * S))
    return layer


def shaded(base, mask, c_top, c_bot, y0=0.0, y1=1.0,
           light=(0.42, 0.30), glow=0.38, sigma=0.18):
    t = np.clip((YN - y0) / max(y1 - y0, 1e-6), 0, 1) ** 1.15
    arr = np.zeros((W, W, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = c_top[c] + (c_bot[c] - c_top[c]) * t
    d2 = (XN - light[0]) ** 2 + (YN - light[1]) ** 2
    g = np.exp(-d2 / (2 * sigma ** 2)) * glow
    warm = np.array([255, 228, 178], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] += (warm[c] - arr[:, :, c]) * g
    img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    img.putalpha(mask.filter(ImageFilter.GaussianBlur(1.2 * S)))
    base.alpha_composite(img)


def inner_shadow(base, mask, strength=90, blur=14, color=(120, 52, 18)):
    eroded = mask.filter(ImageFilter.MinFilter(31))
    edge = np.clip(np.asarray(mask, dtype=np.int16) - np.asarray(eroded, dtype=np.int16), 0, 255)
    edge_img = Image.fromarray(edge.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(blur * S))
    layer = Image.new("RGBA", (W, W), color + (0,))
    layer.putalpha(edge_img.point(lambda v: v * strength // 255))
    base.alpha_composite(layer)


def rim_light(base, mask, strength=110, color=(255, 234, 196)):
    eroded = mask.filter(ImageFilter.MinFilter(21))
    edge = np.clip(np.asarray(mask, dtype=np.int16) - np.asarray(eroded, dtype=np.int16), 0, 255)
    weight = np.clip(0.75 - (XN - 0.5) * 0.9 - (YN - 0.45) * 1.1, 0, 1)
    rim = (edge.astype(np.float32) * weight).astype(np.uint8)
    rim_img = Image.fromarray(rim, "L").filter(ImageFilter.GaussianBlur(5 * S))
    layer = Image.new("RGBA", (W, W), color + (0,))
    layer.putalpha(rim_img.point(lambda v: v * strength // 255))
    base.alpha_composite(layer)


def fur(base, mask, colors, n, cx, cy, len_lo=7, len_hi=15, alpha=(24, 58),
        downward=0.45, width=3):
    marr = np.asarray(mask)
    ys, xs = np.nonzero(marr > 140)
    if len(xs) == 0:
        return
    idx = nprng.integers(0, len(xs), size=n)
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        ang = math.atan2(y - cy * S, x - cx * S) * downward + (math.pi / 2) * (1 - downward)
        ang += rng.uniform(-0.45, 0.45)
        ln = rng.uniform(len_lo, len_hi) * S
        col = colors[rng.randrange(len(colors))]
        a = rng.randint(*alpha)
        d.line([x, y, x + math.cos(ang) * ln, y + math.sin(ang) * ln],
               fill=col + (a,), width=max(2, int(width * S / 2)))
    layer = layer.filter(ImageFilter.GaussianBlur(0.8 * S))
    la = np.asarray(layer.split()[3]).astype(np.float32)
    ma = np.asarray(mask.filter(ImageFilter.GaussianBlur(2 * S))).astype(np.float32) / 255.0
    layer.putalpha(Image.fromarray((la * ma).astype(np.uint8), "L"))
    base.alpha_composite(layer)


def mask_of(draw_fn):
    m = Image.new("L", (W, W), 0)
    d = ImageDraw.Draw(m)
    draw_fn(d)
    return m


# ---------------------------------------------------------------- background

def make_background():
    top = np.array([255, 246, 220], dtype=np.float32)
    mid = np.array([253, 226, 186], dtype=np.float32)
    edge = np.array([240, 190, 148], dtype=np.float32)
    r = np.clip(np.sqrt((XN - 0.5) ** 2 + (YN - 0.42) ** 2) / 0.78, 0, 1)
    img = np.zeros((W, W, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = np.where(r < 0.5, top[c] + (mid[c] - top[c]) * (r / 0.5),
                                mid[c] + (edge[c] - mid[c]) * ((r - 0.5) / 0.5))
    return Image.fromarray(img.astype(np.uint8), "RGB").convert("RGBA")


def bokeh(base):
    bok = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(bok)
    for (x, y, r, a) in [(120, 130, 44, 80), (910, 110, 36, 75), (960, 420, 24, 60),
                         (60, 470, 28, 60), (200, 60, 20, 65), (830, 40, 18, 60),
                         (70, 800, 30, 45), (960, 760, 34, 45), (330, 90, 14, 55),
                         (690, 70, 15, 55)]:
        d.ellipse(sc([x - r, y - r, x + r, y + r]), fill=(255, 252, 232, a))
        d.ellipse(sc([x - r, y - r, x + r, y + r]), outline=(255, 255, 255, a // 2), width=3 * S)
    base.alpha_composite(bok.filter(ImageFilter.GaussianBlur(3 * S)))
    # a few very soft green leaf blurs at the corners (depth, not clutter)
    leaves = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaves)
    for (x, y, r) in [(40, 60, 90), (950, 300, 80), (30, 620, 70)]:
        ld.ellipse(sc([x - r, y - r // 2, x + r, y + r // 2]), fill=(140, 190, 110, 60))
    base.alpha_composite(leaves.filter(ImageFilter.GaussianBlur(24 * S)))


# ---------------------------------------------------------------- features

def draw_big_eye(layer, ex, ey, ew=96, eh=128):
    """Huge glossy amber eye: bright honey iris, modest pupil, big catchlights."""
    patch = Image.new("RGBA", (int(ew * 2 * S), int(eh * 2 * S)), (0, 0, 0, 0))
    pw, ph = patch.width, patch.height
    # iris gradient: deep amber top -> bright honey bottom
    grad = np.zeros((ph, pw, 3), dtype=np.float32)
    t = (np.arange(ph, dtype=np.float32) / ph)[:, None]
    stops = [(0.0, (96, 48, 20)), (0.45, (168, 96, 38)), (0.8, (226, 152, 66)), (1.0, (246, 188, 96))]
    for c in range(3):
        grad[:, :, c] = np.interp(t[:, 0], [s[0] for s in stops], [s[1][c] for s in stops])[:, None]
    pimg = Image.fromarray(grad.astype(np.uint8), "RGB").convert("RGBA")
    pmask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(pmask).ellipse([3, 3, pw - 3, ph - 3], fill=255)
    pimg.putalpha(pmask)
    patch.alpha_composite(pimg)
    pd = ImageDraw.Draw(patch)
    # dark rim ring
    pd.ellipse([3, 3, pw - 3, ph - 3], outline=(58, 30, 14, 255), width=5 * S)
    # radial iris fibres (barely-there texture)
    cxp, cyp = pw / 2, ph / 2
    for k in range(30):
        ang = k * (2 * math.pi / 30) + rng.uniform(-0.08, 0.08)
        r0, r1 = pw * 0.22, pw * rng.uniform(0.36, 0.45)
        pd.line([cxp + math.cos(ang) * r0, cyp + math.sin(ang) * r0 * 1.25,
                 cxp + math.cos(ang) * r1, cyp + math.sin(ang) * r1 * 1.25],
                fill=(255, 226, 160, 14), width=2 * S)
    # pupil (kept modest so the amber shows)
    pd.ellipse([pw * 0.325, ph * 0.22, pw * 0.675, ph * 0.72], fill=(26, 13, 8, 255))
    # catchlights: one big top-left, one medium bottom-right, one tiny
    pd.ellipse([pw * 0.23, ph * 0.13, pw * 0.47, ph * 0.33], fill=(255, 255, 255, 255))
    pd.ellipse([pw * 0.60, ph * 0.62, pw * 0.80, ph * 0.80], fill=(255, 255, 255, 240))
    pd.ellipse([pw * 0.16, ph * 0.68, pw * 0.28, ph * 0.79], fill=(255, 255, 255, 210))
    # star sparkle top-right of pupil
    sx, sy, sr = pw * 0.70, ph * 0.24, pw * 0.075
    pd.polygon([(sx, sy - sr), (sx + sr * 0.28, sy - sr * 0.28), (sx + sr, sy),
                (sx + sr * 0.28, sy + sr * 0.28), (sx, sy + sr),
                (sx - sr * 0.28, sy + sr * 0.28), (sx - sr, sy),
                (sx - sr * 0.28, sy - sr * 0.28)], fill=(255, 255, 255, 245))
    # warm bottom reflection arc
    pd.arc([pw * 0.12, ph * 0.50, pw * 0.88, ph * 0.96], start=25, end=155,
           fill=(255, 238, 205, 150), width=5 * S)
    layer.alpha_composite(patch, (int((ex - ew) * S), int((ey - eh) * S)))
    # soft upper lid line
    ld = ImageDraw.Draw(layer)
    ld.arc(sc([ex - ew + 5, ey - eh + 2, ex + ew - 5, ey + eh - 26]),
           start=182, end=358, fill=(90, 48, 24, 255), width=7 * S)


def draw_leaf(base, cx, cy, length=170, angle=-28):
    leaf = Image.new("RGBA", (420 * S, 200 * S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    # pointed leaf outline from a sine width profile
    x0, midy, maxw = 14, 100, 44
    top, bot = [], []
    for i in range(41):
        t = i / 40
        x = x0 + length * t
        wdt = maxw * (math.sin(math.pi * t) ** 0.75) * (1 - 0.12 * t)
        top.append((x * S, (midy - wdt) * S))
        bot.append((x * S, (midy + wdt) * S))
    ld.polygon(top + bot[::-1], fill=(104, 178, 82, 255))
    inner = [(x0 + 8 + (length - 16) * (i / 40),
              midy - (maxw * 0.58) * (math.sin(math.pi * (i / 40)) ** 0.75)) for i in range(41)]
    inner_b = [(x0 + 8 + (length - 16) * (i / 40),
                midy + (maxw * 0.58) * (math.sin(math.pi * (i / 40)) ** 0.75)) for i in range(41)]
    ld.polygon([(x * S, y * S) for (x, y) in inner] + [(x * S, y * S) for (x, y) in inner_b[::-1]],
               fill=(130, 202, 100, 255))
    ld.line([18 * S, 100 * S, (x0 + length - 6) * S, 100 * S], fill=(80, 136, 58, 110), width=3 * S)
    for k in range(1, 5):
        x = 26 + k * (length - 48) / 5
        ld.line([x * S, 100 * S, (x + 10) * S, 84 * S], fill=(80, 136, 58, 55), width=2 * S)
        ld.line([x * S, 100 * S, (x + 10) * S, 116 * S], fill=(80, 136, 58, 55), width=2 * S)
    leaf = leaf.filter(ImageFilter.GaussianBlur(1 * S)).rotate(angle, resample=Image.BICUBIC, expand=True)
    base.alpha_composite(leaf, (int((cx - 150) * S), int((cy - 80) * S)))


def paw_pads(base, cx, cy, scale=1.0, toes=((-34, -40), (0, -52), (34, -40))):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ellipse(d, [cx - 30 * scale, cy - 4 * scale, cx + 30 * scale, cy + 38 * scale], fill=(255, 205, 188, 255))
    for (tx, ty) in toes:
        ellipse(d, [cx + (tx - 15) * scale, cy + (ty - 17) * scale,
                    cx + (tx + 15) * scale, cy + (ty + 17) * scale], fill=(255, 205, 188, 255))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1 * S)))


# ---------------------------------------------------------------- scene

def main():
    img = make_background()
    bokeh(img)
    img.alpha_composite(soft_layer(lambda d: ellipse(d, [90, 40, 934, 880], fill=(255, 255, 255, 70)), 60))

    cx, cy = 512, 500

    # ears (behind head, tips peeking at the top corners)
    for ex in (196, 828):
        em = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 128, 30, ex + 128, 286]), fill=255))
        shaded(img, em, (200, 106, 54), (150, 74, 34), 0.03, 0.28, light=(0.35, 0.10), glow=0.3)
        fur(img, em, [(234, 132, 68), (164, 84, 38), (250, 172, 104)], 420, ex, 160)
        im = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 78, 84, ex + 78, 240]), fill=255))
        shaded(img, im, (255, 228, 204), (250, 198, 170), 0.08, 0.24, glow=0.25)
        pm = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 30, 126, ex + 30, 198]), fill=255))
        shaded(img, pm, (250, 180, 174), (240, 150, 148), 0.12, 0.20, glow=0.2)

    # head: fills ~85% of the frame width
    def head_shape(d):
        d.ellipse(sc([cx - 442, cy - 412, cx + 442, cy + 412]), fill=255)
        for (bx, by, br) in [(cx - 408, cy + 190, 46), (cx + 408, cy + 190, 46)]:
            d.ellipse(sc([bx - br, by - br, bx + br, by + br]), fill=255)
    hm = mask_of(head_shape)
    shaded(img, hm, ORANGE_TOP, ORANGE_BOT, 0.09, 0.90, light=(0.44, 0.26), glow=0.42, sigma=0.22)
    inner_shadow(img, hm, strength=85, blur=16)
    rim_light(img, hm, strength=115)

    # cream face patch (large, soft)
    fm = mask_of(lambda d: d.ellipse(sc([cx - 348, cy - 252, cx + 348, cy + 372]), fill=255))
    shaded(img, fm, FACE_TOP, FACE_BOT, 0.25, 0.88, light=(0.47, 0.36), glow=0.30, sigma=0.19)
    inner_shadow(img, fm, strength=30, blur=12, color=(196, 140, 84))

    ring = Image.fromarray(np.clip(np.asarray(hm, dtype=np.int16) - np.asarray(fm, dtype=np.int16), 0, 255).astype(np.uint8), "L")
    fur(img, ring, [(250, 168, 98), (216, 114, 54), (255, 200, 136), (174, 86, 38)],
        3000, cx, cy, len_lo=9, len_hi=19, alpha=(30, 64))
    fur(img, fm, [(255, 255, 250), (242, 220, 186)], 550, cx, cy,
        len_lo=6, len_hi=11, alpha=(9, 22))

    d = ImageDraw.Draw(img)
    # (brow spots removed: the blurred-ellipse trick rendered as grey smudges)

    # big soft blush + tiny freckles
    cheeks = soft_layer(lambda dd: (ellipse(dd, [cx - 330, cy + 108, cx - 168, cy + 196], fill=(247, 140, 140, 200)),
                                    ellipse(dd, [cx + 168, cy + 108, cx + 330, cy + 196], fill=(247, 140, 140, 200))), 13)
    img.alpha_composite(cheeks)
    d = ImageDraw.Draw(img)
    for (fx, fy) in [(cx - 268, cy + 138), (cx - 238, cy + 152), (cx - 292, cy + 158),
                     (cx + 268, cy + 138), (cx + 238, cy + 152), (cx + 292, cy + 158)]:
        ellipse(d, [fx - 5, fy - 5, fx + 5, fy + 5], fill=(214, 130, 92, 160))

    # HUGE eyes, direct gaze
    for ex in (cx - 168, cx + 168):
        draw_big_eye(img, ex, cy + 6)
    d = ImageDraw.Draw(img)

    # tiny pink nose: proper rounded-triangle (wide top, soft round tip),
    # not a floating oval -- top corners and tip are rounded with small
    # circles unioned into the mask so the muzzle reads naturally.
    def nose_shape(dd):
        # rounded-triangle / teardrop nose: a wide ellipse on top unioned
        # with a smaller ellipse below -- tapers down to a soft round tip,
        # no sharp corners and no floating-oval look
        dd.ellipse(sc([cx - 30, cy + 146, cx + 30, cy + 176]), fill=255)
        dd.ellipse(sc([cx - 18, cy + 158, cx + 18, cy + 190]), fill=255)
    nm = mask_of(nose_shape)
    shaded(img, nm, (232, 138, 128), (196, 94, 92), 0.64, 0.69, glow=0.25, sigma=0.05)
    d = ImageDraw.Draw(img)
    # glossy highlight on the upper-left of the nose
    ellipse(d, [cx - 15, cy + 150, cx - 3, cy + 159], fill=(255, 255, 255, 150))
    ellipse(d, [cx + 5, cy + 148, cx + 11, cy + 154], fill=(255, 255, 255, 70))
    # short philtrum line from nose tip down to the mouth
    d.line(sc([cx, cy + 190, cx, cy + 199]), fill=(150, 86, 72, 255), width=4 * S)
    om = mask_of(lambda dd: dd.ellipse(sc([cx - 48, cy + 192, cx + 48, cy + 258]), fill=255))
    shaded(img, om, (146, 62, 58), (122, 46, 46), 0.70, 0.75, glow=0.1, sigma=0.05)
    tm = mask_of(lambda dd: dd.ellipse(sc([cx - 32, cy + 226, cx + 32, cy + 262]), fill=255))
    shaded(img, tm, (252, 182, 182), (243, 142, 142), 0.72, 0.76, glow=0.2, sigma=0.05)
    d = ImageDraw.Draw(img)
    d.line(sc([cx, cy + 230, cx, cy + 250]), fill=(218, 118, 118, 255), width=3 * S)
    ellipse(d, [cx - 15, cy + 226, cx - 4, cy + 236], fill=(255, 255, 255, 110))
    # chin highlight under mouth
    img.alpha_composite(soft_layer(lambda dd: ellipse(dd, [cx - 46, cy + 258, cx + 46, cy + 292], fill=(255, 255, 255, 60)), 8))

    # leaf resting on head (drawn over the crown, slight tilt)
    draw_leaf(img, 628, 96, length=180, angle=-24)

    # one paw REACHING toward the viewer, bottom-right corner (foreground,
    # overlapping the head silhouette so it reads as part of the animal)
    pm2 = mask_of(lambda dd: dd.ellipse(sc([730, 716, 1030, 1004]), fill=255))
    shaded(img, pm2, (246, 156, 84), (202, 104, 48), 0.70, 0.98, light=(0.85, 0.74), glow=0.35, sigma=0.10)
    inner_shadow(img, pm2, strength=60, blur=10)
    fur(img, pm2, [(250, 172, 102), (218, 118, 56), (176, 88, 40)], 500, 880, 860)
    paw_pads(img, 884, 846, scale=1.35)
    # soften the paw silhouette: short fur strokes straddling its top-left
    # arc (the edge facing the viewer), so it reads fluffy, not cut-out.
    # Local RNG: deterministic and does not disturb the global sequences.
    edge_rng = random.Random(20261114)
    fluff = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fluff)
    for k in range(46):
        ang = math.radians(edge_rng.uniform(140, 265))
        rx, ry = 150, 144
        bx, by = 880 + math.cos(ang) * rx, 860 + math.sin(ang) * ry
        ln = edge_rng.uniform(7, 13)
        col = [(250, 172, 102), (218, 118, 56), (240, 146, 78)][edge_rng.randrange(3)]
        fd.line([bx * S, by * S, (bx + math.cos(ang) * ln) * S, (by + math.sin(ang) * ln) * S],
                fill=col + (edge_rng.randint(50, 105),), width=3 * S)
    img.alpha_composite(fluff.filter(ImageFilter.GaussianBlur(1.1 * S)))

    # gentle vignette
    vig = np.clip((np.sqrt((XN - 0.5) ** 2 + (YN - 0.5) ** 2) - 0.55) / 0.30, 0, 1) * 40
    vimg = Image.new("RGBA", (W, W), (122, 62, 26, 0))
    vimg.putalpha(Image.fromarray(vig.astype(np.uint8), "L"))
    img.alpha_composite(vimg)

    img = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    img.save(OUT, "PNG", optimize=True)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
