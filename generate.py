"""Generate v2: a richly rendered cute baby red panda hugging bamboo.

Painterly upgrade over v1 (flat vector): every shape gets gradient
shading, an inner edge shadow, a warm rim light and thousands of short
directional fur strokes clipped to its mask. The background is a soft
bamboo grove with depth blur, sun rays and bokeh. Pure PIL + numpy,
no external assets, no text. Renders at 2048 and downscales to 1024.
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2
W = 1024 * S
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry.png")
rng = random.Random(20261113)
nprng = np.random.default_rng(20261113)

# normalised coordinate grids (0..1), reused by all shading
YN, XN = np.mgrid[0:W, 0:W].astype(np.float32) / W

# Palette
ORANGE_TOP = (240, 146, 74)
ORANGE_BOT = (196, 96, 44)
ORANGE_DEEP = (150, 74, 34)
CREAM_TOP = (255, 248, 232)
CREAM_BOT = (243, 222, 190)
FACE_TOP = (255, 250, 238)
FACE_BOT = (247, 228, 198)
PINK = (246, 158, 158)
PAD = (255, 209, 186)
BROWN = (62, 34, 20)


def sc(box):
    return [v * S for v in box]


def ellipse(d, box, **kw):
    d.ellipse(sc(box), **kw)


# ---------------------------------------------------------------- helpers

def soft_layer(draw_fn, blur):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    draw_fn(d)
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur * S))
    return layer


def shaded(base, mask, c_top, c_bot, y0=0.0, y1=1.0,
           light=(0.40, 0.30), glow=0.38, sigma=0.16):
    """Gradient-filled shape with a warm radial light, composited by mask."""
    t = np.clip((YN - y0) / max(y1 - y0, 1e-6), 0, 1) ** 1.15
    arr = np.zeros((W, W, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = c_top[c] + (c_bot[c] - c_top[c]) * t
    d2 = (XN - light[0]) ** 2 + (YN - light[1]) ** 2
    g = np.exp(-d2 / (2 * sigma ** 2)) * glow
    warm = np.array([255, 226, 175], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] += (warm[c] - arr[:, :, c]) * g
    img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    img.putalpha(mask.filter(ImageFilter.GaussianBlur(1.2 * S)))
    base.alpha_composite(img)


def inner_shadow(base, mask, strength=90, blur=14, color=(120, 52, 18)):
    """Dark soft edge inside a shape (ambient occlusion / form shadow)."""
    eroded = mask.filter(ImageFilter.MinFilter(31))
    edge = np.clip(np.asarray(mask, dtype=np.int16) - np.asarray(eroded, dtype=np.int16), 0, 255)
    edge_img = Image.fromarray(edge.astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(blur * S))
    layer = Image.new("RGBA", (W, W), color + (0,))
    layer.putalpha(edge_img.point(lambda v: v * strength // 255))
    base.alpha_composite(layer)


def rim_light(base, mask, strength=120, color=(255, 232, 190)):
    """Warm rim on the upper-left edge of a shape."""
    eroded = mask.filter(ImageFilter.MinFilter(21))
    edge = np.clip(np.asarray(mask, dtype=np.int16) - np.asarray(eroded, dtype=np.int16), 0, 255)
    weight = np.clip(0.75 - (XN - 0.5) * 0.9 - (YN - 0.45) * 1.1, 0, 1)
    rim = (edge.astype(np.float32) * weight).astype(np.uint8)
    rim_img = Image.fromarray(rim, "L").filter(ImageFilter.GaussianBlur(5 * S))
    layer = Image.new("RGBA", (W, W), color + (0,))
    layer.putalpha(rim_img.point(lambda v: v * strength // 255))
    base.alpha_composite(layer)


def fur(base, mask, colors, n, cx, cy, len_lo=7, len_hi=15, alpha=(26, 62),
        downward=0.45, width=3):
    """Thousands of short soft strokes, radial + downward, clipped to mask."""
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
    # clip strokes to the (slightly soft) mask
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
    top = np.array([255, 244, 214], dtype=np.float32)
    mid = np.array([252, 224, 178], dtype=np.float32)
    edge = np.array([238, 183, 140], dtype=np.float32)
    r = np.clip(np.sqrt((XN - 0.5) ** 2 + (YN - 0.40) ** 2) / 0.75, 0, 1)
    img = np.zeros((W, W, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = np.where(r < 0.5, top[c] + (mid[c] - top[c]) * (r / 0.5),
                                mid[c] + (edge[c] - mid[c]) * ((r - 0.5) / 0.5))
    band = np.clip((YN - 0.78) / 0.22, 0, 1) * 0.5
    mint = np.array([183, 222, 168], dtype=np.float32)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - band) + mint[c] * band
    return Image.fromarray(img.astype(np.uint8), "RGB").convert("RGBA")


def bamboo_layer(stalks, blur, alpha):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (x, wdt, tone) in stalks:
        col = (tone[0], tone[1], tone[2], alpha)
        d.rounded_rectangle(sc([x, -40, x + wdt, 1064]), radius=wdt * S // 2, fill=col)
        for ny in (180, 430, 690, 920):
            d.rectangle(sc([x - 3, ny, x + wdt + 3, ny + 14]),
                        fill=(70, 110, 52, min(255, alpha + 30)))
        d.rectangle(sc([x + wdt * 0.18, -40, x + wdt * 0.34, 1064]),
                    fill=(255, 255, 235, alpha // 3))
    # leaf silhouettes
    for (lx, ly, lw, ang) in [(60, 120, 190, -24), (790, 90, 210, 18),
                              (880, 300, 150, 34), (30, 380, 140, -38),
                              (700, 40, 150, -12), (150, 40, 130, 26)]:
        leaf = Image.new("RGBA", (360 * S, 120 * S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(leaf)
        ld.ellipse([0, 30 * S, lw * S, 90 * S], fill=(96, 156, 74, alpha))
        leaf = leaf.rotate(ang, resample=Image.BICUBIC, expand=True)
        layer.alpha_composite(leaf, (int(lx * S), int(ly * S)))
    return layer.filter(ImageFilter.GaussianBlur(blur * S))


def sun_and_bokeh(base):
    rays = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(rays)
    for (x0, x1) in [(300, 430), (520, 610), (120, 210), (700, 790)]:
        d.polygon([(x0 * S, 0), (x1 * S, 0), ((x1 + 190) * S, W), ((x0 + 150) * S, W)],
                  fill=(255, 246, 205, 34))
    base.alpha_composite(rays.filter(ImageFilter.GaussianBlur(28 * S)))
    bok = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(bok)
    for (x, y, r, a) in [(140, 170, 36, 90), (888, 150, 28, 80), (946, 430, 19, 70),
                         (84, 500, 22, 70), (212, 84, 17, 75), (826, 74, 16, 65),
                         (70, 760, 26, 55), (950, 700, 30, 55), (300, 120, 12, 60),
                         (660, 96, 13, 60)]:
        d.ellipse(sc([x - r, y - r, x + r, y + r]), fill=(255, 252, 230, a))
        d.ellipse(sc([x - r, y - r, x + r, y + r]), outline=(255, 255, 255, a // 2),
                  width=3 * S)
    base.alpha_composite(bok.filter(ImageFilter.GaussianBlur(3 * S)))


# ---------------------------------------------------------------- pieces

def draw_bamboo_shoot(base):
    """The hugged bamboo shoot: shaded segments + pointed leaves."""
    x0, x1 = 268, 336
    for i, y0 in enumerate([648, 730, 812, 884]):
        seg = mask_of(lambda d, y0=y0: d.rounded_rectangle(
            sc([x0, y0, x1, y0 + 86]), radius=17 * S, fill=255))
        c_top = (168, 220, 128) if i % 2 == 0 else (146, 205, 108)
        c_bot = (104, 168, 76) if i % 2 == 0 else (92, 152, 64)
        shaded(base, seg, c_top, c_bot, y0 / 1024, (y0 + 86) / 1024,
               light=(0.28, 0.62), glow=0.30, sigma=0.10)
        inner_shadow(base, seg, strength=60, blur=8)
    d = ImageDraw.Draw(base)
    d.polygon([((x0 + 4) * S, 656 * S), ((x1 - 4) * S, 656 * S), (302 * S, 596 * S)],
              fill=(178, 228, 138, 255))
    # leaves (rotated soft ellipses with a vein)
    for (lx, ly, lw, ang, tone) in [(168, 596, 168, -18, (110, 178, 80)),
                                    (292, 576, 158, 14, (88, 156, 62)),
                                    (214, 540, 148, -4, (140, 205, 100))]:
        leaf = Image.new("RGBA", (380 * S, 150 * S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(leaf)
        ld.ellipse([6 * S, 42 * S, lw * S, 108 * S], fill=tone + (255,))
        ld.line([10 * S, 75 * S, (lw - 8) * S, 75 * S], fill=(70, 120, 48, 160), width=3 * S)
        leaf = leaf.filter(ImageFilter.GaussianBlur(1 * S)).rotate(
            ang, resample=Image.BICUBIC, expand=True)
        base.alpha_composite(leaf, (int(lx * S), int(ly * S)))


def paw_pads(base, cx, cy, scale=1.0, toes=((-30, -34), (0, -44), (30, -34))):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ellipse(d, [cx - 27 * scale, cy - 6 * scale, cx + 27 * scale, cy + 34 * scale],
            fill=PAD + (255,))
    for (tx, ty) in toes:
        ellipse(d, [cx + (tx - 13) * scale, cy + (ty - 15) * scale,
                    cx + (tx + 13) * scale, cy + (ty + 15) * scale], fill=PAD + (255,))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1 * S)))


def draw_eye(d_layer, ex, ey):
    """One big glossy eye: amber-brown iris gradient, pupil, 3 catchlights."""
    ew, eh = 56, 76
    patch = Image.new("RGBA", (int(ew * 2 * S), int(eh * 2 * S)), (0, 0, 0, 0))
    h = patch.height
    grad = np.zeros((h, patch.width, 3), dtype=np.float32)
    t = (np.arange(h, dtype=np.float32) / h)[:, None]
    topc = np.array([52, 28, 16], dtype=np.float32)
    botc = np.array([146, 84, 40], dtype=np.float32)
    for c in range(3):
        grad[:, :, c] = topc[c] + (botc[c] - topc[c]) * t
    pimg = Image.fromarray(grad.astype(np.uint8), "RGB").convert("RGBA")
    pmask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(pmask).ellipse([4, 4, patch.width - 4, patch.height - 4], fill=255)
    pimg.putalpha(pmask)
    patch.alpha_composite(pimg)
    pd = ImageDraw.Draw(patch)
    # pupil
    pw, ph = patch.width, patch.height
    pd.ellipse([pw * 0.30, ph * 0.16, pw * 0.70, ph * 0.72], fill=(24, 12, 8, 255))
    # catchlights: big top-left, medium bottom-right, tiny lower-left
    pd.ellipse([pw * 0.24, ph * 0.14, pw * 0.52, ph * 0.36], fill=(255, 255, 255, 255))
    pd.ellipse([pw * 0.58, ph * 0.60, pw * 0.78, ph * 0.78], fill=(255, 255, 255, 235))
    pd.ellipse([pw * 0.20, ph * 0.66, pw * 0.32, ph * 0.78], fill=(255, 255, 255, 200))
    # bottom reflection arc
    pd.arc([pw * 0.14, ph * 0.52, pw * 0.86, ph * 0.94], start=30, end=150,
           fill=(255, 236, 200, 130), width=4 * S)
    d_layer.alpha_composite(patch, (int((ex - ew) * S), int((ey - eh) * S)))
    # glossy top lid line
    ld = ImageDraw.Draw(d_layer)
    ld.arc(sc([ex - ew + 4, ey - eh + 2, ex + ew - 4, ey + eh - 18]),
           start=185, end=355, fill=(74, 40, 22, 255), width=6 * S)


def draw_head():
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cx, cy = 512, 424

    # ears first (behind head), shaded with pink inners
    for ex in (310, 714):
        em = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 100, 176, ex + 100, 376]), fill=255))
        shaded(layer, em, (196, 102, 52), (146, 70, 32), 0.18, 0.38, light=(0.35, 0.22), glow=0.3)
        fur(layer, em, [(232, 128, 66), (160, 80, 36), (248, 168, 100)], 500, ex, 280)
        im = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 62, 222, ex + 62, 348]), fill=255))
        shaded(layer, im, (255, 226, 200), (250, 196, 168), 0.22, 0.34, glow=0.25)
        pm = mask_of(lambda d, ex=ex: d.ellipse(sc([ex - 30, 254, ex + 30, 322]), fill=255))
        shaded(layer, pm, (250, 178, 172), (240, 148, 146), 0.25, 0.32, glow=0.2)

    # head mask: main ellipse + fluff bumps on cheeks / chin / crown
    def head_shape(d):
        d.ellipse(sc([cx - 288, cy - 256, cx + 288, cy + 256]), fill=255)
        for (bx, by, br) in [(cx - 262, cy + 138, 48), (cx + 262, cy + 138, 48),
                             (cx - 216, cy + 208, 42), (cx + 216, cy + 208, 42)]:
            d.ellipse(sc([bx - br, by - br, bx + br, by + br]), fill=255)
    hm = mask_of(head_shape)
    shaded(layer, hm, ORANGE_TOP, ORANGE_BOT, 0.16, 0.68, light=(0.42, 0.30), glow=0.42, sigma=0.20)
    inner_shadow(layer, hm, strength=95, blur=16)
    rim_light(layer, hm, strength=130)

    # cream face patch (soft-edged)
    fm = mask_of(lambda d: d.ellipse(sc([cx - 224, cy - 156, cx + 224, cy + 240]), fill=255))
    shaded(layer, fm, FACE_TOP, FACE_BOT, 0.26, 0.66, light=(0.46, 0.34), glow=0.30, sigma=0.17)
    inner_shadow(layer, fm, strength=36, blur=12, color=(196, 140, 84))

    # fur: dense on the orange ring, sparse light strokes on the face edge
    ring = Image.fromarray(np.clip(np.asarray(hm, dtype=np.int16) - np.asarray(fm, dtype=np.int16), 0, 255).astype(np.uint8), "L")
    fur(layer, ring, [(248, 164, 96), (214, 112, 52), (256, 196, 132), (172, 84, 36)],
        2600, cx, cy, len_lo=8, len_hi=17, alpha=(30, 66))
    fur(layer, fm, [(255, 255, 248), (240, 218, 184)], 500, cx, cy,
        len_lo=6, len_hi=11, alpha=(10, 24))

    d = ImageDraw.Draw(layer)
    # soft white brow patches
    brows = soft_layer(lambda dd: (ellipse(dd, [cx - 182, cy - 168, cx - 86, cy - 114], fill=(255, 255, 255, 235)),
                                   ellipse(dd, [cx + 86, cy - 168, cx + 182, cy - 114], fill=(255, 255, 255, 235))), 5)
    layer.alpha_composite(brows)
    d = ImageDraw.Draw(layer)

    # rosy cheeks (big, soft, warm)
    cheeks = soft_layer(lambda dd: (ellipse(dd, [cx - 218, cy + 82, cx - 118, cy + 142], fill=(246, 138, 138, 185)),
                                    ellipse(dd, [cx + 118, cy + 82, cx + 218, cy + 142], fill=(246, 138, 138, 185))), 11)
    layer.alpha_composite(cheeks)
    d = ImageDraw.Draw(layer)

    # eyes
    for ex in (cx - 120, cx + 120):
        draw_eye(layer, ex, cy + 8)
    d = ImageDraw.Draw(layer)

    # nose: shaded oval + highlight
    nm = mask_of(lambda dd: dd.ellipse(sc([cx - 31, cy + 94, cx + 31, cy + 136]), fill=255))
    shaded(layer, nm, (150, 84, 74), (104, 52, 48), 0.51, 0.56, glow=0.25, sigma=0.05)
    d = ImageDraw.Draw(layer)
    ellipse(d, [cx - 17, cy + 99, cx + 1, cy + 112], fill=(255, 255, 255, 120))
    # smiling mouth (two soft arcs)
    d.arc(sc([cx - 64, cy + 106, cx, cy + 170]), start=25, end=158, fill=BROWN + (255,), width=7 * S)
    d.arc(sc([cx, cy + 106, cx + 64, cy + 170]), start=22, end=155, fill=BROWN + (255,), width=7 * S)
    # tiny tongue
    tm = mask_of(lambda dd: dd.ellipse(sc([cx - 21, cy + 146, cx + 21, cy + 186]), fill=255))
    shaded(layer, tm, (250, 178, 178), (240, 140, 140), 0.56, 0.60, glow=0.2, sigma=0.05)
    d = ImageDraw.Draw(layer)
    d.line(sc([cx, cy + 152, cx, cy + 176]), fill=(216, 116, 116, 255), width=3 * S)
    ellipse(d, [cx - 12, cy + 150, cx - 2, cy + 160], fill=(255, 255, 255, 110))

    # charming head tilt
    return layer.rotate(6, resample=Image.BICUBIC, center=(cx * S, cy * S))


# ---------------------------------------------------------------- main

def main():
    img = make_background()
    # bamboo grove: far layer (very blurred) + nearer side stalks
    img.alpha_composite(bamboo_layer(
        [(36, 54, (140, 190, 110)), (150, 40, (128, 178, 100)), (838, 58, (134, 184, 104)),
         (716, 38, (146, 196, 114)), (934, 44, (124, 172, 96)), (252, 30, (150, 200, 118))],
        blur=13, alpha=120))
    img.alpha_composite(bamboo_layer(
        [(-16, 64, (104, 164, 82)), (972, 68, (98, 156, 78))], blur=6, alpha=150))
    sun_and_bokeh(img)
    # warm halo right behind the subject
    img.alpha_composite(soft_layer(
        lambda d: ellipse(d, [140, 56, 884, 800], fill=(255, 255, 255, 80)), 60))
    # ground + contact shadow
    img.alpha_composite(soft_layer(
        lambda d: ellipse(d, [196, 838, 828, 1000], fill=(168, 208, 148, 120)), 26))
    img.alpha_composite(soft_layer(
        lambda d: ellipse(d, [226, 906, 798, 994], fill=(120, 66, 30, 130)), 20))

    # waving arm (behind body/head, raised on the right)
    am = mask_of(lambda d: (d.line(sc([688, 742, 836, 512]), fill=255, width=106 * S, joint="curve"),
                            d.ellipse(sc([836 - 80, 512 - 80, 836 + 80, 512 + 80]), fill=255)))
    shaded(img, am, ORANGE_TOP, ORANGE_BOT, 0.48, 0.74, light=(0.78, 0.42), glow=0.35, sigma=0.13)
    inner_shadow(img, am, strength=70, blur=12)
    fur(img, am, [(248, 164, 96), (214, 112, 52), (172, 84, 36)], 700, 770, 620)
    paw_pads(img, 842, 500, scale=1.05)

    # chubby body with fluff bumps along the bottom edge
    def body_shape(d):
        d.ellipse(sc([248, 548, 776, 990]), fill=255)
        for bx in (356, 512, 668):
            d.ellipse(sc([bx - 30, 952, bx + 30, 1006]), fill=255)
    bm = mask_of(body_shape)
    shaded(img, bm, ORANGE_TOP, ORANGE_BOT, 0.54, 0.98, light=(0.42, 0.52), glow=0.36, sigma=0.19)
    inner_shadow(img, bm, strength=88, blur=16)
    rim_light(img, bm, strength=90)
    fur(img, bm, [(248, 164, 96), (214, 112, 52), (256, 196, 132), (172, 84, 36)],
        1900, 512, 740)

    # cream belly
    bel = mask_of(lambda d: d.ellipse(sc([356, 692, 668, 972]), fill=255))
    shaded(img, bel, CREAM_TOP, CREAM_BOT, 0.68, 0.96, light=(0.46, 0.66), glow=0.28, sigma=0.13)
    inner_shadow(img, bel, strength=52, blur=12, color=(190, 130, 80))
    fur(img, bel, [(255, 255, 250), (238, 206, 164)], 650, 512, 830,
        len_lo=6, len_hi=11, alpha=(12, 28))

    # feet with pads
    for fx in (326, 698):
        fm2 = mask_of(lambda d, fx=fx: d.ellipse(sc([fx - 94, 898, fx + 94, 992]), fill=255))
        shaded(img, fm2, (206, 110, 54), (156, 76, 34), 0.88, 0.97, glow=0.22, sigma=0.10)
        fur(img, fm2, [(232, 132, 70), (170, 84, 38)], 260, fx, 945)
        paw_pads(img, fx, 950, scale=1.15)

    # head shadow cast onto the body (grounding / AO under the chin)
    img.alpha_composite(soft_layer(
        lambda d: ellipse(d, [316, 596, 708, 716], fill=(130, 60, 22, 74)), 22))

    # hugged bamboo in front of the body
    draw_bamboo_shoot(img)

    # hugging arm over the bamboo
    hm2 = mask_of(lambda d: (d.line(sc([356, 700, 306, 806]), fill=255, width=94 * S, joint="curve"),
                             d.ellipse(sc([306 - 64, 806 - 54, 306 + 64, 806 + 54]), fill=255)))
    shaded(img, hm2, ORANGE_TOP, ORANGE_BOT, 0.68, 0.84, light=(0.32, 0.66), glow=0.32, sigma=0.11)
    inner_shadow(img, hm2, strength=70, blur=10)
    fur(img, hm2, [(248, 164, 96), (214, 112, 52), (172, 84, 36)], 480, 322, 756)
    paw_pads(img, 310, 792, scale=0.92, toes=((-27, -30), (0, -39), (27, -30)))

    # the tilted head on top
    img.alpha_composite(draw_head())

    # gentle vignette to focus the centre
    vig = np.clip((np.sqrt((XN - 0.5) ** 2 + (YN - 0.5) ** 2) - 0.52) / 0.30, 0, 1) * 46
    vimg = Image.new("RGBA", (W, W), (122, 62, 26, 0))
    vimg.putalpha(Image.fromarray(vig.astype(np.uint8), "L"))
    img.alpha_composite(vimg)

    img = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    img.save(OUT, "PNG", optimize=True)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
