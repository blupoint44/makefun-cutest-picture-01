"""Generate a cute baby red panda hugging a bamboo shoot.

Procedural drawing with PIL only (plus numpy for the background gradient).
Renders at 2048x2048 and downscales to 1024x1024 entry.png for smoothness.
No external assets, no text.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2  # supersample factor
W = 1024 * S
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry.png")

# Palette
CREAM = (255, 241, 220, 255)
FACE = (255, 243, 226, 255)
ORANGE = (224, 122, 63, 255)
ORANGE_DARK = (181, 86, 39, 255)
ORANGE_DEEP = (138, 69, 43, 255)
BROWN = (59, 32, 20, 255)
PINK = (244, 150, 150, 255)
GREEN = (124, 185, 92, 255)
GREEN_DARK = (91, 150, 66, 255)
GREEN_LIGHT = (168, 216, 130, 255)


def sc(box):
    return [v * S for v in box]


def ellipse(d, box, **kw):
    d.ellipse(sc(box), **kw)


def make_background():
    """Soft warm radial glow: cream centre fading to peach edges, mint floor."""
    y, x = np.mgrid[0:W, 0:W].astype(np.float32)
    cx, cy = W * 0.5, W * 0.42
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (W * 0.72)
    r = np.clip(r, 0, 1)
    top = np.array([255, 246, 220], dtype=np.float32)
    mid = np.array([255, 218, 178], dtype=np.float32)
    edge = np.array([250, 183, 148], dtype=np.float32)
    img = np.zeros((W, W, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = np.where(
            r < 0.55,
            top[c] + (mid[c] - top[c]) * (r / 0.55),
            mid[c] + (edge[c] - mid[c]) * ((r - 0.55) / 0.45),
        )
    # gentle mint wash along the very bottom (meadow hint)
    band = np.clip((y - W * 0.80) / (W * 0.20), 0, 1) * 0.45
    mint = np.array([196, 230, 178], dtype=np.float32)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - band) + mint[c] * band
    return Image.fromarray(img.astype(np.uint8), "RGB").convert("RGBA")


def soft_ellipse(base, box, color, blur=18):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse(sc(box), fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(blur * S / 2))
    base.alpha_composite(layer)


def draw_bokeh(base):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for (x, y, r, a) in [
        (150, 180, 34, 70), (880, 150, 26, 60), (940, 420, 18, 55),
        (90, 480, 20, 55), (210, 90, 16, 60), (820, 80, 15, 50),
    ]:
        d.ellipse(sc([x - r, y - r, x + r, y + r]), fill=(255, 255, 255, a))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(6 * S)))


def draw_bamboo(d):
    """A tiny bamboo shoot standing in front of the body, centre-left."""
    x0, x1 = 272, 332
    # stalk segments
    for i, y0 in enumerate([652, 734, 816, 884]):
        fill = GREEN if i % 2 == 0 else (140, 198, 104, 255)
        d.rounded_rectangle(sc([x0, y0, x1, y0 + 84]), radius=18 * S, fill=fill)
    # pointed tip
    d.polygon([sc([x0 + 2, 622])[0], sc([x0 + 2, 622])[1]] if False else
              [( (x0 + 2) * S, 656 * S), ((x1 - 2) * S, 656 * S), (302 * S, 600 * S)],
              fill=GREEN_LIGHT)
    # leaves
    leaf = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    ld.ellipse(sc([196, 614, 322, 664]), fill=GREEN)
    ld.ellipse(sc([286, 598, 410, 648]), fill=GREEN_DARK)
    ld.ellipse(sc([232, 566, 352, 612]), fill=GREEN_LIGHT)
    d.bitmap((0, 0), leaf.split()[3], fill=None) if False else None
    return leaf


def draw_head():
    """Head drawn on its own layer so it can be tilted as one piece."""
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = 512, 428
    # ears (two, round)
    for ex in (322, 702):
        ellipse(d, [ex - 96, 196, ex + 96, 388], fill=ORANGE_DEEP)
        ellipse(d, [ex - 64, 228, ex + 64, 356], fill=(255, 214, 188, 255))
        ellipse(d, [ex - 30, 258, ex + 30, 326], fill=PINK)
    # main head
    ellipse(d, [cx - 282, cy - 252, cx + 282, cy + 252], fill=ORANGE, outline=ORANGE_DEEP, width=5 * S)
    # cream face patch
    ellipse(d, [cx - 222, cy - 158, cx + 222, cy + 238], fill=FACE)
    # white brow patches
    ellipse(d, [cx - 182, cy - 168, cx - 84, cy - 118], fill=(255, 255, 255, 255))
    ellipse(d, [cx + 84, cy - 168, cx + 182, cy - 118], fill=(255, 255, 255, 255))
    # ears inner re-tint over head edge is unnecessary; eyes next
    for ex in (cx - 118, cx + 118):
        # eye white glow ring then big glossy eye
        ellipse(d, [ex - 52, cy - 66, ex + 52, cy + 72], fill=BROWN)
        ellipse(d, [ex - 20, cy - 48, ex + 16, cy - 10], fill=(255, 255, 255, 255))
        ellipse(d, [ex + 8, cy + 22, ex + 30, cy + 44], fill=(255, 255, 255, 230))
        ellipse(d, [ex - 34, cy + 34, ex - 24, cy + 44], fill=(255, 255, 255, 200))
    # rosy cheeks
    cheeks = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cheeks)
    for ex in (cx - 168, cx + 168):
        cd.ellipse(sc([ex - 46, cy + 88, ex + 46, cy + 134]), fill=(244, 138, 138, 175))
    layer.alpha_composite(cheeks.filter(ImageFilter.GaussianBlur(10 * S)))
    d = ImageDraw.Draw(layer)
    # nose + smiling mouth
    ellipse(d, [cx - 30, cy + 96, cx + 30, cy + 134], fill=(122, 63, 58, 255))
    ellipse(d, [cx - 16, cy + 100, cx + 2, cy + 112], fill=(255, 255, 255, 110))
    d.arc(sc([cx - 62, cy + 108, cx, cy + 168]), start=20, end=160, fill=BROWN, width=7 * S)
    d.arc(sc([cx, cy + 108, cx + 62, cy + 168]), start=20, end=160, fill=BROWN, width=7 * S)
    # tiny tongue peeking out
    ellipse(d, [cx - 20, cy + 148, cx + 20, cy + 182], fill=(246, 166, 166, 255))
    d.line(sc([cx, cy + 152, cx, cy + 172]), fill=(214, 120, 120, 255), width=3 * S)
    # rotate the whole head for a charming tilt
    return layer.rotate(7, resample=Image.BICUBIC, center=(cx * S, cy * S))


def main():
    img = make_background()
    draw_bokeh(img)
    # halo glow behind subject
    soft_ellipse(img, [150, 70, 874, 794], (255, 255, 255, 85), blur=110)
    # ground shadow
    soft_ellipse(img, [236, 902, 788, 992], (160, 90, 50, 110), blur=22)

    d = ImageDraw.Draw(img)
    # body: chubby oval
    ellipse(d, [252, 560, 772, 986], fill=ORANGE, outline=ORANGE_DEEP, width=5 * S)
    # belly
    ellipse(d, [362, 700, 662, 968], fill=CREAM)
    # feet (two) with pads
    for fx in (330, 694):
        ellipse(d, [fx - 92, 902, fx + 92, 990], fill=ORANGE_DARK)
        ellipse(d, [fx - 32, 934, fx + 32, 974], fill=(255, 205, 180, 255))
        for tx in (fx - 56, fx, fx + 56):
            ellipse(d, [tx - 13, 902, tx + 13, 926], fill=(255, 205, 180, 255))
    # waving arm (right side, raised)
    d.line(sc([700, 730, 838, 500]), fill=ORANGE, width=104 * S, joint="curve")
    ellipse(d, [838 - 78, 500 - 78, 838 + 78, 500 + 78], fill=ORANGE, outline=ORANGE_DEEP, width=4 * S)
    for tx, ty in [(806, 452), (842, 438), (874, 452)]:
        ellipse(d, [tx - 13, ty - 16, tx + 13, ty + 16], fill=(255, 205, 180, 255))
    ellipse(d, [812, 486, 864, 532], fill=(255, 205, 180, 255))
    # bamboo in front of the body
    leaf = draw_bamboo(d)
    img.alpha_composite(leaf)
    d = ImageDraw.Draw(img)
    # hugging arm over the bamboo
    d.line(sc([352, 706, 304, 812]), fill=ORANGE, width=92 * S, joint="curve")
    ellipse(d, [304 - 62, 812 - 52, 304 + 62, 812 + 52], fill=ORANGE, outline=ORANGE_DEEP, width=4 * S)
    for tx, ty in [(274, 788), (304, 778), (334, 788)]:
        ellipse(d, [tx - 11, ty - 14, tx + 11, ty + 14], fill=(255, 205, 180, 255))
    # head, tilted
    img.alpha_composite(draw_head())
    # soft shading at the very bottom of the body
    soft_ellipse(img, [330, 880, 694, 990], (150, 70, 30, 45), blur=26)

    img = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    img.save(OUT, "PNG", optimize=True)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
