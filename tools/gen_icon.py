# Renders icon.png (NuGet package icon, 128x128): a document page split by a before/after
# slider - shadowed and dim on the left, clean on the right. Usage: python tools/gen_icon.py
# (Pillow; optional args: <out.png> <size> for other sizes). Drawn at 1024 px and downsampled.
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 1024
OUT = Path(__file__).resolve().parent.parent / "icon.png"


def lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def vgradient(size, top, bottom):
    w, h = size
    g = Image.new("RGB", (1, h))
    for y in range(h):
        g.putpixel((0, y), lerp(top, bottom, y / (h - 1)))
    return g.resize((w, h))


def rounded_mask(size, box, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, r, fill=255)
    return m


# background: deep blue rounded square
bg = vgradient((S, S), (37, 99, 235), (30, 58, 138)).convert("RGBA")
icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(bg, (0, 0), rounded_mask((S, S), (0, 0, S - 1, S - 1), 210))

# page, drawn upright, tilted, then split in screen space so the split matches the slider
PW, PH = 600, 760
x0, y0 = (S - PW) // 2, (S - PH) // 2
x1, y1 = x0 + PW, y0 + PH
page = rounded_mask((S, S), (x0, y0, x1, y1), 36)

text = Image.new("L", (S, S), 0)
d = ImageDraw.Draw(text)
lines = [0.80, 0.92, 0.70, 0.88, 0.60, 0.90, 0.76]
lh, gap, mx = 30, 84, 72
for i, frac in enumerate(lines):
    y = y0 + 110 + i * gap
    d.rounded_rectangle((x0 + mx, y, x0 + mx + (PW - 2 * mx) * frac, y + lh), lh // 2, fill=255)

# after: clean white paper, crisp text
clean = Image.new("RGBA", (S, S), (0, 0, 0, 0))
clean.paste((255, 255, 255, 255), (0, 0), page)
clean.paste((30, 41, 59, 255), (0, 0), text)

# before: dim yellowed paper, faded text, a hard diagonal (hand) shadow over both
before = Image.new("RGBA", (S, S), (0, 0, 0, 0))
before.paste(vgradient((S, S), (200, 194, 176), (160, 153, 136)), (0, 0), page)
before.paste((118, 112, 98, 255), (0, 0), text)
shadow = Image.new("L", (S, S), 0)
ImageDraw.Draw(shadow).polygon(
    [(x0 - 60, y0 + 230), (x1 + 60, y0 + 520), (x1 + 60, y1 + 60), (x0 - 60, y1 + 60)], fill=125)
shadow = ImageChops.multiply(shadow.filter(ImageFilter.GaussianBlur(22)), page)
shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
shade.paste((40, 34, 28, 255), (0, 0), shadow)
before = Image.alpha_composite(before, shade)

tilt = -6
rot = lambda im: im.rotate(tilt, resample=Image.BICUBIC, center=(S / 2, S / 2))
left = Image.new("L", (S, S), 0)
ImageDraw.Draw(left).rectangle((0, 0, S // 2 - 1, S), fill=255)
L = Image.composite(rot(before), rot(clean), left)

# drop shadow under the page
ps = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ps.paste((8, 20, 60, 140), (0, 0), page.filter(ImageFilter.GaussianBlur(26)))
ps = rot(ps.transform((S, S), Image.AFFINE, (1, 0, -14, 0, 1, -22)))
icon = Image.alpha_composite(icon, ps)
icon = Image.alpha_composite(icon, L)
# before/after slider: vertical bar + round handle with arrows (not tilted)
SL = Image.new("RGBA", (S, S), (0, 0, 0, 0))
ds = ImageDraw.Draw(SL)
amber = (251, 191, 36, 255)
ds.rounded_rectangle((S // 2 - 11, 96, S // 2 + 11, S - 96), 11, fill=amber)
r = 92
c = S // 2
ds.ellipse((c - r, c - r, c + r, c + r), fill=amber)
a, h = 44, 34
ds.polygon([(c - 14, c - h), (c - 14, c + h), (c - 14 - a, c)], fill=(30, 41, 59, 255))
ds.polygon([(c + 14, c - h), (c + 14, c + h), (c + 14 + a, c)], fill=(30, 41, 59, 255))
glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
glow.paste((8, 20, 60, 110), (0, 0), SL.split()[3].filter(ImageFilter.GaussianBlur(14)))
icon = Image.alpha_composite(icon, glow)
icon = Image.alpha_composite(icon, SL)

# clip to the rounded square and downsample
final = Image.new("RGBA", (S, S), (0, 0, 0, 0))
final.paste(icon, (0, 0), rounded_mask((S, S), (0, 0, S - 1, S - 1), 210))
out, px = (Path(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (OUT, 128)
final.resize((px, px), Image.LANCZOS).save(out, optimize=True)
print(out)
