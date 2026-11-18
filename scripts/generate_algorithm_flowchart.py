"""
Generate Fig. 1 - a monochrome flowchart of the Robust Two-Signal Retraining
Policy, drawn with Pillow so it does not need matplotlib.

Typography matches the rest of the manuscript exactly: the labels are set in
Times New Roman itself (C:/Windows/Fonts/times.ttf on Windows, otherwise the
bundled fonts/Times.TTF and fonts/Timesbd.TTF), the same face Word renders the
body text with. Subscripts are typeset as real reduced-size runs (not Unicode
subscript glyphs), so the label style is the same as IEEE math in the body text.

The canvas is exactly the width the figure is inserted at (3.25 in) at 600 dpi,
so the 8 pt labels keep their true printed size -- above the 7 pt floor IEEE
print review applies to figure lettering.

Output: figures/fig_algorithm_flowchart.png
"""
import os
import re
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The script sits at the project root in the authors' tree and under scripts/ in
# the packaged artifact; find the figures/ directory either way.
ROOT = BASE_DIR if os.path.isdir(os.path.join(BASE_DIR, "figures")) else os.path.dirname(BASE_DIR)
OUT = os.path.join(ROOT, "figures", "fig_algorithm_flowchart.png")

DPI = 600
W_IN, H_IN = 3.25, 3.00
W, H = int(W_IN * DPI), int(H_IN * DPI)
PT = DPI / 72.0


def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)


def _win_fonts(*names):
    """Font files shipped with Windows; %SystemRoot% is set on every install."""
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or "C:\\Windows"
    return [os.path.join(root, "Fonts", n) for n in names]


# Windows first: the inserted raster must use the face Word draws the text with.
SERIF = _first(
    *_win_fonts("times.ttf", "Times.TTF"),
    os.path.join(ROOT, "fonts", "Times.TTF"),
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)
SERIF_BOLD = _first(
    *_win_fonts("timesbd.ttf", "Timesbd.TTF"),
    os.path.join(ROOT, "fonts", "Timesbd.TTF"),
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
)

FS = 8.0
FS_BAR = 7.8
FS_TAG = 7.0
SUB_RATIO = 0.72
LW = max(2, int(0.75 * PT))
GRAY = (237, 237, 237)
BLACK = (0, 0, 0)


def font(size_pt, bold=False):
    return ImageFont.truetype(SERIF_BOLD if bold else SERIF, int(size_pt * PT))


img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def parse_runs(text):
    """'Calibrate \u03b8_{1}, S\u0304_{ref}' -> [('Calibrate \u03b8', 0), ('1', 1), ...]"""
    out, i = [], 0
    while i < len(text):
        j = text.find("_{", i)
        if j < 0:
            out.append((text[i:], 0))
            break
        if j > i:
            out.append((text[i:j], 0))
        k = text.find("}", j)
        out.append((text[j + 2:k], 1))
        i = k + 1
    return [r for r in out if r[0]]


def runs_width(runs, size_pt, bold=False):
    total = 0.0
    for txt, sub in runs:
        f = font(size_pt * (SUB_RATIO if sub else 1.0), bold)
        total += d.textlength(txt, font=f)
    return total


def draw_runs(cx, cy, runs, size_pt, bold=False):
    """Draw runs; U+0304 in a run is drawn as a real macron over the preceding
    glyph, which keeps 'S-bar' correct even when the font lacks the combining mark."""
    total = runs_width(runs, size_pt, bold)
    x = cx - total / 2
    for txt, sub in runs:
        f = font(size_pt * (SUB_RATIO if sub else 1.0), bold)
        bar = txt.endswith("\u0304")
        if bar:
            txt = txt[:-1]
            if not txt:
                continue
        bb = d.textbbox((0, 0), txt, font=f)
        h = bb[3] - bb[1]
        dy = 0.30 * size_pt * PT if sub else 0.0
        d.text((x, cy - h / 2 - bb[1] + dy), txt, font=f, fill=BLACK)
        if bar:
            last_w = d.textlength(txt[-1], font=f)
            left = x + d.textlength(txt[:-1], font=f)
            rx0, rx1 = left + 0.12 * last_w, left + 0.88 * last_w
            ry = cy - h / 2 + dy - int(0.13 * size_pt * PT)
            d.line([rx0, ry, rx1, ry], fill=BLACK, width=max(1, int(size_pt * PT * 0.07)))
        x += d.textlength(txt, font=f)


def fit_size(text, max_w, start, max_lines=1):
    size = start
    while size > 4.2:
        runs = parse_runs(text)
        if runs_width(runs, size) <= max_w and "\n" not in text:
            return size
        size -= 0.1
    return size


def wrap_runs(text, size_pt, max_w):
    """Split on spaces into lines that fit max_w (runs preserved)."""
    words, lines, cur = re.split(r"(\s+)", text), [], ""
    out, cur = [], ""
    for w in words:
        trial = cur + w
        if runs_width(parse_runs(trial), size_pt) <= max_w or not cur:
            cur = trial
        else:
            out.append(cur.strip())
            cur = w.lstrip()
    if cur.strip():
        out.append(cur.strip())
    return out


def box(x0, y0, x1, y1, text):
    d.rounded_rectangle([x0, y0, x1, y1], radius=int(0.028 * DPI),
                        outline=BLACK, width=LW, fill="white")
    max_w = (x1 - x0) - int(0.07 * DPI)
    size = fit_size(text, max_w, FS)
    lines = wrap_runs(text, size, max_w)
    gap = int(1.30 * size * PT)
    heights = [size * PT]
    total = len(lines) * heights[0] + (len(lines) - 1) * gap
    y = (y0 + y1) / 2 - total / 2 + heights[0] / 2
    for ln in lines:
        draw_runs((x0 + x1) / 2, y, parse_runs(ln), size)
        y += heights[0] + gap


def bar(y0, y1, text):
    d.rectangle([int(0.05 * DPI), y0, W - int(0.05 * DPI), y1], fill=GRAY)
    draw_runs(W / 2, (y0 + y1) / 2, parse_runs(text), FS_BAR, bold=True)


def diamond(cx, cy, hw, hh, text):
    d.polygon([(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)],
              outline=BLACK, width=LW, fill="white")
    draw_runs(cx, cy, parse_runs(text), FS)


def arrow(x0, y0, x1, y1):
    d.line([x0, y0, x1, y1], fill=BLACK, width=LW)
    head = int(0.020 * DPI)
    d.polygon([(x1, y1), (x1 - head, y1 - head * 0.5), (x1 - head, y1 + head * 0.5)],
              fill=BLACK)


def px(inch):
    return int(inch * DPI)


MAIN_X0, MAIN_X1 = px(0.05), px(2.28)
SIDE_X0, SIDE_X1 = px(2.38), px(3.20)
MAIN_CX = (MAIN_X0 + MAIN_X1) / 2
DHW, DHH = px(0.96), px(0.20)   # diamond half-width / half-height

# ---- Phase 1 ---------------------------------------------------------------
bar(px(0.03), px(0.21), "Phase 1 - robust calibration")
box(MAIN_X0, px(0.24), MAIN_X1, px(0.57), "W_{ref} reference windows")
box(MAIN_X0, px(0.62), MAIN_X1, px(0.95), "Calibrate \u03b8_{1}, \u03b8_{2}, S\u0304_{ref}")

# ---- Phase 2 ---------------------------------------------------------------
bar(px(1.00), px(1.18), "Phase 2 - online surveillance")
box(MAIN_X0, px(1.21), MAIN_X1, px(1.54), "Distribution shift \u03a8_{t}")
diamond(MAIN_CX, px(1.71), DHW, DHH, "\u03a8_{t} > \u03b8_{1} ?")
box(SIDE_X0, px(1.53), SIDE_X1, px(1.89), "Wait")
box(MAIN_X0, px(1.91), MAIN_X1, px(2.24), "Cluster pair, \u0394S_{t}")
diamond(MAIN_CX, px(2.41), DHW, DHH, "\u0394S_{t} > \u03b8_{2} ?")
box(SIDE_X0, px(2.23), SIDE_X1, px(2.59), "Wait")
box(MAIN_X0, px(2.63), MAIN_X1, px(2.96), "Retrain f_{t}, update S\u0304_{ref}")

# ---- arrows ----------------------------------------------------------------
for y0, y1 in [(0.21, 0.23), (0.57, 0.61), (0.95, 0.99), (1.18, 1.20),
               (1.54, 1.545), (1.875, 1.90), (2.24, 2.245), (2.575, 2.62)]:
    arrow(MAIN_CX, px(y0), MAIN_CX, px(y1))

arrow(MAIN_CX + DHW, px(1.71), SIDE_X0, px(1.71))
arrow(MAIN_CX + DHW, px(2.41), SIDE_X0, px(2.41))

f_tag = font(FS_TAG)
d.text((SIDE_X0 - px(0.34), px(1.71) + int(0.7 * PT)), "no", font=f_tag, fill=BLACK)
d.text((SIDE_X0 - px(0.34), px(2.41) + int(0.7 * PT)), "no", font=f_tag, fill=BLACK)

img.save(OUT, dpi=(DPI, DPI))
print("wrote", OUT, "(%d x %d px @ %d dpi = %.2f x %.2f in)"
      % (W, H, DPI, W / DPI, H / DPI))
