"""Render a reference chart in the WA Fog theme: python3 build_fog_preview.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wa_style_fog import CANVAS, CORAL, HAIR, INK, MUTED, MUTED2, POS_COLOR, SANS, setup_figure, draw_masthead, draw_footer  # noqa: E402

FIGW, FIGH = 11, 7
fig, ax, fx, fy = setup_figure(FIGW, FIGH, dpi=200)
draw_masthead(fig, ax, fx, fy, "WA colour theme  |  Fog", "Fog: cool, light canvas",
              "Reference chart: two rolling-average series plus a role-colour key")
draw_footer(fig, ax, fx, fy, "Illustrative data", "Marc Lamberts | Waltzing Analytics")

p = fig.add_axes([fx(0.9), fy(1.6), 1 - fx(3.6), 1 - fy(2.2) - fy(1.6)])
p.set_facecolor(CANVAS)
rng = np.random.default_rng(7)
x = np.arange(1, 35)
for col, lab, base in [(CORAL, "Series A", 1.6), (POS_COLOR["MID"], "Series B", 1.2)]:
    y = base + np.cumsum(rng.normal(0, 0.08, len(x)))
    roll = np.convolve(y, np.ones(5) / 5, mode="valid")
    p.plot(x, y, color=col, alpha=0.25, lw=1)
    p.plot(x[4:], roll, color=col, lw=2.4, label=lab)
for s in p.spines.values():
    s.set_visible(False)
p.grid(axis="y", color=HAIR, lw=0.8)
p.tick_params(colors=MUTED, labelsize=9, length=0)
p.set_xlabel("Matchweek", color=MUTED, fontsize=9, family=SANS)
p.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")

swatches = [("CANVAS", CANVAS), ("INK", INK), ("MUTED", MUTED), ("MUTED2", MUTED2), ("HAIR", HAIR),
            ("CORAL / FWD", CORAL), ("MID", POS_COLOR["MID"]), ("DEF", POS_COLOR["DEF"])]
for i, (name, hexc) in enumerate(swatches):
    yy = 1 - fy(1.85) - fy(0.55) * i
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (1 - fx(2.45), yy - fy(0.16)), fx(0.32), fy(0.32), transform=ax.transAxes,
        facecolor=hexc, edgecolor=HAIR, lw=0.8))
    ax.text(1 - fx(2.0), yy + fy(0.06), name, transform=ax.transAxes, fontsize=8.5, color=INK,
            family=SANS, va="center", fontweight="bold")
    ax.text(1 - fx(2.0), yy - fy(0.11), hexc.upper(), transform=ax.transAxes, fontsize=7.8, color=MUTED,
            family=SANS, va="center")

fig.savefig(os.path.join(HERE, "wa_style_fog_preview.png"), facecolor=CANVAS)
print("saved wa_style_fog_preview.png")
