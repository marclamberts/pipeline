"""Shared Waltzing Analytics fog/coral style helpers for single-chart exports (cool-light theme)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

CANVAS = "#E7ECEE"
INK = "#1A2226"
MUTED = "#78888D"
MUTED2 = "#A9B7BB"
HAIR = "#CBD6D9"
CORAL = "#E8794C"

POS_COLOR = {"FWD": "#E8794C", "MID": "#4C7A93", "DEF": "#5E8C6A"}

SERIF = "Georgia"
SANS = "Helvetica Neue"
plt.rcParams["font.sans-serif"] = [SANS, "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"


def setup_figure(figw, figh, dpi=140):
    fig = plt.figure(figsize=(figw, figh), dpi=dpi)
    fig.patch.set_facecolor(CANVAS)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fx = lambda inches: inches / figw
    fy = lambda inches: inches / figh
    return fig, ax, fx, fy


def draw_masthead(fig, ax, fx, fy, eyebrow, title, subtitle, title_fontsize=23):
    ax.text(fx(0.5), 1 - fy(0.42), eyebrow.upper(), transform=ax.transAxes, ha="left", va="center",
            fontsize=10, fontweight="bold", color=CORAL, family=SANS, zorder=3)
    ax.text(fx(0.5), 1 - fy(0.90), title, transform=ax.transAxes, ha="left", va="center",
            fontsize=title_fontsize, fontweight="bold", color=INK, family=SERIF, zorder=3)
    ax.text(fx(0.5), 1 - fy(1.28), subtitle, transform=ax.transAxes, ha="left", va="center",
            fontsize=10, color=MUTED2, family=SANS, zorder=3)

    lockup_right = 1 - fx(0.32)
    fig.canvas.draw()
    word = ax.text(lockup_right, 1 - fy(0.38), "WALTZING ANALYTICS", transform=ax.transAxes,
                    ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK, family=SANS, zorder=5)
    ax.text(lockup_right, 1 - fy(0.55), "MARC LAMBERTS", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.8, color=MUTED2, family=SANS, zorder=5)
    bbox = word.get_window_extent(renderer=fig.canvas.get_renderer())
    word_left_frac = bbox.x0 / fig.bbox.width
    divider_x = word_left_frac - fx(0.10)
    ax.plot([divider_x, divider_x], [1 - fy(0.24), 1 - fy(0.68)], color=HAIR, linewidth=1.2,
            transform=ax.transAxes, zorder=5)
    ax.text(divider_x - fx(0.08), 1 - fy(0.46), "WA", transform=ax.transAxes, ha="right", va="center",
            fontsize=19, fontweight="bold", color=CORAL, family=SANS, zorder=5)


def draw_footer(fig, ax, fx, fy, source_text, date_text="Marc Lamberts | Created on 04-09-2026"):
    ax.plot([fx(0.5), 1 - fx(0.32)], [fy(0.50), fy(0.50)], color=HAIR, linewidth=1,
            transform=ax.transAxes, zorder=3)
    ax.text(fx(0.5), fy(0.30), source_text, transform=ax.transAxes, ha="left", va="center",
            fontsize=7.8, color=MUTED, family=SANS, zorder=4)
    ax.text(1 - fx(0.32), fy(0.30), date_text, transform=ax.transAxes, ha="right", va="center",
            fontsize=7.8, color=MUTED, family=SANS, zorder=4)
