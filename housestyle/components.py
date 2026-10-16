"""
Meridian chart anatomy — the recurring pieces that make a chart recognizable
at a glance: a kicker+headline+dek header with the signature accent tick,
a highlight-one/mute-the-rest convention for series emphasis, and a
source/credit footer. Built on plain matplotlib calls, no dependencies
beyond matplotlib itself.
"""
import datetime

import matplotlib.pyplot as plt

from .colors import LIGHT, DARK

MARK = "▪"  # the signature square tick, printed before every kicker
BRAND_NAME = "Waltzing Analytics"
AUTHOR = "Marc Lamberts"


def _draw_tracked(fig, x_right, y, text, fontsize, color, weight="normal",
                   step=0.0105, space_step=0.0155):
    """Draw right-aligned tracked-out caps one glyph at a time (matplotlib has
    no letter-spacing property) — returns the x of the leftmost glyph drawn,
    so callers can butt a divider/monogram up against it precisely."""
    x = x_right
    for ch in reversed(text):
        if ch == " ":
            x -= space_step
            continue
        fig.text(x, y, ch, fontsize=fontsize, fontweight=weight, color=color,
                  family="sans-serif", ha="center", va="center")
        x -= step
    return x + step


def brand_mark(fig, palette=LIGHT, right=0.94, y=0.965):
    """The Waltzing Analytics logo lockup — always top right, on every chart:
    a bold "WA" monogram in the signature accent, a hairline divider, then
    "WALTZING ANALYTICS" over "MARC LAMBERTS", tracked-out caps."""
    line1_left = _draw_tracked(fig, right, y + 0.013, BRAND_NAME.upper(),
                                fontsize=9.5, color=palette["ink_primary"], weight="bold")
    line2_left = _draw_tracked(fig, right, y - 0.016, AUTHOR.upper(),
                                fontsize=7, color=palette["ink_muted"])
    wordmark_left = min(line1_left, line2_left)

    divider_x = wordmark_left - 0.016
    fig.add_artist(plt.Line2D([divider_x, divider_x], [y - 0.028, y + 0.018],
                               transform=fig.transFigure, color=palette["axis"], linewidth=1.1))

    monogram_right = divider_x - 0.016
    _draw_tracked(fig, monogram_right, y - 0.002, "WA", fontsize=20,
                  color=palette["accent"], weight="bold", step=0.021, space_step=0)
    return fig


def header(fig, kicker, title, dek=None, palette=LIGHT, top=0.86, left=0.06, right=0.94):
    """Kicker (accent, small caps) + headline (serif) + dek (sans, muted),
    plus the Waltzing Analytics logo, top right, always.

    Reserve figure space above the axes for this (e.g. fig.subplots_adjust
    (top=top - 0.08) before calling, or build the figure with a GridSpec
    that leaves that band empty).
    """
    y = 0.97
    fig.text(left, y, f"{MARK} {kicker.upper()}", fontsize=10.5, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="left", va="top")
    fig.text(left, y - 0.055, title, fontsize=16, color=palette["ink_primary"],
              family="serif", fontweight="normal", ha="left", va="top")
    if dek:
        fig.text(left, y - 0.055 - 0.055, dek, fontsize=10.5,
                  color=palette["ink_secondary"], family="sans-serif",
                  ha="left", va="top")
    brand_mark(fig, palette=palette, right=right, y=y - 0.005)
    return fig


def footer(fig, source, note=None, palette=LIGHT, left=0.06, right=0.94, author=AUTHOR):
    """Hairline rule + source/note (bottom left) + credit line (bottom right),
    always "<author> | Created on dd-mm-yyyy" with today's date."""
    y = 0.045
    fig.add_artist(plt.Line2D([left, right], [y + 0.028, y + 0.028],
                               transform=fig.transFigure, color=palette["axis"], linewidth=0.8))
    text = f"Source: {source}" + (f"   {note}" if note else "")
    fig.text(left, y, text, fontsize=8.5, color=palette["ink_muted"],
              family="sans-serif", ha="left", va="top")
    credit = f"{author} | Created on {datetime.date.today().strftime('%d-%m-%Y')}"
    fig.text(right, y, credit, fontsize=8.5, color=palette["ink_muted"],
              family="sans-serif", ha="right", va="top")
    return fig


def highlight_lines(ax, accent_index, palette=LIGHT, muted=None):
    """Recolor ax.lines so one series carries the accent and reads as
    "the finding"; every other line recedes to a shared muted gray.
    Call after all lines are plotted.
    """
    muted = muted or palette["axis"]
    for i, line in enumerate(ax.get_lines()):
        if i == accent_index:
            line.set_color(palette["accent"])
            line.set_linewidth(2.6)
            line.set_zorder(5)
        else:
            line.set_color(muted)
            line.set_linewidth(1.6)
            line.set_zorder(2)
    return ax


def highlight_bars(container, accent_index, palette=LIGHT, muted=None):
    """Recolor a BarContainer so one bar carries the accent; the rest recede."""
    muted = muted or palette["axis"]
    for i, patch in enumerate(container.patches):
        patch.set_color(palette["accent"] if i == accent_index else muted)
    return container


def label_endpoint(ax, x, y, text, color, palette=LIGHT, offset=6):
    """Direct label at a line's last point — the house alternative to a legend
    box for 1-3 highlighted series."""
    ax.annotate(text, xy=(x, y), xytext=(offset, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9.5, color=color, fontweight="bold")
    ax.plot([x], [y], marker="o", markersize=5, color=color, zorder=6)
