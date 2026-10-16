"""
Meridian matplotlib style — applies the house style's rcParams.

Usage:
    from housestyle import style
    palette, cats = style.apply("light")   # or "dark"
"""
from cycler import cycler
import matplotlib as mpl

from .colors import LIGHT, DARK, CATEGORICAL_LIGHT, CATEGORICAL_DARK

# Serif for kicker/headline (editorial voice); sans for everything else.
# Each stack degrades to a matplotlib-bundled face if the named fonts aren't installed.
SERIF_STACK = ["Source Serif 4", "PT Serif", "Georgia", "DejaVu Serif"]
SANS_STACK = ["IBM Plex Sans", "Inter", "Arial", "DejaVu Sans"]


def apply(mode: str = "light"):
    """Apply the Meridian rcParams globally and return (palette, categorical_cycle)."""
    if mode not in ("light", "dark"):
        raise ValueError('mode must be "light" or "dark"')

    palette = LIGHT if mode == "light" else DARK
    cats = CATEGORICAL_LIGHT if mode == "light" else CATEGORICAL_DARK

    mpl.rcParams.update({
        # surfaces
        "figure.facecolor": palette["surface"],
        "axes.facecolor": palette["surface"],
        "savefig.facecolor": palette["surface"],
        "savefig.edgecolor": "none",

        # type
        "font.family": "sans-serif",
        "font.sans-serif": SANS_STACK,
        "font.serif": SERIF_STACK,
        "text.color": palette["ink_primary"],
        "axes.labelcolor": palette["ink_secondary"],
        "xtick.color": palette["ink_muted"],
        "ytick.color": palette["ink_muted"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,

        # frame — recessive by design: no box, baseline only
        "axes.edgecolor": palette["axis"],
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,

        # grid — value-axis hairlines only, behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": palette["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,

        # ticks
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,

        # marks
        "axes.prop_cycle": cycler(color=cats),
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "patch.linewidth": 0,
        "boxplot.boxprops.linewidth": 1.4,

        # legend — off by default; components.legend() draws the house version
        "legend.frameon": False,
        "legend.fontsize": 9.5,

        # figure
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "figure.constrained_layout.use": False,
    })
    return palette, cats
