"""MLB 2025 in the Meridian house style (dark / ink-navy):
  1. spray chart — every ball in play on a baseball field, home runs flagged
  2. pitch map  — every pitch at the plate (catcher's view), home-run pitches flagged

Reads the per-game CSVs in "MLB 2025/xy" and writes PNGs to "MLB 2025/viz".
Run from the repo root:  python scripts/mlb_2025_viz.py
"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle, RegularPolygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from housestyle import components, style  # noqa: E402
from housestyle.colors import SEQUENTIAL_BLUE_DARK  # noqa: E402

MODE = "dark"
OUT = ROOT / "MLB 2025" / "viz"
SOURCE = "MLB StatsAPI game feeds via sportsdataverse/baseballr-data"
COLS = ["bat_side", "event", "is_in_play", "hit_coord_x", "hit_coord_y", "plate_x", "plate_z"]

# Gameday hit coordinates -> feet, home plate at the origin, +y towards centre field
HOME_X, HOME_Y, FT_PER_UNIT = 125.42, 198.27, 2.5


def load():
    files = sorted(glob.glob(str(ROOT / "MLB 2025" / "xy" / "*_pitches_xy.csv")))
    return pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)


def density_cmap():
    steps = [SEQUENTIAL_BLUE_DARK[k] for k in (100, 200, 300, 400, 500, 600, 700)]
    return LinearSegmentedColormap.from_list("meridian_blue_dark", steps)


def legend(ax, palette, density_label, accent_label, loc="lower left", marker="h"):
    handles = [
        Line2D([], [], marker=marker, linestyle="none", markersize=11 if marker == "h" else 6,
               markerfacecolor=SEQUENTIAL_BLUE_DARK[500], markeredgewidth=0),
        Line2D([], [], marker="o", linestyle="none", markersize=6,
               markerfacecolor=palette["accent"], markeredgecolor=palette["surface"],
               markeredgewidth=0.8),
    ]
    ax.legend(handles, [density_label, accent_label], loc=loc, frameon=False,
              labelcolor=palette["ink_secondary"], fontsize=9, handletextpad=0.4,
              borderaxespad=0.2)


# --------------------------------------------------------------------------- field
def fence(theta_deg):
    """Generic park: 330 ft down the lines, 400 ft to centre."""
    return 330 + 70 * np.cos(np.radians(2 * theta_deg))


def draw_field(ax, palette):
    line = dict(color=palette["ink_muted"], linewidth=0.9, solid_capstyle="round", zorder=5)
    th = np.linspace(-45, 45, 200)
    r = fence(th)
    fx, fy = r * np.sin(np.radians(th)), r * np.cos(np.radians(th))
    # outfield as a faint plane, one step off the surface
    ax.add_patch(Polygon(np.c_[np.r_[0, fx], np.r_[0, fy]], closed=True,
                         facecolor=palette["grid"], edgecolor="none", alpha=0.45, zorder=0))
    ax.plot(fx, fy, **line)
    ax.plot([fx[0], 0, fx[-1]], [fy[0], 0, fy[-1]], **line)  # foul lines
    # infield dirt arc (95 ft from the rubber) and the diamond
    a = np.linspace(-np.pi / 4 - 0.2, np.pi / 4 + 0.2, 100)
    ax.plot(95 * np.sin(a), 60.5 + 95 * np.cos(a), **{**line, "linewidth": 0.8})
    b = 90 / np.sqrt(2)
    ax.plot([0, b, 0, -b, 0], [0, b, 2 * b, b, 0], **line)
    for x, y in [(b, b), (0, 2 * b), (-b, b)]:
        ax.add_patch(RegularPolygon((x, y), 4, radius=4.5, orientation=np.pi / 4,
                                    facecolor=palette["ink_primary"], edgecolor="none", zorder=6))
    ax.add_patch(plt.Circle((0, 60.5), 4, facecolor=palette["ink_muted"], edgecolor="none", zorder=6))
    # distance markers, tick-ink
    for d, t in [(330, -45), (400, 0), (330, 45)]:
        rr = fence(t) - 16
        ax.text(rr * np.sin(np.radians(t)), rr * np.cos(np.radians(t)), f"{d} ft",
                ha="center", va="center", fontsize=8, color=palette["ink_muted"])


def spray_chart(d, palette):
    b = d[d.hit_coord_x.notna()].copy()
    b["x"] = FT_PER_UNIT * (b.hit_coord_x - HOME_X)
    b["y"] = FT_PER_UNIT * (HOME_Y - b.hit_coord_y)
    b.loc[b.bat_side == "L", "x"] *= -1  # mirror lefties: pull side is always left field
    b = b[b.y > -20]
    ang = np.degrees(np.arctan2(b.x, b.y))
    b["side"] = np.select([ang < -15, ang > 15], ["pull", "oppo"], "centre")
    hr = b[b.event == "Home Run"]
    share = lambda s: s.side.value_counts(normalize=True).reindex(["pull", "centre", "oppo"]) * 100
    s_all, s_hr = share(b), share(hr)

    fig = plt.figure(figsize=(8.5, 8.6))
    ax = fig.add_axes([0.05, 0.09, 0.90, 0.72])
    ax.set_aspect("equal")
    ax.axis("off")
    draw_field(ax, palette)
    ax.hexbin(b.x, b.y, gridsize=70, extent=(-330, 330, -20, 470), mincnt=3, cmap=density_cmap(),
              norm=LogNorm(), linewidths=0.25, edgecolors=palette["surface"], zorder=3)
    ax.scatter(hr.x, hr.y, s=4, color=palette["accent"], linewidths=0, alpha=0.55, zorder=4)

    # direct labels per direction: HR share vs all-BIP share
    for side, (x, y) in [("pull", (-300, 500)), ("centre", (0, 560)), ("oppo", (300, 500))]:
        ax.text(x, y + 9, f"{s_hr[side]:.0f}%", ha="center", va="bottom", fontsize=17,
                fontweight="bold", color=palette["accent"] if side == "pull" else palette["ink_primary"])
        ax.text(x, y + 2, f"{side.upper()} · of home runs\n{s_all[side]:.0f}% of all balls in play",
                ha="center", va="top", fontsize=8.5, color=palette["ink_secondary"], linespacing=1.4)
    ax.set_xlim(-360, 360)
    ax.set_ylim(-30, 600)
    legend(ax, palette, f"Balls in play, density ({len(b):,})", f"Home runs ({len(hr):,})")

    components.header(
        fig, kicker="MLB 2025 · Spray chart",
        title=f"Seven in ten home runs were pulled",
        dek=("Every tracked ball in play, regular season + postseason. Left-handed batters are mirrored\n"
             "so the pull side is always left field. Pull / oppo = more than 15° off dead centre."),
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)
    out = OUT / "mlb_2025_spray_chart_home_runs.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- plate
ZONE_X, ZONE_Z = 17 / 12 / 2 + 0.12, (1.5, 3.5)  # half plate + ball radius; typical zone height


def pitch_map(d, palette):
    p = d[d.plate_x.notna() & d.plate_x.between(-2.5, 2.5) & d.plate_z.between(-0.5, 5.5)]
    hr = p[(p.event == "Home Run") & p.is_in_play.astype(bool)]
    heart = lambda s: ((s.plate_x.abs() <= ZONE_X * 2 / 3)
                       & s.plate_z.between(ZONE_Z[0] + 1 / 6 * 2, ZONE_Z[1] - 1 / 6 * 2)).mean() * 100
    h_all, h_hr = heart(p), heart(hr)

    fig = plt.figure(figsize=(8.5, 8.6))
    ax = fig.add_axes([0.10, 0.17, 0.80, 0.61])
    ax.set_aspect("equal")
    ax.hexbin(p.plate_x, p.plate_z, gridsize=60, extent=(-2.5, 2.5, -0.5, 5.5), mincnt=3,
              cmap=density_cmap(), norm=LogNorm(), linewidths=0.25, edgecolors=palette["surface"],
              zorder=2)
    ax.scatter(hr.plate_x, hr.plate_z, s=4, color=palette["accent"], linewidths=0, alpha=0.6, zorder=4)

    zone = dict(fill=False, linewidth=1.3, zorder=5)
    ax.add_patch(Rectangle((-ZONE_X, ZONE_Z[0]), 2 * ZONE_X, ZONE_Z[1] - ZONE_Z[0],
                           edgecolor=palette["ink_primary"], **zone))
    hx, hz0, hz1 = ZONE_X * 2 / 3, ZONE_Z[0] + 1 / 3, ZONE_Z[1] - 1 / 3
    ax.add_patch(Rectangle((-hx, hz0), 2 * hx, hz1 - hz0, edgecolor=palette["ink_primary"],
                           linestyle=(0, (4, 3)), **{**zone, "linewidth": 1.3, "zorder": 6}))
    ax.add_patch(Polygon([(-0.708, 0), (0.708, 0), (0.708, -0.12), (0, -0.25), (-0.708, -0.12)],
                         facecolor=palette["ink_secondary"], edgecolor="none", zorder=5))

    ax.annotate(f"{h_hr:.0f}% of home-run pitches", xy=(hx, hz1), xytext=(1.2, 4.55),
                fontsize=12, fontweight="bold", color=palette["accent"], ha="left",
                arrowprops=dict(arrowstyle="-", color=palette["ink_muted"], linewidth=0.8, shrinkA=0, shrinkB=0))
    ax.text(1.25, 4.42, f"were over the heart of the plate\n(dashed box) — vs {h_all:.0f}% of all pitches",
            fontsize=8.5, color=palette["ink_secondary"], ha="left", va="top", linespacing=1.4)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("Horizontal location (ft from centre of plate, catcher's view)")
    ax.set_ylabel("Height above ground (ft)")
    ax.grid(axis="both", color=palette["grid"], linewidth=0.6)
    legend(ax, palette, f"All pitches, density ({len(p):,})", f"Home-run pitches ({len(hr):,})", loc="upper left")

    components.header(
        fig, kicker="MLB 2025 · Pitch location",
        title="Home runs came off the middle of the plate",
        dek=("Every tracked pitch at the moment it crossed the plate, regular season + postseason.\n"
             "Solid box = typical strike zone (1.5–3.5 ft); dashed box = its middle third."),
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)
    out = OUT / "mlb_2025_pitch_locations_home_runs.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    palette, _ = style.apply(MODE)
    data = load()
    print(spray_chart(data, palette))
    print(pitch_map(data, palette))
