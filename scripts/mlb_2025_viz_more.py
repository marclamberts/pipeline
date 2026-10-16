"""More MLB 2025 charts in the Meridian house style (dark / ink-navy):
  pitch movement · exit velocity vs launch angle · whiff rate by zone · hitter card ·
  pitcher card · fastball velocity by month · release-point consistency

Reads the per-game CSVs in "MLB 2025/xy" and writes PNGs to "MLB 2025/viz".
Run from the repo root:  python scripts/mlb_2025_viz_more.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from mlb_2025_viz import (FT_PER_UNIT, HOME_X, HOME_Y, MODE, OUT, SOURCE, components, density_cmap,
                          draw_field, glob, legend, ROOT, style)
from housestyle.colors import SEQUENTIAL_BLUE_DARK

SWINGS = {"Swinging Strike", "Swinging Strike (Blocked)", "Foul", "Foul Tip", "In play, out(s)",
          "In play, no out", "In play, run(s)", "Foul Bunt", "Missed Bunt"}
WHIFFS = {"Swinging Strike", "Swinging Strike (Blocked)", "Foul Tip", "Missed Bunt"}
POSTSEASON_START = "2025-09-30"
ZONE_HALF_W, ZONE_Z = 17 / 12 / 2 + 0.12, (1.5, 3.5)

HITTER = "Cal Raleigh"
PITCHER = "Paul Skenes"
RELEASE_PITCHER = "Jacob deGrom"


def load():
    files = sorted(glob.glob(str(ROOT / "MLB 2025" / "xy" / "*_pitches_xy.csv")))
    d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    d["swing"] = d.call.isin(SWINGS)
    d["whiff"] = d.call.isin(WHIFFS)
    d["hr"] = (d.event == "Home Run") & d.is_in_play.astype(bool)
    # horizontal movement from the pitcher's view: + = arm side
    d["arm_break"] = np.where(d.pitch_hand == "R", -d.pfx_x, d.pfx_x)
    return d


def new_fig(rect=(0.10, 0.12, 0.84, 0.62), size=(8.5, 8.0)):
    fig = plt.figure(figsize=size)
    return fig, fig.add_axes(rect)


def finish(fig, palette, name, kicker, title, dek, note=None):
    components.header(fig, kicker=kicker, title=title, dek=dek, palette=palette)
    components.footer(fig, source=SOURCE, note=note, palette=palette)
    out = OUT / name
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out)


# label offsets (points) per pitch type so neighbouring medians don't collide
OFFSETS = {"Sweeper": (-9, 7, "right"), "Slider": (9, 5, "left"), "Curveball": (-9, -12, "right"),
           "Knuckle Curve": (9, -12, "left"), "Changeup": (9, 6, "left"), "Splitter": (9, -13, "left"),
           "Sinker": (9, 7, "left"), "Cutter": (9, 6, "left"), "Four-Seam Fastball": (9, 6, "left")}


def label_pitch(ax, name, x, y, text, color, bold, offsets=None):
    dx, dy, ha = {**OFFSETS, **(offsets or {})}.get(name, (9, 6, "left"))
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9, color=color,
                fontweight="bold" if bold else "normal", zorder=6)


# --------------------------------------------------------------------------- 1. movement
def movement_chart(d, palette, accent_type="Sweeper"):
    top = d.pitch_type_desc.value_counts().index[:9]
    p = d[d.pitch_type_desc.isin(top) & d.arm_break.notna()]
    med = p.groupby("pitch_type_desc")[["arm_break", "pfx_z", "start_speed"]].median()
    sample = p.sample(60000, random_state=1)
    acc = sample[sample.pitch_type_desc == accent_type]
    rest = sample[sample.pitch_type_desc != accent_type]

    fig, ax = new_fig()
    ax.axhline(0, color=palette["axis"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=palette["axis"], linewidth=0.8, zorder=1)
    ax.scatter(rest.arm_break, rest.pfx_z, s=2, color=palette["axis"], alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(acc.arm_break, acc.pfx_z, s=2, color=palette["accent"], alpha=0.35, linewidths=0, zorder=3)
    for name, r in med.iterrows():
        is_acc = name == accent_type
        ax.scatter(r.arm_break, r.pfx_z, s=70, color=palette["accent"] if is_acc else palette["ink_primary"],
                   edgecolors=palette["surface"], linewidths=1.5, zorder=5)
        label_pitch(ax, name, r.arm_break, r.pfx_z, f"{name} · {r.start_speed:.0f} mph",
                    palette["accent"] if is_acc else palette["ink_primary"], is_acc)
    ax.set_xlim(-18, 18)
    ax.set_ylim(-14, 18)
    ax.set_xlabel("Horizontal movement (in)   ← glove side  ·  arm side →")
    ax.set_ylabel("Vertical movement vs. a spinless pitch (in)")
    ax.grid(axis="both", color=palette["grid"], linewidth=0.6)
    s = med.loc[accent_type]
    finish(fig, palette, "mlb_2025_pitch_movement.png", "MLB 2025 · Pitch movement",
           f"Sweepers break {abs(s.arm_break):.0f} inches to the glove side, more than any other pitch",
           "Pitcher's view, both hands combined. Large dots = median for each pitch type with its median speed;\n"
           "small dots = 60,000 randomly sampled pitches.")


# --------------------------------------------------------------------------- 2. EV / LA
def ev_la_chart(d, palette):
    b = d[d.launch_speed.notna() & d.launch_angle.notna()]
    zone = (b.launch_speed >= 98) & b.launch_angle.between(24, 34)
    share_contact, share_hr, hr_rate = zone.mean() * 100, zone[b.hr].mean() * 100, b.hr[zone].mean() * 100

    fig, ax = new_fig()
    hb = ax.hexbin(b.launch_speed, b.launch_angle, C=b.hr.astype(float), reduce_C_function=np.mean,
                   gridsize=(45, 38), extent=(50, 120, -60, 80), mincnt=15, cmap=density_cmap(),
                   linewidths=0.25, edgecolors=palette["surface"], zorder=2)
    ax.add_patch(Rectangle((98, 24), 120 - 98, 10, fill=False, edgecolor=palette["accent"], linewidth=1.8,
                           zorder=4))
    ax.text(55, 66, f"{share_hr:.0f}% of home runs", fontsize=12, fontweight="bold", color=palette["accent"])
    ax.plot([84, 98], [47, 34], color=palette["ink_muted"], linewidth=0.8, zorder=5)
    ax.text(55, 62, f"came from this box: 98+ mph at 24–34°.\nIt holds {share_contact:.0f}% of batted balls; "
            f"{hr_rate:.0f}% of them left the park.", fontsize=8.5, color=palette["ink_secondary"], va="top",
            linespacing=1.4)
    cax = fig.add_axes([0.70, 0.155, 0.22, 0.014])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=palette["ink_muted"], length=0)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cb.set_label("Home-run rate", fontsize=8.5, color=palette["ink_secondary"])
    ax.set_xlim(50, 120)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Exit velocity (mph)")
    ax.set_ylabel("Launch angle (°)")
    ax.grid(axis="both", color=palette["grid"], linewidth=0.6)
    finish(fig, palette, "mlb_2025_exit_velocity_launch_angle.png", "MLB 2025 · Quality of contact",
           f"{share_contact:.0f}% of batted balls produced two-thirds of the home runs",
           f"Every tracked batted ball ({len(b):,}). Each cell is coloured by the share of batted balls in it\n"
           "that were home runs (cells with at least 15 balls).")


# --------------------------------------------------------------------------- 3. whiff grid
def whiff_chart(d, palette):
    s = d[d.swing & d.plate_x.notna()].copy()
    # catcher's view, mirrored so + = away from the batter (RHB stand on the catcher's left)
    s["away_x"] = np.where(s.bat_side == "R", s.plate_x, -s.plate_x)
    cw, ch = 2 * ZONE_HALF_W / 3, (ZONE_Z[1] - ZONE_Z[0]) / 3
    xb = -ZONE_HALF_W - cw + cw * np.arange(6)
    zb = ZONE_Z[0] - ch + ch * np.arange(6)
    s["cx"] = pd.cut(s.away_x, xb, labels=False)
    s["cz"] = pd.cut(s.plate_z, zb, labels=False)
    g = s.dropna(subset=["cx", "cz"]).groupby(["cz", "cx"]).whiff.agg(["mean", "size"])
    grid = g["mean"].unstack().reindex(index=range(5), columns=range(5)).values
    low = grid[0]  # row below the zone
    overall = s.whiff.mean() * 100

    fig, ax = new_fig(rect=(0.16, 0.12, 0.64, 0.62))
    cmap = density_cmap()
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    for z in range(5):
        for x in range(5):
            v = grid[z, x]
            t = (v - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((xb[x] + 0.012, zb[z] + 0.012), cw - 0.024, ch - 0.024,
                                   facecolor=cmap(t), edgecolor="none", zorder=2))
            ax.text(xb[x] + cw / 2, zb[z] + ch / 2, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color=palette["surface"] if t > 0.55 else palette["ink_primary"],
                    fontweight="bold" if z == 0 else "normal", zorder=4)
    ax.add_patch(Rectangle((xb[0], zb[0]), 5 * cw, ch, fill=False, edgecolor=palette["accent"], linewidth=2.2,
                           zorder=5))
    ax.text(xb[-1] + 0.08, zb[0] + ch / 2, "Below\nthe zone", ha="left", va="center", fontsize=10,
            fontweight="bold", color=palette["accent"])
    ax.add_patch(Rectangle((-ZONE_HALF_W, ZONE_Z[0]), 2 * ZONE_HALF_W, ZONE_Z[1] - ZONE_Z[0], fill=False,
                           edgecolor=palette["ink_primary"], linewidth=1.4, zorder=5))
    ax.set_xlim(xb[0] - 0.05, xb[-1] + 0.05)
    ax.set_ylim(zb[0] - 0.05, zb[-1] + 0.05)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([xb[0] + cw / 2, 0, xb[-1] - cw / 2], ["Inside", "Middle", "Away"])
    ax.set_yticks([zb[0] + ch / 2, 2.5, zb[-1] - ch / 2], ["Low", "Mid", "High"])
    ax.spines["bottom"].set_visible(False)
    finish(fig, palette, "mlb_2025_whiff_rate_by_zone.png", "MLB 2025 · Swing and miss",
           f"Below the zone, bats go quiet: up to {np.nanmax(low):.0%} of swings miss",
           f"Whiff rate (misses ÷ swings) by location, catcher's view, mirrored so 'away' is always away\n"
           f"from the batter. Box = typical strike zone. League average: {overall:.0f}% of swings miss.")


# --------------------------------------------------------------------------- 4. hitter card
def hitter_card(d, palette, name=HITTER):
    h = d[(d.batter == name)]
    b = h[h.hit_coord_x.notna()].copy()
    b["x"] = FT_PER_UNIT * (b.hit_coord_x - HOME_X)
    b["y"] = FT_PER_UNIT * (HOME_Y - b.hit_coord_y)
    b.loc[b.bat_side == "L", "x"] *= -1
    hr = b[b.hr]
    ev = h[h.launch_speed.notna() & h.launch_angle.notna()]

    fig = plt.figure(figsize=(11, 7.6))
    ax1 = fig.add_axes([0.04, 0.11, 0.50, 0.64])
    ax2 = fig.add_axes([0.62, 0.17, 0.34, 0.54])
    ax1.set_aspect("equal")
    ax1.axis("off")
    draw_field(ax1, palette)
    ax1.scatter(b[~b.hr].x, b[~b.hr].y, s=9, color=SEQUENTIAL_BLUE_DARK[500], alpha=0.6, linewidths=0, zorder=3)
    ax1.scatter(hr.x, hr.y, s=22, color=palette["accent"], edgecolors=palette["surface"], linewidths=0.6,
                zorder=4)
    ax1.set_xlim(-360, 360)
    ax1.set_ylim(-30, 500)
    legend(ax1, palette, f"Other balls in play ({len(b) - len(hr):,})", f"Home runs ({len(hr)})", marker="o")
    ax1.set_title("Spray chart (pull side always left)", loc="left", fontsize=10, color=palette["ink_secondary"])

    ax2.scatter(ev[~ev.hr].launch_speed, ev[~ev.hr].launch_angle, s=9, color=SEQUENTIAL_BLUE_DARK[500],
                alpha=0.6, linewidths=0, zorder=2)
    ax2.scatter(ev[ev.hr].launch_speed, ev[ev.hr].launch_angle, s=22, color=palette["accent"],
                edgecolors=palette["surface"], linewidths=0.6, zorder=3)
    ax2.add_patch(Rectangle((98, 24), 22, 10, fill=False, edgecolor=palette["ink_muted"], linewidth=1,
                            linestyle=(0, (3, 3)), zorder=4))
    ax2.set_xlim(40, 120)
    ax2.set_ylim(-70, 80)
    ax2.set_xlabel("Exit velocity (mph)")
    ax2.set_ylabel("Launch angle (°)")
    ax2.grid(axis="both", color=palette["grid"], linewidth=0.6)
    ax2.set_title("Exit velocity vs launch angle (dashed = 98+ mph, 24–34°)", loc="left", fontsize=10,
                  color=palette["ink_secondary"])

    pull = (np.degrees(np.arctan2(hr.x, hr.y)) < -15).mean() * 100
    finish(fig, palette, "mlb_2025_hitter_card_cal_raleigh.png", f"MLB 2025 · Hitter card · {name}",
           f"{name} hit {len(hr)} home runs, {pull:.0f}% of them to the pull side",
           f"Every tracked ball in play, regular season + postseason. Switch-hitter: left-handed at-bats are\n"
           f"mirrored so the pull side is always left field. Median home-run exit velocity "
           f"{ev[ev.hr].launch_speed.median():.1f} mph.")


# --------------------------------------------------------------------------- 5. pitcher card
def pitcher_card(d, palette, name=PITCHER):
    k = d[d.pitcher == name]
    a = k.groupby("pitch_type_desc").agg(n=("swing", "size"), mph=("start_speed", "mean"), sw=("swing", "sum"),
                                         wh=("whiff", "sum"), hb=("arm_break", "median"), vb=("pfx_z", "median"))
    a = a[a.n >= 50]
    a["use"] = a.n / a.n.sum()
    a["whiff"] = a.wh / a.sw
    a = a.sort_values("use", ascending=False)
    best = a[a.sw >= 40].whiff.idxmax()

    fig = plt.figure(figsize=(11, 7.6))
    ax1 = fig.add_axes([0.07, 0.17, 0.42, 0.54])
    ax2 = fig.add_axes([0.66, 0.17, 0.28, 0.54])
    pts = k[k.pitch_type_desc.isin(a.index)]
    ax1.axhline(0, color=palette["axis"], linewidth=0.8)
    ax1.axvline(0, color=palette["axis"], linewidth=0.8)
    ax1.scatter(pts[pts.pitch_type_desc != best].arm_break, pts[pts.pitch_type_desc != best].pfx_z, s=6,
                color=palette["axis"], alpha=0.55, linewidths=0, zorder=2)
    ax1.scatter(pts[pts.pitch_type_desc == best].arm_break, pts[pts.pitch_type_desc == best].pfx_z, s=8,
                color=palette["accent"], alpha=0.6, linewidths=0, zorder=3)
    for t, r in a.iterrows():
        c = palette["accent"] if t == best else palette["ink_primary"]
        ax1.scatter(r.hb, r.vb, s=60, color=c, edgecolors=palette["surface"], linewidths=1.4, zorder=5)
        label_pitch(ax1, t, r.hb, r.vb, t, c, t == best,
                    offsets={"Changeup": (10, -15, "left"), "Sinker": (9, 9, "left")})
    ax1.set_xlim(-16, 18)
    ax1.set_ylim(-10, 15)
    ax1.set_xlabel("Horizontal movement (in)   ← glove · arm →")
    ax1.set_ylabel("Vertical movement vs. spinless (in)")
    ax1.grid(axis="both", color=palette["grid"], linewidth=0.6)
    ax1.set_title("Arsenal, pitcher's view", loc="left", fontsize=10, color=palette["ink_secondary"])

    y = np.arange(len(a))[::-1]
    bars = ax2.barh(y, a.whiff * 100, height=0.62)
    for i, (t, r) in enumerate(a.iterrows()):
        bars.patches[i].set_color(palette["accent"] if t == best else palette["axis"])
        ax2.text(r.whiff * 100 + 1, y[i], f"{r.whiff:.0%}", va="center", fontsize=9,
                 color=palette["accent"] if t == best else palette["ink_secondary"],
                 fontweight="bold" if t == best else "normal")
    ax2.set_yticks(y, [f"{t}\n{r.use:.0%} use · {r.mph:.0f} mph" for t, r in a.iterrows()], fontsize=8.5)
    ax2.set_xlim(0, max(a.whiff * 100) * 1.25)
    ax2.set_xlabel("Whiff rate (misses ÷ swings, %)")
    ax2.grid(axis="x", color=palette["grid"], linewidth=0.6)
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Whiff rate by pitch", loc="left", fontsize=10, color=palette["ink_secondary"])

    r = a.loc[best]
    finish(fig, palette, "mlb_2025_pitcher_card_paul_skenes.png", f"MLB 2025 · Pitcher card · {name}",
           f"{name}{chr(39) if name.endswith('s') else chr(39) + 's'} {best.lower()} got a miss on {r.whiff:.0%} of swings",
           f"Every tracked pitch, regular season ({len(k):,} pitches). Large dots = median movement per pitch type;\n"
           f"pitch types thrown fewer than 50 times are left out.")


# --------------------------------------------------------------------------- 6. velocity
def velocity_chart(d, palette):
    f = d[(d.pitch_type == "FF") & d.start_speed.notna()].copy()
    f["post"] = f.date >= POSTSEASON_START
    reg = f[~f.post].assign(m=lambda x: pd.to_datetime(x.date).dt.month)
    reg = reg[reg.m >= 4]  # March is only the opening few days; folded into April
    by = reg.groupby("m").start_speed.mean()
    post = f[f.post].start_speed.mean()
    labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]

    fig, ax = new_fig(rect=(0.10, 0.14, 0.84, 0.58), size=(8.5, 7.0))
    x = np.arange(len(by))
    ax.plot(x, by.values, color=palette["axis"], linewidth=2.4, marker="o", markersize=6, zorder=3)
    ax.plot([x[-1], x[-1] + 1], [by.values[-1], post], color=palette["accent"], linewidth=2.4,
            linestyle=(0, (3, 2)), zorder=3)
    ax.scatter([x[-1] + 1], [post], s=80, color=palette["accent"], edgecolors=palette["surface"], linewidths=1.5,
               zorder=4)
    for xi, v in zip(x, by.values):
        ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=9, color=palette["ink_secondary"])
    ax.annotate(f"{post:.1f} mph", (x[-1] + 1, post), xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color=palette["accent"])
    ax.set_xticks(list(x) + [x[-1] + 1], labels + ["Postseason"])
    ax.set_ylim(93.5, 96.5)
    ax.set_xlim(-0.4, x[-1] + 1.4)
    ax.set_ylabel("Average four-seam fastball (mph)")
    finish(fig, palette, "mlb_2025_fastball_velocity_by_month.png", "MLB 2025 · Velocity",
           "Fastballs gained more than a full mph once October arrived",
           f"Average speed of every four-seam fastball by month (March folded into April); postseason games\n"
           f"from {POSTSEASON_START}. {len(f):,} fastballs.")


# --------------------------------------------------------------------------- 7. release point
def release_chart(d, palette, name=RELEASE_PITCHER):
    r = d[d.x0.notna() & d.z0.notna()]
    per = r.groupby("pitcher").filter(lambda g: len(g) >= 1000).groupby("pitcher").agg(
        x0=("x0", "mean"), z0=("z0", "mean"), sx=("x0", "std"), sz=("z0", "std"))
    per["spread_in"] = np.hypot(per.sx, per.sz) * 12
    pct = (per.spread_in > per.loc[name, "spread_in"]).mean() * 100
    me = r[r.pitcher == name]

    fig, ax = new_fig(rect=(0.10, 0.14, 0.84, 0.58), size=(8.5, 7.6))
    ax.scatter(per.x0, per.z0, s=26, color=SEQUENTIAL_BLUE_DARK[400], alpha=0.8, edgecolors=palette["surface"],
               linewidths=0.6, zorder=2)
    ax.scatter(me.x0, me.z0, s=5, color=palette["accent"], alpha=0.45, linewidths=0, zorder=3)
    ax.annotate(f"{name}: every pitch", (me.x0.mean() - 0.25, me.z0.mean() + 0.2), xytext=(-3.9, 6.75),
                fontsize=10, fontweight="bold", color=palette["accent"],
                arrowprops=dict(arrowstyle="-", color=palette["ink_muted"], linewidth=0.8, shrinkA=2, shrinkB=0))
    ax.axvline(0, color=palette["axis"], linewidth=0.8, zorder=1)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(4, 7.1)
    ax.set_xlabel("Release side (ft, catcher's view: right-handers on the left)")
    ax.set_ylabel("Release height (ft)")
    ax.grid(axis="both", color=palette["grid"], linewidth=0.6)
    legend(ax, palette, f"Average release point per pitcher (1,000+ pitches, {len(per)})",
           f"{name}, {len(me):,} pitches", loc="lower left", marker="o")
    finish(fig, palette, "mlb_2025_release_point_consistency.png", "MLB 2025 · Release point",
           f"{name} lets go from the same spot, pitch after pitch",
           f"His release points sit within {per.loc[name, 'spread_in']:.1f} in. of each other on average (one standard deviation),\n"
           f"tighter than {pct:.0f}% of the {len(per)} pitchers with 1,000+ pitches. League median: "
           f"{per.spread_in.median():.1f} in.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    palette, _ = style.apply(MODE)
    data = load()
    for fn in (movement_chart, ev_la_chart, whiff_chart, hitter_card, pitcher_card, velocity_chart, release_chart):
        fn(data, palette)
