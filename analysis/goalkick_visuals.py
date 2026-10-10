"""Navy-themed visuals for the goal-kick routine analysis.

Run goalkick_extract.py and goalkick_routines.py first. Writes PNGs to analysis/output/visuals/.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
VIS = os.path.join(OUT, "visuals")

# ---------------------------------------------------------------- navy theme
NAVY = "#0b1d3a"        # page + chart surface
PANEL = "#10264a"       # pitch / raised surface
GRID = "#21385f"
LINE = "#3a5480"        # axes, pitch lines
TEXT = "#eef2f8"
MUTED = "#9fb0c8"
# validated against NAVY (validate_palette.js --mode dark --surface #0b1d3a): all checks pass
C_LONG, C_BAIT, C_BUILD = "#3987e5", "#e8642d", "#1aa776"
STRATEGY = {
    "Direct long": (C_LONG, "o"),
    "Short, then long": (C_BAIT, "s"),
    "Short, play out": (C_BUILD, "D"),
}
SOURCE = "Impect event data · 216 matches · Bundesliga, 2. Bundesliga, 3. Liga, Eredivisie 2026-27"

plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY, "savefig.facecolor": NAVY,
    "text.color": TEXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
    "legend.frameon": False, "legend.labelcolor": TEXT,
})


def header(fig, title, subtitle):
    fig.text(0.02, 0.975, title, fontsize=14, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.925, subtitle, fontsize=9.5, color=MUTED, va="top")
    fig.text(0.02, 0.015, SOURCE, fontsize=7.5, color=MUTED, va="bottom")


def save(fig, name):
    os.makedirs(VIS, exist_ok=True)
    fig.savefig(os.path.join(VIS, name), dpi=170)
    plt.close(fig)


def wilson(p, n, z=1.645):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load():
    g = pd.read_csv(os.path.join(OUT, "goalkicks_labelled.csv"))
    g["strategy"] = np.where(g.gk_zone == "long", "Direct long",
                    np.where(g.plan.str.startswith("then long"), "Short, then long", "Short, play out"))
    g["path3"] = g.path_pos.str.split("-").str[:3].str.join("-") + " | " + g.plan
    return g


# ---------------------------------------------------------------- 1. strategy summary
def strategy_summary(g):
    t = g.groupby("strategy").agg(n=("shot", "size"), shot=("shot", "mean"), lost=("lost_own3", "mean"),
                                  xg=("xg_chain", "mean"), xga=("xg_against", "mean")).reindex(list(STRATEGY))
    lo, hi = wilson(t.shot, t.n)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.14, wspace=0.55)
    header(fig, "Short goal kicks create more shots, but cost about as much as they gain",
           "Each goal kick followed until the opponent wins the ball · bars show 90% intervals on shot rate")
    y = np.arange(len(t))[::-1]
    cols = [STRATEGY[s][0] for s in t.index]

    ax = axes[0]
    ax.barh(y, t.shot, color=cols, height=0.55)
    ax.errorbar(t.shot, y, xerr=[t.shot - lo, hi - t.shot], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
    for yi, v in zip(y, t.shot):
        ax.text(hi.max() + 0.004, yi, f"{v:.1%}", va="center", color=TEXT)
    ax.set_xlim(0, 0.12); ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.set_title("Possession ends in a shot", loc="left")

    ax = axes[1]
    ax.barh(y, t.lost, color=cols, height=0.55)
    for yi, v in zip(y, t.lost):
        ax.text(v + 0.004, yi, f"{v:.0%}", va="center", color=TEXT)
    ax.set_xlim(0, 0.17); ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.set_title("Ball lost in own third", loc="left")

    ax = axes[2]
    h = 0.32
    ax.barh(y + h / 2, t.xg * 100, color=cols, height=h)
    ax.barh(y - h / 2, t.xga * 100, color=MUTED, height=h)
    for yi, a, b in zip(y, t.xg * 100, t.xga * 100):
        ax.text(a + 0.03, yi + h / 2, f"{a:.2f} created", va="center", color=TEXT, fontsize=8)
        ax.text(b + 0.03, yi - h / 2, f"{b:.2f} conceded", va="center", color=MUTED, fontsize=8)
    ax.set_xlim(0, 2.1)
    ax.set_title("xG per 100 goal kicks", loc="left")

    for ax in axes:
        ax.set_yticks(y); ax.grid(axis="y", visible=False)
    axes[0].set_yticklabels([f"{s}\n(n={n:,})" for s, n in zip(t.index, t.n)], color=TEXT)
    axes[1].set_yticklabels([]); axes[2].set_yticklabels([])
    save(fig, "01_strategy_summary.png")


# ---------------------------------------------------------------- 2. risk / reward scatter
LABELS = {
    "GK short outside box → FB | build via left": "GK → LB outside box, build left",
    "GK long → opp half (right) | direct": "GK long, right side",
    "GK long → opp half (centre) | direct": "GK long, centre",
    "CB short in box → GK | then long by GK (centre)": "CB tap → GK, GK long to centre",
    "GK short in box → CB | build via right": "GK → CB in box, build right",
    "CB short in box → GK | build via left": "CB tap → GK, build left",
}


def risk_reward(g):
    t = g.groupby(["routine", "strategy"]).agg(n=("shot", "size"), shot=("shot", "mean"),
                                                lost=("lost_own3", "mean")).reset_index()
    t = t[t.n >= 30]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.12)
    header(fig, "Goal-kick routines: reward vs risk",
           "Each marker is a routine used at least 30 times · marker size = number of goal kicks")
    base = g.shot.mean()
    ax.axhline(base, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.06, base + 0.0025, f"average goal kick {base:.1%}", color=MUTED, fontsize=8, ha="left")
    for s, (c, m) in STRATEGY.items():
        d = t[t.strategy == s]
        ax.scatter(d.lost, d.shot, s=d.n * 1.1, c=c, marker=m, edgecolor=NAVY, linewidth=2, label=s, zorder=3)
    for _, r in t.iterrows():
        if r.routine in LABELS:
            right_edge = r.lost > 0.17
            ax.annotate(f"{LABELS[r.routine]} (n={r.n})", (r.lost, r.shot), xytext=(10, -34) if right_edge else (9, 11),
                        textcoords="offset points", fontsize=8.5, color=TEXT, ha="right" if right_edge else "left",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=7) if right_edge else None)
    ax.set_xlabel("Ball lost in own third (share of goal kicks)")
    ax.set_ylabel("Possession ends in a shot (share of goal kicks)")
    ax.set_xlim(-0.01, 0.235); ax.set_ylim(-0.005, 0.125)
    ax.set_xticks(np.arange(0, 0.21, 0.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, 0)); ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.text(0.0, 0.118, "safe & productive", color=MUTED, fontsize=8, style="italic")
    ax.text(0.232, 0.118, "productive but risky", color=MUTED, fontsize=8, style="italic", ha="right")
    leg = ax.legend(loc="lower right", markerscale=0.55, title="Strategy", title_fontsize=8.5, fontsize=8.5)
    leg.get_title().set_color(MUTED)
    save(fig, "02_risk_reward.png")


# ---------------------------------------------------------------- 3. top routines on the pitch
def draw_pitch(ax):
    ax.set_facecolor(PANEL)
    kw = dict(color=LINE, lw=1.1, zorder=1)
    ax.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, **kw))
    ax.plot([0, 0], [-34, 34], **kw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, **kw))
    for sgn in (-1, 1):
        x0 = 52.5 * sgn
        ax.add_patch(Rectangle((x0 - 16.5 * (sgn > 0), -20.16), 16.5, 40.32, fill=False, **kw))
        ax.add_patch(Rectangle((x0 - 5.5 * (sgn > 0), -9.16), 5.5, 18.32, fill=False, **kw))
        ax.add_patch(Arc((x0 - 11 * sgn, 0), 18.3, 18.3, theta1=127 if sgn > 0 else -53,
                         theta2=233 if sgn > 0 else 53, **kw))
    ax.set_xlim(-54, 54); ax.set_ylim(-35.5, 35.5)
    ax.set_aspect("equal"); ax.axis("off")


def top_routines_pitch(g):
    specs = [
        ("1  Short to the left-back, build down the left",
         g.routine == "GK short outside box → FB | build via left", C_BUILD),
        ("2  Bait the press, then GK goes long to the striker",
         g.path3 == "CB-GK-CF | then long by GK (centre)", C_BAIT),
        ("3  CB tap → GK → defensive midfielder through the middle",
         g.path3 == "CB-GK-DM | build via centre", C_BUILD),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.74, bottom=0.17, wspace=0.05)
    header(fig, "The three goal-kick routines that stood out",
           "Faint lines = every instance · bold arrows = typical (median) pass for each step · attacking left → right")
    for ax, (title, mask, col) in zip(axes, specs):
        draw_pitch(ax)
        s = g[mask]
        steps = {}
        for p in s.path:
            for k, (pos, sx, sy, ex, ey, _a, _r) in enumerate(json.loads(p)[:3]):
                ax.plot([sx, ex], [sy, ey], color=col, alpha=0.13, lw=0.8, zorder=2)
                steps.setdefault(k, []).append((pos, sx, sy, ex, ey))
        nodes = []  # [x, y, step numbers, positions]; passes starting within 5 m share a marker
        for k, rows in sorted(steps.items()):
            if len(rows) < 0.5 * len(s):
                continue
            arr = np.array([r[1:] for r in rows], dtype=float)
            sx, sy, ex, ey = np.median(arr, axis=0)
            pos = pd.Series([r[0] for r in rows]).mode()[0]
            if np.hypot(ex - sx, ey - sy) > 3:
                ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=14,
                                             color=TEXT, lw=2.2, zorder=4))
            near = next((nd for nd in nodes if np.hypot(nd[0] - sx, nd[1] - sy) < 5), None)
            if near:
                near[2].append(k + 1); near[3].append(pos)
            else:
                nodes.append([sx, sy, [k + 1], [pos]])
        for x, y_, ks, ps in nodes:
            ax.scatter([x], [y_], s=170 if len(ks) == 1 else 260, color=col, edgecolor=TEXT, linewidth=1.5, zorder=5)
            ax.text(x, y_, "·".join(map(str, ks)), ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=NAVY, zorder=6)
            ax.text(x, y_ - 4.5, " → ".join(ps), ha="center", va="top", fontsize=7.5, color=TEXT, zorder=6)
        ax.set_title(title, loc="left", fontsize=9.5, pad=6)
        n, shot, xg, lost = len(s), s.shot.mean(), s.xg_chain.mean(), s.lost_own3.mean()
        ax.text(-52.5, -38.5,
                f"n={n}   shot rate {shot:.0%}   xG/kick {xg:.3f}   lost in own third {lost:.0%}",
                fontsize=8.5, color=MUTED, va="top")
    save(fig, "03_top_routines_pitch.png")


# ---------------------------------------------------------------- 4. team ranking
def team_ranking(g):
    t = g.groupby("team_name").agg(n=("shot", "size"), net=("net_xg", "mean"), shot=("shot", "mean"),
                                   league=("league", "first"))
    t = t[t.n >= 30].sort_values("net")
    t = pd.concat([t.head(6), t.tail(10)])
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.85, bottom=0.1)
    header(fig, "Which teams get the most out of their goal kicks",
           "Net xG per 100 goal kicks (created minus conceded after a turnover) · teams with ≥30 goal kicks · top 10 and bottom 6")
    y = np.arange(len(t))
    v = t.net * 100
    ax.barh(y, v, color=[C_BUILD if x > 0 else C_BAIT for x in v], height=0.62)
    ax.axvline(0, color=MUTED, lw=1)
    for yi, x, s in zip(y, v, t.shot):
        ax.text(x + (0.06 if x >= 0 else -0.06), yi, f"{x:+.1f}  ({s:.0%} shots)",
                va="center", ha="left" if x >= 0 else "right", fontsize=8, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ·  {lg}" for n, lg in zip(t.index, t.league)], color=TEXT, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(v.min() - 1.2, v.max() + 1.2)
    ax.set_xlabel("Net xG per 100 goal kicks")
    gap = 5.5  # separator between bottom and top groups
    ax.axhline(gap, color=GRID, lw=1, ls=":")
    save(fig, "04_team_ranking.png")


def main():
    g = load()
    strategy_summary(g)
    risk_reward(g)
    top_routines_pitch(g)
    team_ranking(g)
    print("saved to", VIS)


if __name__ == "__main__":
    main()
