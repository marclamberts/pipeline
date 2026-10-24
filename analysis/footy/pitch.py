"""Pitch drawing and the standard charts used by the reports (matplotlib, static PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

from .data import BOX_Y, HALF_LEN, HALF_WID, POST_Y  # noqa: E402

SURFACE = "#fcfcfb"
LINE = "#c9c8c2"
INK = "#0b0b0b"
INK_2 = "#52514e"
TEAM_COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2
GOOD, BAD, NEUTRAL = "#1baf7a", "#e34948", "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": LINE, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def draw_pitch(ax, half: bool = False):
    lw, c = 1.2, LINE
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-HALF_LEN, -HALF_WID), 2 * HALF_LEN, 2 * HALF_WID, fill=False, ec=c, lw=lw))
    ax.plot([0, 0], [-HALF_WID, HALF_WID], c=c, lw=lw)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec=c, lw=lw))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * HALF_LEN - (s > 0) * 16.5, -BOX_Y), 16.5, 2 * BOX_Y, fill=False, ec=c, lw=lw))
        ax.add_patch(Rectangle((s * HALF_LEN - (s > 0) * 5.5, -9.16), 5.5, 18.32, fill=False, ec=c, lw=lw))
        ax.add_patch(Rectangle((s * HALF_LEN - (s < 0) * 1.2, -POST_Y), 1.2, 2 * POST_Y, fill=False, ec=c, lw=lw))
        ax.plot(s * (HALF_LEN - 11), 0, "o", c=c, ms=2)
        ax.add_patch(Arc((s * (HALF_LEN - 11), 0), 18.3, 18.3, theta1=127 if s > 0 else -53,
                         theta2=233 if s > 0 else 53, ec=c, lw=lw))
    if half:
        ax.set_xlim(10, HALF_LEN + 2)
    else:
        ax.set_xlim(-HALF_LEN - 2, HALF_LEN + 2)
    ax.set_ylim(-HALF_WID - 2, HALF_WID + 2)
    return ax


def _size(xg):
    return 40 + 900 * np.asarray(xg, float)


def shot_map(shots: pd.DataFrame, teams: tuple[str, str], path, title: str):
    """Both teams on one pitch: first team attacks right, second attacks left."""
    fig, ax = plt.subplots(figsize=(10, 6.8))
    draw_pitch(ax)
    for k, team in enumerate(teams):
        s = shots[shots.team == team]
        sign = 1 if k == 0 else -1
        x, y = sign * s.x, sign * s.y
        col = TEAM_COLORS[k]
        miss = ~s.goal
        ax.scatter(x[miss], y[miss], s=_size(s.xg[miss]), facecolor="none", edgecolor=col, lw=1.6, zorder=3)
        ax.scatter(x[s.goal], y[s.goal], s=_size(s.xg[s.goal]), facecolor=col, edgecolor=SURFACE, lw=2, zorder=4)
        ax.text(sign * HALF_LEN * 0.5, HALF_WID + 4, f"{team}  {s.xg.sum():.2f} xG from {len(s)} shots · {int(s.goal.sum())} scored",
                ha="center", color=INK, fontsize=10)
        ax.scatter([], [], s=80, facecolor=col, edgecolor=col, label=team)
    for v, lab in ((0.05, "0.05"), (0.2, "0.20"), (0.5, "0.50 xG")):
        ax.scatter([], [], s=_size(v), facecolor="none", edgecolor=INK_2, label=lab)
    ax.scatter([], [], s=80, facecolor=INK_2, label="goal (filled)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False, fontsize=9)
    ax.set_title(title, pad=24)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def single_team_shot_map(shots: pd.DataFrame, color: str, path, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 6.8))
    draw_pitch(ax, half=True)
    miss = ~shots.goal
    ax.scatter(shots.x[miss], shots.y[miss], s=_size(shots.xg[miss]), facecolor="none", edgecolor=color, lw=1.6, zorder=3)
    ax.scatter(shots.x[shots.goal], shots.y[shots.goal], s=_size(shots.xg[shots.goal]),
               facecolor=color, edgecolor=SURFACE, lw=2, zorder=4)
    ax.set_title(title)
    ax.text(31, -HALF_WID - 5, f"{len(shots)} shots · {shots.xg.sum():.2f} xG · {int(shots.goal.sum())} goals · filled = goal, size = xG",
            ha="center", color=INK_2, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def xg_race(shots: pd.DataFrame, goals: list[dict], teams: tuple[str, str], path, title: str):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    end = max(95.0, float(shots.t.max()) + 2 if len(shots) else 95.0)
    labelled: list[float] = []
    for k, team in enumerate(teams):
        s = shots[shots.team == team].sort_values("t")
        ts, cum = s["t"].to_numpy(), s["xg"].cumsum().to_numpy()
        prev = np.r_[0, cum[:-1]]
        t = np.r_[0, np.repeat(ts, 2), end]
        c = np.r_[0, np.column_stack([prev, cum]).ravel(), cum[-1] if len(cum) else 0]
        ax.plot(t, c, color=TEAM_COLORS[k], lw=2, label=team)
        ax.text(end + 0.5, c[-1], f"{team} {c[-1]:.2f}", color=INK, va="center", fontsize=9)
        for g in (g for g in goals if g["team"] == team):
            yv = s[s["t"] <= g["t"] + 0.05]["xg"].sum()
            ax.plot(g["t"], yv, "o", ms=9, color=TEAM_COLORS[k], mec=SURFACE, mew=2, zorder=4)
            who = g["scorer"] + (" (og)" if g["own_goal"] else "")
            near = sum(1 for p in labelled if abs(p - g["t"]) < 8)
            labelled.append(g["t"])
            ax.annotate(who, (g["t"], yv), xytext=(0, 9 + 12 * near), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
    ax.axvline(45, color=LINE, lw=1, ls="--")
    ax.set_xlim(0, end)
    ax.set_xlabel("minute")
    ax.set_ylabel("cumulative xG")
    ax.grid(axis="y", color=LINE, lw=0.5, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def lane_bars(counts: dict[str, float], path, title: str, against: dict | None = None,
              labels: tuple[str, str] = ("for", "against"), subtitle: str = ""):
    """Final-third entries per lane, shown in pitch order (left wing at the top)."""
    lanes = list(counts)
    names = [l.replace("_", " ").title() for l in lanes]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    yy = np.arange(len(lanes))[::-1]
    h = 0.38 if against else 0.6
    series = [(counts, TEAM_COLORS[0], labels[0], h / 2 if against else 0)]
    if against:
        series.append((against, TEAM_COLORS[1], labels[1], -h / 2))
    for vals, col, lab, off in series:
        ax.barh(yy + off, [vals[l] for l in lanes], height=h * 0.92, color=col, label=lab)
        for y0, l in zip(yy, lanes):
            ax.text(vals[l], y0 + off, f" {vals[l]:.1f}", va="center", fontsize=8, color=INK_2)
    if against:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_yticks(yy, names)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("per match")
    ax.set_title(title, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def event_map(points: pd.DataFrame, path, title: str, color: str, subtitle: str = "", half: bool = False,
              arrows: bool = False, outcome_col: str | None = None):
    """Scatter (optionally arrows start→end) of events in the acting team's frame (attacking right)."""
    fig, ax = plt.subplots(figsize=(6.5 if half else 10, 6.8))
    draw_pitch(ax, half=half)
    if arrows and len(points):
        for _, r in points.iterrows():
            c = color if outcome_col is None else (GOOD if r[outcome_col] else NEUTRAL)
            ax.annotate("", (r.end_x, r.end_y), (r.x, r.y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, alpha=0.75, shrinkA=0, shrinkB=0))
        if outcome_col:
            ax.plot([], [], color=GOOD, lw=2, label="won / led to shot")
            ax.plot([], [], color=NEUTRAL, lw=2, label="other")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=9)
    else:
        ax.scatter(points.x, points.y, s=36, color=color, edgecolor=SURFACE, lw=1, alpha=0.85, zorder=3)
    ax.set_title(title, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.0, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
