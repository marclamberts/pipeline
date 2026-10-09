"""Risk/reward chart of goal-kick routines (run goalkick_routines.py first)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
g = pd.read_csv(os.path.join(OUT, "goalkicks_labelled.csv"))
g["strategy"] = np.where(g.gk_zone == "long", "Direct long",
                np.where(g.plan.str.startswith("then long"), "Short, then long", "Short, play out"))
t = g.groupby(["routine", "strategy"]).agg(n=("shot", "size"), shot=("shot", "mean"),
                                            lost=("lost_own3", "mean"), net=("net_xg", "mean")).reset_index()
t = t[t.n >= 30]
STYLE = {"Direct long": ("#2a78d6", "o"), "Short, then long": ("#eb6834", "s"), "Short, play out": ("#1baf7a", "D")}
LABELS = {
    "GK short outside box → FB | build via left": "GK → LB outside box,\nbuild down the left",
    "GK long → opp half (right) | direct": "GK long, right side",
    "GK long → opp half (centre) | direct": "GK long, centre",
    "CB short in box → GK | then long by GK (centre)": "CB tap → GK, GK long to CF",
    "GK short in box → CB | build via right": "GK → CB in box, build right",
    "CB short in box → GK | build via left": "CB tap → GK, build left",
}
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
base = g.shot.mean()
ax.axhline(base, color="#a3a29d", lw=1, ls="--", zorder=0)
ax.text(0.23, base + 0.002, f"all goal kicks {base:.1%}", color="#52514e", fontsize=8, ha="right")
for s, (c, m) in STYLE.items():
    d = t[t.strategy == s]
    ax.scatter(d.lost, d.shot, s=d.n * 0.9, c=c, marker=m, edgecolor="#fcfcfb", linewidth=2, label=s, alpha=0.9, zorder=3)
for _, r in t.iterrows():
    if r.routine in LABELS:
        ax.annotate(f"{LABELS[r.routine]} (n={r.n})", (r.lost, r.shot), xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="#0b0b0b")
ax.set_xlabel("Ball lost in own third (share of goal kicks)", color="#52514e")
ax.set_ylabel("Possession ends in a shot (share of goal kicks)", color="#52514e")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color("#c3c2b7")
ax.tick_params(colors="#52514e", labelsize=8)
ax.set_xlim(-0.01, 0.235); ax.set_ylim(-0.005, 0.125)
ax.set_xticks(np.arange(0, 0.21, 0.05))
ax.legend(frameon=False, fontsize=8, loc="lower right", markerscale=0.6, title="Strategy (marker size = n)", title_fontsize=8)
ax.set_title("Goal-kick routines: shot rate vs own-third risk\n216 matches, Bundesliga / 2. BL / 3. Liga / Eredivisie 2026-27, routines with n≥30",
             loc="left", fontsize=10, color="#0b0b0b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "goalkick_routines_risk_reward.png"))
