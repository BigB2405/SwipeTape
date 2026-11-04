"""D3 figures from reports/d3_summary.json (run scripts/evaluate.py first).

Usage: uv run --frozen python scripts/plot_d3.py
Writes reports/figures/d3_*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("reports/figures")
S = json.loads(Path("reports/d3_summary.json").read_text())
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
GRAY = "#9a9994"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = [m for m in S["validation"] if not m.startswith("Oracle")]
COLOR = {m: SERIES[i] for i, m in enumerate(MODELS)}  # color follows the model in every figure
SEL = S["selected"]


def finish(ax, fig, name, title):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# 1. Model comparison: P@10 with bootstrap 95% CI, chance level marked
fig, ax = plt.subplots(figsize=(7, 3.4))
for y, m in enumerate(MODELS[::-1]):
    v = S["validation"][m]
    lo, hi = v["p10_ci95"]
    ax.barh(y, v["p10"], height=0.55, color=COLOR[m] if m == SEL else GRAY)
    ax.errorbar(v["p10"], y, xerr=[[v["p10"] - lo], [hi - v["p10"]]], color=INK, lw=1.2, capsize=3)
    ax.text(hi + 0.006, y, f"{v['p10']:.3f}", va="center", fontsize=9, color=INK2)
ax.axvline(S["chance_p10"], color=INK2, ls="--", lw=1)
ax.text(S["chance_p10"], -0.75, f" chance {S['chance_p10']:.3f}", fontsize=8, color=INK2, va="center")
ax.set_ylim(-1, len(MODELS) - 0.5)
ax.set_yticks(range(len(MODELS)), [m + ("  (selected)" if m == SEL else "") for m in MODELS[::-1]])
ax.set_xlabel(f"Precision@10 on validation users (n = {S['protocol']['n_users']}), 95% bootstrap CI")
finish(ax, fig, "d3_1_model_comparison.png", "Genre-match precision@10 by model")

# 2. P@10 by genre family: selected model vs popularity baseline
fam = S["error_by_family"]
order = sorted(fam, key=lambda f: fam[f][SEL])
fig, ax = plt.subplots(figsize=(7, 4))
base = "B1 Popularity"
for y, f in enumerate(order):
    ax.plot([fam[f][base], fam[f][SEL]], [y, y], color=GRID, lw=2, zorder=1)
ax.scatter([fam[f][base] for f in order], range(len(order)), s=40, color=COLOR[base], label=base, zorder=2,
           edgecolor=SURF, lw=1.5)
ax.scatter([fam[f][SEL] for f in order], range(len(order)), s=40, color=COLOR[SEL], label=SEL, zorder=3,
           edgecolor=SURF, lw=1.5)
ax.set_yticks(range(len(order)), [f"{f}  (n={fam[f]['n_users']})" for f in order])
ax.set_xlabel("Mean precision@10 (validation users whose seeds come from that family)")
ax.legend(frameon=False, loc="lower right", fontsize=9)
finish(ax, fig, "d3_2_p10_by_family.png", "Where the selected model works and where it fails")

# 3. Cold start: P@10 vs number of liked seeds
cold = S["cold_start_p10"]
ns = sorted(cold, key=int)
fig, ax = plt.subplots(figsize=(7, 4.2))
for m in MODELS:
    ax.plot([int(n) for n in ns], [cold[n][m] for n in ns], color=COLOR[m], lw=2, marker="o", ms=5,
            markeredgecolor=SURF, markeredgewidth=1.5, label=m)
ax.set_xticks([int(n) for n in ns])
ax.set_xlabel("Number of liked seed tracks (dislikes = same number)")
ax.set_ylabel("Precision@10")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
finish(ax, fig, "d3_3_cold_start.png", "Even one liked song beats the popularity baseline")
print("wrote", *sorted(p.name for p in FIG.glob("d3_*.png")))
