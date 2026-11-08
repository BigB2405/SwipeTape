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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc7"]  # fixed order
GRAY = "#9a9994"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = [m for m in S["validation"] if "ablation" not in m]
COLOR = {m: SERIES[i] for i, m in enumerate(MODELS)}  # color follows the model in every figure
SEL, APP, BASE = S["selected"], "M1 Cosine centroid (app)", "B1 Popularity"
P = S["protocol"]


def finish(ax, fig, name, title, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# 1. Model comparison: Recall@50 with bootstrap 95% CI, chance level marked
fig, ax = plt.subplots(figsize=(7, 3.6))
for y, m in enumerate(MODELS[::-1]):
    v = S["validation"][m]
    lo, hi = v["r50_ci95"]
    ax.barh(y, v["r50"], height=0.55, color=COLOR[m] if m == SEL else GRAY)
    ax.errorbar(v["r50"], y, xerr=[[v["r50"] - lo], [hi - v["r50"]]], color=INK, lw=1.2, capsize=3)
    ax.text(hi + 0.003, y, f"{v['r50']:.3f}", va="center", fontsize=9, color=INK2)
ax.axvline(P["chance_r50"], color=INK2, ls="--", lw=1)
ax.text(P["chance_r50"], -0.8, f" chance {P['chance_r50']:.3f}", fontsize=8, color=INK2, va="center")
ax.set_ylim(-1.1, len(MODELS) - 0.5)
ax.set_yticks(range(len(MODELS)), [m + ("  (selected)" if m == SEL else "") for m in MODELS[::-1]])
ax.set_xlabel(f"Recall@50 of held-out songs (n = {P['n_users']} listeners), 95% bootstrap CI")
finish(ax, fig, "d3_1_model_comparison.png", "Finding the listener's other songs: Recall@50 by model")

# 2. Recall@50 by genre family (single-taste listeners): selected model vs the app's current centroid
fam = S["error_by_family"]
order = sorted(fam, key=lambda f: fam[f][SEL])
fig, ax = plt.subplots(figsize=(7, 4.2))
for y, f in enumerate(order):
    ax.plot([fam[f][APP], fam[f][SEL]], [y, y], color=GRID, lw=2, zorder=1)
for m, z in ((BASE, 2), (APP, 3), (SEL, 4)):
    ax.scatter([fam[f][m] for f in order], range(len(order)), s=40, color=COLOR[m], label=m, zorder=z,
               edgecolor=SURF, lw=1.5)
ax.set_yticks(range(len(order)), [f"{f}  (n={fam[f]['n_users']})" for f in order])
ax.set_xlabel("Recall@50, single-taste listeners grouped by artist's family")
ax.legend(frameon=False, loc="lower right", fontsize=8)
finish(ax, fig, "d3_2_r50_by_family.png", "Where the selected model works and where it struggles")

# 3. Cold start: Recall@50 vs number of liked songs
cold = S["cold_start_r50"]
ns = sorted(cold, key=int)
fig, ax = plt.subplots(figsize=(7, 4.2))
for m in MODELS:
    ax.plot([int(n) for n in ns], [cold[n][m] for n in ns], color=COLOR[m], lw=2, marker="o", ms=5,
            markeredgecolor=SURF, markeredgewidth=1.5, label=m)
ax.set_xticks([int(n) for n in ns])
ax.set_xlabel("Number of liked songs (single-taste listeners; dislikes fixed at 5)")
ax.set_ylabel("Recall@50")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
finish(ax, fig, "d3_3_cold_start.png", "Even one liked song beats the popularity baseline", grid="y")

# 4. Single vs mixed taste
fig, ax = plt.subplots(figsize=(7, 3.6))
for y, m in enumerate(MODELS[::-1]):
    v = S["validation"][m]
    ax.plot([v["r50_mixed"], v["r50_single"]], [y, y], color=GRID, lw=2, zorder=1)
    ax.scatter(v["r50_single"], y, s=40, color=INK2, zorder=2, edgecolor=SURF, lw=1.5,
               label="one artist" if y == 0 else None)
    ax.scatter(v["r50_mixed"], y, s=40, color=COLOR[m] if m == SEL else GRAY, marker="D", zorder=3,
               edgecolor=SURF, lw=1.5, label="two artists, different families" if y == 0 else None)
ax.set_yticks(range(len(MODELS)), MODELS[::-1])
ax.set_xlabel(f"Recall@50 (single n = {P['n_single']}, mixed n = {P['n_mixed']})")
ax.legend(frameon=False, fontsize=8, loc="upper right")
finish(ax, fig, "d3_4_single_vs_mixed.png", "Mixed tastes: song-to-song models lose less than the centroid")
print("wrote", *sorted(p.name for p in FIG.glob("d3_*.png")))
