"""D6 figures from reports/d6_summary.json (run scripts/final_test.py first).

Usage: uv run --frozen python scripts/plot_d6.py
Writes reports/figures/d6_*.png. Same style and model colors as the D3 figures.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("reports/figures")
S = json.loads(Path("reports/d6_summary.json").read_text())
INK, INK2, GRID, SURF, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9994"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
MODELS = list(S["test"])
SEL, APP, BASE = S["protocol"]["artifact"]["model"], "M1 Cosine centroid (app)", "B1 Popularity"
COLOR = {BASE: "#eb6834", APP: "#1baf7a", SEL: "#7a5cc7"}   # same hues as these models in the D3 figures
P = S["protocol"]
LABEL = {"B0 Random": "Random", BASE: "Popularity (baseline)", APP: "App's original method (M1)",
         SEL: "M5 hybrid kNN (frozen, selected)"}
LABEL.update({m: m.split(" ")[0] + " audio-only kNN" for m in MODELS if m.startswith("M4")})


def finish(ax, fig, name, title, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# 1. Final test Recall@50 with 95% CI; the D3 validation value as a hollow marker
fig, ax = plt.subplots(figsize=(7, 3.4))
for y, m in enumerate(MODELS):
    v = S["test"][m]
    lo, hi = v["r50_ci95"]
    ax.barh(y, v["r50"], height=0.55, color=COLOR[SEL] if m == SEL else GRAY)
    ax.errorbar(v["r50"], y, xerr=[[v["r50"] - lo], [hi - v["r50"]]], color=INK, lw=1.2, capsize=3)
    ax.scatter(S["validation_r50"][m], y, s=48, facecolor=SURF, edgecolor=INK, lw=1.5, zorder=4,
               label="validation (D3)" if y == 0 else None)
    ax.text(hi + 0.003, y, f"{v['r50']:.3f}", va="center", fontsize=9, color=INK2)
ax.axvline(P["chance_r50"], color=INK2, ls="--", lw=1)
ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
ax.set_xlabel(f"Recall@50, reserved test split (n = {P['n_users']} listeners), 95% CI", fontsize=9)
ax.legend(loc="lower right", frameon=False, fontsize=8)
finish(ax, fig, "d6_1_test_results.png", "Final test vs. validation: Recall@50 by model")

# 2. Recall@50 by genre family (single-taste test listeners): M5 with CI vs the app's method and popularity
fam, ci = S["error_by_family"], S["m5_by_family_ci"]
order = sorted(fam, key=lambda f: fam[f][SEL])
fig, ax = plt.subplots(figsize=(7, 4.2))
for y, f in enumerate(order):
    lo, hi = ci[f]["ci95"]
    ax.plot([lo, hi], [y, y], color=COLOR[SEL], lw=1.2, alpha=0.45, zorder=1)
for m, z in ((BASE, 2), (APP, 3), (SEL, 4)):
    ax.scatter([fam[f][m] for f in order], range(len(order)), s=42, color=COLOR[m], label=LABEL[m], zorder=z,
               edgecolor=SURF, lw=1.5)
ax.set_yticks(range(len(order)), [f"{f}  (n={fam[f]['n_users']})" for f in order])
ax.set_xlabel("Recall@50, single-taste test listeners (line = M5 95% CI)")
ax.legend(loc="lower right", frameon=False, fontsize=8)
finish(ax, fig, "d6_2_by_family.png", "Every genre family gains; small families have wide intervals")

# 3. Discovery check: how popular are the songs each model would show (top-10 playlists)?
ex = S["exposure_top10"]
fig, ax = plt.subplots(figsize=(7, 2.8))
for y, m in enumerate(MODELS):
    ax.barh(y, ex[m]["rec_pop_median"], height=0.55, color=COLOR.get(m, GRAY) if m in (BASE, SEL) else GRAY)
    ax.text(max(ex[m]["rec_pop_median"], P["catalog_pop_median"]) + 2, y, f"{ex[m]['rec_pop_median']:.0f}  ·  "
            f"{ex[m]['share_below_catalog_median']:.0%} below median", va="center", fontsize=8.5, color=INK2)
ax.axvline(P["catalog_pop_median"], color=INK2, ls="--", lw=1)
ax.text(P["catalog_pop_median"], -0.85, f" catalog median {P['catalog_pop_median']:.0f}", fontsize=8,
        color=INK2)
ax.set_xlim(0, 140)
ax.set_xticks(range(0, 101, 20))
ax.set_ylim(-1.1, len(MODELS) - 0.5)
ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
ax.set_xlabel("Median Spotify popularity of recommended songs (0-100)")
finish(ax, fig, "d6_3_popularity_exposure.png", "Popularity bias check: songs each model would show")
print("wrote", *sorted(p.name for p in FIG.glob("d6_*.png")))
