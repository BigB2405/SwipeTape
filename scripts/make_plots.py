"""D2 EDA figures, computed on the TRAIN split only (val/test untouched).

Usage: uv run python scripts/make_plots.py
Writes reports/figures/*.png and reports/eda_stats.json (numbers quoted in the data note).
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from genres import FAMILIES

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
FEATS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "speechiness", "liveness", "tempo", "loudness"]  # same as FEAT_COLS in public/app.js
sns.set_theme(style="whitegrid", context="notebook")

d = pd.read_parquet("data/processed/tracks_clean.parquet")
d = d[d.split == "train"].copy()
stats = {"train_rows": len(d)}

# 1. Popularity distribution vs. the app's filters
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(d.popularity, bins=range(0, 102, 2), color="#4c72b0")
for x, lab, h in [(40, "build_data.py cutoff (40)", 0.92), (62, "swipe deck cutoff (62)", 0.80)]:
    ax.axvline(x, color="#c44e52", ls="--", lw=1.5)
    ax.text(x + 1, ax.get_ylim()[1] * h, lab, color="#c44e52", fontsize=9)
ax.set(title="Track popularity (train split)", xlabel="Spotify popularity (0-100)", ylabel="Tracks")
fig.tight_layout(); fig.savefig(FIG / "1_popularity.png", dpi=150); plt.close(fig)
stats["popularity"] = {
    "median": float(d.popularity.median()), "share_zero": round(float((d.popularity == 0).mean()), 3),
    "share_ge_40": round(float((d.popularity >= 40).mean()), 3), "share_ge_62": round(float((d.popularity >= 62).mean()), 3),
    "artists_ge_62_share": round(d[d.popularity >= 62].primary_artist.nunique() / d.primary_artist.nunique(), 3),
}

# 2. Feature correlation heatmap
corr = d[FEATS + ["popularity"]].corr()
fig, ax = plt.subplots(figsize=(7.5, 6.2))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .7}, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation of audio features (train split)")
fig.tight_layout(); fig.savefig(FIG / "2_feature_correlation.png", dpi=150); plt.close(fig)
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
stats["top_correlations"] = {f"{a}~{b}": round(v, 2) for (a, b), v in pairs.head(5).items()}
stats["max_abs_corr_with_popularity"] = round(float(corr.popularity.drop("popularity").abs().max()), 2)

# 3. Feature distributions by genre family
order = [f for f in FAMILIES if f != "Other"]
long = d[d.genre_family.isin(order)].melt(id_vars="genre_family", value_vars=["energy", "danceability", "valence", "acousticness"])
g = sns.catplot(data=long, x="value", y="genre_family", col="variable", kind="box", order=order,
                col_wrap=2, height=3.4, aspect=1.35, showfliers=False, color="#8da0cb", sharex=True)
g.set_titles("{col_name}"); g.set_axis_labels("", "")
g.figure.suptitle("Audio features by genre family (train split)", y=1.02)
g.savefig(FIG / "3_features_by_family.png", dpi=150, bbox_inches="tight"); plt.close(g.figure)
# eta^2: share of feature variance explained by genre (fine-grained) and family
def eta2(col, by):
    m = d[col].mean(); grp = d.groupby(by)[col]
    return float(((grp.mean() - m) ** 2 * grp.size()).sum() / ((d[col] - m) ** 2).sum())
stats["eta2_genre"] = {c: round(eta2(c, "primary_genre"), 2) for c in FEATS}
stats["eta2_family"] = {c: round(eta2(c, "genre_family"), 2) for c in FEATS}

# 4. Genre label overlap (tracks listed under more than one genre)
pc = Counter(p for gs in d.genres for p in combinations(gs, 2))
top = pd.Series({f"{a} + {b}": n for (a, b), n in pc.most_common(15)}).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
top.plot.barh(ax=ax, color="#55a868")
ax.set(title="Most common genre pairs on the same track (train split)", xlabel="Tracks with both genres")
fig.tight_layout(); fig.savefig(FIG / "4_genre_overlap.png", dpi=150); plt.close(fig)
stats["multi_genre_share"] = round(float((d.n_genres > 1).mean()), 3)
stats["top_genre_pairs"] = {k: int(v) for k, v in top.iloc[::-1].head(5).items()}

# 5. PCA of standardized features, colored by family
X = StandardScaler().fit_transform(d[FEATS])
pca = PCA(2, random_state=42).fit(X)
P = pca.transform(X)
sample = d.assign(pc1=P[:, 0], pc2=P[:, 1]).sample(8000, random_state=42)
fig, ax = plt.subplots(figsize=(8.5, 6))
sns.scatterplot(data=sample, x="pc1", y="pc2", hue="genre_family", hue_order=list(FAMILIES), s=7, alpha=.55, linewidth=0, palette="tab20", ax=ax)
ax.set(title=f"PCA of 9 audio features ({pca.explained_variance_ratio_.sum():.0%} variance), 8k-track sample",
       xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=3, fontsize=8, title="Family")
fig.tight_layout(); fig.savefig(FIG / "5_pca_by_family.png", dpi=150); plt.close(fig)
stats["pca_var"] = [round(float(v), 3) for v in pca.explained_variance_ratio_]
stats["pca_loadings"] = {f"PC{i+1}": dict(zip(FEATS, np.round(c, 2).tolist())) for i, c in enumerate(pca.components_)}

Path("reports/eda_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
