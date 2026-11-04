"""D3 model evaluation: popularity baseline vs. audio-feature recommenders.

Usage: uv run --frozen python scripts/evaluate.py
Reads data/processed/tracks_clean.parquet (train + val only; test.parquet is never opened).
Writes reports/d3_results.csv, reports/d3_summary.json, reports/d3_spot_checks.md.

Protocol (identical for every model)
  * Simulated listener = 5 liked tracks from one genre g + 5 disliked tracks from other families.
  * Candidate pool = same split, minus every track by a seed artist.
  * Each model returns a top-10 (max 2 per artist, like the app). Hit = g in the track's genres.
  * Scalers / PCA / classifiers are fit on train only. Hyperparameters are tuned with
    3-fold GroupKFold (by artist) *inside* train; val is used once for the final comparison.
  * Candidate models see only the 9 audio features: never genre, popularity, or artist.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from genres import FAMILY_OF

FEATS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "speechiness", "liveness", "tempo", "loudness"]          # FEAT_COLS in public/app.js
APP_W = np.array([1.0, 1.2, 1.0, 1.0, 0.8, 0.6, 0.3, 0.5, 0.6])  # FEAT_W in public/app.js
K, N_SEED, EPS_PER_GENRE, ARTIST_CAP, SEED = 10, 5, 5, 2, 42
N_BOOT = 2000
REP = Path("reports")


# ----------------------------------------------------------------- data
class Data:
    """One split as arrays, with genre / family membership masks."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.X = self.df[FEATS].to_numpy(float)
        self.pop = self.df.popularity.to_numpy(float)
        self.artist = pd.factorize(self.df.primary_artist)[0]
        self.tie = self.df.track_id.rank(method="first").to_numpy()  # deterministic tie-break
        ex = self.df.genres.explode()
        self.member = {g: np.zeros(self.n, bool) for g in ex.unique()}
        for g, idx in ex.groupby(ex).groups.items():
            self.member[g][np.asarray(idx)] = True
        self.fam_member = {}
        for g, m in self.member.items():
            f = FAMILY_OF.get(g, "Other")
            self.fam_member[f] = self.fam_member.get(f, np.zeros(self.n, bool)) | m
        self.family = self.df.genre_family.to_numpy()


def make_episodes(d: Data, n_seed: int, per_genre: int, seed: int, scaler: StandardScaler):
    rng = np.random.default_rng(seed)
    Zs = scaler.transform(d.X)
    eps = []
    for g in sorted(d.member):
        m = d.member[g]
        fam = FAMILY_OF.get(g, "Other")
        likes_from = np.flatnonzero(m)
        dislikes_from = np.flatnonzero(~d.fam_member[fam])
        if len(likes_from) < n_seed + 2 * K:
            continue
        for _ in range(per_genre):
            like = rng.choice(likes_from, n_seed, replace=False)
            dislike = rng.choice(dislikes_from, n_seed, replace=False)
            pool = ~np.isin(d.artist, d.artist[np.concatenate([like, dislike])])
            if (pool & m).sum() < K:
                continue
            z = Zs[like]
            coh = (np.mean([np.linalg.norm(z[i] - z[j]) for i in range(n_seed) for j in range(i + 1, n_seed)])
                   if n_seed > 1 else np.nan)
            eps.append(dict(id=len(eps), genre=g, family=fam, like=like, dislike=dislike, pool=pool,
                            seed_pop=float(np.median(d.pop[like])), coherence=float(coh),
                            rng_seed=int(rng.integers(2**31))))
    return eps


# --------------------------------------------------------------- models
def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class RandomRec:
    name, complexity = "B0 Random", 0
    def fit(self, tr): return self
    def prepare(self, d): return d.n
    def score(self, n, ep): return np.random.default_rng(ep["rng_seed"]).random(n)


class PopularityRec:
    name, complexity = "B1 Popularity", 0
    def fit(self, tr): return self
    def prepare(self, d): return d.pop
    def score(self, pop, ep): return pop


class OracleRec:
    """Sanity check only: reads the label. Shows the metric works and what leakage would do."""
    name, complexity = "Oracle (leaks label)", 99
    def fit(self, tr): return self
    def prepare(self, d): return d
    def score(self, d, ep): return d.member[ep["genre"]] + d.pop / 1000


class CentroidRec:
    """Cosine similarity to the liked centroid (minus lam x the disliked centroid), as in app.js."""

    def __init__(self, name, complexity, weights=None, n_pca=None, lam=0.0):
        self.name, self.complexity = name, complexity
        self.weights, self.n_pca, self.lam = weights, n_pca, lam

    def fit(self, tr):
        self.scaler = StandardScaler().fit(tr.X)
        self.pca = (PCA(self.n_pca, whiten=True, random_state=SEED).fit(self.scaler.transform(tr.X))
                    if self.n_pca else None)
        return self

    def prepare(self, d):
        Z = self.scaler.transform(d.X)
        if self.weights is not None:
            Z = Z * self.weights
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return Z, unit(Z)

    def score(self, R, ep):
        Z, Zn = R
        s = Zn @ unit(Z[ep["like"]].mean(0))
        if self.lam:
            s = s - self.lam * (Zn @ unit(Z[ep["dislike"]].mean(0)))
        return s


class GenreModelRec:
    """Classifier predicts a genre distribution per track; rank by similarity to the seeds' mean."""

    def __init__(self, name, complexity, est):
        self.name, self.complexity, self.est = name, complexity, est

    def fit(self, tr):
        self.scaler = StandardScaler().fit(tr.X)
        ex = tr.df[["genres", "n_genres"]].reset_index(drop=True).explode("genres")
        Xs = self.scaler.transform(tr.X)[ex.index.to_numpy()]
        self.est.fit(Xs, ex.genres.to_numpy(str), sample_weight=1.0 / ex.n_genres.to_numpy())
        return self

    def prepare(self, d):
        return self.est.predict_proba(self.scaler.transform(d.X))

    def score(self, P, ep):
        return P @ P[ep["like"]].mean(0)


# ----------------------------------------------------------- evaluation
def top_k(s, ep, d):
    s = np.where(ep["pool"], s, -np.inf)
    for width in (400, d.n):
        cand = np.argpartition(-s, min(width, d.n - 1))[:width]
        cand = cand[np.lexsort((d.tie[cand], -s[cand]))]
        picks, per = [], {}
        for i in cand:
            if not np.isfinite(s[i]):
                break
            a = d.artist[i]
            if per.get(a, 0) < ARTIST_CAP:
                per[a] = per.get(a, 0) + 1
                picks.append(i)
                if len(picks) == K:
                    return np.array(picks)
    return np.array(picks)


IDCG = np.cumsum(1 / np.log2(np.arange(2, K + 2)))


def run(models, d, eps, keep_picks=False):
    rows = []
    for m in models:
        R = m.prepare(d)
        for ep in eps:
            picks = top_k(m.score(R, ep), ep, d)
            rel = d.member[ep["genre"]][picks]
            n_rel = int((d.member[ep["genre"]] & ep["pool"]).sum())
            row = dict(model=m.name, episode=ep["id"], genre=ep["genre"], family=ep["family"],
                       p10=rel.mean(), ndcg10=float((rel / np.log2(np.arange(2, K + 2))).sum() / IDCG[min(n_rel, K) - 1]),
                       hit10=float(rel.any()), family_p10=d.fam_member[ep["family"]][picks].mean(),
                       chance=n_rel / ep["pool"].sum(), rec_pop_median=float(np.median(d.pop[picks])),
                       seed_pop=ep["seed_pop"], coherence=ep["coherence"])
            if keep_picks:
                row["picks"] = picks
            rows.append(row)
    return pd.DataFrame(rows)


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    return [round(float(np.percentile(x[idx].mean(1), q)), 4) for q in (2.5, 97.5)]


def paired(a, b, rng):
    diff = a - b
    stat = wilcoxon(a, b, zero_method="zsplit") if np.any(diff) else None
    return dict(mean_diff=round(float(diff.mean()), 4), ci95=boot_ci(diff, rng),
                wilcoxon_p=float(f"{stat.pvalue:.3g}") if stat else 1.0)


# ------------------------------------------------------------------ main
def candidate_grid():
    yield "M1d", [CentroidRec(f"M1d lam={l}", 2, APP_W, lam=l) for l in (0.15, 0.3, 0.5, 1.0)]
    yield "M2", [CentroidRec(f"M2 pca={k}", 3, n_pca=k) for k in (3, 5, 7, 9)]
    yield "M3", [GenreModelRec(f"M3 logreg C={c}", 4, LogisticRegression(C=c, max_iter=2000))
                 for c in (0.01, 1.0)] + \
                [GenreModelRec(f"M3 forest leaf={l}", 4, RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=l, n_jobs=-1, random_state=SEED)) for l in (20, 50)]


def tune(train_df):
    """3-fold GroupKFold by artist inside train. Returns per-config mean P@10 and the chosen configs."""
    folds = list(GroupKFold(3).split(train_df, groups=train_df.primary_artist))
    scores = {}
    for f, (a, b) in enumerate(folds):
        tr, ho = Data(train_df.iloc[a]), Data(train_df.iloc[b])
        eps = make_episodes(ho, N_SEED, EPS_PER_GENRE, SEED + f, StandardScaler().fit(tr.X))
        for fam, configs in candidate_grid():
            for m in configs:
                p = run([m.fit(tr)], ho, eps).p10.mean()
                scores.setdefault((fam, m.name), []).append(p)
                print(f"  fold {f} {m.name:<22} P@10={p:.4f}", flush=True)
    table = {f"{k[1]}": round(float(np.mean(v)), 4) for k, v in scores.items()}
    best = {}
    for (fam, name), v in scores.items():
        if fam not in best or np.mean(v) > np.mean(scores[(fam, best[fam])]):
            best[fam] = name
    return table, best


def build(name):
    for _, configs in candidate_grid():
        for m in configs:
            if m.name == name:
                return m
    raise KeyError(name)


def main():
    df = pd.read_parquet("data/processed/tracks_clean.parquet")
    assert set(df.split) == {"train", "val"}, "test rows must never be loaded before D6"
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    assert not set(train_df.primary_artist) & set(val_df.primary_artist), "artist leak across splits"

    print("Tuning inside train (GroupKFold by artist)...")
    tuning, chosen = tune(train_df)

    tr, va = Data(train_df), Data(val_df)
    models = [RandomRec(), PopularityRec(),
              CentroidRec("M1 Cosine (app weights)", 1, APP_W),
              build(chosen["M1d"]), build(chosen["M2"]), build(chosen["M3"]), OracleRec()]
    for m in models:
        m.fit(tr)
    scaler = StandardScaler().fit(tr.X)
    eps = make_episodes(va, N_SEED, EPS_PER_GENRE, SEED, scaler)
    for ep in eps:
        assert not ep["pool"][np.concatenate([ep["like"], ep["dislike"]])].any()
    print(f"Validation: {len(eps)} simulated users, {va.n} tracks")
    res = run(models, va, eps, keep_picks=True)

    rng = np.random.default_rng(SEED)
    by = {m: g.sort_values("episode") for m, g in res.groupby("model")}
    summary_rows = {}
    for m in models:
        g = by[m.name]
        recs = np.concatenate(g.picks.to_numpy())
        summary_rows[m.name] = dict(
            p10=round(g.p10.mean(), 4), p10_ci95=boot_ci(g.p10.to_numpy(), rng),
            ndcg10=round(g.ndcg10.mean(), 4), hit10=round(g.hit10.mean(), 4),
            family_p10=round(g.family_p10.mean(), 4),
            rec_pop_median=float(np.median(va.pop[recs])),
            coverage=round(len(np.unique(recs)) / va.n, 4))
    chance = round(float(by[models[0].name].chance.mean()), 4)

    # Selection rule (fixed before looking at val): best P@10; if a simpler candidate is not
    # significantly worse (paired 95% CI of the difference includes 0), take the simpler one.
    cands = [m for m in models if 1 <= m.complexity <= 4]
    best = max(cands, key=lambda m: summary_rows[m.name]["p10"])
    vs_best = {m.name: paired(by[m.name].p10.to_numpy(), by[best.name].p10.to_numpy(), rng)
               for m in cands if m is not best}
    tied = [m for m in cands if m is best or vs_best[m.name]["ci95"][1] >= 0]
    selected = min(tied, key=lambda m: (m.complexity, -summary_rows[m.name]["p10"]))
    base = by["B1 Popularity"].p10.to_numpy()
    vs_base = {m.name: paired(by[m.name].p10.to_numpy(), base, rng) for m in cands}
    dislike_effect = paired(by[build(chosen["M1d"]).name].p10.to_numpy(),
                            by["M1 Cosine (app weights)"].p10.to_numpy(), rng)

    # Error analysis on val (selected model vs. baseline and the other candidates)
    names = [m.name for m in models if m.complexity <= 4]
    r = res[res.model.isin(names)].copy()
    r["seed_pop_band"] = pd.cut(r.seed_pop, [-1, 19, 39, 61, 100], labels=["0-19", "20-39", "40-61", "62+"])
    r["coherence_q"] = pd.qcut(r.coherence, 4, labels=["Q1 tight", "Q2", "Q3", "Q4 spread"])

    def table(col):
        t = r.pivot_table(index=col, columns="model", values="p10", aggfunc="mean", observed=True)[names]
        t.insert(0, "n_users", r[r.model == selected.name].groupby(col, observed=True).size())
        return json.loads(t.round(4).to_json(orient="index"))

    sel = res[res.model == selected.name]
    miss_pairs, near = {}, [0, 0]
    for _, row in sel.iterrows():
        for i in row.picks:
            if not va.member[row.genre][i]:
                pg = va.df.primary_genre.iat[i]
                miss_pairs[(row.genre, pg)] = miss_pairs.get((row.genre, pg), 0) + 1
                near[0] += 1
                near[1] += int(va.fam_member[row.family][i])
    top_conf = sorted(miss_pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:15]

    print("Cold-start curve...")
    cold = {}
    for n in (1, 3, 5, 10):
        e = make_episodes(va, n, EPS_PER_GENRE, SEED + 100 + n, scaler)
        c = run([m for m in models if m.complexity <= 4], va, e)
        cold[n] = {"n_users": len(e), **c.groupby("model").p10.mean().round(4)[names].to_dict()}

    summary = dict(
        protocol=dict(k=K, n_seed=N_SEED, episodes_per_genre=EPS_PER_GENRE, artist_cap=ARTIST_CAP,
                      seed=SEED, n_users=len(eps), val_tracks=va.n, train_tracks=tr.n, bootstrap=N_BOOT),
        tuning_train_cv_p10=tuning, chosen_configs=chosen, chance_p10=chance,
        validation=summary_rows, paired_vs_popularity=vs_base, paired_vs_best=vs_best,
        dislike_effect_M1d_minus_M1=dislike_effect,
        best_by_p10=best.name, selected=selected.name,
        error_by_family=table("family"), error_by_seed_popularity=table("seed_pop_band"),
        error_by_seed_coherence=table("coherence_q"),
        misses_selected=dict(total=near[0], same_family_share=round(near[1] / max(near[0], 1), 4),
                             top_confusions=[dict(target=a, recommended_primary=b, n=n) for (a, b), n in top_conf]),
        cold_start_p10=cold)
    REP.mkdir(exist_ok=True)
    (REP / "d3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    res.drop(columns="picks").round(5).to_csv(REP / "d3_results.csv", index=False)

    # Spot checks: three hand-readable episodes
    lines = ["# D3 spot checks (validation)\n"]
    for fam in ("Metal", "Pop", "Jazz & Classical"):
        ep = next(e for e in eps if e["family"] == fam)
        g = ep["genre"]
        pk = sel[sel.episode == ep["id"]].picks.iat[0]
        show = lambda idx: "\n".join(f"| {va.df.track_name.iat[i][:40]} | {va.df.primary_artist.iat[i]} | "
                                     f"{', '.join(va.df.genres.iat[i])} |" for i in idx)
        lines += [f"## Target genre: {g} ({selected.name})\n", "**Liked seeds**\n", "| Track | Artist | Genres |",
                  "|---|---|---|", show(ep["like"]), "\n**Top 10**\n", "| Track | Artist | Genres |", "|---|---|---|",
                  show(pk), ""]
    (REP / "d3_spot_checks.md").write_text("\n".join(lines) + "\n")

    print(f"\nChance P@10 = {chance}")
    for k, v in summary_rows.items():
        print(f"{k:<26} P@10={v['p10']:.4f} {v['p10_ci95']}  nDCG={v['ndcg10']:.4f}  "
              f"famP@10={v['family_p10']:.4f}  recPop={v['rec_pop_median']}")
    print(f"Best by P@10: {best.name}; selected: {selected.name}")


if __name__ == "__main__":
    main()
