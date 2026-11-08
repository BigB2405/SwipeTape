"""D3 model evaluation: song-level recommendation quality (popularity baseline vs. audio models).

Usage: uv run --frozen python scripts/evaluate.py
Reads data/processed/tracks_clean.parquet (train + val only; test.parquet is never opened).
Writes reports/d3_results.csv, reports/d3_summary.json, reports/d3_spot_checks.md.

PRIMARY protocol: held-out songs ("taste fingerprint")
  * A simulated listener likes 5 songs by one artist (or 3 + 3 by two artists from different
    genre families = a "mixed taste" listener) and dislikes 5 songs from other families.
  * The artist's OTHER songs are held out and hidden in the whole candidate pool. A good model
    ranks those specific songs near the top: Recall@50, Recall@10, MRR.
  * Same-title copies of a seed (re-releases) are removed from both pool and held-out set.
  * Models see only the 9 audio features: never artist, album, genre, or popularity.
SECONDARY protocol (continuity with D2/first D3 draft): genre-match P@10 of a 10-song playlist.

Leakage controls: artist-grouped split (asserted); scalers / forests fit on train only;
hyperparameters tuned with 3-fold GroupKFold by artist inside train; val used once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swipetape.model import APP_W, FEATS, SongEmbedder, fit_genre_forest, taste_scores, unit  # noqa: E402
from genres import FAMILY_OF  # noqa: E402

K, N_SEED, ARTIST_CAP, SEED = 10, 5, 2, 42
MIN_TRACKS, MIN_HELD = 8, 3
N_BOOT = 2000
REP = Path("reports")


# ----------------------------------------------------------------- data
class Data:
    """One split as arrays, with genre / family membership masks."""

    def __init__(self, df: pd.DataFrame, tag: str):
        self.tag = tag
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.X = self.df[FEATS].to_numpy(float)
        self.pop = self.df.popularity.to_numpy(float)
        self.artist = pd.factorize(self.df.primary_artist)[0]
        self.album = pd.factorize(self.df.album_name)[0]
        self.key = pd.factorize(self.df.track_name.str.lower().str.strip() + "|" + self.df.primary_artist)[0]
        self.tie = self.df.track_id.rank(method="first").to_numpy()  # deterministic tie-break
        self.family = self.df.genre_family.to_numpy()
        ex = self.df.genres.explode()
        self.member = {g: np.zeros(self.n, bool) for g in ex.unique()}
        for g, idx in ex.groupby(ex).groups.items():
            self.member[g][np.asarray(idx)] = True
        self.fam_member = {}
        for g, m in self.member.items():
            f = FAMILY_OF.get(g, "Other")
            self.fam_member[f] = self.fam_member.get(f, np.zeros(self.n, bool)) | m


# ---------------------------------------------------- song-level episodes
def make_song_episodes(d: Data, n_seed: int, seed: int, max_single=None, n_mixed=0, min_tracks=MIN_TRACKS):
    rng = np.random.default_rng(seed)
    counts = np.bincount(d.artist)
    artists = np.flatnonzero(counts >= min_tracks)
    if max_single and len(artists) > max_single:
        artists = np.sort(rng.choice(artists, max_single, replace=False))
    tracks_of = {a: np.flatnonzero(d.artist == a) for a in artists}
    fam_of = {a: pd.Series(d.family[t]).mode().iat[0] for a, t in tracks_of.items()}
    eps = []

    def episode(kind, groups, n_each):
        like, held, held_other = [], [], []
        for a in groups:
            lk = rng.choice(tracks_of[a], n_each, replace=False)
            rest = np.setdiff1d(tracks_of[a], lk)
            rest = rest[~np.isin(d.key[rest], d.key[lk])]                       # drop re-releases of seeds
            like.append(lk), held.append(rest)
            held_other.append(rest[~np.isin(d.album[rest], d.album[lk])])
        fams = {fam_of[a] for a in groups}
        if min(len(h) for h in held) < MIN_HELD:
            return
        like = np.concatenate(like)
        dislike_from = ~np.isin(d.family, list(fams)) & ~np.isin(d.artist, groups)
        dislike = rng.choice(np.flatnonzero(dislike_from), n_seed, replace=False)
        excl = np.isin(np.arange(d.n), np.concatenate([like, dislike])) | np.isin(d.key, d.key[like])
        eps.append(dict(id=len(eps), kind=kind, family=" + ".join(sorted(fams)), like=like, dislike=dislike,
                        held=held, held_other_album=held_other, excl=excl,
                        artist=" + ".join(d.df.primary_artist.iat[tracks_of[a][0]] for a in groups),
                        seed_pop=float(np.median(d.pop[like])), rng_seed=int(rng.integers(2**31))))

    for a in artists:
        episode("single", [a], n_seed)
    tries = 0
    while sum(e["kind"] == "mixed" for e in eps) < n_mixed and tries < 50 * max(n_mixed, 1):
        tries += 1
        a, b = rng.choice(artists, 2, replace=False)
        if fam_of[a] != fam_of[b]:
            episode("mixed", [a, b], 3)
    return eps


# ------------------------------------------------- genre-protocol episodes
def make_genre_episodes(d: Data, n_seed: int, per_genre: int, seed: int):
    rng = np.random.default_rng(seed)
    eps = []
    for g in sorted(d.member):
        m = d.member[g]
        fam = FAMILY_OF.get(g, "Other")
        likes_from, dislikes_from = np.flatnonzero(m), np.flatnonzero(~d.fam_member[fam])
        if len(likes_from) < n_seed + 2 * K:
            continue
        for _ in range(per_genre):
            like = rng.choice(likes_from, n_seed, replace=False)
            dislike = rng.choice(dislikes_from, n_seed, replace=False)
            pool = ~np.isin(d.artist, d.artist[np.concatenate([like, dislike])])
            if (pool & m).sum() < K:
                continue
            eps.append(dict(id=len(eps), genre=g, family=fam, like=like, dislike=dislike, pool=pool,
                            rng_seed=int(rng.integers(2**31))))
    return eps


# --------------------------------------------------------------- models
FORESTS: dict = {}


def get_forest(tr: Data, leaf: int):
    """Genre forest fit on train-only standardized audio; cached so tuning doesn't refit it."""
    if (tr.tag, leaf) not in FORESTS:
        Xs = StandardScaler().fit(tr.X).transform(tr.X)
        FORESTS[(tr.tag, leaf)] = fit_genre_forest(Xs, tr.df.genres, leaf)
    return FORESTS[(tr.tag, leaf)]


class RandomRec:
    name, complexity = "B0 Random", 0
    def fit(self, tr): return self
    def prepare(self, d): return d.n
    def score(self, n, ep): return np.random.default_rng(ep["rng_seed"]).random(n)


class PopularityRec:
    name, complexity = "B1 Popularity", 0
    def fit(self, tr): return self
    def prepare(self, d): return d.pop
    def score(self, pop, ep): return pop.copy()


class CentroidRec:
    """Cosine similarity to the liked centroid (minus lam x the disliked centroid): the app's audio core."""

    def __init__(self, name, complexity, lam=0.0):
        self.name, self.complexity, self.lam = name, complexity, lam

    def fit(self, tr):
        self.scaler = StandardScaler().fit(tr.X)
        return self

    def prepare(self, d):
        Z = self.scaler.transform(d.X) * APP_W
        return Z, unit(Z)

    def score(self, R, ep):
        Z, Zn = R
        s = Zn @ unit(Z[ep["like"]].mean(0))
        if self.lam:
            s = s - self.lam * (Zn @ unit(Z[ep["dislike"]].mean(0)))
        return s


class GenreModelRec:
    """First D3 pick: forest predicts each song's genre mix; rank by overlap with the seeds' mean mix."""

    def __init__(self, name, complexity, leaf):
        self.name, self.complexity, self.leaf = name, complexity, leaf

    def fit(self, tr):
        self.scaler, self.forest = StandardScaler().fit(tr.X), get_forest(tr, self.leaf)
        return self

    def prepare(self, d):
        return self.forest.predict_proba(self.scaler.transform(d.X))

    def score(self, P, ep):
        return P @ P[ep["like"]].mean(0)


class SongKNNRec:
    """M4 (alpha=0): song-to-song similarity on audio. M5 (alpha>0): audio + forest genre-mix embedding."""

    def __init__(self, name, complexity, tau, lam, alpha=0.0, leaf=20):
        self.name, self.complexity = name, complexity
        self.tau, self.lam, self.alpha, self.leaf = tau, lam, alpha, leaf

    def fit(self, tr):
        forest = get_forest(tr, self.leaf) if self.alpha else None
        self.embed = SongEmbedder(alpha=self.alpha, min_leaf=self.leaf).fit(tr.df, forest=forest)
        return self

    def prepare(self, d):
        return self.embed.transform(d.df)

    def score(self, E, ep):
        return taste_scores(E, ep["like"], ep["dislike"], self.tau, self.lam)


# ----------------------------------------------------------- evaluation
def ranks_of(s, d):
    order = np.lexsort((d.tie, -s))
    pos = np.empty(d.n, int)
    pos[order] = np.arange(d.n)
    return pos


def song_metrics(pos, ep):
    held = np.concatenate(ep["held"])
    r = pos[held]
    norm = lambda k, n: min(n, k)
    other = np.concatenate(ep["held_other_album"])
    row = dict(r10=float((r < 10).sum() / norm(10, len(r))), r50=float((r < 50).sum() / norm(50, len(r))),
               mrr=float(1 / (r.min() + 1)), hit10=float((r < 10).any()),
               r50_other_album=float((pos[other] < 50).sum() / norm(50, len(other))) if len(other) else np.nan)
    if ep["kind"] == "mixed":
        per = [float((pos[h] < 50).sum() / norm(50, len(h))) for h in ep["held"]]
        row.update(both_served50=float(all((pos[h] < 50).any() for h in ep["held"])), min_taste_r50=min(per))
    return row


def run_song(models, d, eps):
    rows = []
    for m in models:
        R = m.prepare(d)
        for ep in eps:
            s = np.asarray(m.score(R, ep), float).copy()
            s[ep["excl"]] = -np.inf
            rows.append(dict(model=m.name, episode=ep["id"], kind=ep["kind"], family=ep["family"],
                             artist=ep["artist"], seed_pop=ep["seed_pop"], n_like=len(ep["like"]),
                             **song_metrics(ranks_of(s, d), ep)))
    return pd.DataFrame(rows)


def top_k(s, pool, d, k=K):
    s = np.where(pool, s, -np.inf)
    order = np.lexsort((d.tie, -s))
    picks, per = [], {}
    for i in order:
        if not np.isfinite(s[i]):
            break
        if per.get(d.artist[i], 0) < ARTIST_CAP:
            per[d.artist[i]] = per.get(d.artist[i], 0) + 1
            picks.append(i)
            if len(picks) == k:
                break
    return np.array(picks)


def run_genre(models, d, eps):
    rows = []
    for m in models:
        R = m.prepare(d)
        for ep in eps:
            picks = top_k(m.score(R, ep), ep["pool"], d)
            rows.append(dict(model=m.name, episode=ep["id"], genre=ep["genre"], family=ep["family"],
                             p10=d.member[ep["genre"]][picks].mean(),
                             family_p10=d.fam_member[ep["family"]][picks].mean(), picks=picks))
    return pd.DataFrame(rows)


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    return [round(float(np.percentile(x[idx].mean(1), q)), 4) for q in (2.5, 97.5)]


def paired(a, b, rng):
    diff = a - b
    stat = wilcoxon(a, b, zero_method="zsplit") if np.any(diff) else None
    return dict(mean_diff=round(float(diff.mean()), 4), ci95=boot_ci(diff, rng),
                wilcoxon_p=float(f"{stat.pvalue:.3g}") if stat else 1.0)


# ---------------------------------------------------------------- tuning
def grid_stage1():
    yield "M1d", [CentroidRec(f"M1d lam={l}", 2, lam=l) for l in (0.15, 0.3, 0.5, 1.0)]
    yield "M3", [GenreModelRec(f"M3 forest leaf={l}", 4, l) for l in (10, 20, 50)]
    yield "M4", [SongKNNRec(f"M4 tau={t} lam={l}", 3, t, l) for t in (0.03, 0.1, 0.3, 1.0) for l in (0, 0.15, 0.3)]


def grid_stage2(tau, lam):
    # D4 serving budget: a 100-tree forest with min_samples_leaf=5 is ~111 MB saved / ~1.5 GB in RAM
    # on its own, so leaf >= 10 only (full leaf-10 pipeline: ~102 MB saved, ~0.9 GB RAM).
    yield "M5", [SongKNNRec(f"M5 alpha={a} leaf={l}", 5, tau, lam, alpha=a, leaf=l)
                 for a in (0.5, 1.0, 2.0, 3.0, 4.0) for l in (10, 20, 50)]


def tune(train_df):
    """3-fold GroupKFold by artist inside train; metric = mean Recall@50 on song episodes."""
    folds = list(GroupKFold(3).split(train_df, groups=train_df.primary_artist))
    data = []
    for f, (a, b) in enumerate(folds):
        tr, ho = Data(train_df.iloc[a], f"fold{f}"), Data(train_df.iloc[b], f"fold{f}-ho")
        data.append((tr, ho, make_song_episodes(ho, N_SEED, SEED + f, max_single=300, n_mixed=150)))
    scores = {}

    def stage(grid_fn):
        for f, (tr, ho, eps) in enumerate(data):
            for fam, configs in grid_fn():
                for m in configs:
                    r = run_song([m.fit(tr)], ho, eps).r50.mean()
                    scores.setdefault((fam, m.name), []).append(r)
                    print(f"  fold {f} {m.name:<24} R@50={r:.4f}", flush=True)

    def best_of(fam):
        cands = {n: np.mean(v) for (fm, n), v in scores.items() if fm == fam}
        return max(cands, key=cands.get)

    stage(grid_stage1)
    m4 = next(m for _, c in grid_stage1() for m in c if m.name == best_of("M4"))
    stage(lambda: grid_stage2(m4.tau, m4.lam))
    table = {n: round(float(np.mean(v)), 4) for (_, n), v in scores.items()}
    chosen = {fam: best_of(fam) for fam in ("M1d", "M3", "M4", "M5")}
    return table, chosen, (m4.tau, m4.lam)


def build(name, m4_params):
    for grid in (grid_stage1(), grid_stage2(*m4_params)):
        for _, configs in grid:
            for m in configs:
                if m.name == name:
                    return m
    raise KeyError(name)


# ------------------------------------------------------------------ main
def main():
    df = pd.read_parquet("data/processed/tracks_clean.parquet")
    assert set(df.split) == {"train", "val"}, "test rows must never be loaded before D6"
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    assert not set(train_df.primary_artist) & set(val_df.primary_artist), "artist leak across splits"

    print("Tuning inside train (GroupKFold by artist)...")
    tuning, chosen, m4p = tune(train_df)

    tr, va = Data(train_df, "train"), Data(val_df, "val")
    models = [RandomRec(), PopularityRec(), CentroidRec("M1 Cosine centroid (app)", 1),
              build(chosen["M1d"], m4p), build(chosen["M4"], m4p), build(chosen["M3"], m4p),
              build(chosen["M5"], m4p)]
    # ablation (not a candidate): the selected kNN family with the dislike term switched off / on
    abl_lam = 0.3 if m4p[1] == 0 else 0.0
    ablation = SongKNNRec(f"M4 ablation lam={abl_lam}", 99, m4p[0], abl_lam)
    for m in models + [ablation]:
        m.fit(tr)

    eps = make_song_episodes(va, N_SEED, SEED, n_mixed=400)
    for ep in eps:
        held = np.concatenate(ep["held"])
        assert not ep["excl"][held].any() and ep["excl"][ep["like"]].all() and ep["excl"][ep["dislike"]].all()
    n_single = sum(e["kind"] == "single" for e in eps)
    print(f"Validation: {len(eps)} listeners ({n_single} single, {len(eps) - n_single} mixed), {va.n} tracks")
    res = run_song(models + [ablation], va, eps)

    rng = np.random.default_rng(SEED)
    by = {m: g.sort_values("episode") for m, g in res.groupby("model")}
    chance = round(float(np.mean([50 / (~e["excl"]).sum() for e in eps])), 4)

    def summarize(g):
        mx = g[g.kind == "mixed"]
        return dict(r50=round(g.r50.mean(), 4), r50_ci95=boot_ci(g.r50.to_numpy(), rng),
                    r10=round(g.r10.mean(), 4), mrr=round(g.mrr.mean(), 4), hit10=round(g.hit10.mean(), 4),
                    r50_single=round(g[g.kind == "single"].r50.mean(), 4), r50_mixed=round(mx.r50.mean(), 4),
                    r50_other_album=round(g.r50_other_album.mean(), 4),
                    both_tastes_served50=round(mx.both_served50.mean(), 4),
                    min_taste_r50=round(mx.min_taste_r50.mean(), 4))

    summary_rows = {m.name: summarize(by[m.name]) for m in models + [ablation]}

    # Selection rule (fixed before looking at val): best mean Recall@50; if a simpler candidate is not
    # significantly worse (paired 95% CI of the difference includes 0), take the simpler one.
    cands = [m for m in models if 1 <= m.complexity <= 5]
    best = max(cands, key=lambda m: summary_rows[m.name]["r50"])
    vs_best = {m.name: paired(by[m.name].r50.to_numpy(), by[best.name].r50.to_numpy(), rng)
               for m in cands if m is not best}
    tied = [m for m in cands if m is best or vs_best[m.name]["ci95"][1] >= 0]
    selected = min(tied, key=lambda m: (m.complexity, -summary_rows[m.name]["r50"]))
    base = by["B1 Popularity"].r50.to_numpy()
    vs_base = {m.name: paired(by[m.name].r50.to_numpy(), base, rng) for m in cands}
    sel_knn = build(chosen["M4"], m4p).name
    dislikes = dict(
        M1d_minus_M1=paired(by[chosen["M1d"]].r50.to_numpy(), by["M1 Cosine centroid (app)"].r50.to_numpy(), rng),
        M4_with_vs_without=paired(*(by[n].r50.to_numpy() for n in
                                    ((sel_knn, ablation.name) if m4p[1] else (ablation.name, sel_knn))), rng),
        M4_chosen_lam=m4p[1])
    mixed_gain = paired(by[sel_knn][by[sel_knn].kind == "mixed"].r50.to_numpy(),
                        by["M1 Cosine centroid (app)"][by["M1 Cosine centroid (app)"].kind == "mixed"].r50.to_numpy(),
                        rng)

    # Error analysis (validation)
    names = [m.name for m in models]
    r = res[res.model.isin(names)].copy()
    r["seed_pop_band"] = pd.cut(r.seed_pop, [-1, 19, 39, 61, 100], labels=["0-19", "20-39", "40-61", "62+"])

    def table(col, sub=None, metric="r50"):
        x = r if sub is None else r[sub(r)]
        t = x.pivot_table(index=col, columns="model", values=metric, aggfunc="mean", observed=True)[names]
        t.insert(0, "n_users", x[x.model == selected.name].groupby(col, observed=True).size())
        return json.loads(t.round(4).to_json(orient="index"))

    print("Cold-start curve...")
    cold = {}
    for n in (1, 3, 5):
        e = make_song_episodes(va, n, SEED + 100 + n)
        c = run_song(models, va, e)
        cold[n] = {"n_users": len(e), **c.groupby("model").r50.mean().round(4)[names].to_dict()}

    print("Secondary: genre-match playlist protocol...")
    geps = make_genre_episodes(va, N_SEED, 5, SEED)
    gres = run_genre(models, va, geps)
    secondary = {}
    for m in models:
        g = gres[gres.model == m.name]
        recs = np.concatenate(g.picks.to_numpy())
        secondary[m.name] = dict(p10=round(g.p10.mean(), 4), p10_ci95=boot_ci(g.p10.to_numpy(), rng),
                                 family_p10=round(g.family_p10.mean(), 4),
                                 rec_pop_median=float(np.median(va.pop[recs])),
                                 coverage=round(len(np.unique(recs)) / va.n, 4))

    summary = dict(
        protocol=dict(primary="held-out songs: Recall@50 (normalized by min(|held|, 50))", n_seed=N_SEED,
                      mixed_seeds="3 + 3", min_artist_tracks=MIN_TRACKS, min_held=MIN_HELD, seed=SEED,
                      n_users=len(eps), n_single=n_single, n_mixed=len(eps) - n_single, val_tracks=va.n,
                      train_tracks=tr.n, bootstrap=N_BOOT, chance_r50=chance, genre_protocol_users=len(geps)),
        tuning_train_cv_r50=tuning, chosen_configs=chosen, validation=summary_rows,
        paired_vs_popularity=vs_base, paired_vs_best=vs_best, dislike_effect=dislikes,
        mixed_taste_knn_minus_centroid=mixed_gain, best_by_r50=best.name, selected=selected.name,
        error_by_family=table("family", lambda x: x.kind == "single"),
        error_by_seed_popularity=table("seed_pop_band"), error_by_kind=table("kind"),
        error_other_album_r50=table("kind", metric="r50_other_album"),
        cold_start_r50=cold, secondary_genre_protocol=secondary)
    REP.mkdir(exist_ok=True)
    (REP / "d3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    res.round(5).to_csv(REP / "d3_results.csv", index=False)

    # Spot checks: the discovery playlist the app would show (seed artists excluded, max 2 per artist)
    sel = next(m for m in models if m.name == selected.name)
    R = sel.prepare(va)
    En = unit(R) if isinstance(R, np.ndarray) else None
    show = lambda i: f"{va.df.track_name.iat[i][:40]} | {va.df.primary_artist.iat[i]} | {', '.join(va.df.genres.iat[i])}"
    lines = [f"# D3 spot checks (validation, {selected.name})\n",
             "Playlist = top 10 with seed artists excluded (discovery mode), max 2 per artist.\n"]
    picks_eps = [next(e for e in eps if e["kind"] == "single" and e["family"] == "Metal"),
                 next(e for e in eps if e["kind"] == "single" and e["family"] == "Jazz & Classical"),
                 next(e for e in eps if e["kind"] == "mixed")]
    for ep in picks_eps:
        pool = ~np.isin(va.artist, va.artist[np.concatenate([ep["like"], ep["dislike"]])])
        pk = top_k(sel.score(R, ep), pool, va)
        lines += [f"## Listener: likes {ep['artist']} ({ep['family']})\n", "**Liked seeds**\n",
                  "| Track | Artist | Genres |", "|---|---|---|", *[f"| {show(i)} |" for i in ep["like"]],
                  "\n**Disliked seeds**\n", "| Track | Artist | Genres |", "|---|---|---|",
                  *[f"| {show(i)} |" for i in ep["dislike"]],
                  "\n**Playlist**\n", "| Track | Artist | Genres | Because you liked |", "|---|---|---|---|"]
        for i in pk:
            b = ep["like"][int(np.argmax(En[ep["like"]] @ En[i]))] if En is not None else ep["like"][0]
            lines.append(f"| {show(i)} | {va.df.track_name.iat[b][:30]} |")
        lines.append("")
    (REP / "d3_spot_checks.md").write_text("\n".join(lines) + "\n")

    print(f"\nChance R@50 = {chance}")
    for k, v in summary_rows.items():
        print(f"{k:<28} R@50={v['r50']:.4f} {v['r50_ci95']}  R@10={v['r10']:.4f}  MRR={v['mrr']:.4f}  "
              f"single={v['r50_single']:.4f} mixed={v['r50_mixed']:.4f} otherAlbum={v['r50_other_album']:.4f}")
    for k, v in secondary.items():
        print(f"  [genre] {k:<28} P@10={v['p10']:.4f} famP@10={v['family_p10']:.4f} "
              f"recPop={v['rec_pop_median']} cov={v['coverage']}")
    print(f"Best by R@50: {best.name}; selected: {selected.name}")


if __name__ == "__main__":
    main()
