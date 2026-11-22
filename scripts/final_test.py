"""D6 final test: the frozen model vs. baselines on the reserved test split (run once).

Usage: uv run --frozen python scripts/final_test.py
Reads  pipeline.joblib (the D4 artifact, fit on train + val; never refit here),
       data/processed/tracks_clean.parquet (train + val, to fit the comparison models the same way),
       data/processed/test.parquet (opened for the first time in D6; only this script reads it).
Writes reports/d6_summary.json, reports/d6_results.csv, reports/d6_spot_checks.md.

Protocol = D3's song-level protocol unchanged (scripts/evaluate.py): simulated single-taste listeners
(5 likes by one test artist) and 400 mixed-taste listeners (3 + 3 from two artists of different genre
families), 5 dislikes each; the artists' other songs are hidden among all test tracks; Recall@50.
Nothing is tuned here: M5 uses the saved pipeline's parameters, M4/M1 use the D3-chosen settings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (N_SEED, REP, SEED, CentroidRec, Data, PopularityRec, RandomRec,  # noqa: E402
                      SongKNNRec, boot_ci, make_genre_episodes, make_song_episodes, paired, run_genre,
                      run_song, top_k)
from swipetape.model import taste_scores, unit  # noqa: E402

D3 = json.loads((REP / "d3_summary.json").read_text())


class FrozenM5:
    """The D4 artifact's fitted SongEmbedder + its recommender settings, used as-is."""
    complexity = 5

    def __init__(self, bundle):
        self.pipe, self.meta = bundle["pipeline"], bundle["metadata"]
        self.name = self.meta["model"]
        rec = self.pipe.named_steps["rec"]
        self.tau, self.lam = rec.tau, rec.lam

    def fit(self, tr):          # frozen: nothing to fit
        return self

    def prepare(self, d):
        return self.pipe.named_steps["embed"].transform(d.df)

    def score(self, E, ep):
        return taste_scores(E, ep["like"], ep["dislike"], self.tau, self.lam)


def main():
    bundle = joblib.load("pipeline.joblib")
    m5 = FrozenM5(bundle)
    p = bundle["metadata"]["params"]
    assert m5.name == D3["selected"], f"artifact {m5.name} != D3 selection {D3['selected']}"
    assert (p["tau"], p["lam"]) == (m5.tau, m5.lam) == (0.3, 0.0)

    fit_df = pd.read_parquet("data/processed/tracks_clean.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    assert set(fit_df.split) == {"train", "val"} and set(test_df.split) == {"test"}
    assert not set(fit_df.primary_artist) & set(test_df.primary_artist), "artist leak into test"
    served = set(bundle["pipeline"].named_steps["rec"].meta_.track_id)
    assert not served & set(test_df.track_id), "a test track is in the served catalog"

    fit, te = Data(fit_df, "train+val"), Data(test_df, "test")
    m4 = D3["chosen_configs"]["M4"]
    models = [RandomRec(), PopularityRec(), CentroidRec("M1 Cosine centroid (app)", 1),
              SongKNNRec(m4, 3, m5.tau, m5.lam), m5]
    for m in models:
        m.fit(fit)          # baselines refit on train + val, like the served model; M5 untouched
    names = [m.name for m in models]

    eps = make_song_episodes(te, N_SEED, SEED, n_mixed=400)
    for ep in eps:
        assert not ep["excl"][np.concatenate(ep["held"])].any()
    n_single = sum(e["kind"] == "single" for e in eps)
    print(f"Test: {len(eps)} listeners ({n_single} single, {len(eps) - n_single} mixed), {te.n} tracks")
    res = run_song(models, te, eps)

    rng = np.random.default_rng(SEED)
    by = {m: g.sort_values("episode") for m, g in res.groupby("model")}
    chance = round(float(np.mean([50 / (~e["excl"]).sum() for e in eps])), 4)

    def summarize(g):
        mx = g[g.kind == "mixed"]
        return dict(r50=round(g.r50.mean(), 4), r50_ci95=boot_ci(g.r50.to_numpy(), rng),
                    r10=round(g.r10.mean(), 4), mrr=round(g.mrr.mean(), 4), hit10=round(g.hit10.mean(), 4),
                    r50_single=round(g[g.kind == "single"].r50.mean(), 4), r50_mixed=round(mx.r50.mean(), 4),
                    r50_other_album=round(g.r50_other_album.mean(), 4),
                    both_tastes_served50=round(mx.both_served50.mean(), 4))

    test = {n: summarize(by[n]) for n in names}
    r50 = lambda n: by[n].r50.to_numpy()
    comparisons = {
        "M5_vs_popularity": paired(r50(m5.name), r50("B1 Popularity"), rng),
        "M5_vs_app_centroid": paired(r50(m5.name), r50("M1 Cosine centroid (app)"), rng),
        "M5_vs_M4": paired(r50(m5.name), r50(m4), rng),
        "M5_vs_app_centroid_mixed": paired(by[m5.name][by[m5.name].kind == "mixed"].r50.to_numpy(),
                                           by["M1 Cosine centroid (app)"]
                                           [by["M1 Cosine centroid (app)"].kind == "mixed"].r50.to_numpy(), rng),
    }

    # error analysis, each cell with its n
    r = res.copy()
    r["seed_pop_band"] = pd.cut(r.seed_pop, [-1, 19, 39, 61, 100], labels=["0-19", "20-39", "40-61", "62+"])

    def table(col, sub=None, metric="r50"):
        x = r if sub is None else r[sub(r)]
        t = x.pivot_table(index=col, columns="model", values=metric, aggfunc="mean", observed=True)[names]
        t.insert(0, "n_users", x[x.model == m5.name].groupby(col, observed=True).size())
        return json.loads(t.round(4).to_json(orient="index"))

    fam_ci = {}
    for f, g in r[(r.kind == "single") & (r.model == m5.name)].groupby("family"):
        fam_ci[f] = dict(n=len(g), r50=round(g.r50.mean(), 4), ci95=boot_ci(g.r50.to_numpy(), rng))

    cold = {}
    for n in (1, 3, 5):
        e = make_song_episodes(te, n, SEED + 100 + n)
        c = run_song(models, te, e)
        cold[n] = {"n_users": len(e), **c.groupby("model").r50.mean().round(4)[names].to_dict()}

    # bias check 1 (popularity / discovery): the playlist the app would show = top 10, seed artists
    # excluded, max 2 per artist. How popular are the songs each model puts in front of people?
    pop_med = float(np.median(te.pop))
    exposure = {}
    for m in models:
        R = m.prepare(te)
        recs, same_fam = [], []
        for ep in eps:
            pool = ~np.isin(te.artist, te.artist[np.concatenate([ep["like"], ep["dislike"]])])
            pk = top_k(m.score(R, ep), pool, te)
            recs.append(pk)
            same_fam.append(np.isin(te.family[pk], ep["family"].split(" + ")).mean())
        recs = np.concatenate(recs)
        exposure[m.name] = dict(rec_pop_median=float(np.median(te.pop[recs])),
                                share_below_catalog_median=round(float((te.pop[recs] < pop_med).mean()), 4),
                                coverage=round(len(np.unique(recs)) / te.n, 4),
                                same_family_share=round(float(np.mean(same_fam)), 4))

    # secondary genre protocol (continuity with D2/D3)
    geps = make_genre_episodes(te, N_SEED, 5, SEED)
    gres = run_genre(models, te, geps)
    secondary = {m: dict(p10=round(g.p10.mean(), 4), family_p10=round(g.family_p10.mean(), 4))
                 for m, g in gres.groupby("model")}

    val = D3["validation"]
    summary = dict(
        protocol=dict(source="D3 song-level protocol (scripts/evaluate.py), unchanged", n_seed=N_SEED, seed=SEED,
                      n_users=len(eps), n_single=n_single, n_mixed=len(eps) - n_single, test_tracks=te.n,
                      fit_tracks=fit.n, chance_r50=chance, catalog_pop_median=pop_med,
                      artifact=dict(model=m5.name, built_at=m5.meta["built_at"], git_commit=m5.meta["git_commit"])),
        test=test, validation_r50={n: val[n]["r50"] for n in names if n in val},
        paired=comparisons, error_by_family=table("family", lambda x: x.kind == "single"),
        m5_by_family_ci=fam_ci, error_by_seed_popularity=table("seed_pop_band"), error_by_kind=table("kind"),
        cold_start_r50=cold, exposure_top10=exposure, secondary_genre_protocol=secondary)
    REP.mkdir(exist_ok=True)
    (REP / "d6_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    res.round(5).to_csv(REP / "d6_results.csv", index=False)

    # spot checks (test): what the app would actually show three listeners
    E = m5.prepare(te)
    En = unit(E)
    show = lambda i: (f"{te.df.track_name.iat[i][:40]} | {te.df.primary_artist.iat[i]} | "
                      f"{', '.join(te.df.genres.iat[i])} | {int(te.pop[i])}")
    lines = [f"# D6 spot checks (test split, frozen {m5.name})\n",
             "Playlist = top 10 with seed artists excluded (discovery mode), max 2 per artist.\n"]
    fams = [e["family"] for e in eps if e["kind"] == "single"]
    pick = [next(e for e in eps if e["kind"] == "single" and e["family"] == f)
            for f in pd.Series(fams).value_counts().index[:2]] + [next(e for e in eps if e["kind"] == "mixed")]
    for ep in pick:
        pool = ~np.isin(te.artist, te.artist[np.concatenate([ep["like"], ep["dislike"]])])
        pk = top_k(m5.score(E, ep), pool, te)
        lines += [f"## Listener: likes {ep['artist']} ({ep['family']})\n", "**Liked seeds**\n",
                  "| Track | Artist | Genres | Popularity |", "|---|---|---|---|",
                  *[f"| {show(i)} |" for i in ep["like"]],
                  "\n**Playlist**\n", "| Track | Artist | Genres | Popularity | Because you liked |",
                  "|---|---|---|---|---|"]
        for i in pk:
            b = ep["like"][int(np.argmax(En[ep["like"]] @ En[i]))]
            lines.append(f"| {show(i)} | {te.df.track_name.iat[b][:30]} |")
        lines.append("")
    (REP / "d6_spot_checks.md").write_text("\n".join(lines) + "\n")

    print(f"Chance R@50 = {chance}")
    for n, v in test.items():
        print(f"{n:<26} test R@50={v['r50']:.4f} {v['r50_ci95']}  (val {val.get(n, {}).get('r50', float('nan')):.4f})"
              f"  single={v['r50_single']:.4f} mixed={v['r50_mixed']:.4f}  otherAlbum={v['r50_other_album']:.4f}")
    for k, v in comparisons.items():
        print(f"  {k}: {v}")
    for n, v in exposure.items():
        print(f"  [exposure] {n:<26} {v}")


if __name__ == "__main__":
    main()
