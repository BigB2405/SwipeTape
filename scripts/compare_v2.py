"""v2 model check: M5 (forest genre guess) vs. real genre tags (+ optional popularity boost).

Usage: uv run --frozen python scripts/compare_v2.py
Same protocol as scripts/evaluate.py (D3): fit on train, score the validation listeners
(single-artist + mixed-taste), test split never opened. Writes reports/v2_compare.{csv,md}.

Metrics per model
  r50          Recall@50 of the held-out songs (D3 primary metric; same-artist proxy)
  r50_mixed    Recall@50 for the mixed-taste listeners only
  genre_agree  share of the 10-song playlist (artist cap 2) that shares a genre tag with a liked song
  kids_rate    share of the playlist tagged children/kids/sleep when no liked song is
  n_genres     distinct genre tags in the playlist (variety)

Caveat: real tags raise genre_agree almost by construction, and the held-out songs share their
artist's tags, so r50 also gets easier. Read the numbers together with the spot checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import (K, N_SEED, SEED, Data, SongKNNRec, make_song_episodes, run_song,  # noqa: E402
                      top_k)
from pipeline_def import SongEmbedder, taste_scores  # noqa: E402

KIDS = {"children", "kids", "sleep"}
REP = ROOT / "reports"


class LabelKNNRec(SongKNNRec):
    """Song kNN on [audio | alpha * real genre tags], plus pop_w * popularity/100."""

    def __init__(self, name, tau, alpha, pop_w):
        super().__init__(name, 6, tau, 0.0, alpha=alpha)
        self.pop_w = pop_w

    def fit(self, tr):
        self.embed = SongEmbedder(alpha=self.alpha, genre_source="labels").fit(tr.df)
        return self

    def prepare(self, d):
        return self.embed.transform(d.df), d.pop / 100

    def score(self, R, ep):
        E, pop = R
        return taste_scores(E, ep["like"], ep["dislike"], self.tau, self.lam) + self.pop_w * pop


class PopBoostM5(SongKNNRec):
    """M5 unchanged except the popularity boost (separates the two effects)."""

    def __init__(self, name, pop_w):
        super().__init__(name, 6, 0.3, 0.0, alpha=2.0, leaf=10)
        self.pop_w = pop_w

    def prepare(self, d):
        return self.embed.transform(d.df), d.pop / 100

    def score(self, R, ep):
        E, pop = R
        return taste_scores(E, ep["like"], ep["dislike"], self.tau, self.lam) + self.pop_w * pop


def playlist_metrics(m, d, eps):
    R = m.prepare(d)
    tags = d.df.genres.map(set).to_numpy()
    rows = []
    for ep in eps:
        s = np.asarray(m.score(R, ep), float).copy()
        picks = top_k(s, ~ep["excl"], d, K)
        liked = set().union(*tags[ep["like"]])
        agree = np.mean([bool(tags[i] & liked) for i in picks])
        kids = np.nan if liked & KIDS else np.mean([bool(tags[i] & KIDS) for i in picks])
        rows.append(dict(model=m.name, episode=ep["id"], genre_agree=agree, kids_rate=kids,
                         n_genres=len(set().union(*tags[picks]))))
    return pd.DataFrame(rows)


def spot(m, d, title_artist):
    R = m.prepare(d)
    df = d.df
    i = np.flatnonzero((df.track_name == title_artist[0]) & (df.primary_artist == title_artist[1]))
    if not len(i):
        return None
    ep = dict(like=i[:1], dislike=np.array([], int))
    s = np.asarray(m.score(R, ep), float).copy()
    s[i[:1]] = -np.inf
    s[(df.track_name.str.lower() == title_artist[0].lower()).to_numpy()] = -np.inf
    return [f"{df.track_name.iat[j][:28]} – {df.primary_artist.iat[j][:18]}" for j in top_k(s, np.ones(d.n, bool), d, 6)]


def md_table(df):
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return head + "".join("\n| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))


def main():
    df = pd.read_parquet(ROOT / "data/processed/tracks_clean.parquet")
    assert set(df.split) == {"train", "val"}, "test rows must never be loaded here"
    tr, va = Data(df[df.split == "train"], "train"), Data(df[df.split == "val"], "val")

    models = [SongKNNRec("M5 (live) alpha=2 leaf=10", 5, 0.3, 0.0, alpha=2.0, leaf=10),
              PopBoostM5("M5 + pop 0.05", 0.05)]
    models += [LabelKNNRec(f"Tags alpha={a} pop={p}", 0.3, a, p) for a in (0.1, 0.2, 0.3, 0.5, 1.0) for p in (0.0, 0.05)]
    for m in models:
        m.fit(tr)

    eps = make_song_episodes(va, N_SEED, SEED, n_mixed=400)
    print(f"validation: {len(eps)} listeners, {va.n} tracks", flush=True)
    rows = []
    for m in models:
        r = run_song([m], va, eps)
        p = playlist_metrics(m, va, eps)
        rows.append(dict(model=m.name, r50=r.r50.mean(), r50_mixed=r[r.kind == "mixed"].r50.mean(),
                         genre_agree=p.genre_agree.mean(), kids_rate=p.kids_rate.mean(), n_genres=p.n_genres.mean()))
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float) else str(v) for k, v in rows[-1].items()), flush=True)
    res = pd.DataFrame(rows).round(4)
    REP.mkdir(exist_ok=True)
    res.to_csv(REP / "v2_compare.csv", index=False)

    # spot checks use the full catalog (train + val), like the live app
    full = Data(df, "full")
    best = res.iloc[2:].sort_values("r50", ascending=False).model.iat[0]
    picks = {m.name: m for m in models}
    lines = ["# v2 comparison (validation; test split not opened)", "", md_table(res), "",
             f"Best tag model by R@50: **{best}**", "", "## Spot checks (full catalog)"]
    for song in [("Gangsta's Paradise", "Coolio"), ("Never Enough", "Loren Allred"), ("Unholy (feat. Kim Petras)", "Sam Smith")]:
        lines.append(f"\n**{song[0]}**")
        for name in ("M5 (live) alpha=2 leaf=10", best):
            m = picks[name]
            m.fit(full)
            lines.append(f"- {name}: " + "; ".join(spot(m, full, song) or ["(not found)"]))
    (REP / "v2_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
