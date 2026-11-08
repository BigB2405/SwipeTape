"""SwipeTape song-to-song recommender (shared by D3 evaluation and the D4 API).

Pipeline([("embed", SongEmbedder(...)), ("rec", SwipeRecommender(...))])

* SongEmbedder (custom transformer) turns a track's 9 audio features into a song embedding:
      [ unit(weighted standardized audio) | alpha * unit(sqrt(forest genre probabilities)) ]
  Learned state: the StandardScaler and (if alpha > 0) a random forest that predicts a
  track's genre mix from audio. The genre labels are used only to *fit* the forest; at
  transform time only the 9 audio features are read.
* SwipeRecommender stores the catalog embeddings and ranks every catalog song by how close it
  is to the songs the listener liked (a soft maximum over liked songs, so two different tastes
  are both served instead of being averaged away), minus lam x closeness to disliked songs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "speechiness", "liveness", "tempo", "loudness"]           # FEAT_COLS in public/app.js
APP_W = np.array([1.0, 1.2, 1.0, 1.0, 0.8, 0.6, 0.3, 0.5, 0.6])   # FEAT_W in public/app.js
SEED = 42


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def fit_genre_forest(Xs: np.ndarray, genres: pd.Series, min_leaf: int, n_estimators: int = 100,
                     random_state: int = SEED) -> RandomForestClassifier:
    """Forest on standardized audio -> genre. Multi-genre tracks appear once per genre, weight 1/n."""
    g = pd.Series(list(genres), dtype=object).reset_index(drop=True)
    ex = g.explode()
    w = 1.0 / g.map(len).to_numpy()
    idx = ex.index.to_numpy()
    est = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_leaf, n_jobs=-1,
                                 random_state=random_state)
    return est.fit(Xs[idx], ex.to_numpy(str), sample_weight=w[idx])


class SongEmbedder(BaseEstimator, TransformerMixin):
    """Audio features -> song embedding (see module docstring)."""

    def __init__(self, alpha: float = 1.0, min_leaf: int = 20, n_estimators: int = 100,
                 weights=tuple(APP_W), random_state: int = SEED):
        self.alpha, self.min_leaf, self.n_estimators = alpha, min_leaf, n_estimators
        self.weights, self.random_state = weights, random_state

    def fit(self, X: pd.DataFrame, y=None, forest: RandomForestClassifier | None = None):
        """X needs FEATS columns, plus a `genres` (list) column when alpha > 0.
        `forest` lets the caller pass a forest already fit on the same X (used to share work in tuning)."""
        self.scaler_ = StandardScaler().fit(X[FEATS].to_numpy(float))
        self.forest_ = None
        if self.alpha:
            self.forest_ = forest if forest is not None else fit_genre_forest(
                self.scaler_.transform(X[FEATS].to_numpy(float)), X["genres"], self.min_leaf,
                self.n_estimators, self.random_state)
        self.n_features_in_ = len(FEATS)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Zs = self.scaler_.transform(X[FEATS].to_numpy(float))
        parts = [unit(Zs * np.asarray(self.weights))]
        if self.forest_ is not None:
            parts.append(self.alpha * unit(np.sqrt(self.forest_.predict_proba(Zs))))
        return np.hstack(parts)


def taste_scores(E: np.ndarray, like: np.ndarray, dislike: np.ndarray | None = None,
                 tau: float = 0.1, lam: float = 0.0) -> np.ndarray:
    """Score every row of E against liked / disliked rows of E.

    like-score = tau * log mean_j exp(cos(i, like_j) / tau): tau -> 0 is "closest liked song",
    large tau is "average similarity". Dislike penalty = lam * max_j cos(i, dislike_j).
    """
    En = unit(E)
    S = En @ En[like].T / tau
    m = S.max(1, keepdims=True)
    s = tau * (m[:, 0] + np.log(np.exp(S - m).mean(1)))
    if lam and dislike is not None and len(dislike):
        s = s - lam * (En @ En[dislike].T).max(1)
    return s


class SwipeRecommender(BaseEstimator):
    """Holds catalog embeddings + metadata; `recommend` turns swipes into a playlist."""

    def __init__(self, tau: float = 0.1, lam: float = 0.0, k: int = 10, artist_cap: int = 2):
        self.tau, self.lam, self.k, self.artist_cap = tau, lam, k, artist_cap

    def fit(self, E: np.ndarray, y=None, meta: pd.DataFrame | None = None):
        """meta: one row per catalog track, same order as E, with track_id / track_name / primary_artist."""
        assert meta is not None and len(meta) == len(E), "pass rec__meta=<catalog dataframe>"
        self.E_ = np.asarray(E, np.float32)  # float32 keeps the saved artifact under the 100 MB budget
        self.meta_ = meta[["track_id", "track_name", "primary_artist"]].reset_index(drop=True)
        self.pos_ = {t: i for i, t in enumerate(self.meta_.track_id)}
        return self

    def unknown(self, ids) -> list[str]:
        return [t for t in ids if t not in self.pos_]

    def recommend(self, liked_ids, disliked_ids=(), k: int | None = None) -> pd.DataFrame:
        k = k or self.k
        like = np.array([self.pos_[t] for t in liked_ids])
        dislike = np.array([self.pos_[t] for t in disliked_ids], int)
        if not len(like):
            raise ValueError("need at least one liked track")
        s = taste_scores(self.E_, like, dislike, self.tau, self.lam)
        swiped = np.concatenate([like, dislike])
        s[swiped] = -np.inf
        names = self.meta_.track_name.str.lower().str.strip().to_numpy()
        artists = self.meta_.primary_artist.to_numpy()
        seen = {(names[i], artists[i]) for i in swiped}
        En = unit(self.E_)
        picks, per = [], {}
        for i in np.argsort(-s, kind="stable"):
            if not np.isfinite(s[i]):
                break
            a = artists[i]
            if per.get(a, 0) >= self.artist_cap or (names[i], a) in seen:
                continue
            per[a] = per.get(a, 0) + 1
            seen.add((names[i], a))
            picks.append(i)
            if len(picks) == k:
                break
        picks = np.array(picks)
        because = like[(En[picks] @ En[like].T).argmax(1)]
        out = self.meta_.iloc[picks].copy()
        out["score"] = s[picks].round(4)
        out["because_you_liked"] = self.meta_.track_id.to_numpy()[because]
        return out.reset_index(drop=True)
