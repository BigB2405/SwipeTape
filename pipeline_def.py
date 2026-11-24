"""SwipeTape song-to-song recommender: the custom transformer + recommender step (HW4 `pipeline_def.py`).

Shared by the D3 evaluation (via swipetape/model.py), the build script, serve.py and modal_serve.py.
The saved pipeline pickles these classes by module name, so this file must ship with the API.

Pipeline([("embed", SongEmbedder(...)), ("rec", SwipeRecommender(...))])

* SongEmbedder (custom transformer) turns a track's 9 audio features into a song embedding:
      [ unit(weighted standardized audio) | alpha * unit(sqrt(forest genre probabilities)) ]
  Learned state: the StandardScaler and (if alpha > 0) a random forest that predicts a
  track's genre mix from audio. The genre labels are used only to *fit* the forest; at
  transform time only the 9 audio features are read.
  v2 option genre_source="labels": the genre half is built from the track's real genre tags
  (unit(sqrt(multi-hot))) instead of the forest's guess from audio. The API only serves catalog
  songs, which all carry tags. No forest is fit in that mode.
* SwipeRecommender stores the catalog embeddings and ranks every catalog song by how close it
  is to the songs the listener liked (a soft maximum over liked songs, so two different tastes
  are both served instead of being averaged away), minus lam x closeness to disliked songs,
  plus (v2) pop_w x popularity/100. v2 also adds two optional filters: hide_unless_seeded (genre tags,
  e.g. children/kids/sleep, that are left out unless a liked song has one) and lang_filter (an
  estimated language per track, so a request can ask for songs in chosen languages).
  Defaults (genre_source="forest", pop_w=0, no filters) reproduce M5 exactly.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FEATS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "speechiness", "liveness", "tempo", "loudness"]           # FEAT_COLS in public/app.js
APP_W = np.array([1.0, 1.2, 1.0, 1.0, 0.8, 0.6, 0.3, 0.5, 0.6])   # FEAT_W in public/app.js
SEED = 42

# ---- estimated language (the dataset has no language column) ---------------------------------
# From genre tags, plus the title's alphabet. A track with neither signal is "en" (= English or other).
# This is an ESTIMATE: a latin-tagged song can be sung in English, and a Spanish song tagged only
# "pop" reads as "en".
LANG_OF_GENRE = {
    **dict.fromkeys(["spanish", "latin", "latino", "reggaeton", "salsa", "tango"], "es"),
    **dict.fromkeys(["brazil", "mpb", "pagode", "samba", "sertanejo", "forro"], "pt"),
    "indian": "hi",
    **dict.fromkeys(["j-pop", "j-rock", "j-idol", "j-dance", "anime"], "ja"),
    "k-pop": "ko",
    **dict.fromkeys(["cantopop", "mandopop"], "zh"),
    "french": "fr", "german": "de", "swedish": "sv", "turkish": "tr", "malay": "ms", "iranian": "fa",
}
SCRIPT_LANG = [(re.compile("[\uac00-\ud7af\u1100-\u11ff]"), "ko"),
               (re.compile("[\u3040-\u30ff]"), "ja"),
               (re.compile("[\u0900-\u097f\u0a00-\u0a7f]"), "hi"),
               (re.compile("[\u4e00-\u9fff]"), "zh"),
               (re.compile("[\u0600-\u06ff]"), "ar"),
               (re.compile("[\u0400-\u04ff]"), "ru")]
LANGS = ["en", "es", "pt", "hi", "ja", "ko", "zh", "fr", "it", "de", "sv", "tr", "ms", "fa", "ar", "ru"]

# Third signal, per ARTIST: common words in all of the artist's song + album titles. It catches e.g.
# Brazilian funk or Spanish songs whose only tag is "funk"/"pop". Spot check (Nov 23): right for about
# 4 in 5 artists it changes. Romanized Punjabi with mostly English titles is still often missed.
TITLE_WORDS = {
    "en": "the you i me my your love and of to in is it on don't i'm it's we be all what so this that baby night like up down with for".split(),
    "es": "el la los las de que y en mi tu te amor no con por para una un lo como más quiero corazón vida eres mí yo noche bien todo".split(),
    "pt": "o os da do das dos que e não com você meu minha pra tá é ao vivo uma um eu te amor sem coração mais quem".split(),
    "hi": ("tera tere teri mera mere meri dil pyaar pyar ve hai nahi main yaar jatt kudi mahi sohneya ishq ni tenu "
           "menu sanu mainu tainu jind gal naal rabba munde jaan").split(),
    "fr": "le les des une et je tu pas est mon ma pour dans qui sur toi moi c'est j'ai ne au ce".split(),
    "it": "il gli della di che non per sono mio mia cuore ti amo sei io più come".split(),
}
_TITLE_SETS = {k: set(v) for k, v in TITLE_WORDS.items()}


def artist_title_langs(artists, titles, albums) -> dict:
    """artist -> language code guessed from words in their titles (only when clearly not English)."""
    words: dict = {}
    seen = set()
    for a, *texts in zip(artists, titles, albums):
        for t in texts:
            if (a, t) in seen:
                continue
            seen.add((a, t))
            t = re.sub(r"\(.*?\)|\[.*?\]| - .*", "", t or "").lower()   # drop (feat. ...), - Remix, ...
            words.setdefault(a, []).extend(re.findall(r"[a-zà-ÿ']+", t))
    out = {}
    for a, ws in words.items():
        n = {k: sum(w in v for w in ws) for k, v in _TITLE_SETS.items()}
        best = max((k for k in n if k != "en"), key=n.get)
        if n[best] >= 2 and n[best] > 1.5 * n["en"]:
            out[a] = best
    return out


def track_langs(genres, title: str = "", artist_lang: str | None = None) -> tuple[str, ...]:
    """Estimated language codes of one track: genre tags, title alphabet, then the artist's title words."""
    out = {LANG_OF_GENRE[g] for g in genres if g in LANG_OF_GENRE}
    title = title or ""
    for rx, code in SCRIPT_LANG:
        if rx.search(title):
            if code == "zh" and "ja" in out:      # kanji inside a Japanese title
                continue
            out.add(code)
    if not out and artist_lang:
        out.add(artist_lang)
    return tuple(sorted(out)) or ("en",)


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
                 weights=tuple(APP_W), random_state: int = SEED, genre_source: str = "forest"):
        self.alpha, self.min_leaf, self.n_estimators = alpha, min_leaf, n_estimators
        self.weights, self.random_state = weights, random_state
        self.genre_source = genre_source

    def fit(self, X: pd.DataFrame, y=None, forest: RandomForestClassifier | None = None):
        """X needs FEATS columns, plus a `genres` (list) column when alpha > 0.
        `forest` lets the caller pass a forest already fit on the same X (used to share work in tuning)."""
        if self.genre_source not in ("forest", "labels"):
            raise ValueError(f"genre_source must be 'forest' or 'labels', got {self.genre_source!r}")
        self.scaler_ = StandardScaler().fit(X[FEATS].to_numpy(float))
        self.forest_ = None
        self.genres_ = None
        if self.alpha and self.genre_source == "labels":
            self.genres_ = sorted({g for gs in X["genres"] for g in gs})
        elif self.alpha:
            self.forest_ = forest if forest is not None else fit_genre_forest(
                self.scaler_.transform(X[FEATS].to_numpy(float)), X["genres"], self.min_leaf,
                self.n_estimators, self.random_state)
        self.n_features_in_ = len(FEATS)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Zs = self.scaler_.transform(X[FEATS].to_numpy(float))
        parts = [unit(Zs * np.asarray(self.weights))]
        if getattr(self, "genres_", None) is not None:
            col = {g: j for j, g in enumerate(self.genres_)}
            L = np.zeros((len(X), len(col)))
            for i, gs in enumerate(X["genres"]):
                for g in gs:
                    if g in col:
                        L[i, col[g]] = 1.0
            parts.append(self.alpha * unit(np.sqrt(L)))
        elif self.forest_ is not None:
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

    def __init__(self, tau: float = 0.1, lam: float = 0.0, k: int = 10, artist_cap: int = 2, pop_w: float = 0.0,
                 hide_unless_seeded=(), lang_filter: bool = False):
        self.tau, self.lam, self.k, self.artist_cap = tau, lam, k, artist_cap
        self.pop_w, self.hide_unless_seeded, self.lang_filter = pop_w, hide_unless_seeded, lang_filter

    def fit(self, E: np.ndarray, y=None, meta: pd.DataFrame | None = None):
        """meta: one row per catalog track, same order as E, with track_id / track_name / primary_artist."""
        assert meta is not None and len(meta) == len(E), "pass rec__meta=<catalog dataframe>"
        self.E_ = np.asarray(E, np.float32)  # float32 keeps the saved artifact under the 100 MB budget
        self.meta_ = meta[["track_id", "track_name", "primary_artist"]].reset_index(drop=True)
        self.pop_ = None
        if self.pop_w:
            assert "popularity" in meta, "pop_w > 0 needs a popularity column in meta"
            self.pop_ = meta["popularity"].to_numpy(np.float32) / 100
        self.hidden_, self.seed_hides_ = None, None
        if self.hide_unless_seeded:
            hide = set(self.hide_unless_seeded)
            tags = [set(g) for g in meta["genres"]]
            self.hidden_ = np.array([bool(t) and t <= hide for t in tags])     # every tag is a hidden one
            self.seed_hides_ = np.array([bool(t & hide) for t in tags])        # liking this unlocks them
        self.lang_mask_ = None
        if self.lang_filter:
            albums = meta["album_name"] if "album_name" in meta else [""] * len(meta)
            al = artist_title_langs(meta["primary_artist"], meta["track_name"], albums)
            langs = [track_langs(g, n, al.get(a)) for g, n, a in zip(meta["genres"], meta["track_name"], meta["primary_artist"])]
            self.lang_mask_ = {c: np.array([c in t for t in langs]) for c in LANGS}
        self.pos_ = {t: i for i, t in enumerate(self.meta_.track_id)}
        return self

    def unknown(self, ids) -> list[str]:
        return [t for t in ids if t not in self.pos_]

    def recommend(self, liked_ids, disliked_ids=(), k: int | None = None, languages=None) -> pd.DataFrame:
        """languages: estimated language codes to prefer (None = any). Songs in other languages are
        used only if fewer than k songs in the chosen languages are left. Ignored by bundles built
        without lang_filter."""
        k = k or self.k
        like = np.array([self.pos_[t] for t in liked_ids])
        dislike = np.array([self.pos_[t] for t in disliked_ids], int)
        if not len(like):
            raise ValueError("need at least one liked track")
        s = taste_scores(self.E_, like, dislike, self.tau, self.lam)
        if getattr(self, "pop_w", 0.0) and getattr(self, "pop_", None) is not None:   # old bundles lack these
            s = s + self.pop_w * self.pop_
        hidden = getattr(self, "hidden_", None)
        if hidden is not None and not self.seed_hides_[like].any():
            s = np.where(hidden, -np.inf, s)
        lang_mask = getattr(self, "lang_mask_", None)
        if languages and lang_mask is not None:
            ok = np.zeros(len(s), bool)
            for c in languages:
                ok |= lang_mask.get(c, False)
            s = np.where(ok, s, s - 1000.0)       # other languages rank after every match (still finite)
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
