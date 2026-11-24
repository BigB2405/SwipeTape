"""v2 model options: real genre tags (genre_source="labels") and the popularity boost (pop_w)."""
import numpy as np
import pytest
from conftest import make_catalog

from pipeline_def import SongEmbedder, SwipeRecommender


def test_defaults_are_m5():
    e, r = SongEmbedder(), SwipeRecommender()
    assert e.genre_source == "forest" and r.pop_w == 0.0


def test_labels_mode_uses_tags_and_fits_no_forest():
    cat = make_catalog()
    e = SongEmbedder(alpha=0.5, genre_source="labels").fit(cat)
    assert e.forest_ is None and e.genres_ == sorted({g for gs in cat.genres for g in gs})
    E = e.transform(cat)
    assert E.shape == (len(cat), 9 + len(e.genres_))
    tag_part = E[:, 9:] / 0.5
    assert np.allclose(np.linalg.norm(tag_part, axis=1), 1.0)       # every song has a tag -> unit genre half


def test_unknown_genre_source_rejected():
    with pytest.raises(ValueError):
        SongEmbedder(genre_source="nope").fit(make_catalog())


def test_labels_mode_keeps_recs_in_the_seed_genre():
    cat = make_catalog()
    E = SongEmbedder(alpha=1.0, genre_source="labels").fit(cat).transform(cat)
    rec = SwipeRecommender(tau=0.3).fit(E, meta=cat)
    seed = cat.track_id[0]
    out = rec.recommend([seed], k=10)
    seed_genre = cat.genres[0][0]
    got = cat.set_index("track_id").loc[out.track_id, "genres"].map(lambda g: g[0])
    assert (got == seed_genre).all()


def test_pop_boost_moves_popular_songs_up_and_zero_is_a_no_op():
    cat = make_catalog()
    cat["popularity"] = 0
    E = SongEmbedder(alpha=0).fit(cat).transform(cat)
    seed = cat.track_id[0]
    base = SwipeRecommender(tau=0.3).fit(E, meta=cat).recommend([seed], k=10)
    zero = SwipeRecommender(tau=0.3, pop_w=0.0).fit(E, meta=cat).recommend([seed], k=10)
    assert base.track_id.tolist() == zero.track_id.tolist()
    last = base.track_id.iat[-1]
    cat.loc[cat.track_id == last, "popularity"] = 100
    boosted = SwipeRecommender(tau=0.3, pop_w=1.0).fit(E, meta=cat).recommend([seed], k=10)
    assert boosted.track_id.iat[0] == last


# ---- v2 filters: hide kids/sleep unless seeded, estimated language --------------------------------
from pipeline_def import LANGS, track_langs  # noqa: E402


def tagged_catalog():
    """genre0 = 'kids' songs; genre1 songs get a Spanish tag; the rest are plain."""
    cat = make_catalog()
    tag = {"genre0": ["kids"], "genre1": ["latin"], "genre2": ["pop"], "genre3": ["rock"]}
    cat["genres"] = [tag[g[0]] for g in cat.genres]
    return cat


def fit_rec(cat, **kw):
    E = SongEmbedder(alpha=0).fit(cat).transform(cat)        # audio only, so tags act only via the filters
    return SwipeRecommender(tau=0.3, **kw).fit(E, meta=cat)


def test_track_langs():
    assert track_langs(["pop"], "Hello") == ("en",)
    assert track_langs(["latin", "pop"], "Despacito") == ("es",)
    assert track_langs(["pop"], "사랑") == ("ko",)
    assert track_langs(["j-pop"], "夜に駆ける") == ("ja",)          # kanji in a Japanese song is not Chinese
    assert set(track_langs(["indian"], "")) <= set(LANGS)


def test_kids_hidden_unless_a_liked_song_is_kids():
    cat = tagged_catalog()
    kids = set(cat.track_id[cat.genres.map(lambda g: g == ["kids"])])
    rec = fit_rec(cat, hide_unless_seeded=("children", "kids", "sleep"))
    plain_seed = cat.track_id[cat.genres.map(lambda g: g == ["rock"])].iat[0]
    assert not set(rec.recommend([plain_seed], k=40).track_id) & kids
    kid_seed = sorted(kids)[0]
    assert set(rec.recommend([kid_seed], k=10).track_id) & kids
    # default (no filter): kids songs can appear for a seed that sounds like them
    assert set(fit_rec(cat).recommend([kid_seed], k=10).track_id) & kids


def test_language_filter_prefers_chosen_and_tops_up():
    cat = tagged_catalog()
    es = set(cat.track_id[cat.genres.map(lambda g: g == ["latin"])])
    rec = fit_rec(cat, lang_filter=True)
    seed = cat.track_id[cat.genres.map(lambda g: g == ["rock"])].iat[0]
    got = rec.recommend([seed], k=10, languages=["es"]).track_id
    assert set(got) <= es
    n_es = len(es)
    many = rec.recommend([seed], k=n_es + 20, languages=["es"])     # more than exist -> filled with others
    assert len(many) == n_es + 20 or len(many) == len(cat) - 1
    assert rec.recommend([seed], k=10).track_id.tolist() == fit_rec(cat).recommend([seed], k=10).track_id.tolist()


def test_http_languages_field(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import importlib
    import sys

    import joblib
    from fastapi.testclient import TestClient

    from conftest import build_bundle
    b = build_bundle()
    cat = tagged_catalog()
    b["pipeline"].steps[1] = ("rec", fit_rec(cat, lang_filter=True, hide_unless_seeded=("kids",)))
    b["pipeline"].steps[0] = ("embed", SongEmbedder(alpha=0).fit(cat))
    p = tmp_path / "v2.joblib"
    joblib.dump(b, p)
    monkeypatch.setenv("PIPELINE_PATH", str(p))
    sys.modules.pop("serve", None)
    c = TestClient(importlib.import_module("serve").app)
    seed = cat.track_id[cat.genres.map(lambda g: g == ["rock"])].iat[0]
    ok = c.post("/recommend", json={"liked_ids": [seed], "k": 5, "languages": ["es"]})
    assert ok.status_code == 200
    es = set(cat.track_id[cat.genres.map(lambda g: g == ["latin"])])
    assert {r["track_id"] for r in ok.json()["recommendations"]} <= es
    assert c.post("/recommend", json={"liked_ids": [seed], "languages": ["xx"]}).status_code == 422
    assert c.post("/recommend", json={"liked_ids": [seed]}).status_code == 200
