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
