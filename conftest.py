"""Builds a tiny fitted bundle so the tests need neither the Kaggle data nor the real pipeline.joblib."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_def import FEATS, SongEmbedder, SwipeRecommender  # noqa: E402


def make_catalog(n=240, seed=0):
    rng = np.random.default_rng(seed)
    fam = rng.integers(0, 4, n)                                  # 4 fake genres with different audio profiles
    df = pd.DataFrame(np.clip(rng.normal(0.25 + 0.15 * fam[:, None], 0.1, (n, len(FEATS))), 0, 1), columns=FEATS)
    df["tempo"] = 80 + 25 * fam + rng.normal(0, 8, n)
    df["loudness"] = -14 + 2 * fam + rng.normal(0, 1, n)
    df["track_id"] = [f"T{i:021d}" for i in range(n)]
    df["track_name"] = [f"Song {i}" for i in range(n)]
    df["primary_artist"] = [f"Artist {i // 3}" for i in range(n)]
    df["genres"] = [[f"genre{f}"] for f in fam]
    return df


def build_bundle(sklearn_version=None):
    cat = make_catalog()
    pipe = Pipeline([("embed", SongEmbedder(alpha=2.0, min_leaf=3, n_estimators=20)),
                     ("rec", SwipeRecommender(tau=0.3, lam=0.0))])
    pipe.fit(cat, rec__meta=cat)
    meta = dict(steps=[{"name": n, "class": type(s).__name__} for n, s in pipe.steps], built_at="2026-01-01T00:00:00+00:00",
                sklearn_version=sklearn_version or sklearn.__version__, model="M5 test", params={"alpha": 2.0},
                catalog_size=len(cat), git_commit="test")
    return {"pipeline": pipe, "metadata": meta}


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("bundle") / "pipeline.joblib"
    joblib.dump(build_bundle(), p, compress=3)
    return p


@pytest.fixture(scope="session")
def ids():
    return list(make_catalog().track_id)
