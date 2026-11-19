"""Logic tests that run without a web framework, plus HTTP tests that need fastapi + httpx (skipped otherwise)."""
import importlib
import sys

import joblib
import pytest
import sklearn
from sklearn.utils.validation import check_is_fitted

from conftest import build_bundle


def test_custom_transformer_has_learned_state(bundle_path):
    emb = joblib.load(bundle_path)["pipeline"].named_steps["embed"]
    check_is_fitted(emb, ["scaler_", "forest_"])          # learned state: wrong if rebuilt from scratch at boot
    assert emb.__class__.__module__ == "pipeline_def"     # lives in pipeline_def.py (HW4)
    assert emb.get_params()["alpha"] == 2.0               # __init__ only stores its arguments


def test_fit_returns_self_and_init_only_assigns():
    from pipeline_def import SongEmbedder
    t = SongEmbedder(alpha=0)
    assert vars(t).keys() == {"alpha", "min_leaf", "n_estimators", "weights", "random_state"}
    from conftest import make_catalog
    assert t.fit(make_catalog()) is t


def test_bundle_is_dict_with_metadata(bundle_path):
    b = joblib.load(bundle_path)
    assert set(b) >= {"pipeline", "metadata"}
    assert {"steps", "built_at", "sklearn_version"} <= set(b["metadata"])


def test_recommend_excludes_swiped_and_is_deterministic(bundle_path, ids):
    rec = joblib.load(bundle_path)["pipeline"].named_steps["rec"]
    a = rec.recommend(ids[:3], ids[100:102], 7)
    b = rec.recommend(ids[:3], ids[100:102], 7)
    assert list(a.track_id) == list(b.track_id) and len(a) == 7
    assert not set(a.track_id) & set(ids[:3] + ids[100:102])


@pytest.fixture()
def serve_mod(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    def load(path):
        monkeypatch.setenv("PIPELINE_PATH", str(path))
        sys.modules.pop("serve", None)
        return importlib.import_module("serve")      # the bundle loads at import
    return load


def client_for(mod):
    from fastapi.testclient import TestClient
    return TestClient(mod.app)


def test_http_happy_path(serve_mod, bundle_path, ids):
    c = client_for(serve_mod(bundle_path))
    assert c.get("/health").json() == {"status": "ok", "artifact_loaded": True}
    info = c.get("/pipeline/info").json()
    assert info["sklearn_version"] == sklearn.__version__ and [s["name"] for s in info["steps"]] == ["embed", "rec"]
    assert len(c.get("/sample?n=3").json()["tracks"]) == 3
    r = c.post("/recommend", json={"liked_ids": ids[:2], "disliked_ids": ids[50:52], "k": 5})
    assert r.status_code == 200 and len(r.json()["recommendations"]) == 5


@pytest.mark.parametrize("body", [{"liked_ids": []}, {"k": 3}, {"liked_ids": ["bad"]}, {"liked_ids": ["Z" * 22]},
                                  {"liked_ids": ["A" * 22], "k": 0}, {"liked_ids": ["A" * 22], "k": 51},
                                  {"liked_ids": ["A" * 22], "disliked_ids": ["A" * 22]}, {"liked_ids": ["A" * 22], "x": 1}])
def test_http_invalid_is_422(serve_mod, bundle_path, body):
    assert client_for(serve_mod(bundle_path)).post("/recommend", json=body).status_code == 422


def test_http_missing_artifact_is_503(serve_mod, tmp_path, ids):
    c = client_for(serve_mod(tmp_path / "missing.joblib"))
    assert c.get("/health").json()["artifact_loaded"] is False
    assert c.get("/pipeline/info").status_code == 503
    assert c.post("/recommend", json={"liked_ids": ids[:1]}).status_code == 503


def test_http_corrupt_or_wrong_version_is_503(serve_mod, tmp_path, ids):
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"garbage")
    assert client_for(serve_mod(bad)).get("/pipeline/info").status_code == 503
    old = tmp_path / "old.joblib"
    joblib.dump(build_bundle(sklearn_version="0.0.1"), old)
    c = client_for(serve_mod(old))
    assert c.post("/recommend", json={"liked_ids": ids[:1]}).status_code == 503


def test_http_search(serve_mod, bundle_path):
    c = client_for(serve_mod(bundle_path))
    r = c.get("/search", params={"q": "song 12", "limit": 5})
    assert r.status_code == 200
    t = r.json()["tracks"]
    assert t and t[0]["track_name"] == "Song 12"                        # exact title ranks first
    assert all("12" in x["track_name"] for x in t)
    by_artist = c.get("/search", params={"q": "artist 7"}).json()["tracks"]
    assert {x["artist"] for x in by_artist} >= {"Artist 7"}
    assert c.get("/search", params={"q": "zzzz"}).json()["tracks"] == []


@pytest.mark.parametrize("params", [{}, {"q": "a"}, {"q": "  "}, {"q": "song", "limit": 0}, {"q": "song", "limit": 51}])
def test_http_search_invalid_is_422(serve_mod, bundle_path, params):
    assert client_for(serve_mod(bundle_path)).get("/search", params=params).status_code == 422


def test_http_search_missing_artifact_is_503(serve_mod, tmp_path):
    assert client_for(serve_mod(tmp_path / "missing.joblib")).get("/search?q=song").status_code == 503


def test_cors_allows_vercel_and_blocks_others(serve_mod, bundle_path):
    c = client_for(serve_mod(bundle_path))
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    ok = c.options("/recommend", headers={"Origin": "https://swipetape.vercel.app", **pre})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://swipetape.vercel.app"
    bad = c.options("/recommend", headers={"Origin": "https://evil.example.com", **pre})
    assert "access-control-allow-origin" not in bad.headers
