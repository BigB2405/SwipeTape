"""SwipeTape recommender API (HW4 `serve.py`).

    uv run uvicorn serve:app --reload        ->  http://localhost:8000/docs

The bundle (pipeline.joblib: {"pipeline": fitted Pipeline, "metadata": {...}}) is loaded ONCE, at import.
Nothing is re-fitted while serving. If the file is missing, unreadable or built with a different
scikit-learn, the service still starts and answers 503 (not 500) on the model endpoints.

GET  /health         liveness; says whether the model loaded (always 200)
GET  /pipeline/info  describes the artifact: steps, built_at, versions, parameters, D3 validation score
GET  /sample         a few catalog tracks (ids to try /recommend with)
POST /recommend      liked / disliked track ids -> ranked recommendations (422 bad input, 503 no model)
"""
import os
from pathlib import Path
from typing import Annotated

import joblib
import sklearn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

import pipeline_def  # noqa: F401  (the pickled pipeline refers to its classes by module name)

PIPELINE_PATH = Path(os.environ.get("PIPELINE_PATH", Path(__file__).with_name("pipeline.joblib")))


def load_bundle(path: Path = PIPELINE_PATH) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found")
    bundle = joblib.load(path)
    want = bundle["metadata"]["sklearn_version"]
    if want != sklearn.__version__:           # a pickle from another scikit-learn version is not safe to use
        raise RuntimeError(f"scikit-learn mismatch: bundle built with {want}, server runs {sklearn.__version__}")
    steps = [n for n, _ in bundle["pipeline"].steps]
    if steps != ["embed", "rec"]:
        raise RuntimeError(f"unexpected pipeline steps {steps}")
    return bundle


try:                                           # loaded once per server process
    BUNDLE, LOAD_ERROR = load_bundle(), None
except Exception as e:                         # noqa: BLE001  any failure -> 503, never a 500
    BUNDLE, LOAD_ERROR = None, f"{type(e).__name__}: {e}"

app = FastAPI(title="SwipeTape recommender API", version="0.4.0",
              description="Song-to-song recommender selected in D3 (hybrid audio + genre-mix embedding).")


def _bundle() -> dict:
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail=f"model unavailable: {LOAD_ERROR}")
    return BUNDLE


# Spotify track ids are 22 base-62 characters (docs/data_dictionary.md).
TrackId = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9]{22}$")]


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    liked_ids: list[TrackId] = Field(min_length=1, max_length=25, description="Tracks swiped right (at least 1).")
    disliked_ids: list[TrackId] = Field(default_factory=list, max_length=25, description="Tracks swiped left.")
    k: int = Field(default=10, ge=1, le=50, description="How many recommendations to return.")

    @model_validator(mode="after")
    def _consistent(self):
        if len(set(self.liked_ids)) != len(self.liked_ids) or len(set(self.disliked_ids)) != len(self.disliked_ids):
            raise ValueError("liked_ids and disliked_ids must not contain duplicates")
        both = set(self.liked_ids) & set(self.disliked_ids)
        if both:
            raise ValueError(f"tracks cannot be both liked and disliked: {sorted(both)}")
        return self


class Recommendation(BaseModel):
    track_id: str
    track_name: str
    artist: str
    score: float
    because_you_liked: str = Field(description="track_id of the liked song this pick is closest to")


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    model: str
    k: int


def recommend_items(bundle: dict, liked: list[str], disliked: list[str], k: int) -> list[dict]:
    """Run swipes through the fitted pipeline's recommender step. ValueError on unknown track ids."""
    rec = bundle["pipeline"].named_steps["rec"]
    unknown = rec.unknown(liked + disliked)
    if unknown:
        raise ValueError(f"unknown track ids: {unknown}")
    out = rec.recommend(liked, disliked, k)
    return [{"track_id": r.track_id, "track_name": r.track_name, "artist": r.primary_artist,
             "score": float(r.score), "because_you_liked": r.because_you_liked} for r in out.itertuples()]


@app.get("/health")
def health():
    return {"status": "ok", "artifact_loaded": BUNDLE is not None}


@app.get("/pipeline/info")
def pipeline_info():
    return _bundle()["metadata"]


@app.get("/sample")
def sample(n: int = Query(5, ge=1, le=20), seed: int = Query(0, ge=0, le=100_000)):
    meta = _bundle()["pipeline"].named_steps["rec"].meta_
    rows = meta.sample(n=min(n, len(meta)), random_state=seed)
    return {"tracks": [{"track_id": r.track_id, "track_name": r.track_name, "artist": r.primary_artist}
                       for r in rows.itertuples()]}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    bundle = _bundle()
    try:
        items = recommend_items(bundle, body.liked_ids, body.disliked_ids, body.k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"recommendations": items, "model": bundle["metadata"]["model"], "k": body.k}
