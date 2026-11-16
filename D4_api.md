# D4: fitted pipeline and API (Group 10)

**Repository:** https://github.com/BigB2405/SwipeTape · **Branch/tag:** `D4` / `d4` · **Commit:** `[add hash]`
**Modal API URL:** `[paste from modal deploy]` · **API docs (/docs):** `[Modal URL]/docs`

## What is served
The model selected in D3 (**M5, hybrid song-kNN**) as one scikit-learn `Pipeline`:

| Step | Class (file) | Learned state |
|---|---|---|
| `embed` | `SongEmbedder` (custom transformer, `pipeline_def.py`) | `StandardScaler` + a random forest that predicts a track's genre mix from its 9 audio features. `transform` returns `[unit(weighted audio) ‖ α·unit(√genre mix)]`. |
| `rec` | `SwipeRecommender` (`pipeline_def.py`) | Embeddings of the 69k-track catalog (train + val; test stays reserved for D6). `recommend` ranks songs by soft-max similarity to the liked songs, minus a dislike term (λ = 0 after D3 tuning), at most 2 per artist. |

`pipeline.joblib` is a **dict bundle**: `{"pipeline": <fitted Pipeline>, "metadata": {steps, built_at, sklearn_version, python/joblib/pandas/numpy versions, git_commit, model, params, features, catalog_size, d3_validation}}`.
It is about 100 MB, so it is **not committed**. Rebuild it with `scripts/train_pipeline.py`.

## Endpoints (`serve.py`)
| Method / path | Purpose | Errors |
|---|---|---|
| `GET /health` | liveness; `artifact_loaded` says whether the model loaded | always 200 |
| `GET /pipeline/info` | the bundle metadata (steps, `built_at`, `sklearn_version`, D3 validation score) | 503 if the model is unavailable |
| `GET /sample?n=5&seed=0` | a few catalog tracks, to get ids for `/recommend` and the D5 frontend | 503 |
| `POST /recommend` | `{"liked_ids": [...1–25], "disliked_ids": [...0–25], "k": 1–50}` → ranked `recommendations` (`track_id, track_name, artist, score, because_you_liked`) | **422**: empty/missing `liked_ids`, malformed id (must be 22 alphanumeric characters), duplicates, an id both liked and disliked, `k` out of range, extra fields, unknown track id. **503**: model unavailable. |

The bundle is loaded **once, at import** of `serve.py`; nothing is re-fitted while serving. A missing file, an unreadable file, or a scikit-learn version different from the one in the metadata leaves the service running and returns **503** (not 500) on the model endpoints.

## Reproduce
```bash
uv sync                                                    # first run `uv lock` once: D4 added fastapi, uvicorn, httpx to pyproject.toml
uv run --frozen python scripts/prepare_data.py             # D2: cleaned data + split (data/raw/dataset.csv from Kaggle)
uv run --frozen python scripts/train_pipeline.py           # fit M5 -> pipeline.joblib (about 1-2 min)
uv run --frozen pytest -q tests                            # 4 logic tests; the HTTP tests need fastapi + httpx

uv run uvicorn serve:app --reload                          # local API: http://localhost:8000/docs
uv run --frozen python scripts/check_deployed.py --url http://127.0.0.1:8000
```
**Test the 503 locally** (does not touch the live app):
```bash
PIPELINE_PATH=/nonexistent uv run uvicorn serve:app --port 8001
uv run --frozen python scripts/check_deployed.py --url http://127.0.0.1:8001 --expect-unavailable
```

## Deploy to Modal (`modal_serve.py`)
```bash
uv run modal setup                                         # one time
uv run modal deploy modal_serve.py                         # prints the public URL
uv run --frozen python scripts/check_deployed.py --url <that URL>
```
The image contains exactly `serve.py`, `pipeline_def.py` and `pipeline.joblib`. scikit-learn (plus joblib, pandas, numpy and the Python minor version) is pinned to what `metadata` records, and `serve` is imported inside the Modal function. The first request after idle takes a few seconds while the container loads the model.

## Local vs deployed outputs
`scripts/check_deployed.py` loads the local `pipeline.joblib`, sends the same requests to the given URL, and compares: pipeline info fields, then, for three swipe sets (1 like; 5 likes + 3 dislikes; mixed taste, k = 20), the **recommended track ids and order (exact)** and **scores (within 1e-4)**; it also confirms 7 kinds of invalid input return 422. Result for the deployed URL: `[paste the final "ALL CHECKS PASSED" output]`.

## Postman
`postman/swipetape_api.postman_collection.json`. Import it into Postman, set the `baseUrl` collection variable to the Modal URL, and run the collection (7 requests): health, pipeline info, sample (stores ids), valid recommend (200), and three invalid recommends (422). Every request has assertions. Screenshots go with the D5 submission.

## Notes and limits
- `SongEmbedder.__init__` only stores its arguments and `fit` returns `self`, as the sklearn API (and HW4) require.
- The artifact is a pickle: load only files you built yourself, with the matching scikit-learn version.
- `GET /sample` and the id format check assume ids come from the catalog (`/sample`); ids outside it get a 422.
- AI assistance: Claude helped draft `serve.py`, `modal_serve.py`, the tests, the check script and this note. _Contributions: [to confirm with group]._
