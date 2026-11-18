# SwipeTape

SwipeTape is a music discovery web app built around a Tinder-style swipe. Users listen to 30-second song previews and swipe left or right (or up to "love" a track), and the app builds a personalized playlist from their taste. Our goal is to help listeners get past trending hits and discover smaller artists they would otherwise miss.

**DTSC 3602 · Fall 2026 · Group 10:** Devanshi Parmar, Britain Russell

**Research question:** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

---

## D4 submission: fitted pipeline and API

| Item | Link |
|---|---|
| Modal API URL | https://bigb2405--swipetape-api-web.us-east.modal.run |
| API docs (/docs) | https://bigb2405--swipetape-api-web.us-east.modal.run/docs |
| Branch / tag | branch [`D4`](https://github.com/BigB2405/SwipeTape/tree/D4) · tag [`d4-final`](https://github.com/BigB2405/SwipeTape/tree/d4-final) |
| D4 note (endpoints, errors, rebuild, verification) | [`docs/D4_api.md`](docs/D4_api.md) |
| Pipeline (custom transformer) | [`pipeline_def.py`](pipeline_def.py) · build: [`scripts/train_pipeline.py`](scripts/train_pipeline.py) |
| API code | [`serve.py`](serve.py) (FastAPI) · [`modal_serve.py`](modal_serve.py) (Modal) |
| Verification | [`scripts/check_deployed.py`](scripts/check_deployed.py) · [`tests/`](tests/) · [`postman/`](postman/swipetape_api.postman_collection.json) |

- **Pipeline:** `Pipeline([("embed", SongEmbedder), ("rec", SwipeRecommender)])`, the M5 model selected in D3. `SongEmbedder` is the custom transformer: it learns a `StandardScaler` and a random forest that predicts each track's genre mix from its 9 audio features, and outputs `[weighted audio | genre mix]` embeddings. The bundle stores the fitted pipeline together with its metadata (versions, parameters, git commit, D3 validation score). It is about 100 MB, so it is not committed; `scripts/train_pipeline.py` rebuilds it in under a minute.
- **API:** `GET /health`, `GET /pipeline/info`, `GET /sample`, `POST /recommend`. Invalid input returns **422** and a missing or incompatible model returns **503**; the server never returns a 500.
- **Local = deployed:** `check_deployed.py` sends the same swipes to the Modal URL and to the local pipeline. The recommended tracks and their order match exactly, scores agree within 1e-4, and all 7 invalid inputs return 422: **ALL CHECKS PASSED** (Nov 18, 2026). 15/15 pytest tests pass.

## D3 submission: model evaluation

| Item | Link |
|---|---|
| Repository | https://github.com/BigB2405/SwipeTape |
| Branch / tag | branch [`D3`](https://github.com/BigB2405/SwipeTape/tree/D3) · tag [`d3-final`](https://github.com/BigB2405/SwipeTape/tree/d3-final) |
| Modeling note (3-page PDF for submission) | [`docs/D3_modeling_note.pdf`](docs/D3_modeling_note.pdf) |
| Modeling note (full version) | [`docs/D3_modeling_note.md`](docs/D3_modeling_note.md) |
| Evaluation code | [`scripts/evaluate.py`](scripts/evaluate.py) · [`scripts/plot_d3.py`](scripts/plot_d3.py) · [`swipetape/model.py`](swipetape/model.py) |
| Results | [`reports/d3_summary.json`](reports/d3_summary.json) · [`reports/d3_results.csv`](reports/d3_results.csv) · [`reports/d3_spot_checks.md`](reports/d3_spot_checks.md) · figures `reports/figures/d3_*.png` |

### What D3 covers
- **Evaluation:** simulated listeners like songs by one artist (368 single-taste listeners) or by two artists from different genre families (400 mixed-taste listeners). Each artist's other songs are hidden among the 11,612 validation tracks. The primary metric is **Recall@50**, the share of hidden songs ranked in the top 50. Genre P@10 from the first draft is kept as a secondary check.
- **Models:** random and popularity baselines; the app's current centroid method (M1), with and without dislikes (M1d); a random-forest genre model (M3); song-to-song kNN (M4); and a hybrid audio + genre-mix kNN (M5). All hyperparameters are tuned with artist-grouped 3-fold CV inside train.
- **Selected: M5.** Recall@50 is **0.139 [0.127, 0.151]**, against 0.007 for popularity (about 19×) and 0.086 for the app's current method. Every other candidate is significantly worse. Dislikes did not help.
- **Error analysis:** by genre family, seed popularity, and number of likes, plus spot-checked playlists.
- **Honest caveats:** validation was used twice (the first draft's genre comparison, then this redesigned comparison), so the reserved test set in D6 is the only unbiased estimate. The test set has never been opened.

## D2 submission: data preparation and EDA

| Item | Link |
|---|---|
| Branch / tag | branch [`d2`](https://github.com/BigB2405/SwipeTape/tree/d2) · tag [`d2-final`](https://github.com/BigB2405/SwipeTape/tree/d2-final) |
| Streamlit dashboard (Modal, reads from Supabase) | https://bigb2405--swipetape-dashboard-dashboard.us-east.modal.run |
| Data note | [`docs/D2_data_note.md`](docs/D2_data_note.md) |
| Data dictionary | [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| EDA figures (5) | [`reports/figures/`](reports/figures/) · numbers in [`reports/eda_stats.json`](reports/eda_stats.json) |
| Preparation code | [`scripts/prepare_data.py`](scripts/prepare_data.py) · [`scripts/make_plots.py`](scripts/make_plots.py) · [`scripts/load_supabase.py`](scripts/load_supabase.py) |
| Dashboard code | [`dashboard/app.py`](dashboard/app.py) · [`dashboard/modal_app.py`](dashboard/modal_app.py) · [`sql/schema.sql`](sql/schema.sql) |

The first load of the dashboard takes about 15 seconds while it reads the tracks from Supabase.

- **Audit and cleaning:** 114,000 raw rows → **81,043 clean tracks**. We drop the 1 row with missing text and 450 exact duplicates, collapse 16,299 tracks listed under several genres into one row each (keeping every genre), remove 8,397 re-releases of the same song, and drop 300 rows with invalid audio values.
- **Reserved test data:** `StratifiedGroupKFold` (seed 42), grouped by primary artist and stratified by genre. **Train 57,801 · validation 11,612 · test 11,630.** No artist appears in more than one split (asserted). The test split is never uploaded, plotted, or used until D6.
- **EDA (train split only):** popularity vs. the app's filters, feature correlations, features by genre family, genre-label overlap, and a PCA of the audio features, each interpreted in the data note and on the dashboard.

---

## Data

**Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, maharshipandya, 2022). 114,000 tracks × 21 columns across 114 genres, collected from the Spotify Web API. License: Open Database License (ODbL 1.0) for the database; the track contents remain with their original authors.

**Access:** sign in to Kaggle, download the dataset, and unzip `dataset.csv` into `data/raw/`:

```text
data/raw/dataset.csv
```

Raw data, processed data, and model artifacts are not committed (`data/` and `artifacts/` are gitignored). Everything can be rebuilt with the steps below.

---

## How to reproduce

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (Python 3.11–3.13), plus free [Supabase](https://supabase.com) and [Modal](https://modal.com) accounts for the dashboard.

### 1. Environment and data preparation (D2)
```bash
uv sync                                   # install pinned dependencies (pyproject.toml + uv.lock)
uv run python scripts/prepare_data.py     # audit + clean + split -> data/processed/{tracks_clean,test}.parquet, audit.json
uv run python scripts/make_plots.py       # EDA figures -> reports/figures/, stats -> reports/eda_stats.json
```
Both scripts are deterministic: rerunning them produces identical outputs.

### 2. Model evaluation (D3)
```bash
uv run --frozen python scripts/evaluate.py         # tuning + validation comparison -> reports/d3_summary.json, d3_results.csv, d3_spot_checks.md
uv run --frozen python scripts/plot_d3.py          # D3 figures -> reports/figures/d3_*.png
uv run --frozen python scripts/train_pipeline.py   # fit selected model -> pipeline.joblib (fitted pipeline + metadata)
```
`evaluate.py` takes several minutes (about 7 on a MacBook) and is deterministic: a rerun reproduced the committed `reports/d3_*` files byte for byte. It reads only train and validation rows; `test.parquet` is never opened (asserted).

### 3. API (D4)
```bash
uv run --frozen --with pytest pytest -q tests                                 # 15 tests
uv run --frozen uvicorn serve:app --reload                                    # local API: http://localhost:8000/docs
uv run --frozen python scripts/check_deployed.py --url http://127.0.0.1:8000  # local API vs. saved pipeline
uv run --frozen modal deploy modal_serve.py                                   # deploy; prints the public URL
uv run --frozen python scripts/check_deployed.py --url <modal URL>            # deployed vs. local: ALL CHECKS PASSED
```
See [`docs/D4_api.md`](docs/D4_api.md) for the endpoints, error codes, and the 503 test.

### 4. Load Supabase
1. Create a Supabase project and run [`sql/schema.sql`](sql/schema.sql) in **SQL Editor → New query**. This creates the `tracks`, `audit_steps`, and `split_summary` tables, which are read-only for the public key.
2. Copy `.env.example` to `.env` and fill in your values from **Project Settings**:
   ```text
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_SERVICE_KEY=<secret key>        # used only by the loader (writes)
   SUPABASE_ANON_KEY=<publishable key>      # used by the dashboard (read-only)
   ```
3. Upload the train and validation rows (the test split is never uploaded):
   ```bash
   uv run python scripts/load_supabase.py
   ```

### 5. Run the dashboard
```bash
uv run streamlit run dashboard/app.py                 # local, http://localhost:8501

uv run modal setup                                    # one-time Modal login
uv run modal secret create swipetape-supabase \
  SUPABASE_URL=... SUPABASE_ANON_KEY=...              # read-only key only
uv run modal deploy dashboard/modal_app.py            # prints the public URL
```

Never commit `.env`: it is gitignored, and no keys are stored in the code.

### 6. Swipe app (Vercel frontend)
```bash
uv run python scripts/build_data.py       # regenerate public/songs.json from data/raw/dataset.csv
```
`public/` is a static site and `api/preview.js` is a Vercel serverless function. Deploy with `vercel`, or preview locally with `vercel dev`.

---

## Model (D3, served in D4)

The recommender is song-to-song. `pipeline_def.py` defines a scikit-learn Pipeline of two steps (`swipetape/model.py` re-exports them for the D3 code):
- `SongEmbedder`, a custom transformer that combines standardized audio features with a random forest's predicted genre mix
- `SwipeRecommender`, which ranks songs by similarity to each liked song and allows at most 2 per artist

The saved artifact (~100 MB) is not committed. Rebuild it with `scripts/train_pipeline.py`. The bundle's `metadata` records the versions, parameters, git commit, and D3 validation score.

```python
import joblib
pipe = joblib.load("pipeline.joblib")["pipeline"]
pipe.named_steps["rec"].recommend(liked_ids=["<track_id>", ...], disliked_ids=[...])
```

---

## Repository structure

```text
.
├── api/
│   └── preview.js            # Vercel function: Deezer preview lookup fallback
├── dashboard/
│   ├── app.py                # Streamlit dashboard (audit, explorer, EDA, dictionary), reads Supabase
│   └── modal_app.py          # Modal deployment of the dashboard
├── docs/
│   ├── D2_data_note.md       # D2 data note
│   ├── D3_modeling_note.md   # D3 modeling note (full version)
│   ├── D3_modeling_note.pdf  # D3 modeling note (3-page submission version)
│   ├── D4_api.md             # D4 pipeline + API note
│   └── data_dictionary.md    # column definitions, roles, prediction-time availability
├── postman/
│   └── swipetape_api.postman_collection.json  # API requests with assertions
├── public/
│   ├── app.js                # swipe logic, audio playback, recommendation engine
│   ├── index.html            # app shell
│   ├── songs.json            # compiled song data for the app
│   └── style.css
├── reports/
│   ├── eda_stats.json        # D2 numbers quoted in the data note
│   ├── d3_summary.json       # D3 metrics, CIs, tests, error analysis
│   ├── d3_results.csv        # D3 per-listener results
│   ├── d3_spot_checks.md     # D3 example playlists
│   └── figures/              # 5 EDA plots + 4 D3 plots
├── scripts/
│   ├── prepare_data.py       # D2: audit, clean, split (reserves test data)
│   ├── make_plots.py         # D2: EDA figures
│   ├── load_supabase.py      # D2: upload train/val to Supabase
│   ├── evaluate.py           # D3: model tuning and comparison
│   ├── plot_d3.py            # D3: figures
│   ├── train_pipeline.py     # D3/D4: fit and save the selected pipeline -> pipeline.joblib
│   ├── check_deployed.py     # D4: compare a running API with the local pipeline
│   ├── genres.py             # genre -> genre-family map (shared)
│   └── build_data.py         # builds public/songs.json for the app
├── sql/
│   └── schema.sql            # Supabase tables + read-only policies
├── swipetape/
│   └── model.py              # re-exports pipeline_def (keeps D3 imports working)
├── tests/                    # D4: pipeline + API tests (pytest)
├── pipeline_def.py           # SongEmbedder (custom transformer) + SwipeRecommender
├── serve.py                  # D4: FastAPI app
├── modal_serve.py            # D4: Modal deployment of the API
├── .env.example              # template for Supabase keys
├── pyproject.toml / uv.lock  # Python environment
├── package.json / vercel.json
└── README.md
```

---

## The app

- Tinder-style swipe interface with 30-second audio previews (iTunes, with Deezer as a fallback)
- Playlist recommendations from likes, loves, and dislikes, using the audio features and genre
- "Because you liked …" explanations and a taste profile
- Spotify links for the final playlist
- Responsive single-page frontend (HTML, CSS, and JavaScript) deployed on Vercel

## Project roadmap

| Deliverable | Due | Status |
|---|---|---|
| D1 Proposal | Oct 21 | ✅ |
| D2 Data preparation and EDA | Nov 2 | ✅ tag `d2-final` |
| D3 Model evaluation | Nov 16 | ✅ tag `d3-final` |
| D4 Fitted scikit-learn pipeline + FastAPI on Modal | Nov 23 | ✅ tag `d4-final` |
| D5 Vercel frontend calling the live API + Postman tests | Nov 30 | planned |
| D6 Final report and demo (frozen model on reserved test data) | Dec 2 | planned |

## Contributions and AI use

- **D2:** Devanshi Parmar: data preparation, plotting, data dictionary, dashboard code, first draft of the data note. Britain Russell: Supabase setup and data load, Modal deployment, dashboard chart fix, finalizing the data note.
- **D3:** Britain Russell: evaluation design, model and evaluation code, experiments, first versions of the modeling note. Devanshi Parmar: code review, reproducing the results, revising the modeling note.
- **D4:** _[to confirm with group]_ pipeline refactor (`pipeline_def.py`), FastAPI service, Modal files, tests, Postman collection, and D4 note (GitHub `ashroff23`). Britain Russell: building and verifying the artifact, Modal deployment and fix, local-vs-deployed check.

**AI assistance:** Claude Code helped draft the preparation, plotting, dashboard, model, and evaluation code, the notes, and this README, and helped review and reproduce the D3 results, and drafted the D4 API, deployment, tests, and checks. The group reviewed and ran all of it.
