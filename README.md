# SwipeTape

SwipeTape is a music discovery web app built around a Tinder-style swipe. Users listen to 30-second song previews and swipe left or right (or up to "love" a track), and the app builds a personalized playlist from their taste. Our goal is to help listeners get past trending hits and discover smaller artists they would otherwise miss.

**DTSC 3602 · Fall 2026 · Group 10:** Devanshi Parmar, Britain Russell

**Research question:** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

---

## D6 submission: final report and test results

| Item | Link |
|---|---|
| Final report (PDF, 5 pages) | [`docs/D6_final_report.pdf`](docs/D6_final_report.pdf) · source [`docs/D6_final_report.md`](docs/D6_final_report.md) |
| Branch / tag | branch `D6` · tag `d6-final` (after review) |
| Final test code | [`scripts/final_test.py`](scripts/final_test.py) · [`scripts/plot_d6.py`](scripts/plot_d6.py) |
| Results | [`reports/d6_summary.json`](reports/d6_summary.json) · [`reports/d6_results.csv`](reports/d6_results.csv) · [`reports/d6_spot_checks.md`](reports/d6_spot_checks.md) · figures `reports/figures/d6_*.png` |
| Website / API / docs | https://swipetape-group10.vercel.app · https://bigb2405--swipetape-api-web.us-east.modal.run · [/docs](https://bigb2405--swipetape-api-web.us-east.modal.run/docs) |
| Postman | [`postman/`](postman/swipetape_api.postman_collection.json) · screenshots in [`docs/d5_screenshots/`](docs/d5_screenshots/) |

- **Final test (reserved split, opened once on Nov 22, 2026; frozen `pipeline.joblib`, no refit):** 766 simulated listeners, 11,630 unseen-artist songs. M5 Recall@50 **0.145 [0.133, 0.158]** (validation 0.139) vs. popularity **0.006** (~26×) and the app's original method **0.088** (all p < 10⁻³⁴).
- **Answer:** shared audio features of *liked* songs clearly improve the playlist; *disliked* songs do not (centroid −0.001, p = 0.99; kNN penalty −0.012, p < 0.001).
- **Bias checks:** recommended songs have the catalog's median popularity (35 vs. 87 for the popularity baseline); M5 beats the app's method in all 11 genre families (reported with n and CIs); other-album Recall@50 0.119 shows it is not just matching album production.

```bash
uv run --frozen python scripts/final_test.py   # ~25 s, deterministic
uv run --frozen python scripts/plot_d6.py
```

## D5 submission: web app calling the live API

| Item | Link |
|---|---|
| Website (Vercel) | https://swipetape-group10.vercel.app |
| Modal API / docs | https://bigb2405--swipetape-api-web.us-east.modal.run · [/docs](https://bigb2405--swipetape-api-web.us-east.modal.run/docs) |
| Branch / tag | branch `D5` · tag `d5-final` (after review) |
| D5 note (tabs, API calls, loading/error states, tests) | [`docs/D5_app.md`](docs/D5_app.md) · screenshots in [`docs/d5_screenshots/`](docs/d5_screenshots/) |
| Frontend code | [`public/`](public/) (`index.html`, `style.css`, `js/*.js`, `config.js`) |
| Accounts schema | [`sql/accounts.sql`](sql/accounts.sql) (Supabase Auth + row level security) |
| API tests | [`postman/`](postman/swipetape_api.postman_collection.json) (10 requests with assertions) · [`scripts/check_deployed.py`](scripts/check_deployed.py) · [`tests/`](tests/) |

The app has four tabs:
- **Home:** a Netflix-style catalog with a featured song, then rows you scroll sideways: Trending, top songs per genre family, Hidden gems (random less-played songs, a counterweight to popularity bias) and, once you like songs, **"Because you liked …"** rows from the live model plus **My List**.
- **Swipe:** the original Tinder-style deck. The playlist at the end now comes from the **live model** (`POST /recommend`, loves first, at most 25 likes and 25 dislikes). If the API can't be reached, the old in-browser formula makes "offline picks" and the page says so.
- **Find similar:** search any of the 69,413 catalog songs (`GET /search`) or tap a quick pick. The song starts playing, and the model returns the **5 closest songs** (`POST /recommend` with one liked id, `k=5`), each with a preview, % similarity, My List and Spotify. "More like this" chains to a new seed.
- **Me:** sign up or sign in with email/password or Google (Supabase Auth). Shows swipe stats, your taste profile, My List and liked songs. Guests can use every tab; signing in saves swipes and My List, and the deck skips songs you already rated.

**Loading and error states:** each API call shows a spinner, says the model is waking up after 3 s (Modal cold start), times out after 60 s, and turns 422/503/network errors into readable messages with a retry button.

**API changes for D5 (v0.5.0, same fitted `pipeline.joblib`, no retraining):** CORS for `*.vercel.app` and localhost (plus `ALLOWED_ORIGINS`), and `GET /search?q=&limit=` (422 if the query is under 2 characters, 503 without a model). `/recommend` is unchanged. 23/23 pytest tests pass, and `check_deployed.py` also checks search and CORS.

**Data fix:** `public/songs.json` used to be built from the raw CSV and held 5,010 songs from the reserved test split. It is now built from `tracks_clean.parquet` (train + val only, asserted), so the app never shows a D6 test song and every song in it is known to the model.

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
- **Local = deployed:** `check_deployed.py` sends the same swipes to the Modal URL and to the local pipeline. The recommended tracks and their order match exactly, scores agree within 1e-4, and all 7 invalid inputs return 422: **ALL CHECKS PASSED** (Nov 18, re-run Nov 22, 2026). 15/15 pytest tests pass.
- **Note for grading:** since Nov 18 the same Modal URL runs the D5 version of `serve.py` (v0.5.0), which adds `GET /search` and CORS for the website. `/health`, `/pipeline/info`, `/sample` and `/recommend` and the fitted `pipeline.joblib` are unchanged, so the D4 checks above still apply. The bundle's `git_commit` (12019c5) is the commit the artifact was built from; `d4-final` (eacac83) changed only the Modal startup code.

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

### 6. Web app (Vercel frontend, D5)
```bash
uv run --frozen python scripts/build_data.py          # public/songs.json from the train+val catalog
uv run --frozen uvicorn serve:app --port 8000         # optional: local API
vercel dev                                            # http://localhost:3000  (add ?api=local to use the local API)
vercel --prod                                         # deploy
```
`public/` is a static site (plain HTML/CSS/JS modules, no build step) and `api/preview.js` is a Vercel serverless function.

**Accounts (one-time):** run [`sql/accounts.sql`](sql/accounts.sql) in the Supabase SQL editor. Put the **publishable** key in `public/config.js` (it is public by design; row level security keeps each user's rows private; never the secret key). In Supabase → Authentication: enable Email and Google, turn **off** "Confirm email" (the free built-in mailer sends only ~2 emails per hour, which blocked sign-ups), set the Site URL to the Vercel URL, and add it plus `http://localhost:3000` to the redirect URLs. For Google, create an OAuth client in Google Cloud Console with redirect URI `https://<project>.supabase.co/auth/v1/callback` and paste its ID and secret into the Google provider. Without a key, the app runs in guest mode.

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
│   ├── D5_app.md             # D5 app + tests note
│   ├── d5_screenshots/       # D5 app and Postman screenshots
│   └── data_dictionary.md    # column definitions, roles, prediction-time availability
├── postman/
│   └── swipetape_api.postman_collection.json  # API requests with assertions
├── public/
│   ├── index.html            # app shell: nav + 4 tabs (Home, Swipe, Find similar, Me)
│   ├── config.js             # public settings: API URL, Supabase URL + publishable key
│   ├── js/                   # main (router), home, swipe, similar, profile, api, auth, preview, ui, data
│   ├── songs.json            # train+val songs (popularity >= 40) for the app
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
│   └── build_data.py         # builds public/songs.json (train+val only) for the app
├── sql/
│   ├── schema.sql            # Supabase tables + read-only policies
│   └── accounts.sql          # D5: profiles, swipes, my_list + row level security
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

- Netflix-style Home, Tinder-style Swipe, a kNN "Find similar" tab, and accounts (Supabase Auth: email or Google)
- 30-second audio previews (iTunes, with Deezer as a fallback)
- Playlists and "5 similar songs" from the live model API on Modal
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
| D5 Vercel frontend calling the live API + Postman tests | Nov 30 | in progress, branch `D5` |
| D6 Final report and demo (frozen model on reserved test data) | Dec 2 | ✅ report on branch `D6`; presentation Dec 2 |

## Contributions and AI use

- **D2:** Devanshi Parmar: data preparation, plotting, data dictionary, dashboard code, first draft of the data note. Britain Russell: Supabase setup and data load, Modal deployment, dashboard chart fix, finalizing the data note.
- **D3:** Britain Russell: evaluation design, model and evaluation code, experiments, first versions of the modeling note. Devanshi Parmar: code review, reproducing the results, revising the modeling note.
- **D5:** _[to confirm with group]_ Britain Russell: app redesign (tabs, Home, Find similar, accounts), search endpoint and CORS, tests and Postman. Claude Code drafted the code.
- **D4:** GitHub user `ashroff23`: pipeline refactor (`pipeline_def.py`), FastAPI service, Modal files, tests, Postman collection, and D4 note. Britain Russell: building and verifying the artifact, Modal deployment and fix, local-vs-deployed check.

**AI assistance:** Claude Code helped draft the preparation, plotting, dashboard, model, and evaluation code, the notes, and this README, and helped review and reproduce the D3 results, and drafted the D4 API, deployment, tests, and checks. The group reviewed and ran all of it.
