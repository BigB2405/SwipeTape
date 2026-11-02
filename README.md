# SwipeTape

SwipeTape is a music discovery web app built around a Tinder-style swipe. Users listen to 30-second song previews and swipe left or right (or up to "love" a track), and the app builds a personalized playlist from their taste. Our goal is to help listeners get past trending hits and discover smaller artists they would otherwise miss.

**DTSC 3602 · Fall 2026 · Group 10:** Devanshi Parmar, Britain Russell

**Research question:** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

---

## D2 submission: data preparation and EDA

| Item | Link |
|---|---|
| Repository | https://github.com/BigB2405/SwipeTape |
| Branch / tag | branch [`d2`](https://github.com/BigB2405/SwipeTape/tree/d2) · tag [`d2-final`](https://github.com/BigB2405/SwipeTape/tree/d2-final) |
| Streamlit dashboard (Modal, reads from Supabase) | https://bigb2405--swipetape-dashboard-dashboard.us-east.modal.run |
| Data note (2–3 pages) | [`docs/D2_data_note.md`](docs/D2_data_note.md) |
| Data dictionary | [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| EDA figures (5) | [`reports/figures/`](reports/figures/) · numbers in [`reports/eda_stats.json`](reports/eda_stats.json) |
| Preparation code | [`scripts/prepare_data.py`](scripts/prepare_data.py) · [`scripts/make_plots.py`](scripts/make_plots.py) · [`scripts/load_supabase.py`](scripts/load_supabase.py) |
| Dashboard code | [`dashboard/app.py`](dashboard/app.py) · [`dashboard/modal_app.py`](dashboard/modal_app.py) · [`sql/schema.sql`](sql/schema.sql) |

The first load of the dashboard takes about 15 seconds while it reads the tracks from Supabase.

### What D2 covers
- **Audit and cleaning:** 114,000 raw rows → **81,043 clean tracks**. We drop the 1 row with missing text and 450 exact duplicates, collapse 16,299 tracks listed under several genres into one row each (keeping every genre), remove 8,397 re-releases of the same song, and drop 300 rows with invalid audio values. Every step is logged to `data/processed/audit.json` and shown on the dashboard.
- **Reserved test data:** `StratifiedGroupKFold` (seed 42), grouped by primary artist and stratified by genre. **Train 57,801 · validation 11,612 · test 11,630.** No artist appears in more than one split, and the script asserts this. The test split is never uploaded, plotted, or used until D6.
- **EDA (train split only):** popularity distribution vs. the app's filters, feature correlations, features by genre family, genre-label overlap, and a PCA of the audio features. Each comes with an interpretation in the data note and on the dashboard.
- **Data-quality and sampling concerns:** artificial 1,000-tracks-per-genre balance, search-based (noisy) genre labels, popularity bias, no real user feedback (genre is used as a proxy), repeated entities, and frozen Spotify audio features. See section 5 of the data note.

---

## Data

**Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, maharshipandya, 2022). 114,000 tracks × 21 columns across 114 genres, collected from the Spotify Web API. License: Open Database License (ODbL 1.0) for the database; the track contents remain with their original authors.

**Access:** sign in to Kaggle, download the dataset, and unzip `dataset.csv` into `data/raw/`:

```text
data/raw/dataset.csv
```

Raw and processed data are not committed (`data/` is gitignored), and every derived file can be rebuilt with the steps below.

---

## How to reproduce

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (Python 3.11–3.13), plus free [Supabase](https://supabase.com) and [Modal](https://modal.com) accounts for the dashboard.

### 1. Environment and data preparation
```bash
uv sync                                   # install pinned dependencies (pyproject.toml + uv.lock)
uv run python scripts/prepare_data.py     # audit + clean + split -> data/processed/{tracks_clean,test}.parquet, audit.json
uv run python scripts/make_plots.py       # EDA figures -> reports/figures/, stats -> reports/eda_stats.json
```
Both scripts are deterministic: rerunning them produces identical outputs.

### 2. Load Supabase
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

### 3. Run the dashboard
```bash
uv run streamlit run dashboard/app.py                 # local, http://localhost:8501

uv run modal setup                                    # one-time Modal login
uv run modal secret create swipetape-supabase \
  SUPABASE_URL=... SUPABASE_ANON_KEY=...              # read-only key only
uv run modal deploy dashboard/modal_app.py            # prints the public URL
```

Never commit `.env`: it is gitignored, and no keys are stored in the code.

### 4. Swipe app (Vercel frontend)
```bash
uv run python scripts/build_data.py       # regenerate public/songs.json from data/raw/dataset.csv
```
`public/` is a static site and `api/preview.js` is a Vercel serverless function. Deploy with `vercel`, or preview locally with `vercel dev`.

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
│   └── data_dictionary.md    # column definitions, roles, prediction-time availability
├── public/
│   ├── app.js                # swipe logic, audio playback, recommendation engine
│   ├── index.html            # app shell
│   ├── songs.json            # compiled song data for the app
│   └── style.css
├── reports/
│   ├── eda_stats.json        # numbers quoted in the data note
│   └── figures/              # 5 EDA plots
├── scripts/
│   ├── prepare_data.py       # D2: audit, clean, split (reserves test data)
│   ├── make_plots.py         # D2: EDA figures
│   ├── load_supabase.py      # D2: upload train/val to Supabase
│   ├── genres.py             # genre -> genre-family map (shared)
│   └── build_data.py         # builds public/songs.json for the app
├── sql/
│   └── schema.sql            # Supabase tables + read-only policies
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
| D2 Data preparation and EDA | Nov 2 | ✅ this tag |
| D3 Model evaluation: popularity baseline vs. feature-based kNN and classifiers | Nov 16 | planned |
| D4 Fitted scikit-learn pipeline + FastAPI on Modal | Nov 23 | planned |
| D5 Vercel frontend calling the live API + Postman tests | Nov 30 | planned |
| D6 Final report and demo (frozen model on reserved test data) | Dec 2 | planned |

## Contributions and AI use

- **Devanshi Parmar:** data preparation, plotting, data dictionary, dashboard code, first draft of the data note.
- **Britain Russell:** Supabase setup and data load, Modal deployment, dashboard chart fix, finalizing the data note.

**AI assistance:** Claude Code helped draft the preparation, plotting, and dashboard code, the data note, and this README. The group reviewed and ran all of it.
