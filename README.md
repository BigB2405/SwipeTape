# SwipeTape

SwipeTape is a music discovery web app inspired by the classic Tinder-style swipe experience. Users listen to 30-second song previews, swipe left or right (or up to “love” a track), and the app builds a personalized playlist based on their taste.

The project combines a curated music dataset, client-side recommendation logic, and a lightweight preview lookup API to make music discovery feel interactive and fun.

## Features

- Tinder-style swipe interface for songs
- 30-second audio preview playback
- Personalized playlist generation from likes and dislikes
- Genre and mood-aware recommendation logic
- Responsive single-page frontend
- Vercel-ready API for preview lookup fallback
- Spotify link generation for final playlist

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Data processing: Python + pandas
- API: Vercel serverless function
- Deployment: Vercel
- Data source: processed music metadata dataset

## Repository Structure

```text
.
├── api/
│   └── preview.js            # Preview fallback API for Deezer lookup
├── public/
│   ├── app.js                # Swipe logic, audio playback, recommendation engine
│   ├── index.html            # App shell and UI structure
│   ├── songs.json            # Compiled song dataset
│   └── style.css             # App styling
├── scripts/
│   └── build_data.py         # Converts CSV music data into compact JSON
├── .gitignore
├── package.json
├── vercel.json               # Vercel configuration and cache headers
├── README.md

## Data preparation & EDA (D2)

**Data access:** download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle and unzip `dataset.csv` into `data/raw/` (raw and processed data are gitignored).

```bash
uv sync                                         # Python env (pyproject.toml)
uv run python scripts/prepare_data.py           # audit + clean + split -> data/processed/
uv run python scripts/make_plots.py             # EDA figures -> reports/figures/
uv run --frozen python scripts/evaluate.py      # D3 model comparison (~3 min) -> reports/d3_*.json/csv/md
uv run --frozen python scripts/plot_d3.py       # D3 figures -> reports/figures/d3_*.png
uv run --frozen python scripts/train_pipeline.py  # fit selected model -> artifacts/swipetape_pipeline.joblib + bundle.json
uv run python scripts/build_data.py             # (app) regenerate public/songs.json
```

**Supabase:** create a project, run `sql/schema.sql` in the SQL editor, copy `.env.example` to `.env`, then `uv run python scripts/load_supabase.py`. Only train/val rows are uploaded; the test split stays in `data/processed/test.parquet` until D6.

**Dashboard:** `uv run streamlit run dashboard/app.py` locally, or deploy on Modal:

```bash
uv run modal setup
uv run modal secret create swipetape-supabase SUPABASE_URL=... SUPABASE_ANON_KEY=...
uv run modal deploy dashboard/modal_app.py
```

## Model (D3)

The recommender is song-to-song: `swipetape/model.py` defines a scikit-learn Pipeline of `SongEmbedder`, a custom transformer that combines standardized audio features with a random forest's predicted genre mix, and `SwipeRecommender`, which ranks songs by similarity to each liked song and allows at most 2 per artist. The saved artifact (~100 MB) is not committed. Rebuild it with `scripts/train_pipeline.py`. `artifacts/bundle.json` records the versions, parameters, and D3 validation score.

```python
import joblib
pipe = joblib.load("artifacts/swipetape_pipeline.joblib")
pipe.named_steps["rec"].recommend(liked_ids=["<track_id>", ...], disliked_ids=[...])
```

Docs: [`docs/data_dictionary.md`](docs/data_dictionary.md) · [`docs/D2_data_note.md`](docs/D2_data_note.md) · [`docs/D3_modeling_note.md`](docs/D3_modeling_note.md)
