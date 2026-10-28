# Data dictionary

**Source:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, M. Pandya, 2022). The data was collected through the Spotify Web API by searching for 114 genre keywords and keeping 1,000 tracks per genre.
**Raw:** 114,000 rows × 21 columns. **Clean (`scripts/prepare_data.py`):** 81,043 tracks, one row per `track_id`.
**Unit of observation:** one track. **Target (proxy label):** genre (`primary_genre` / `genres`).

Roles: **id** = identifier, not a model input · **feature** = model input · **target** = label for evaluation · **context** = shown in the app or used for analysis but not a model input · **derived** = created during preparation.

## Original columns

| Column | Type | Range in clean data | Description | Role | Available at prediction time? |
|---|---|---|---|---|---|
| `track_id` | text | 22-char Spotify ID | Spotify track ID (primary key after cleaning) | id | yes |
| `track_name` | text | — | Track title | context | yes. Hidden from users while swiping (D1 bias control) |
| `artists` | text | — | Credited artists; raw data separates them with `;`, cleaned data with `, ` | context | yes. Hidden while swiping |
| `album_name` | text | — | Album title | context | yes |
| `popularity` | int | 0–100 | Spotify's popularity score at scrape time (2022), driven by recent play counts | context / baseline | ⚠ snapshot only. Used for the popularity baseline but not as a similarity feature, since it isn't something the listener hears |
| `duration_ms` | int | 30,080–896,445 | Track length in ms | context | yes |
| `explicit` | bool | — | Spotify explicit-lyrics flag | context | yes |
| `danceability` | float | 0–1 | How suitable the track is for dancing (tempo, rhythm stability, beat strength) | feature | yes |
| `energy` | float | 0–1 | Perceived intensity/activity | feature | yes |
| `key` | int | 0–11 | Pitch class (0 = C … 11 = B) | excluded | yes. Categorical with weak genre signal |
| `loudness` | float | −46.6 to 4.5 dB | Average loudness | feature | yes |
| `mode` | int | 0/1 | Minor (0) / major (1) | excluded | yes. Binary with weak signal |
| `speechiness` | float | 0–1 | Presence of spoken words | feature | yes |
| `acousticness` | float | 0–1 | Confidence the track is acoustic | feature | yes |
| `instrumentalness` | float | 0–1 | Likelihood of no vocals | feature | yes |
| `liveness` | float | 0–1 | Likelihood of a live audience | feature | yes |
| `valence` | float | 0–1 | Musical positiveness | feature | yes |
| `tempo` | float | 30–243 BPM | Estimated tempo (rows with 0 removed) | feature | yes |
| `time_signature` | int | 1–5 | Estimated beats per bar (rows with 0 removed) | excluded | yes |
| `track_genre` | text | 114 values | The genre keyword whose search returned the track. Replaced by `genres` / `primary_genre` | target (raw) | — |

## Derived columns

| Column | Type | Description |
|---|---|---|
| `genres` | text[] | Every genre the `track_id` was listed under in the raw data (16% of clean tracks have more than one) |
| `n_genres` | int | Length of `genres` (1–9) |
| `primary_genre` | text | Alphabetically first entry in `genres`. A deterministic tie-break, **not** a "true" genre. Used only to stratify the split |
| `genre_family` | text | One of 11 broad families (Pop, Rock, Metal, …), mapped from `primary_genre` via `scripts/genres.py` (the same map the app uses) |
| `primary_artist` | text | First credited artist. Used as the grouping key for the train/val/test split |
| `duration_s` | float | `duration_ms / 1000` |
| `pop_bucket` | text | Popularity band: `0`, `1-19`, `20-39`, `40-61`, `62+` (40 and 62 are the app's current cutoffs) |
| `split` | text | `train` / `val` / `test` (test is stored only in `data/processed/test.parquet`) |

## Model features (planned for D3)
`danceability, energy, valence, acousticness, instrumentalness, speechiness, liveness, tempo, loudness`. These are the same nine columns the app's recommender uses (`FEAT_COLS` in `public/app.js`). They get standardized using statistics fitted **on the training split only**.
