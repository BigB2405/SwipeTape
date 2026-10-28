# D2 Data Note: SwipeTape (Group 10)

**Repository:** _<GitHub URL>_ · **Tag:** `d2` · **Dashboard (Modal):** _<URL>_
**Research question (D1):** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

## 1. Source and access
We use the **Spotify Tracks Dataset** on Kaggle (maharshipandya, 2022): 114,000 rows × 21 columns, collected from the Spotify Web API by searching 114 genre keywords and keeping 1,000 tracks for each. Anyone can download it with a free Kaggle account. The README explains where to put `dataset.csv`, and raw data is not committed. _(TODO: confirm the license listed on the Kaggle page.)_ Spotify stopped giving new apps access to its audio-features endpoint in Nov 2024, so this file is a fixed snapshot that we can't refresh or extend with the same features.

**Labels.** There are no real user swipes, so D3–D6 evaluate against a **genre proxy**. A simulated listener is a set of seed tracks from one genre, and a recommendation counts as a hit if it shares a genre with the seeds. The baseline is popularity ranking. The unit of observation is one track.

## 2. Cleaning (reproducible: `uv run python scripts/prepare_data.py`)
| Step | Rows after | Why |
|---|---|---|
| Raw | 114,000 | 114 genres × 1,000 rows |
| Drop missing text | 113,999 | 1 row had no artist, album, or title |
| Drop exact duplicates | 113,549 | 450 identical rows |
| Collapse to one row per `track_id` | 89,740 | 16,299 tracks appeared under 2–9 genres. We keep the full list in `genres` and the highest popularity value |
| Drop re-release duplicates | 81,343 | 8,397 songs had several IDs (single vs. album, remasters) with the same title and artist. We keep the most popular one so the "same" song can't land in both train and test |
| Drop invalid audio values | 81,043 | `tempo = 0` or `time_signature = 0` (failed analysis), duration < 30 s or > 15 min |

Running the script twice gives a byte-identical `audit.json`. See `docs/data_dictionary.md` for every column, its role, and whether it's available at prediction time.

## 3. Reserved test data
We split with `StratifiedGroupKFold` (seed 42), **grouped by primary artist** and **stratified by primary genre**:

| Split | Tracks | Artists | Genres |
|---|---|---|---|
| train | 57,801 | 12,602 | 113 |
| val | 11,612 | 2,522 | 113 |
| test | 11,630 | 2,505 | 113 |

Grouping by artist means no artist appears in more than one split (the script asserts this). Otherwise a model could "recommend" another song by an artist it already saw, which would inflate the score. The test set is written only to `data/processed/test.parquet`. It isn't uploaded to Supabase, plotted, or used for any decision until D6. Every plot below uses the training split only.

## 4. EDA: plots and interpretation
**Figure 1: Popularity distribution** (`reports/figures/1_popularity.png`). Median popularity is 35, and 5.9% of tracks score exactly 0. Only 43% of tracks pass the current `songs.json` cutoff (40) and **9% pass the swipe-deck cutoff (62), which keeps just 16% of artists**. The prototype app therefore filters out the niche artists our problem statement wants to surface. Popularity filtering becomes an explicit, tested design choice in D3 rather than a hard-coded default.

**Figure 2: Feature correlations** (`2_feature_correlation.png`). Energy correlates with loudness (r = 0.76) and against acousticness (r = −0.73). An unweighted distance would count the "loud/energetic vs. acoustic" axis about three times, so the features need standardizing and the redundant ones need down-weighting or decorrelating (e.g., PCA). Popularity is nearly unrelated to every audio feature (|r| ≤ 0.18), so the popularity baseline and a feature-based model capture different signals, which makes the comparison meaningful.

**Figure 3: Features by genre family** (`3_features_by_family.png`). Some families separate clearly. Metal is high-energy and almost never acoustic, while Jazz & Classical and Chill & Acoustic are low-energy and highly acoustic. Pop, Rock, Latin, and World overlap a lot. The fine-grained genre explains 44–51% of the variance in danceability, energy, acousticness, speechiness, and loudness, but only 9% for tempo and 17% for liveness. This is direct evidence for the research question: audio features carry real but incomplete genre information, so a feature model should beat chance but won't be perfect.

**Figure 4: Genre overlap** (`4_genre_overlap.png`). 16% of training tracks carry more than one genre. Some pairs are synonyms (`songwriter` and `singer-songwriter` are the identical 1,000 tracks in the raw data; punk + punk-rock; indie + indie-pop). Others look like artifacts of keyword search (dub + dubstep, reggae + reggaeton). The genre proxy is noisy, so D3 will count a hit when *any* genre matches. `primary_genre` (alphabetical first) is only used for stratification.

**Figure 5: PCA of the nine features** (`5_pca_by_family.png`). Two components explain 48% of the variance. PC1 is roughly loud/energetic vs. acoustic, and PC2 is danceable/positive vs. instrumental. Acoustic and classical tracks sit in a distinct tail and metal clusters low on PC2, but most families share one dense central cloud. Nearest-neighbor similarity should work well for distinctive tastes and worse for mainstream pop/rock listeners. We'll check this group difference in D3's error analysis.

## 5. Data-quality and sampling concerns
1. **Artificial balance.** Exactly 1,000 tracks per genre means niche genres (e.g., grindcore, forro) are hugely over-represented compared to Spotify's catalog or a real listener. Results describe this curated sample, not typical listening.
2. **Search-based labels.** The genres come from keyword search, not editorial tags, which produces multi-genre and inconsistent labels (Fig. 4). Genre match is only a proxy for "the user would like it."
3. **Popularity bias.** Popularity is a 2022 snapshot and is heavily skewed. Any popularity bonus in the recommender pushes against the project's niche-discovery goal (Fig. 1).
4. **No real user feedback.** Simulated genre-based users can't capture taste that cuts across genres, so we'll state this limitation in D6 and not claim real user satisfaction.
5. **Repeated entities.** Duplicate IDs and re-releases (≈33k rows removed) would leak across splits if not handled. Collaborations can still connect splits through secondary artists. That's a minor, disclosed risk.
6. **Frozen features.** Spotify's audio-features API is no longer available to new apps, so new tracks can't be scored the same way.

## 6. Dashboard and next steps
The cleaned train and validation tracks plus the audit tables live in **Supabase** (`sql/schema.sql`, `scripts/load_supabase.py`). The **Streamlit dashboard** (`dashboard/app.py`, deployed on **Modal** via `dashboard/modal_app.py`) has four pages: the cleaning audit and split summary; an interactive explorer filtered by genre family, popularity, and split; the EDA findings above; and the data dictionary.

**D3 plan:** baseline = popularity ranking. Candidates = (a) cosine kNN on standardized features (the app's current approach), (b) kNN on PCA-decorrelated features, (c) logistic regression or random forest that predicts genre from features. All of them are scored with precision@10 on validation users built from held-out artists.

_Contributions: <member: tasks>. AI assistance: Claude Code helped draft the preparation, plotting, and dashboard code and this note. The group reviewed and ran all of it._
