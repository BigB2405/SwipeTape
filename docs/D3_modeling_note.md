# D3 Modeling Note: SwipeTape (Group 10)

**Repository:** https://github.com/BigB2405/SwipeTape · **Branch/tag:** `d3` · **Run:** `uv run --frozen python scripts/evaluate.py && uv run --frozen python scripts/plot_d3.py`
**Research question:** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

## 1. Evaluation design
**Simulated listeners.** We have no real swipe logs, so we use the genre proxy from D2. For each genre we create 5 listeners. Each listener likes 5 random tracks from that genre and dislikes 5 random tracks from *other* genre families. The 113 genres give **560 validation listeners** (a few genres are too small in val). A recommended track is a **hit** if its `genres` list contains the listener's genre.

**Identical protocol for every model.** Every model sees exactly the same listeners. Each returns a top-10 list with at most 2 tracks per artist, the same diversity rule the app uses. All models are scored with the same metrics, so every comparison is **paired**: we compare per-listener differences, not just averages.

**Leakage controls.**
| Risk | Control |
|---|---|
| Same artist in fit and evaluation | D2 artist-grouped split. The script asserts no artist overlaps train/val. |
| "Recommending" the seed artist's other songs | Candidate pool drops **every track by any seed artist**. A per-episode assert confirms no seed is ever recommended. |
| Preprocessing fit on evaluation data | StandardScaler, PCA, and classifiers are fit on **train only**. |
| Tuning on the comparison data | Hyperparameters are tuned with 3-fold GroupKFold **by artist inside train**. Val is used once, for the final comparison. |
| Label in the inputs | Candidate models get **only the 9 audio features**: no genre, popularity, or artist. The live app's scorer (`public/app.js`) adds a genre-match bonus, which reads our label directly. An "oracle" that does the same scores P@10 = 0.997, so we evaluate the app's **audio-only** core instead. |
| Test-set peeking | `test.parquet` is never opened, and the script asserts that only train/val rows are loaded. It stays reserved for D6. |

**Metrics.** The **primary metric is precision@10**: the share of a 10-song playlist that matches the listener's taste. That is the decision the app makes, and it's the metric we committed to in D2. Secondary metrics: nDCG@10 (are hits near the top), hit-rate@10 (did the listener get at least one good song), family-level P@10 (a hit counts if the track is in the same broad family), the median popularity of recommended tracks, and catalog coverage. The last two are discovery checks, because the project aims to surface niche music. Uncertainty comes from 2,000 bootstrap resamples over listeners and Wilcoxon signed-rank tests on paired per-listener P@10. **Chance P@10 = 0.0065** (a genre is ~0.65% of the pool).

## 2. Models
| ID | Model | Tuned on train-CV (P@10) |
|---|---|---|
| B0 | Random ranking (floor) | — |
| B1 | **Popularity baseline**: the same most-popular tracks for everyone (the app's prior default) | — |
| M1 | **Cosine similarity to the liked centroid**, standardized features with the app's `FEAT_W` weights (the app's audio-only core) | — |
| M1d | M1 **minus λ × cosine to the disliked centroid** (tests the "dislikes" half of the research question) | λ ∈ {0.15, **0.3 (app)**, 0.5, 1.0} → **0.15** (0.097 vs. 0.084 for the app's 0.3) |
| M2 | Cosine on **PCA-whitened** features (addresses the correlated energy, loudness, and acousticness found in D2) | components ∈ {3, 5, 7, 9} → **9** (0.102; fewer was worse) |
| M3 | **Genre model**: a classifier predicts each track's genre distribution from audio. Listener profile = mean of the seeds' distributions. Rank by dot product | LogReg C ∈ {0.01, 1}, Random forest min_leaf ∈ {20, 50} → **forest, leaf = 20** (0.151 vs. 0.107 LogReg) |

Multi-genre tracks are used once per genre in M3's training, with weight 1/`n_genres`, so the label ambiguity from D2 isn't hidden by an arbitrary primary genre.

## 3. Results (validation, 560 listeners; Figure `reports/figures/d3_1_model_comparison.png`)
| Model | P@10 [95% CI] | nDCG@10 | Hit@10 | Family P@10 | Median pop. of recs | Coverage |
|---|---|---|---|---|---|---|
| B0 Random | 0.006 [0.004, 0.008] | 0.006 | 0.06 | 0.12 | 36 | 38% |
| B1 Popularity | 0.014 [0.009, 0.020] | 0.013 | 0.07 | 0.21 | 92 | 0.1% |
| M1 Cosine (app) | 0.084 [0.072, 0.098] | 0.085 | 0.41 | 0.34 | 36 | 34% |
| M1d + dislikes | 0.087 [0.076, 0.099] | 0.088 | **0.44** | 0.34 | 36 | 34% |
| M2 PCA-whitened | 0.083 [0.070, 0.095] | 0.085 | 0.41 | 0.34 | 37 | 35% |
| **M3 Genre forest** | **0.141 [0.120, 0.162]** | **0.145** | 0.38 | **0.38** | **30** | 7% |

- **Every feature model beats the baseline.** The paired gain over popularity is +0.069 to +0.127 P@10. Every CI excludes 0, and every p < 10⁻²⁴. The popularity baseline is barely above chance and recommends tracks with median popularity 92: the opposite of discovery.
- **Dislikes did not measurably help.** M1d − M1 = +0.003, 95% CI [−0.003, +0.009], p = 0.18. Weights above 0.15 *hurt* on train-CV, and the app's current 0.3 does worse than 0.15.
- **Decorrelating with PCA did not help.** M2 ≈ M1 (−0.001). The D2 concern about the redundant energy axis didn't translate into ranking quality.
- **The learned genre model is clearly best.** M3 − next best ≥ +0.054, CI [+0.038, +0.071], p < 0.01 against every candidate.

## 4. Selection
**The rule was fixed before looking at val:** take the highest P@10. If a simpler model's paired CI against it includes 0, take the simpler model instead. **M3 (random-forest genre model) is selected.** No simpler model is statistically tied with it.

The selection comes with two trade-offs we report rather than hide:
1. M3 has a lower hit-rate@10 than M1d (0.38 vs. 0.44). It is more "all-or-nothing": when it identifies the genre it fills the playlist, and when it doesn't, it misses completely.
2. M3 recommends from only **7% of the catalog**, versus 34% for the cosine models. Listeners with similar tastes receive overlapping playlists.

On the project's discovery goal, M3 still surfaces *less* popular music (median 30) than any other model. In D4 it becomes a scikit-learn Pipeline: scaler → a custom transformer that turns the forest's genre probabilities into a listener profile → ranking.

## 5. Error analysis (validation; P@10, n = listeners)
**By genre family** (`d3_2_p10_by_family.png`). Apart from the small catch-all "Other" family (children, kids, comedy, disco: 0.31, n = 20), M3 does best on **Latin (0.24, n = 60)**, **Chill & Acoustic (0.23, n = 55)**, and **Jazz & Classical (0.18, n = 20)**, the families D2 found to be acoustically distinct. It does worst on **World (0.04, n = 65)**, **Rock (0.05, n = 75)**, **Country & Roots (0.05, n = 20)**, and **Hip-Hop & R&B (0.07, n = 30)**. These are families defined more by culture, language, or lyrics than by sound. For World, M3 is no better than M1. This matches the D2 prediction that mainstream and culturally defined listeners would be harder. Groups of 20 listeners are small, so differences under about ±0.05 there are not reliable.

**By seed popularity.** Feature models work best for niche listeners: M3 scores 0.17 (pop 0–19, n = 86) and 0.22 (20–39, n = 196). They collapse for **listeners who like hits (62+, n = 59): M3 0.007, where popularity wins at 0.085**. This is the one group where the baseline is better. A practical fix (for D4/D5, to be validated) is to blend in popularity when the seeds are themselves popular.

**By seed coherence.** Listeners whose 5 likes are acoustically tight (Q1, n = 140) get 0.17. Spread-out tastes (Q3, n = 140) get 0.12. The dip is not monotone (Q4 = 0.14), so coherence matters less than genre.

**By number of swipes** (`d3_3_cold_start.png`). One liked song already gives M3 0.12, versus 0.018 for popularity. The gain flattens after 3–5 likes (0.135–0.137), so the app's 5-like threshold is reasonable.

**What the misses are.** **28% of M3's wrong picks are in the listener's own genre family.** Family-level P@10 is 0.38, compared with 0.14 for exact genre. The most common confusions are adjacent genres, not nonsense: death-metal → grindcore (34), samba → pagode (28), detroit-techno ↔ chicago-house (23/22), black-metal → grindcore (23), bluegrass → honky-tonk (20). Some confusions point to label noise from Spotify's keyword search (D2, Fig. 4), e.g., "electro"/"trip-hop" → "study", where "study" is a mood playlist rather than a sound. Some of what P@10 counts as error, a real listener might enjoy. `reports/d3_spot_checks.md` lists three listeners' seeds and recommendations for inspection.

## 6. Conclusions and limitations
Shared audio features of liked songs **do** improve the playlist over a popularity ranking: about 10× the baseline's genre precision (0.141 vs. 0.014), using only information available at swipe time. The features of *disliked* songs added nothing measurable in our setup. A learned genre representation beats raw-feature similarity.

The limitations are real:
- **The labels are a proxy.** Genre match isn't satisfaction, and genre keywords are noisy (adjacent-genre confusions are probably partly label error).
- **Simulated listeners.** Taste is single-genre and dislikes come from clearly different families. Real dislikes are subtler, which may be why they didn't help.
- **Artificial balance.** 1,000 tracks per genre over-represents niche genres compared with real listening, so absolute P@10 won't transfer.
- **Tuning grid edge.** The forest's best leaf size (20) was at the edge of the grid, so it may be slightly under-tuned.
- **Test set untouched.** The reserved test set will be used once, in D6, for the frozen M3 and the popularity baseline.

_Contributions: [to confirm with group]. AI assistance: Claude Code helped draft `scripts/evaluate.py`, `scripts/plot_d3.py`, and this note. The group reviewed the code, ran it, and checked the numbers against `reports/d3_summary.json`._
