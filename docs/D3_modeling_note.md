# D3 Modeling Note: SwipeTape (Group 10)

**Repository:** https://github.com/BigB2405/SwipeTape · **Branch/tag:** `D3` / `d3` · **Run:** `uv run --frozen python scripts/evaluate.py && uv run --frozen python scripts/plot_d3.py`
**Research question:** Can shared musical features among songs a user likes or dislikes improve the quality of the personalized playlist produced by a swipe-based recommender?

**What changed since the first D3 draft.** SwipeTape is a *song* recommender: after a few swipes, it should find specific songs that sound like the ones you liked. Our first draft scored a playlist by whether each song carried the listener's *genre label*. That rewarded a model that sorts songs into genre buckets (7% catalog coverage), not one that matches songs. We therefore changed the primary evaluation to a song-level test (below) and kept the genre test as a secondary check. Both sets of results are reported.

## 1. Evaluation design
**Simulated listeners with held-out songs.** We have no real swipe logs, so we need a stand-in for "this listener would like *this* song" that the models cannot see. We use the listener's favorite artist's other songs:
- **Single-taste listener:** likes 5 songs by one artist and dislikes 5 songs from other genre families. Built for every validation artist with ≥8 tracks: **368 listeners**.
- **Mixed-taste listener:** likes 3 + 3 songs by two artists from *different* families, with 5 dislikes from neither family: **400 listeners**. This tests whether a model can serve two tastes at once instead of averaging them.
- The artist's **remaining songs are held out** (≥3 per artist) and hidden in the whole validation pool of 11,612 tracks. The model ranks the pool, and we measure where the held-out songs land.

**Metrics.** The **primary metric is Recall@50**: the share of held-out songs ranked in the top 50, normalized by min(#held, 50). Fifty is about the number of songs a user sees across a few playlists. Secondary metrics: Recall@10, MRR (how early the first held-out song appears), and Recall@50 restricted to held-out songs from a **different album** than every seed. That last one checks that we are matching taste, not just shared album production. We also report the old **genre P@10** under the original protocol, plus the median popularity of recommended songs and catalog coverage, which are discovery checks. Uncertainty comes from 2,000 bootstrap resamples over listeners and paired Wilcoxon tests on per-listener Recall@50. **Chance Recall@50 ≈ 0.004.**

**Leakage controls.**
| Risk | Control |
|---|---|
| Model knows the artist | Models get **only the 9 audio features**: no artist, album, genre, or popularity. |
| Artist seen in training | The D2 split is grouped by artist (asserted), so scalers and forests never saw the held-out artists. |
| Re-releases of a seed counted as a hit | Same-title songs by the seed artist (33 duplicates in val) are removed from the pool and the held-out set (asserted). |
| Tuning on the comparison data | Every hyperparameter is tuned with 3-fold GroupKFold **by artist inside train**, using the same song-level protocol. Val is used once. |
| Test-set peeking | `test.parquet` is never opened (asserted). It is reserved for D6. |

## 2. Models
| ID | Model | Tuned on train-CV (Recall@50) |
|---|---|---|
| B0 / B1 | Random / **Popularity baseline** (the same most-popular songs for everyone) | — |
| M1 | **Cosine to the liked centroid**, using the app's feature weights (the current app's audio core) | — |
| M1d | M1 **minus λ × cosine to the disliked centroid** | λ ∈ {0.15, 0.3, 0.5, 1} → 0.15 |
| M3 | **Genre forest** (first-draft pick): a random forest predicts each song's genre mix from audio; rank by overlap with the seeds' mix | min_leaf ∈ {10, 20, 50} → 10 |
| M4 | **Song-to-song kNN**: each candidate is scored by its similarity to *each* liked song (a soft maximum with temperature τ, so the nearest liked song counts most), minus λ × similarity to the nearest disliked song | τ ∈ {0.03, 0.1, 0.3, 1}, λ ∈ {0, 0.15, 0.3} → **τ = 0.3, λ = 0** |
| **M5** | **Hybrid song embedding + M4 scoring**: each song = [weighted standardized audio ‖ α × forest genre-mix vector]. The forest's learned "what does this sound like" signal is used *per song*, not as a genre filter | α ∈ {0.5, 1, 2, 3, 4} × min_leaf ∈ {10, 20, 50}, τ/λ from M4 → **α = 2, leaf = 10** |

Two grid notes, disclosed: (1) After the first run, α = 2 was at the edge of the grid, so we widened it to 3–4 (train-CV only; α = 2 stayed best). (2) min_leaf = 5 scored the same on train-CV (0.1339 vs. 0.1333) but doubles the forest (111 MB saved, ~1.5 GB of RAM for the forest alone). We excluded it *before* selection to keep the D4 server practical. With leaf = 10, the full saved pipeline (forest + 69k catalog embeddings) is 102 MB, loads in 2.4 s, and uses ~0.9 GB of RAM. It is rebuilt by `scripts/train_pipeline.py`, not committed.

## 3. Results (validation; Figures `d3_1_model_comparison.png`, `d3_4_single_vs_mixed.png`)
| Model | Recall@50 [95% CI] | R@10 | MRR | Single | Mixed | Other-album R@50 | Genre P@10 | Coverage |
|---|---|---|---|---|---|---|---|---|
| B0 Random | 0.004 [0.003, 0.005] | 0.001 | 0.009 | 0.003 | 0.004 | 0.004 | 0.006 | 38% |
| B1 Popularity | 0.007 [0.005, 0.010] | 0.001 | 0.010 | 0.008 | 0.007 | 0.010 | 0.014 | 0.1% |
| M1 Centroid (app) | 0.086 [0.075, 0.095] | 0.046 | 0.108 | 0.128 | 0.047 | 0.073 | 0.084 | 34% |
| M1d + dislikes | 0.085 [0.075, 0.096] | 0.041 | 0.104 | 0.130 | 0.044 | 0.075 | 0.087 | 34% |
| M3 Genre forest | 0.090 [0.080, 0.100] | 0.036 | 0.091 | 0.109 | 0.072 | 0.085 | **0.153** | 8% |
| M4 Song kNN | 0.108 [0.097, 0.119] | 0.067 | 0.151 | 0.139 | 0.080 | 0.087 | 0.086 | 32% |
| **M5 Hybrid kNN** | **0.139 [0.127, 0.151]** | **0.087** | **0.201** | **0.188** | **0.095** | **0.121** | 0.097 | 29% |

- **Every audio model beats popularity by a wide margin.** The paired gain in Recall@50 is +0.078 to +0.132, every CI excludes 0, and every p < 10⁻⁵³. Popularity barely beats chance and recommends songs with median popularity 92.
- **Matching song to song beats averaging.** M4 − M1 = +0.022 overall. For mixed tastes it is +0.033 [+0.025, +0.041], p < 10⁻¹³: the centroid lands *between* two tastes and serves neither. M5 gets at least one held-out song from *both* artists into the top 50 for 12% of mixed listeners, versus 3.5% for the centroid.
- **The learned genre-mix signal helps when it is used per song.** M5 − M4 = +0.031 [+0.023, +0.039]. Used as a genre bucket (M3), the same forest wins the old genre test (P@10 0.153) but finds the listener's actual songs no better than the centroid.
- **Dislikes did not help.** M1d − M1 = −0.000 [−0.005, +0.005], p = 0.50. Turning on a dislike penalty in M4 (λ = 0.3) *hurts*: −0.011 [−0.019, −0.003], p = 0.005. Train-CV chose λ = 0.
- **Not just album production.** Restricted to held-out songs from other albums, M5 still has Recall@50 of 0.121, against 0.139 overall.

## 4. Selection
**The rule was fixed before looking at val:** take the highest Recall@50. If a simpler model's paired CI against it includes 0, take the simpler model instead. **M5 (α = 2, min_leaf = 10, τ = 0.3, λ = 0) is selected.** Every other candidate is significantly worse (all paired CIs exclude 0, p < 10⁻¹⁴).

Trade-offs we accept: M5's genre P@10 (0.097) is lower than M3's (0.153). That is the point of the change: M5 recommends songs that sound like your songs even when they carry a different genre tag. It also covers 29% of the catalog, against 8% for M3, so playlists are much less bucketed. In D4, M5 becomes a scikit-learn Pipeline: `SongEmbedder` (custom transformer: scaler + genre-mix forest) → `SwipeRecommender` (soft-max similarity ranking, 2-per-artist cap) in `swipetape/model.py`. D3's numbers come from that same code.

## 5. Error analysis (validation; Recall@50, n = listeners)
**By genre family** (single-taste, `d3_2_r50_by_family.png`). M5 is strongest on **Chill & Acoustic (0.26, n = 33)**, **Latin (0.26, n = 42)**, and **Metal (0.24, n = 30)**, families with a distinctive sound. It is weakest on **Rock (0.09, n = 36)**, **Country & Roots (0.09, n = 13)**, and **Pop (0.14, n = 57)**. For Rock, M5 barely beats the centroid (0.08). For Country, the simpler M4 and M3 do slightly better (0.10–0.11), but n = 13 is too small to rely on. These are families where artists vary more than the audio features capture (lyrics, era, instrumentation).

**By seed popularity.** M5 works best for niche listeners: 0.19 (seed popularity 0–19, n = 123), falling to 0.12 for 40–61 (n = 280) and 62+ (n = 42). In the first draft, popularity *beat* the genre model for hit-lovers. On the song-level test it does not: popularity scores 0.06 for 62+, against 0.12 for M5.

**By number of likes** (`d3_3_cold_start.png`). One liked song already gives 0.12, against 0.008 for popularity. Three likes give 0.18, and five give 0.19, so returns flatten after about 3 swipes right.

**Spot checks** (`reports/d3_spot_checks.md`, discovery playlists with seed artists excluded). A Bring Me The Horizon listener gets metalcore/heavy-metal acts (For Today, Fit For A King, Novelists FR). An *Encanto* score listener gets film-score and ambient piano (Abel Korzeniowski, Patrick Doyle, Secret Garden). A mixed pagode + honky-tonk listener gets both sides (sertanejo, Webb Pierce, MPB), plus two Bollywood tracks that match the honky-tonk seeds on audio alone: a typical miss where the language and culture are invisible to the features. Most picks are explained by one "hub" seed ("because you liked *Crooked Young*"), which suggests diversifying across seeds in D5.

## 6. Conclusions and limitations
Shared audio features of liked songs **do** improve the playlist: the selected model finds a listener's held-out songs about **19× more often than the popularity ranking** (Recall@50 0.139 vs. 0.007) and 1.6× more often than the app's current centroid. Matching **song to song** matters most for listeners with more than one taste. The features of **disliked** songs added nothing, and a dislike penalty slightly hurt.

Limitations:
- **"Same artist" stands in for taste.** Real listeners like many artists, and an artist's songs are more alike than a listener's favorites. Absolute Recall values will not transfer to real users. The comparison between models is the claim, not the level.
- **Simulated listeners and dislikes.** Dislikes come from clearly different families. Real dislikes are subtler, which may be why they don't help here.
- **Only artists with ≥8 val tracks** are used as listeners, which over-represents prolific artists.
- **Audio features miss language, lyrics, and era** (the Bollywood miss above). Genre labels from Spotify search are noisy (D2).
- **Deployment-driven grid limit** (min_leaf ≥ 10) is disclosed above. The cost on train-CV was 0.0006.
- **One taste can crowd out another.** In a manual check with 3 metal likes and 2 jazz likes, all 10 picks were metal. In D5 we plan to fill the playlist round-robin across liked songs and will validate that on train-CV first.
- **Test set untouched.** The frozen M5 and the popularity baseline will be scored once on `test.parquet` in D6, on both the song-level and genre tests.

_Contributions: [to confirm with group]. AI assistance: Claude Code helped draft `swipetape/model.py`, `scripts/evaluate.py`, `scripts/plot_d3.py`, and this note. The group reviewed the code, ran it, and checked the numbers against `reports/d3_summary.json`._
