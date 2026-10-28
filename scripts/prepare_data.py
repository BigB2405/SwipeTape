"""D2 data preparation: audit, clean, and split the Spotify Tracks dataset.

Usage: uv run python scripts/prepare_data.py [data/raw/dataset.csv]

Writes (all under data/processed/, gitignored):
  tracks_clean.parquet  train + val rows (split column), safe to upload/explore
  test.parquet          reserved final test set; do not open until D6
  audit.json            row counts for every cleaning step + split summary
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from genres import FAMILY_OF

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/dataset.csv")
OUT = Path("data/processed")
SEED = 42

AUDIO = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "time_signature"]

audit = {"source": str(SRC), "steps": []}

def log(step, df, note):
    audit["steps"].append({"step": step, "rows": len(df), "note": note})
    print(f"{step:<28} {len(df):>7,}  {note}")


raw = pd.read_csv(SRC, index_col=0)
audit["raw"] = {
    "rows": len(raw), "cols": raw.shape[1], "genres": int(raw.track_genre.nunique()),
    "rows_per_genre": sorted(raw.track_genre.value_counts().unique().tolist()),
    "unique_track_ids": int(raw.track_id.nunique()),
    "missing": {k: int(v) for k, v in raw.isna().sum().items() if v},
}
log("raw", raw, f"{raw.track_genre.nunique()} genres x {raw.track_genre.value_counts().iloc[0]} rows")

# 1. Missing identifiers / text
d = raw.dropna(subset=["track_name", "artists", "album_name"])
log("drop_missing_text", d, "null artists/album_name/track_name")

# 2. Exact duplicate rows
d = d.drop_duplicates()
log("drop_exact_duplicates", d, "identical rows (incl. genre)")

# 3. One row per track_id; keep every genre the track was listed under.
#    Rows for the same id can differ slightly in popularity, so keep the max.
genres_of = d.groupby("track_id").track_genre.agg(lambda s: sorted(set(s)))
n_genres = genres_of.str.len()
audit["multi_genre_track_ids"] = int((n_genres > 1).sum())
d = d.sort_values(["popularity", "track_genre"], ascending=[False, True]).drop_duplicates("track_id")
d["genres"] = d.track_id.map(genres_of)
d["n_genres"] = d.genres.str.len()
d["primary_genre"] = d.genres.str[0]  # alphabetical first; the label is ambiguous for n_genres > 1
d = d.drop(columns="track_genre")
log("collapse_track_id", d, f"{audit['multi_genre_track_ids']:,} ids appeared in >1 genre")

# 4. Same song released under several ids (single vs album, remasters): keep the most popular.
d["is_rerelease_dup"] = d.duplicated(["track_name", "artists"], keep="first")  # sorted by popularity desc
audit["rerelease_duplicates"] = int(d.is_rerelease_dup.sum())
d = d[~d.is_rerelease_dup].drop(columns="is_rerelease_dup")
log("drop_rerelease_duplicates", d, "same track_name + artists, different id")

# 5. Invalid / suspicious audio values: flag, then drop from the modeling table.
bad = (d.tempo == 0) | (d.time_signature == 0) | (d.duration_ms < 30_000) | (d.duration_ms > 15 * 60_000)
audit["invalid_audio"] = {
    "tempo_0": int((d.tempo == 0).sum()), "time_signature_0": int((d.time_signature == 0).sum()),
    "duration_lt_30s": int((d.duration_ms < 30_000).sum()), "duration_gt_15min": int((d.duration_ms > 15 * 60_000).sum()),
}
d = d[~bad]
log("drop_invalid_audio", d, "tempo=0, time_signature=0, duration <30s or >15min")

# 6. Derived columns
d["primary_artist"] = d.artists.str.split(";").str[0].str.strip()
d["artists"] = d.artists.str.replace(";", ", ")
d["genre_family"] = d.primary_genre.map(FAMILY_OF)
d["duration_s"] = (d.duration_ms / 1000).round(1)
d["pop_bucket"] = pd.cut(d.popularity, [-1, 0, 19, 39, 61, 100], labels=["0", "1-19", "20-39", "40-61", "62+"]).astype(str)
d["explicit"] = d.explicit.astype(bool)
assert d.genre_family.notna().all()
assert d.track_id.is_unique

# 7. Reserve test data: grouped by primary artist so no artist spans splits,
#    stratified by primary genre so every genre is represented.
d = d.sort_values("track_id").reset_index(drop=True)
def one_fold(df, n_splits):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    _, hold = next(sgkf.split(df, df.primary_genre, df.primary_artist))
    return df.index[hold]

test_idx = one_fold(d, 7)
d["split"] = "train"
d.loc[test_idx, "split"] = "test"
rest = d[d.split != "test"]
d.loc[one_fold(rest, 6), "split"] = "val"

arts = {s: set(d.loc[d.split == s, "primary_artist"]) for s in ("train", "val", "test")}
assert not (arts["test"] & (arts["train"] | arts["val"])), "artist leak into test"
assert not (arts["train"] & arts["val"]), "artist leak into val"

audit["split"] = {
    s: {"rows": int((d.split == s).sum()), "artists": len(arts[s]),
        "genres": int(d.loc[d.split == s, "primary_genre"].nunique())}
    for s in ("train", "val", "test")
}
audit["split"]["method"] = "StratifiedGroupKFold(y=primary_genre, groups=primary_artist, seed=42): test = 1/7, val = 1/6 of rest"
print("split", {s: v for s, v in audit["split"].items() if s != "method"})

cols = ["track_id", "track_name", "artists", "primary_artist", "album_name", "popularity", "pop_bucket",
        "duration_ms", "duration_s", "explicit", *AUDIO, "primary_genre", "genres", "n_genres", "genre_family", "split"]
d = d[cols]

OUT.mkdir(parents=True, exist_ok=True)
d[d.split == "test"].to_parquet(OUT / "test.parquet", index=False)
d[d.split != "test"].to_parquet(OUT / "tracks_clean.parquet", index=False)
(OUT / "audit.json").write_text(json.dumps(audit, indent=2))
print("wrote", *sorted(p.name for p in OUT.iterdir()))
