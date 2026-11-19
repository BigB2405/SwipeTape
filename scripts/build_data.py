"""Convert the model catalog (train + val) into a compact JSON the browser can load.

The app may only show songs the served model knows: the reserved test split stays out until D6.
"""
import json, sys
import pandas as pd

from genres import FAMILIES, FAMILY_OF

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/processed/tracks_clean.parquet"
TEST = "data/processed/test.parquet"
OUT = "public/songs.json"
MIN_POP = 40

fam_of = FAMILY_OF

d = pd.read_parquet(SRC).dropna(subset=["track_name", "artists"])
assert set(d.split) <= {"train", "val"}, "test rows must never reach the app before D6"
assert not set(d.track_id) & set(pd.read_parquet(TEST, columns=["track_id"]).track_id), "test track ids in catalog"
d = d[d.popularity >= MIN_POP]
d = d.sort_values("popularity", ascending=False).drop_duplicates(["track_name", "artists"])

genres = sorted(d.primary_genre.unique())
fams = list(FAMILIES)
missing = [g for g in genres if g not in fam_of]
assert not missing, missing
gi = {g: i for i, g in enumerate(genres)}

rows = []
for r in d.itertuples():
    rows.append([
        r.track_id, r.track_name, r.artists.replace(";", ", "), r.album_name,
        int(r.popularity), gi[r.primary_genre], int(r.duration_ms // 1000), int(bool(r.explicit)),
        round(r.danceability, 3), round(r.energy, 3), round(r.valence, 3),
        round(r.acousticness, 3), round(r.instrumentalness, 3), round(r.speechiness, 3),
        round(r.liveness, 3), round(r.tempo, 1), round(r.loudness, 1),
    ])

json.dump({
    "genres": genres,
    "genreFamily": [fams.index(fam_of[g]) for g in genres],
    "families": fams,
    "tracks": rows,
}, open(OUT, "w"), separators=(",", ":"), ensure_ascii=False)
print(len(rows), "tracks ->", OUT)
