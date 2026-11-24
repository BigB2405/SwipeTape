"""Convert the model catalog (train + val) into a compact JSON the browser can load.

The app may only show songs the served model knows: the reserved test split stays out until D6.
"""
import json, sys
from pathlib import Path
import pandas as pd

from genres import APP_FAMILIES as FAMILIES, APP_FAMILY_OF as FAMILY_OF   # v2 app groups (see genres.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_def import LANGS, artist_title_langs, track_langs  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/processed/tracks_clean.parquet"
TEST = "data/processed/test.parquet"
OUT = "public/songs.json"
MIN_POP = 40

fam_of = FAMILY_OF

d = pd.read_parquet(SRC).dropna(subset=["track_name", "artists"])
ARTIST_LANG = artist_title_langs(d.primary_artist, d.track_name, d.album_name)   # before the popularity cut, like the model
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
        [LANGS.index(c) for c in track_langs(r.genres, r.track_name, ARTIST_LANG.get(r.primary_artist))],     # v2: estimated languages (index 17)
    ])

json.dump({
    "genres": genres,
    "genreFamily": [fams.index(fam_of[g]) for g in genres],
    "families": fams,
    "langs": LANGS,
    "tracks": rows,
}, open(OUT, "w"), separators=(",", ":"), ensure_ascii=False)
print(len(rows), "tracks ->", OUT)
