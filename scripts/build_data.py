"""Convert Music_info.csv into a compact JSON the browser can load."""
import json, sys
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "Music_info.csv"
OUT = "public/songs.json"
MIN_POP = 40

FAMILIES = {
    "Pop": "pop indie-pop power-pop synth-pop pop-film k-pop j-pop j-idol mandopop cantopop swedish british happy party romance".split(),
    "Rock": "rock alt-rock alternative indie grunge psych-rock hard-rock punk punk-rock rock-n-roll rockabilly j-rock emo garage guitar".split(),
    "Metal": "metal heavy-metal black-metal death-metal metalcore grindcore hardcore industrial goth".split(),
    "Hip-Hop & R&B": "hip-hop r-n-b soul funk groove gospel".split(),
    "Electronic": "edm electronic electro house deep-house progressive-house chicago-house techno detroit-techno minimal-techno trance dubstep drum-and-bass breakbeat hardstyle idm trip-hop club dance j-dance".split(),
    "Latin": "latin latino reggaeton salsa samba tango brazil mpb pagode sertanejo forro spanish".split(),
    "Chill & Acoustic": "acoustic chill ambient new-age sleep study piano singer-songwriter songwriter folk sad".split(),
    "Country & Roots": "country bluegrass honky-tonk blues".split(),
    "Jazz & Classical": "jazz classical opera show-tunes disney".split(),
    "World": "afrobeat reggae dancehall dub ska indian iranian turkish malay world-music french german anime".split(),
    "Other": "children kids comedy disco".split(),
}
fam_of = {g: f for f, gs in FAMILIES.items() for g in gs}

d = pd.read_csv(SRC, index_col=0).dropna(subset=["track_name", "artists"])
d = d[d.popularity >= MIN_POP]
d = d.sort_values("popularity", ascending=False).drop_duplicates(["track_name", "artists"])

genres = sorted(d.track_genre.unique())
fams = list(FAMILIES)
missing = [g for g in genres if g not in fam_of]
assert not missing, missing
gi = {g: i for i, g in enumerate(genres)}

rows = []
for r in d.itertuples():
    rows.append([
        r.track_id, r.track_name, r.artists.replace(";", ", "), r.album_name,
        int(r.popularity), gi[r.track_genre], int(r.duration_ms // 1000), int(bool(r.explicit)),
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
