"""SwipeTape D2 dashboard: data audit + EDA, reading from Supabase.

Local:  uv run streamlit run dashboard/app.py   (needs SUPABASE_URL + SUPABASE_ANON_KEY in .env)
Modal:  modal deploy dashboard/modal_app.py
"""
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
st.set_page_config(page_title="SwipeTape · D2 Data", page_icon="🎧", layout="wide")

FEATS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
         "speechiness", "liveness", "tempo", "loudness"]
FAMILIES = ["Pop", "Rock", "Metal", "Hip-Hop & R&B", "Electronic", "Latin", "Chill & Acoustic",
            "Country & Roots", "Jazz & Classical", "World", "Other"]
DICT_MD = Path(__file__).resolve().parent.parent / "docs" / "data_dictionary.md"
PAGE = 1000  # PostgREST max rows per request


@st.cache_resource
def client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


@st.cache_data(ttl=3600, show_spinner="Loading tracks from Supabase…")
def load_tracks() -> pd.DataFrame:
    cols = "track_id,track_name,artists,primary_artist,popularity,pop_bucket,explicit,duration_s," \
           + ",".join(FEATS) + ",primary_genre,genres,n_genres,genre_family,split"
    rows, start = [], 0
    while True:
        chunk = client().table("tracks").select(cols).order("track_id").range(start, start + PAGE - 1).execute().data
        rows += chunk
        if len(chunk) < PAGE:
            break
        start += PAGE
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_table(name: str, order: str) -> pd.DataFrame:
    return pd.DataFrame(client().table(name).select("*").order(order).execute().data)


try:
    tracks = load_tracks()
    audit = load_table("audit_steps", "ord")
    splits = load_table("split_summary", "split")
except Exception as e:  # missing secrets or Supabase down
    st.error(f"Could not load data from Supabase: {e}")
    st.stop()

st.sidebar.title("🎧 SwipeTape")
st.sidebar.caption("DTSC 3602 · Group 10 · D2 data preparation & EDA")
page = st.sidebar.radio("Page", ["Overview", "Explore", "EDA findings", "Data dictionary"])
st.sidebar.divider()
st.sidebar.caption(
    "Source: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). "
    "Only train + validation rows are stored here; the reserved test split stays offline until D6."
)

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("Data audit & cleaning")
    raw_rows, final_rows = int(audit.rows.iloc[0]), int(splits.rows.sum())
    c = st.columns(4)
    c[0].metric("Raw rows", f"{raw_rows:,}")
    c[1].metric("Clean tracks", f"{final_rows:,}", f"{final_rows - raw_rows:,}")
    c[2].metric("Genres", tracks.primary_genre.nunique())
    c[3].metric("Artists (train+val)", f"{tracks.primary_artist.nunique():,}")

    st.subheader("Cleaning steps")
    a = audit.assign(removed=(audit.rows.shift(1) - audit.rows).fillna(0).astype(int))
    st.dataframe(a[["step", "rows", "removed", "note"]], hide_index=True, width="stretch")
    st.altair_chart(
        alt.Chart(a).mark_bar(color="#4c72b0").encode(
            x=alt.X("rows:Q", title="Rows remaining"), y=alt.Y("step:N", sort=None, title=None),
            tooltip=["step", "rows", "removed", "note"]).properties(height=220),
        width="stretch")

    st.subheader("Train / validation / test split")
    st.dataframe(splits, hide_index=True, width="stretch")
    st.caption(
        "StratifiedGroupKFold (seed 42): grouped by primary artist so no artist appears in more than one split, "
        "stratified by primary genre so all genres are represented. Test = 1/7 of tracks, validation = 1/6 of the rest.")

# ------------------------------------------------------------------- Explore
elif page == "Explore":
    st.title("Explore the tracks")
    f1, f2, f3 = st.columns([2, 2, 1])
    fams = f1.multiselect("Genre family", FAMILIES, default=[])
    pop = f2.slider("Popularity", 0, 100, (0, 100))
    split = f3.selectbox("Split", ["train", "val", "both"])
    v = tracks[tracks.popularity.between(*pop)]
    if fams:
        v = v[v.genre_family.isin(fams)]
    if split != "both":
        v = v[v.split == split]
    st.caption(f"{len(v):,} tracks · {v.primary_artist.nunique():,} artists")
    if v.empty:
        st.info("No tracks match these filters.")
        st.stop()

    feat = st.selectbox("Feature", FEATS + ["popularity", "duration_s"], index=1)
    st.altair_chart(
        alt.Chart(v).mark_bar(opacity=0.75).encode(
            x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=40)), y=alt.Y("count():Q", stack=None, title="Tracks"),
            color=alt.Color("genre_family:N", sort=FAMILIES, title="Family")).properties(height=320),
        width="stretch")
    st.dataframe(
        v.sort_values("popularity", ascending=False)[["track_name", "artists", "primary_genre", "popularity", *FEATS[:4]]].head(500),
        hide_index=True, width="stretch")

# -------------------------------------------------------------- EDA findings
elif page == "EDA findings":
    st.title("EDA findings (train split)")
    tr = tracks[tracks.split == "train"]

    st.subheader("1 · Popularity is skewed, and the app's filters hide most artists")
    rules = pd.DataFrame({"x": [40, 62], "label": ["build_data.py cutoff", "swipe deck cutoff"]})
    hist = alt.Chart(tr).mark_bar(color="#4c72b0").encode(
        x=alt.X("popularity:Q", bin=alt.Bin(step=2), title="Spotify popularity"), y=alt.Y("count():Q", title="Tracks"))
    st.altair_chart((hist + alt.Chart(rules).mark_rule(color="#c44e52", strokeDash=[5, 4], size=2).encode(x="x:Q", tooltip="label"))
                    .properties(height=280), width="stretch")
    ge40, ge62 = (tr.popularity >= 40).mean(), (tr.popularity >= 62).mean()
    art62 = tr[tr.popularity >= 62].primary_artist.nunique() / tr.primary_artist.nunique()
    st.markdown(
        f"Median popularity is **{tr.popularity.median():.0f}**, and **{(tr.popularity == 0).mean():.1%}** of tracks sit at exactly 0. "
        f"Only **{ge40:.0%}** of tracks pass the current `songs.json` cutoff (40) and **{ge62:.0%}** pass the swipe-deck cutoff (62), "
        f"which keeps just **{art62:.0%}** of artists. That works against the D1 goal of surfacing niche artists, so the app's "
        "popularity filter and popularity bonus have to be revisited and treated as a modeling choice.")

    st.subheader("2 · Several audio features are redundant")
    corr = tr[FEATS + ["popularity"]].corr().stack().rename("r").reset_index()
    st.altair_chart(
        alt.Chart(corr).mark_rect().encode(
            x=alt.X("level_0:N", title=None, sort=FEATS), y=alt.Y("level_1:N", title=None, sort=FEATS),
            color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
            tooltip=["level_0", "level_1", alt.Tooltip("r:Q", format=".2f")]).properties(height=380),
        width="stretch")
    st.markdown(
        "Energy is strongly tied to loudness (r ≈ 0.76) and inversely to acousticness (r ≈ −0.73), so an unweighted "
        "distance counts \"loud/energetic vs. acoustic\" about three times. Popularity is barely correlated with any audio "
        "feature (|r| ≤ 0.18), so audio similarity alone won't reproduce a popularity ranking. That makes popularity a fair, "
        "independent baseline for D3.")

    st.subheader("3 · Features separate some genre families, not all")
    feat = st.selectbox("Feature", ["energy", "danceability", "valence", "acousticness", "instrumentalness", "speechiness"])
    st.altair_chart(
        alt.Chart(tr[tr.genre_family != "Other"]).mark_boxplot(extent="min-max", color="#8da0cb").encode(
            x=alt.X(f"{feat}:Q"), y=alt.Y("genre_family:N", sort=FAMILIES, title=None)).properties(height=330),
        width="stretch")
    st.markdown(
        "Metal (high energy, near-zero acousticness) and Jazz & Classical / Chill & Acoustic (low energy, high acousticness) "
        "are well separated. Pop, Rock, Latin, and World overlap heavily. Genre explains roughly 45–50% of the variance in "
        "energy, acousticness, and loudness but under 10% for tempo, so tempo carries little genre signal.")

    st.subheader("4 · Genre labels overlap")
    pc = Counter(p for gs in tr.genres for p in combinations(gs, 2))
    top = pd.DataFrame([{"pair": f"{a} + {b}", "tracks": n} for (a, b), n in pc.most_common(15)])
    st.altair_chart(
        alt.Chart(top).mark_bar(color="#55a868").encode(
            x="tracks:Q", y=alt.Y("pair:N", sort="-x", title=None)).properties(height=360),
        width="stretch")
    st.markdown(
        f"**{(tr.n_genres > 1).mean():.0%}** of tracks are listed under more than one genre. Some pairs are near-synonyms "
        "(singer-songwriter / songwriter, punk / punk-rock), and `songwriter` / `singer-songwriter` are the exact same 1,000 tracks. "
        "Others look like search-keyword artifacts (dub / dubstep, reggae / reggaeton). In D3, a recommendation counts as a "
        "genre match if it shares *any* genre with the seed, not just the primary one.")

# ----------------------------------------------------------- Data dictionary
else:
    st.title("Data dictionary")
    st.markdown(DICT_MD.read_text() if DICT_MD.exists() else "_docs/data_dictionary.md not found._")
