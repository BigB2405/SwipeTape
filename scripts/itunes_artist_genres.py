"""Look up each ambiguous artist's genre on the public iTunes Search API (v2 language estimate).

Usage: uv run --frozen python scripts/itunes_artist_genres.py [--limit N]
Candidates: artists with a song of popularity >= 55 (>= 40 in mixed-language genres) whose tracks carry no language signal from genre
tags or title alphabet (they would read as "en"). One request every 3 s (Apple asks for about 20 per
minute); the cache lang_data/itunes_artist_genres.json is saved as it goes, so the script can stop and
resume. Only public artist names are sent.

Cache format: {"<artist>": [["<iTunes artistName>", "<primaryGenreName>"], ...]}  ([] = no result)
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_def import track_langs  # noqa: E402

OUT = ROOT / "lang_data" / "itunes_artist_genres.json"
MIN_POP, MIN_POP_MIXED, PAUSE = 55, 40, 3.0
MIXED_GENRES = {"hip-hop", "pop", "dance", "r-n-b", "edm", "party", "club", "world-music", "chill", "sad",
                "romance", "acoustic", "electro", "house", "reggae", "dancehall", "funk", "soul"}


def candidates() -> list[str]:
    c = pd.read_parquet(ROOT / "data/processed/tracks_clean.parquet")
    c["L"] = [track_langs(g, n) for g, n in zip(c.genres, c.track_name)]
    amb = c[c.L == ("en",)]
    top = amb.groupby("primary_artist").popularity.max()
    # artists in genres where non-English songs hide (e.g. Punjabi hip-hop) go first, down to popularity 40
    # (the app's catalog cut); then everyone else with a song of popularity >= 55
    mixed = set(amb[amb.genres.map(lambda g: bool(set(g) & MIXED_GENRES))].primary_artist)
    keep = [a for a in top.index if top[a] >= (MIN_POP_MIXED if a in mixed else MIN_POP)]
    return sorted(keep, key=lambda a: (a not in mixed, -top[a]))


def lookup(artist: str):
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"entity": "musicArtist", "limit": 3, "term": artist})
    req = urllib.request.Request(url, headers={"User-Agent": "swipetape-class-project/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.load(r).get("results", [])
    return [[x.get("artistName", ""), x.get("primaryGenreName", "")] for x in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [a for a in candidates() if a not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cache)} cached, {len(todo)} to look up", flush=True)
    backoff = PAUSE
    for i, a in enumerate(todo, 1):
        try:
            cache[a] = lookup(a)
            backoff = PAUSE
        except Exception as e:  # noqa: BLE001  rate limit / network: wait longer, retry later
            print(f"  {a}: {e}; pausing", flush=True)
            backoff = min(backoff * 2, 120)
        if i % 25 == 0 or i == len(todo):
            OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=0, sort_keys=True))
            print(f"  {i}/{len(todo)} done", flush=True)
        time.sleep(backoff)
    OUT.write_text(json.dumps(cache, ensure_ascii=False, indent=0, sort_keys=True))


if __name__ == "__main__":
    main()
