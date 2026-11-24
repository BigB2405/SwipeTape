"""Fit the D3-selected recommender as one scikit-learn Pipeline and save it for the D4 API.

Usage: uv run --frozen python scripts/train_pipeline.py          # M5 (D3/D4 artifact) -> pipeline.joblib
       uv run --frozen python scripts/train_pipeline.py --v2     # M6 (real genre tags + popularity) -> pipeline_v2.joblib
Reads data/processed/tracks_clean.parquet (train + val = the serving catalog; test stays reserved).
Writes pipeline.joblib: a dict bundle {"pipeline": fitted Pipeline, "metadata": {...}} (HW4 format).

    bundle = joblib.load("pipeline.joblib")
    bundle["pipeline"].named_steps["rec"].recommend(liked_ids=[...], disliked_ids=[...])
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_def import FEATS, SongEmbedder, SwipeRecommender  # noqa: E402  (pickled by this module name)

OUT = Path("pipeline.joblib")   # repo root, next to serve.py (gitignored; rebuild with this script)
OUT_V2 = Path("pipeline_v2.joblib")
V2 = dict(genre_source="labels", alpha=0.5, tau=0.3, lam=0.0, pop_w=0.05,   # chosen Nov 23 from reports/v2_compare.csv
          hide_unless_seeded=["children", "kids", "sleep"], lang_filter=True)
V2_NAME = "M6 tags alpha=0.5 pop=0.05"


def selected_params():
    """Hyperparameters chosen in D3 (reports/d3_summary.json), so the artifact can't drift from the note."""
    S = json.loads(Path("reports/d3_summary.json").read_text())
    sel, m4 = S["selected"], S["chosen_configs"]["M4"]
    kv = dict(p.split("=") for p in sel.split()[1:])
    kv4 = dict(p.split("=") for p in m4.split()[1:])
    assert sel.startswith("M5"), f"train_pipeline expects M5 to be selected, got {sel}"
    return dict(alpha=float(kv["alpha"]), min_leaf=int(kv["leaf"]), tau=float(kv4["tau"]), lam=float(kv4["lam"])), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="build the v2 model (real genre tags + popularity boost)")
    v2 = ap.parse_args().v2
    p, S = selected_params()
    cat = pd.read_parquet("data/processed/tracks_clean.parquet")
    assert set(cat.split) == {"train", "val"}, "test rows must never enter the catalog before D6"
    cat = cat.reset_index(drop=True)

    if v2:
        p = dict(V2)
        pipe = Pipeline([("embed", SongEmbedder(alpha=p["alpha"], genre_source="labels")),
                         ("rec", SwipeRecommender(tau=p["tau"], lam=p["lam"], pop_w=p["pop_w"],
                                                  hide_unless_seeded=tuple(p["hide_unless_seeded"]),
                                                  lang_filter=p["lang_filter"]))])
    else:
        pipe = Pipeline([("embed", SongEmbedder(alpha=p["alpha"], min_leaf=p["min_leaf"])),
                         ("rec", SwipeRecommender(tau=p["tau"], lam=p["lam"]))])
    pipe.fit(cat, rec__meta=cat)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    metadata = dict(
        steps=[{"name": n, "class": type(s).__name__} for n, s in pipe.steps],
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(), joblib_version=joblib.__version__,
        pandas_version=pd.__version__, numpy_version=np.__version__, git_commit=sha,
        model=S["selected"], params=p, features=FEATS, catalog_size=len(cat),
        d3_validation=dict(recall50=S["validation"][S["selected"]]["r50"],
                           recall50_ci95=S["validation"][S["selected"]]["r50_ci95"],
                           popularity_recall50=S["validation"]["B1 Popularity"]["r50"]))
    out = OUT
    if v2:
        out = OUT_V2
        row = pd.read_csv("reports/v2_compare.csv").set_index("model").loc["Tags alpha=0.5 pop=0.05"]
        metadata.update(model=V2_NAME, previous_model=S["selected"],
                        v2_validation={k: float(row[k]) for k in ("r50", "r50_mixed", "genre_agree", "kids_rate", "n_genres")},
                        v2_note="r50 is inflated for tag models: held-out same-artist songs share the seeds' genre tags")
    joblib.dump({"pipeline": pipe, "metadata": metadata}, out, compress=3)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
