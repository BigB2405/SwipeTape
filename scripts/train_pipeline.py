"""Fit the D3-selected recommender as one scikit-learn Pipeline and save it for the D4 API.

Usage: uv run --frozen python scripts/train_pipeline.py
Reads data/processed/tracks_clean.parquet (train + val = the serving catalog; test stays reserved).
Writes artifacts/swipetape_pipeline.joblib and artifacts/bundle.json.

    pipe = joblib.load("artifacts/swipetape_pipeline.joblib")
    pipe.named_steps["rec"].recommend(liked_ids=[...], disliked_ids=[...])
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swipetape.model import FEATS, SongEmbedder, SwipeRecommender  # noqa: E402

OUT = Path("artifacts")


def selected_params():
    """Hyperparameters chosen in D3 (reports/d3_summary.json), so the artifact can't drift from the note."""
    S = json.loads(Path("reports/d3_summary.json").read_text())
    sel, m4 = S["selected"], S["chosen_configs"]["M4"]
    kv = dict(p.split("=") for p in sel.split()[1:])
    kv4 = dict(p.split("=") for p in m4.split()[1:])
    assert sel.startswith("M5"), f"train_pipeline expects M5 to be selected, got {sel}"
    return dict(alpha=float(kv["alpha"]), min_leaf=int(kv["leaf"]), tau=float(kv4["tau"]), lam=float(kv4["lam"])), S


def main():
    p, S = selected_params()
    cat = pd.read_parquet("data/processed/tracks_clean.parquet")
    assert set(cat.split) == {"train", "val"}, "test rows must never enter the catalog before D6"
    cat = cat.reset_index(drop=True)

    pipe = Pipeline([("embed", SongEmbedder(alpha=p["alpha"], min_leaf=p["min_leaf"])),
                     ("rec", SwipeRecommender(tau=p["tau"], lam=p["lam"]))])
    pipe.fit(cat, rec__meta=cat)

    OUT.mkdir(exist_ok=True)
    path = OUT / "swipetape_pipeline.joblib"
    joblib.dump(pipe, path, compress=3)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    bundle = dict(
        artifact=path.name, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=sha, python=platform.python_version(), sklearn_version=sklearn.__version__,
        joblib_version=joblib.__version__, pandas_version=pd.__version__,
        model=S["selected"], params=p, features=FEATS, catalog_size=len(cat),
        artifact_mb=round(path.stat().st_size / 1e6, 1),
        d3_validation=dict(recall50=S["validation"][S["selected"]]["r50"],
                           recall50_ci95=S["validation"][S["selected"]]["r50_ci95"],
                           popularity_recall50=S["validation"]["B1 Popularity"]["r50"]))
    (OUT / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
