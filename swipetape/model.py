"""Kept so D3 code (`from swipetape.model import ...`) keeps working. The classes live in pipeline_def.py (HW4 layout)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_def import (APP_W, FEATS, SEED, SongEmbedder, SwipeRecommender,  # noqa: E402,F401
                          fit_genre_forest, taste_scores, unit)
