"""Upload the cleaned train/val tracks and the audit summary to Supabase.

Usage: uv run python scripts/load_supabase.py
Needs SUPABASE_URL and SUPABASE_SERVICE_KEY in .env (service role: the tables are read-only for anon).
Run sql/schema.sql in the Supabase SQL editor first.
"""
import json, os

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

BATCH = 1000

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

d = pd.read_parquet("data/processed/tracks_clean.parquet")
assert set(d.split) <= {"train", "val"}, "test rows must never be uploaded"
d["genres"] = d.genres.map(list)
records = json.loads(d.to_json(orient="records"))  # numpy -> plain JSON types

for i in range(0, len(records), BATCH):
    sb.table("tracks").upsert(records[i:i + BATCH]).execute()
    print(f"tracks {min(i + BATCH, len(records)):,}/{len(records):,}", end="\r")
print()

audit = json.load(open("data/processed/audit.json"))
sb.table("audit_steps").upsert([{"ord": i, **s} for i, s in enumerate(audit["steps"])]).execute()
sb.table("split_summary").upsert(
    [{"split": s, **audit["split"][s]} for s in ("train", "val", "test")]
).execute()
print("uploaded", len(records), "tracks,", len(audit["steps"]), "audit steps")
