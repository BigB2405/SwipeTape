"""Check that a running API (local uvicorn or the Modal URL) returns the same outputs as the saved pipeline run locally.

    uv run --frozen python scripts/check_deployed.py --url https://<workspace>--swipetape-api-web.modal.run
    uv run --frozen python scripts/check_deployed.py --url http://127.0.0.1:8000
    PIPELINE_PATH=/nonexistent uv run uvicorn serve:app --port 8001      # 503 test, local only
    uv run --frozen python scripts/check_deployed.py --url http://127.0.0.1:8001 --expect-unavailable

Needs the local pipeline.joblib (rebuild with scripts/train_pipeline.py). Exit code 1 if anything differs.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve  # noqa: E402  (same code the API runs; loads the local pipeline.joblib)


def call(base, method, path, body=None, timeout=120, headers=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (r.status, r.headers) if raw else (r.status, json.loads(r.read()))
    except urllib.error.HTTPError as e:
        return (e.code, e.headers) if raw else (e.code, json.loads(e.read() or b"null"))


failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-unavailable", action="store_true", help="server was started without a model: expect 503s")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    base = a.url.rstrip("/")

    st, h = call(base, "GET", "/health")
    check("GET /health -> 200", st == 200, st)
    if a.expect_unavailable:
        check("/health reports artifact_loaded=false", h and h.get("artifact_loaded") is False, h)
        ids = ["A" * 22]
        st, _ = call(base, "POST", "/recommend", {"liked_ids": ids})
        check("POST /recommend -> 503 without model", st == 503, st)
        st, _ = call(base, "GET", "/pipeline/info")
        check("GET /pipeline/info -> 503 without model", st == 503, st)
        st, _ = call(base, "GET", "/search?q=yellow")
        check("GET /search -> 503 without model", st == 503, st)
        return finish()

    check("/health reports artifact_loaded=true", h and h.get("artifact_loaded") is True, h)
    bundle = serve.load_bundle()
    st, info = call(base, "GET", "/pipeline/info")
    check("GET /pipeline/info -> 200", st == 200, st)
    for key in ("sklearn_version", "model", "params", "catalog_size", "git_commit", "built_at", "steps"):
        check(f"/pipeline/info.{key} matches local bundle", info and info.get(key) == bundle["metadata"].get(key),
              f"{info and info.get(key)} vs {bundle['metadata'].get(key)}")

    ids = sorted(bundle["pipeline"].named_steps["rec"].meta_.track_id)
    step = max(len(ids) // 40, 1)
    pick = ids[::step]
    cases = {
        "1 like": dict(liked_ids=pick[0:1], disliked_ids=[], k=10),
        "5 likes + 3 dislikes": dict(liked_ids=pick[1:6], disliked_ids=pick[6:9], k=10),
        "mixed taste, k=20": dict(liked_ids=pick[10:13] + pick[20:23], disliked_ids=pick[30:35], k=20),
    }
    for name, body in cases.items():
        st, out = call(base, "POST", "/recommend", body)
        local = serve.recommend_items(bundle, body["liked_ids"], body["disliked_ids"], body["k"])
        ok = st == 200 and [r["track_id"] for r in out["recommendations"]] == [r["track_id"] for r in local]
        check(f"recommend [{name}]: same tracks, same order as local", ok, st)
        if ok:
            worst = max(abs(r["score"] - l["score"]) for r, l in zip(out["recommendations"], local))
            check(f"recommend [{name}]: scores within {a.tol}", worst <= a.tol, f"max diff {worst:.2e}")

    bad = {
        "empty liked_ids": dict(liked_ids=[]),
        "missing liked_ids": dict(k=5),
        "malformed id": dict(liked_ids=["not-a-track-id"]),
        "unknown id": dict(liked_ids=["Z" * 22]),
        "k out of range": dict(liked_ids=pick[:1], k=0),
        "liked and disliked overlap": dict(liked_ids=pick[:1], disliked_ids=pick[:1]),
        "extra field": dict(liked_ids=pick[:1], bogus=1),
    }
    for name, body in bad.items():
        st, _ = call(base, "POST", "/recommend", body)
        check(f"invalid input [{name}] -> 422", st == 422, st)

    # D5: search and browser access from the Vercel frontend
    for q in ("yellow", "bad bunny"):
        st, out = call(base, "GET", f"/search?q={q.replace(' ', '%20')}&limit=10")
        local = serve.search_items(bundle, q, 10)
        check(f"search [{q}]: same tracks, same order as local", st == 200 and out["tracks"] == local, st)
    for q in ("a", "%20%20"):
        st, _ = call(base, "GET", f"/search?q={q}")
        check(f"search [{q!r}] too short -> 422", st == 422, st)
    pre = {"Origin": "https://swipetape.vercel.app", "Access-Control-Request-Method": "POST",
           "Access-Control-Request-Headers": "content-type"}
    st, hdr = call(base, "OPTIONS", "/recommend", headers=pre, raw=True)
    check("CORS preflight from a vercel.app origin is allowed",
          st == 200 and hdr.get("access-control-allow-origin") == pre["Origin"], st)
    return finish()


def finish():
    print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
