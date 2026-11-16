"""Deploy serve.py on Modal (HW4 `modal_serve.py`).

    uv run --frozen python scripts/train_pipeline.py     # builds pipeline.joblib
    uv run modal deploy modal_serve.py                    # prints the public URL; API docs at <url>/docs

The image gets exactly three files (serve.py, pipeline_def.py, pipeline.joblib) and a scikit-learn
pinned to the version recorded in the bundle metadata. `serve` is imported inside the Modal function.
"""
import sys
from pathlib import Path

import joblib
import modal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                      # so pipeline.joblib can be unpickled locally to read its metadata
META = joblib.load(ROOT / "pipeline.joblib")["metadata"]

pins = [f"scikit-learn=={META['sklearn_version']}", f"joblib=={META['joblib_version']}",
        f"pandas=={META['pandas_version']}", f"numpy=={META['numpy_version']}"]

image = (
    modal.Image.debian_slim(python_version=".".join(META["python_version"].split(".")[:2]))
    .pip_install(*pins, "fastapi>=0.115", "pydantic>=2.7")
    .add_local_file(str(ROOT / "serve.py"), remote_path="/root/serve.py")
    .add_local_file(str(ROOT / "pipeline_def.py"), remote_path="/root/pipeline_def.py")
    .add_local_file(str(ROOT / "pipeline.joblib"), remote_path="/root/pipeline.joblib")
)

app = modal.App("swipetape-api", image=image)


@app.function(memory=2048)                         # forest + catalog embeddings need ~1 GB of RAM
@modal.asgi_app()
def web():
    from serve import app as fastapi_app           # imported inside the function: the bundle loads once per container

    return fastapi_app
