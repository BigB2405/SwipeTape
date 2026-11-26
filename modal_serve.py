"""Deploy serve.py on Modal (HW4 `modal_serve.py`).

    uv run --frozen python scripts/train_pipeline.py     # builds pipeline.joblib
    uv run modal deploy modal_serve.py                    # prints the public URL; API docs at <url>/docs

v2 (SoundSwipe live site since Nov 2026) is a SEPARATE Modal app, so the D4 app and URL never change:

    uv run --frozen python scripts/train_pipeline.py --v2                 # builds pipeline_v2.joblib
    SWIPETAPE_V2=1 uv run --frozen modal deploy modal_serve.py            # app swipetape-api-v2

The image gets exactly three files (serve.py, pipeline_def.py, pipeline.joblib) and a scikit-learn
pinned to the version recorded in the bundle metadata. `serve` is imported inside the Modal function.
"""
import os
import sys
from pathlib import Path

import joblib
import modal

ROOT = Path(__file__).resolve().parent
V2 = os.environ.get("SWIPETAPE_V2") == "1"                 # set when deploying; the image passes it to the container
BUNDLE = ROOT / ("pipeline_v2.joblib" if V2 else "pipeline.joblib")
APP_NAME = "swipetape-api-v2" if V2 else "swipetape-api"   # default = the D4 app, unchanged

if modal.is_local():
    # Read the version pins from the bundle on the deploying machine only. Modal re-imports this file
    # inside the container, and loading the 100 MB pickle there (before pyarrow exists) crashed startup.
    import pyarrow

    sys.path.insert(0, str(ROOT))                  # so pipeline.joblib can be unpickled locally to read its metadata
    META = joblib.load(BUNDLE)["metadata"]
    PY = ".".join(META["python_version"].split(".")[:2])
    pins = [f"scikit-learn=={META['sklearn_version']}", f"joblib=={META['joblib_version']}",
            f"pandas=={META['pandas_version']}", f"numpy=={META['numpy_version']}",
            f"pyarrow=={pyarrow.__version__}"]     # pandas' string columns in the catalog need pyarrow to unpickle
else:
    PY, pins = None, []                            # inside the container the image is already built
    BUNDLE = ROOT / "pipeline.joblib"              # unused in the container: the image is already built

image = (
    modal.Image.debian_slim(python_version=PY)
    .pip_install(*pins, "fastapi>=0.115", "pydantic>=2.7")
    .env({"SWIPETAPE_V2": "1" if V2 else "0"})     # so the container picks the same app name
    .add_local_file(str(ROOT / "serve.py"), remote_path="/root/serve.py")
    .add_local_file(str(ROOT / "pipeline_def.py"), remote_path="/root/pipeline_def.py")
    .add_local_file(str(BUNDLE), remote_path="/root/pipeline.joblib")    # v2 bundle is copied under the D4 name
)

app = modal.App(APP_NAME, image=image)


@app.function(memory=2048)                         # forest + catalog embeddings need ~1 GB of RAM
@modal.asgi_app()
def web():
    from serve import app as fastapi_app           # imported inside the function: the bundle loads once per container

    return fastapi_app
