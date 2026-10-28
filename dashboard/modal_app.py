"""Serve the Streamlit dashboard on Modal.

One-time:  modal setup
           modal secret create swipetape-supabase SUPABASE_URL=... SUPABASE_ANON_KEY=...
Deploy:    uv run modal deploy dashboard/modal_app.py
Dev:       uv run modal serve dashboard/modal_app.py
"""
import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("streamlit>=1.40", "pandas>=2.2", "altair>=5", "supabase>=2.10", "python-dotenv>=1.0")
    .add_local_file(ROOT / "dashboard" / "app.py", "/root/dashboard/app.py")
    .add_local_file(ROOT / "docs" / "data_dictionary.md", "/root/docs/data_dictionary.md")
)

app = modal.App("swipetape-dashboard", image=image)


@app.function(secrets=[modal.Secret.from_name("swipetape-supabase")], scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=120)
def dashboard():
    cmd = ("streamlit run /root/dashboard/app.py --server.port 8000 --server.address 0.0.0.0 "
           "--server.headless true --server.enableCORS=false --server.enableXsrfProtection=false")
    subprocess.Popen(shlex.split(cmd))
