"""Streamlit Community Cloud entry point.

Set the app's main file path to  Prototype Codebase/olist/streamlit_app.py
Community Cloud then installs requirements.txt from this same directory.

This delegates to the full nine-panel decision surface in app/dashboard.py —
the same screen the figures in the paper were captured from.

dashboard.py derives the project root from its own __file__, and runpy sets
__file__ to the path passed in, so the data and model paths resolve exactly as
they do under `streamlit run app/dashboard.py`.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "app" / "dashboard.py"),
    run_name="__main__",
)
