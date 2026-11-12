"""Streamlit Community Cloud entry point — set the app's main file path to
``streamlit_app.py`` (this file, at the repository root).

It lives here rather than in Prototype Codebase/olist/ because Community Cloud
installs the requirements file beside the entry point without quoting its
path, and the space in "Prototype Codebase" breaks that install. The root
requirements.txt beside this file is the one Cloud uses.

This delegates to the full nine-panel decision surface in
Prototype Codebase/olist/app/dashboard.py. dashboard.py derives the project
root from its own __file__, which runpy sets to the path passed in, so data
and model paths resolve exactly as under `streamlit run app/dashboard.py`.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "Prototype Codebase" / "olist" / "app" / "dashboard.py"),
    run_name="__main__",
)
