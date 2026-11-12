"""Pre-flight check for a Streamlit Community Cloud deploy.

Run from the project root:   python scripts/check_deploy.py

Answers the three questions that decide whether the deploy will work:
  1. Are the artefacts the app needs actually present, and will git commit
     them, or is .gitignore going to leave them behind?
  2. Which scikit-learn version trained each artefact, so requirements.txt
     can be pinned to it?
  3. Does the service actually load and score, with only the serving
     dependencies installed?

It writes nothing and changes nothing.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "
problems: list[str] = []


def say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def git_will_commit(path: Path) -> bool | None:
    """True if git would commit this file, False if ignored, None if no git.

    ``git check-ignore`` exits 0 when the path is ignored. Files already
    tracked are never reported as ignored, which is the right answer here.
    """
    try:
        r = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=ROOT, capture_output=True,
        )
    except FileNotFoundError:
        return None
    return r.returncode != 0


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def check_environment(env) -> None:
    print(f"\n--- {env.label} ({env.key}) ---")
    model = env.model_path()
    history = env.processed_path
    background = history.parent / "background.parquet"

    for label, path in (("model", model), ("feature table", history),
                        ("SHAP background", background)):
        if not path.exists():
            say(BAD, f"{label} missing: {path.relative_to(ROOT)}")
            problems.append(f"{env.key}: {label} not built")
            continue

        size = mb(path)
        note = f"{label}: {path.relative_to(ROOT)} ({size:.1f} MB)"
        if git_will_commit(path) is False:
            say(BAD, note + "  — .gitignore EXCLUDES this; the deploy will not see it")
            problems.append(f"{env.key}: {label} is gitignored")
        elif size > 100:
            say(BAD, note + "  — over GitHub's 100 MB file limit")
            problems.append(f"{env.key}: {label} too large for GitHub")
        elif size > 50:
            say(WARN, note + "  — GitHub warns above 50 MB; consider Git LFS")
        else:
            say(OK, note)

    if not model.exists():
        return

    import joblib

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artefact = joblib.load(model)
        versions = {
            str(w.message).split("from version ")[1].split(" ")[0]
            for w in caught if "from version" in str(w.message)
        }

    import sklearn

    trained_with = sorted(versions)[0] if versions else sklearn.__version__
    say(OK, f"artefact scenario={artefact.get('scenario')} "
            f"threshold={artefact.get('threshold')} "
            f"env={artefact.get('environment_key', 'olist (unmarked)')}")
    print(f"       pin in requirements.txt:  scikit-learn=={trained_with}")
    if versions:
        say(WARN, f"local scikit-learn is {sklearn.__version__}, artefact was "
                  f"trained on {trained_with}")

    if not (history.exists() and background.exists()):
        return

    from olist.bll import PredictionService
    from olist.bll.brief import BusinessCostModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        service = PredictionService.from_files(
            model, history, background, environment=env,
            costs=BusinessCostModel(
                currency=env.currency, refund_share=0.42, churn_share=0.46,
                support_cost=1200, intervention_cost=400),
        )
        risk = service.predict_probability(service.history)

    say(OK, f"loads and scores: {len(service.history):,} orders, "
            f"risk {risk.min():.3f}–{risk.max():.3f}")


def main() -> int:
    from olist import environments

    print(f"Project root: {ROOT}")
    print(f"Python {sys.version.split()[0]}")

    for env in environments.ENVIRONMENTS.values():
        try:
            check_environment(env)
        except Exception as exc:                      # noqa: BLE001
            say(BAD, f"{env.key}: {type(exc).__name__}: {exc}")
            problems.append(f"{env.key}: {exc}")

    print("\n" + "=" * 68)
    if problems:
        print("Not ready to deploy:")
        for p in problems:
            print(f"  - {p}")
        print("\nAn environment that fails here simply will not appear in the "
              "app's sidebar. That is fine if you meant to deploy only one.")
        return 1
    print("Ready to deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
