"""Data-environment registry.

The project carries two environments with different evidential status, and
confusing them would be a research-integrity failure rather than a bug:

**Empirical Olist** -- the Brazilian E-Commerce Public Dataset. Every measured
result in the paper comes from here. Real observations of a real marketplace.

**Synthetic Nigerian** -- generated data for public demonstration. Every
relationship in it was written by hand in
:mod:`olist.synthetic_ng.generate`. A model trained on it will recover those
relationships, and that recovery is a property of the simulation, not a
finding about Nigerian commerce.

Separation is enforced structurally rather than by convention. Each
environment owns its own raw directory, processed file and model directory, so
there is no code path that can write synthetic output into an Olist location
or load one environment's model against the other's data. The model artefact
records its environment key, and :class:`PredictionService` refuses a
mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .config import DATA_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR

#: Marker written into every model artefact so provenance survives the pickle.
ENVIRONMENT_KEY = "environment_key"


@dataclass(frozen=True)
class Environment:
    """Everything that differs between the two data worlds.

    Anything *not* on this object is shared: the cleaning rules, the feature
    definitions, the leakage controls, the split policy, the model, SHAP and
    the Business Language Layer. That is the point of the abstraction -- the
    analytical machinery is identical, only the data and its labelling differ.
    """

    key: str
    label: str
    kind: str                      # "empirical" | "synthetic"
    raw_dir: Path
    processed_path: Path
    models_dir: Path
    geo_bounds: tuple[float, float, float, float]
    geo_region: str
    currency: str
    currency_symbol: str
    banner: str
    description: str
    disclosure: str | None         # None for empirical environments

    @property
    def is_synthetic(self) -> bool:
        return self.kind == "synthetic"

    @property
    def calendar(self) -> ModuleType:
        """The holiday module for this environment.

        Imported on access rather than held as a field, because a frozen
        dataclass holding a module makes the registry awkward to compare and
        impossible to serialise.
        """
        if self.key == "synthetic_ng":
            from . import calendar_ng

            return calendar_ng
        from . import calendar_br

        return calendar_br

    def model_path(self, scenario: str = "post_dispatch") -> Path:
        return self.models_dir / f"model_{scenario}.joblib"

    def provenance(self) -> str:
        """One line stating what the figures on screen rest on."""
        if self.is_synthetic:
            return (
                f"{self.label} — simulated data. Relationships shown were "
                f"injected by the generator, not discovered."
            )
        return f"{self.label} — real observations from a public dataset."


OLIST = Environment(
    key="olist",
    label="Empirical Olist",
    kind="empirical",
    raw_dir=RAW_DIR,
    processed_path=PROCESSED_DIR / "orders_features.parquet",
    models_dir=MODELS_DIR,
    geo_bounds=(-34.0, 5.3, -74.0, -34.0),
    geo_region="Brazil",
    currency="BRL",
    currency_symbol="R$",
    banner="ACTIVE ENVIRONMENT · EMPIRICAL OLIST",
    description="Real-world public dataset — empirical research evaluation",
    disclosure=None,
)

SYNTHETIC_NG = Environment(
    key="synthetic_ng",
    label="Synthetic Nigerian",
    kind="synthetic",
    raw_dir=DATA_DIR / "synthetic_ng" / "raw",
    processed_path=DATA_DIR / "synthetic_ng" / "processed" / "orders_features.parquet",
    models_dir=MODELS_DIR / "synthetic_ng",
    geo_bounds=(4.0, 14.0, 2.5, 15.0),
    geo_region="Nigeria",
    currency="NGN",
    currency_symbol="₦",
    banner="ACTIVE ENVIRONMENT · SYNTHETIC NIGERIAN — RESEARCH DEMONSTRATION",
    description="Simulated data — research demonstration only",
    disclosure=(
        "Synthetic research demonstration. Orders, locations and financial "
        "values are simulated. Every relationship the model finds here was "
        "written into the generator by hand, so results must not be read as "
        "observed Nigerian business outcomes or as evidence of model "
        "performance in the Nigerian market."
    ),
)

ENVIRONMENTS: dict[str, Environment] = {
    OLIST.key: OLIST,
    SYNTHETIC_NG.key: SYNTHETIC_NG,
}

#: The empirical environment is the default everywhere. A caller that does not
#: name an environment gets the one the paper's results come from.
DEFAULT = OLIST


def get(key: str | None = None) -> Environment:
    """Look up an environment, defaulting to the empirical one."""
    if key is None:
        return DEFAULT
    if key not in ENVIRONMENTS:
        raise KeyError(
            f"Unknown environment {key!r}. Available: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[key]


def check_artefact(artefact: dict, environment: Environment) -> None:
    """Refuse a model trained in a different environment.

    Scoring synthetic Nigerian orders with the Olist model, or the reverse,
    would produce numbers that look entirely plausible and mean nothing. The
    feature names match, so nothing else would catch it.

    Artefacts trained before this key existed carry no marker; those are
    assumed Olist, which is what they are.
    """
    found = artefact.get(ENVIRONMENT_KEY, OLIST.key)
    if found != environment.key:
        raise ValueError(
            f"Model was trained in the {found!r} environment but is being "
            f"loaded for {environment.key!r}. The feature names match, so this "
            f"would score silently and produce meaningless results. Train a "
            f"model for {environment.key!r} instead."
        )
