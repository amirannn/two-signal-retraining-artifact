"""Classifier builders used in the reference protocol."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(name: str = "random", params: dict | None = None, seed: int = 42):
    """Build a classifier wrapped in a StandardScaler pipeline.

    Supported names: "random"/"random_forest"/"rf", "mlp", "svm"/"svc",
    "xgboost"/"xgb" (requires the optional ``xgboost`` package).
    """
    params = dict(params or {})
    if name in ("random", "random_forest", "rf"):
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 10),
            random_state=seed,
        )
    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (50,)),
            max_iter=params.get("max_iter", 300),
            random_state=seed,
        )
    elif name in ("svm", "svc"):
        clf = SVC(kernel="linear", C=params.get("C", 1.0), random_state=seed)
    elif name in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "xgboost is not installed; run `pip install xgboost` to use it"
            ) from exc
        clf = XGBClassifier(
            n_estimators=params.get("n_estimators", 50),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=seed,
            eval_metric="logloss",
        )
    else:
        raise ValueError(
            f"unknown model {name!r}; use 'random', 'mlp', 'svm' or 'xgboost'"
        )
    return make_pipeline(StandardScaler(), clf)