"""Estimators and Optuna search spaces (Versuemer et al., Table X)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

MODELS = ("LR", "RF", "XGBoost")

#: n_trials used by the paper for the inner-loop Optuna search
PAPER_TRIALS = {"LR": 0, "RF": 400, "XGBoost": 300}


def make_model(name: str, params: dict | None = None, seed: int = 0, n_jobs: int = -1):
    """Predictors are z-standardised inside the pipeline, so scaling is fitted
    on the training fold only and never leaks across the CV boundary."""
    params = dict(params or {})
    if name == "LR":
        est = LinearRegression()
    elif name == "RF":
        est = RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    elif name == "XGBoost":
        est = XGBRegressor(
            random_state=seed, n_jobs=n_jobs, tree_method="hist",
            objective="reg:squarederror", verbosity=0, **params,
        )
    else:
        raise ValueError(f"unknown model: {name!r}")
    return Pipeline([("scale", StandardScaler()), ("est", est)])


def suggest_params(name: str, trial, n_features: int) -> dict:
    """Search spaces from Table X, clipped to this feature count where needed."""
    if name == "LR":
        return {}
    if name == "RF":
        return {
            "n_estimators":            trial.suggest_int("n_estimators", 2, 7000),
            "max_depth":               trial.suggest_int("max_depth", 5, 100),
            "min_samples_split":       trial.suggest_int("min_samples_split", 2, 1000, log=True),
            "min_samples_leaf":        trial.suggest_int("min_samples_leaf", 2, 5000, log=True),
            "max_samples":             trial.suggest_float("max_samples", 0.0021, 0.9),
            "max_features":            trial.suggest_int("max_features", 1, max(1, n_features)),
            "max_leaf_nodes":          trial.suggest_int("max_leaf_nodes", 10, 40000),
            "min_impurity_decrease":   trial.suggest_float("min_impurity_decrease", 1e-8, 0.02, log=True),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
            "ccp_alpha":               trial.suggest_float("ccp_alpha", 1e-8, 0.01, log=True),
        }
    if name == "XGBoost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 5, 5000),
            "max_depth":        trial.suggest_int("max_depth", 2, 700),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 100),
            "subsample":        trial.suggest_float("subsample", 0.005, 0.97),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0, step=0.1),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0, step=0.1),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 500.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.001, 800.0),
        }
    raise ValueError(f"unknown model: {name!r}")
