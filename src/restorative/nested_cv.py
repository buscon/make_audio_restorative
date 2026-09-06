"""Nested cross-validation driver (Versuemer et al., Fig. 1).

Outer loop (K=5) estimates generalisation; inner loop (K=5) selects
hyperparameters by minimising the objective function of Eq. (4), which
penalises overfitting and underfitting rather than validation error alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .models import make_model, suggest_params
from .objective import mse, objective_function, r2_out_of_sample
from .splitting import leakage_report, make_splitter

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_mse: float
    test_mse: float
    test_r2: float
    n_test: int
    leaked_groups: int
    best_params: dict = field(default_factory=dict)


@dataclass
class CVResult:
    model: str
    target: str
    outer_rule: str
    inner_rule: str
    folds: list[FoldResult]

    @property
    def mse(self) -> float:
        return float(np.mean([f.test_mse for f in self.folds]))

    @property
    def r2(self) -> float:
        return float(np.mean([f.test_r2 for f in self.folds]))

    @property
    def r2_se(self) -> float:
        r2s = [f.test_r2 for f in self.folds]
        return float(np.std(r2s, ddof=1) / np.sqrt(len(r2s)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.folds])

    def summary(self) -> str:
        leak = sum(f.leaked_groups for f in self.folds)
        return (
            f"{self.model:>8s}  {self.target}  {self.outer_rule}-{self.inner_rule}  "
            f"MSE={self.mse:.4f}  R2={self.r2:.4f} (SE {self.r2_se:.4f})  "
            f"leaked_groups={leak}"
        )


def _tune(model_name, X, y, groups, inner_rule, n_trials, seed, n_jobs):
    """Inner loop: Optuna over the OF. Returns the best hyperparameters."""
    if n_trials <= 0:
        return {}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    splitter = make_splitter(inner_rule, n_splits=5, seed=seed)
    splits = list(splitter.split(X, y, groups))

    def objective(trial):
        params = suggest_params(model_name, trial, n_features=X.shape[1])
        ofs = []
        for tr, va in splits:
            model = make_model(model_name, params, seed=seed, n_jobs=n_jobs)
            model.fit(X.iloc[tr], y[tr])
            ofs.append(objective_function(
                valid_mse=mse(y[va], model.predict(X.iloc[va])),
                train_mse=mse(y[tr], model.predict(X.iloc[tr])),
            ))
        return float(np.mean(ofs))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_nested_cv(
    data,
    model_name: str,
    target: str = "ISOPl",
    outer_rule: str = "sgkf",
    inner_rule: str = "gkf",
    n_trials: int = 0,
    seed: int = 0,
    n_jobs: int = -1,
) -> CVResult:
    X, y, groups = data.X, data.y, data.groups
    outer = make_splitter(outer_rule, n_splits=5, seed=seed)

    folds: list[FoldResult] = []
    for k, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        best = _tune(model_name, X.iloc[tr], y[tr], groups[tr],
                     inner_rule, n_trials, seed, n_jobs)
        model = make_model(model_name, best, seed=seed, n_jobs=n_jobs)
        model.fit(X.iloc[tr], y[tr])

        folds.append(FoldResult(
            fold=k,
            train_mse=mse(y[tr], model.predict(X.iloc[tr])),
            test_mse=mse(y[te], model.predict(X.iloc[te])),
            test_r2=r2_out_of_sample(y[te], model.predict(X.iloc[te])),
            n_test=len(te),
            leaked_groups=leakage_report(groups, tr, te),
            best_params=best,
        ))
        log.info("fold %d/5  test MSE %.4f  R2 %.4f", k, folds[-1].test_mse, folds[-1].test_r2)

    return CVResult(model_name, target, outer_rule, inner_rule, folds)
