"""Model-fit measures from Versuemer et al. (2025), Sec. II D 2."""
from __future__ import annotations

import numpy as np

D_ERROR = 0.05


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def objective_function(valid_mse: float, train_mse: float,
                       d_error: float = D_ERROR) -> float:
    """Eq. (4).

        OF = validMSE + 0.5*|dMSE| + 2*max(0, dMSE - dError) + max(0, -dMSE)

    with dMSE = validMSE - trainMSE. Minimising this penalises overfitting
    (dMSE large and positive) and underfitting (dMSE negative) at the same
    time, so hyperparameter selection does not chase validation error alone.
    """
    d_mse = valid_mse - train_mse
    return (
        valid_mse
        + 0.5 * abs(d_mse)
        + 2.0 * max(0.0, d_mse - d_error)
        + max(0.0, -d_mse)
    )


def r2_out_of_sample(y_true, y_pred) -> float:
    """Eq. (5): R^2 = 1 - MSE / VAR(Y), with VAR taken over the test fold.

    Computed per outer fold, not pooled over the whole dataset. Can go
    negative when the fold variance is small relative to the error.
    """
    y_true = np.asarray(y_true, dtype=float)
    var = float(np.var(y_true))
    if var == 0.0:
        return float("nan")
    return 1.0 - mse(y_true, y_pred) / var
