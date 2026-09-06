"""ISO/TS 12913-3 affective projections.

Pleasantness and Eventfulness are computed from the eight ISO/TS 12913-2
attribute items exactly as in Versuemer et al. (2025), Eqs. (1) and (2):

    Pl = (4 + sqrt(32))^-1 * [ (pleasant - annoying)
                             + cos(45) * (calm - chaotic)
                             + cos(45) * (vibrant - monotonous) ]

    Ev = (4 + sqrt(32))^-1 * [ (eventful - uneventful)
                             + cos(45) * (chaotic - calm)
                             + cos(45) * (vibrant - monotonous) ]

The normaliser only yields the intended [-1, 1] range if the Likert items are
*centred* first (a 5-point item 1..5 becomes -2..+2). ARAUS ships raw 1..5
values, so `center_likert` must be applied before projecting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAQ_ITEMS = (
    "pleasant", "vibrant", "eventful", "chaotic",
    "annoying", "monotonous", "uneventful", "calm",
)

NORMALISER = 4.0 + np.sqrt(32.0)   # 9.65685...
COS45 = float(np.cos(np.pi / 4.0))  # 0.70710...


def center_likert(df: pd.DataFrame, items=PAQ_ITEMS, n_points: int = 5) -> pd.DataFrame:
    """Centre Likert items on zero. A 5-point 1..5 item becomes -2..+2."""
    midpoint = (n_points + 1) / 2.0
    return df.loc[:, list(items)].astype(float) - midpoint


def iso_pleasant(centred: pd.DataFrame) -> pd.Series:
    return (
        (centred["pleasant"] - centred["annoying"])
        + COS45 * (centred["calm"] - centred["chaotic"])
        + COS45 * (centred["vibrant"] - centred["monotonous"])
    ) / NORMALISER


def iso_eventful(centred: pd.DataFrame) -> pd.Series:
    return (
        (centred["eventful"] - centred["uneventful"])
        + COS45 * (centred["chaotic"] - centred["calm"])
        + COS45 * (centred["vibrant"] - centred["monotonous"])
    ) / NORMALISER


def add_iso_targets(df: pd.DataFrame, n_points: int = 5) -> pd.DataFrame:
    """Return `df` with ISOPl / ISOEv columns in [-1, 1]."""
    centred = center_likert(df, n_points=n_points)
    out = df.copy()
    out["ISOPl"] = iso_pleasant(centred)
    out["ISOEv"] = iso_eventful(centred)
    return out


def to_model_range(x) -> np.ndarray:
    """Transpose [-1, 1] to [0, 4].

    Versuemer et al. do not standardise the targets; they train on this
    transposed range. Reproducing their MSE values requires the same scaling.
    """
    return (np.asarray(x, dtype=float) + 1.0) * 2.0


def from_model_range(x) -> np.ndarray:
    """Inverse of `to_model_range`."""
    return np.asarray(x, dtype=float) / 2.0 - 1.0
