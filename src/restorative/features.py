"""The acoustic / psychoacoustic predictor set of Versuemer et al. (2025), Table II.

Mapped onto ARAUS v1 `responses.csv` column names. ARAUS reports every metric
for the right ear channel (`_r` suffix) as time-aggregated statistics.

    Versuemer indicator      meaning                     ARAUS column(s)
    ---------------------    -------------------------   -------------------
    L_Aeq                    loudness                    LAavg_r
    L_A10 - L_A90            time variability            LA10_r - LA90_r
    Relative Approach (RA)   saliency                    NOT IN ARAUS
    L_Ceq - L_Aeq            low-frequency content       LCavg_r - LAavg_r
    Sharpness (acum)         high-frequency content      Savg_r
    Tonality (tu)            tonal components            Tavg_r
    Roughness (asper)        envelope modulation         Ravg_r

Relative Approach is unavailable: Versuemer et al. computed it with ArtemiS
SUITE from the original calibrated stimuli, which ARAUS does not distribute.
We therefore model with six of the seven indicators and expect a small
shortfall against the published R^2 values. `MISSING_INDICATORS` records this
so it stays visible in every report rather than being silently absorbed.
"""
from __future__ import annotations

import pandas as pd

MISSING_INDICATORS = ("RA",)

#: acoustic / psychoacoustic predictors, as {feature name: builder}
ACOUSTIC_FEATURES = {
    "LAeq":        lambda d: d["LAavg_r"],
    "LA10_LA90":   lambda d: d["LA10_r"] - d["LA90_r"],
    "LCeq_LAeq":   lambda d: d["LCavg_r"] - d["LAavg_r"],
    "Sharpness":   lambda d: d["Savg_r"],
    "Tonality":    lambda d: d["Tavg_r"],
    "Roughness":   lambda d: d["Ravg_r"],
}

#: person-related predictors, Table III (joined from participants.csv)
PERSON_FEATURES = ("age", "gender", "wellbeing")

REQUIRED_SOURCE_COLUMNS = (
    "LAavg_r", "LA10_r", "LA90_r", "LCavg_r", "Savg_r", "Tavg_r", "Ravg_r",
)


def build_acoustic_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"responses table is missing ARAUS columns: {missing}")
    return pd.DataFrame(
        {name: fn(df) for name, fn in ACOUSTIC_FEATURES.items()},
        index=df.index,
    )


def build_feature_matrix(df: pd.DataFrame, include_person: bool = False) -> pd.DataFrame:
    X = build_acoustic_features(df)
    if include_person:
        for col in PERSON_FEATURES:
            if col not in df.columns:
                raise KeyError(
                    f"person feature '{col}' not present - join participants.csv first"
                )
            X[col] = df[col].astype(float)
    return X
