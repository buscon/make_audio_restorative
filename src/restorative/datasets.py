"""Loading and assembling the ARAUS v1 modelling table."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import targets as T

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "araus_v1"

#: ARAUS fold codes that are not part of the cross-validation set.
#: -1 = practice and attention-check trials, 0 = the separate test fold
#: (5 participants x 48 stimuli, based on 6 recordings held out entirely).
CV_FOLDS = (1, 2, 3, 4, 5)
TEST_FOLD = 0


@dataclass
class ModellingData:
    X: pd.DataFrame
    y: np.ndarray            # target on the [0, 4] modelling range
    y_iso: np.ndarray        # same target on the native ISO [-1, 1] range
    groups: np.ndarray       # participant id - the clustering variable
    frame: pd.DataFrame      # full joined table, for diagnostics

    def __len__(self) -> int:
        return len(self.y)


def load_responses(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / "responses.csv.gz"
    if not path.exists():
        path = data_dir / "responses.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no responses.csv[.gz] in {data_dir} - see data/README.md"
        )
    return pd.read_csv(path, low_memory=False)


def load_participants(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    p = pd.read_csv(Path(data_dir) / "participants.csv")
    # WHO-5 wellbeing, rescaled to [0, 4] as in Versuemer et al. Table III
    out = pd.DataFrame({
        "participant": p["participant"],
        "age": p["age"].astype(float),
        # Table III codes gender as 1 = male, 0 = female or non-binary
        "gender": (p["gender"].astype(str).str.lower().str[0] == "m").astype(float),
        "wellbeing": p["who"].astype(float),
    })
    if out["wellbeing"].max() > 4.0:
        out["wellbeing"] = out["wellbeing"] / out["wellbeing"].max() * 4.0
    return out


def build(
    target: str = "ISOPl",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    include_person: bool = False,
    folds: tuple[int, ...] = CV_FOLDS,
) -> ModellingData:
    """Assemble the modelling table for one target.

    Keeps only the ARAUS cross-validation folds (25,200 ratings from 600
    participants); practice trials, attention checks and the separate test
    fold are dropped.
    """
    if target not in ("ISOPl", "ISOEv"):
        raise ValueError("target must be 'ISOPl' or 'ISOEv'")

    df = load_responses(data_dir)
    df = df[df["fold_r"].isin(folds) & (df["is_attention"] == 0)].copy()

    if include_person:
        df = df.merge(load_participants(data_dir), on="participant", how="left")

    df = T.add_iso_targets(df)
    df = df.dropna(subset=[target, *F.REQUIRED_SOURCE_COLUMNS])

    X = F.build_feature_matrix(df, include_person=include_person)
    y_iso = df[target].to_numpy(dtype=float)

    return ModellingData(
        X=X,
        y=T.to_model_range(y_iso),
        y_iso=y_iso,
        groups=df["participant"].to_numpy(),
        frame=df,
    )
