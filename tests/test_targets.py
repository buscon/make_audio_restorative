import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative import targets as T


def _row(**kw):
    base = {k: 3 for k in T.PAQ_ITEMS}   # neutral on a 1..5 scale
    base.update(kw)
    return pd.DataFrame([base])


def test_neutral_ratings_project_to_zero():
    c = T.center_likert(_row())
    assert T.iso_pleasant(c).iloc[0] == pytest.approx(0.0)
    assert T.iso_eventful(c).iloc[0] == pytest.approx(0.0)


def test_maximally_pleasant_reaches_plus_one():
    """All three pleasantness-loading contrasts at their extreme must hit +1;
    this is what the (4 + sqrt(32)) normaliser is for."""
    c = T.center_likert(_row(pleasant=5, annoying=1, calm=5, chaotic=1,
                             vibrant=5, monotonous=1))
    assert T.iso_pleasant(c).iloc[0] == pytest.approx(1.0)


def test_maximally_annoying_reaches_minus_one():
    c = T.center_likert(_row(pleasant=1, annoying=5, calm=1, chaotic=5,
                             vibrant=1, monotonous=5))
    assert T.iso_pleasant(c).iloc[0] == pytest.approx(-1.0)


def test_model_range_roundtrip():
    x = np.array([-1.0, -0.5, 0.0, 0.75, 1.0])
    assert T.to_model_range(x).min() == 0.0
    assert T.to_model_range(x).max() == 4.0
    np.testing.assert_allclose(T.from_model_range(T.to_model_range(x)), x)
