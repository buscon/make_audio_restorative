import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative.objective import objective_function, r2_out_of_sample


def test_perfect_fit_returns_validation_mse():
    # train == valid: dMSE = 0, so no penalty term fires
    assert objective_function(0.5, 0.5) == pytest.approx(0.5)


def test_overfitting_is_penalised_more_than_underfitting():
    over = objective_function(valid_mse=0.6, train_mse=0.1)   # dMSE = +0.5
    under = objective_function(valid_mse=0.6, train_mse=1.1)  # dMSE = -0.5
    assert over > under > 0.6


def test_small_gap_within_derror_avoids_the_hard_penalty():
    inside = objective_function(0.55, 0.51)   # dMSE = 0.04 < dError = 0.05
    outside = objective_function(0.55, 0.49)  # dMSE = 0.06 > dError
    assert outside - inside > 0.02


def test_r2_matches_definition():
    y = [0.0, 1.0, 2.0, 3.0]
    assert r2_out_of_sample(y, y) == pytest.approx(1.0)
    # predicting the mean everywhere gives R^2 = 0
    assert r2_out_of_sample(y, [1.5] * 4) == pytest.approx(0.0)
