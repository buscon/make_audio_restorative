import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative import acoustics as ac

FS = 48000


def sine(f, seconds=2.0, amp=1.0, fs=FS):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * f * t)


def weighting_gain_db(f, kind):
    """Gain the weighting network applies at frequency f."""
    x = sine(f, 4.0)
    y = ac.apply_weighting(x, FS, kind)
    # discard filter start-up before measuring
    n = FS // 2
    return 20 * np.log10(np.std(y[n:]) / np.std(x[n:]))


# IEC 61672-1 Table 2, tolerance class 1 is +/- 1 dB in this range
@pytest.mark.parametrize("freq,expected", [
    (31.5, -39.4), (125, -16.1), (1000, 0.0), (4000, 1.0), (8000, -1.1),
])
def test_a_weighting_matches_iec61672(freq, expected):
    assert weighting_gain_db(freq, "A") == pytest.approx(expected, abs=0.7)


@pytest.mark.parametrize("freq,expected", [
    (31.5, -3.0), (125, -0.2), (1000, 0.0), (4000, -0.8), (8000, -3.0),
])
def test_c_weighting_matches_iec61672(freq, expected):
    assert weighting_gain_db(freq, "C") == pytest.approx(expected, abs=0.7)


def test_weightings_are_unity_at_1khz():
    assert weighting_gain_db(1000, "A") == pytest.approx(0.0, abs=0.05)
    assert weighting_gain_db(1000, "C") == pytest.approx(0.0, abs=0.05)


def test_calibration_reproduces_target_level():
    x = sine(1000, 3.0, amp=0.1)
    cal = ac.calibration_for_target_laeq(x, FS, 65.0)
    p = ac.calibrate(x, cal)
    assert ac.leq(ac.apply_weighting(p, FS, "A")) == pytest.approx(65.0, abs=0.1)


def test_leq_of_known_pressure():
    # 1 Pa RMS is 20*log10(1/20e-6) = 93.979 dB SPL exactly.
    # "94 dB" is the rounded convention, not the definition.
    p = np.full(FS, 1.0)
    assert ac.leq(p) == pytest.approx(93.9794, abs=0.001)


def test_exceedance_convention_is_l10_above_l90():
    """L_10 is the level exceeded 10% of the time, so it must exceed L_90.
    ARAUS uses the same convention; reversing it would flip the sign of the
    LA10-LA90 feature."""
    rng = np.random.default_rng(0)
    levels = rng.normal(60, 5, 10000)
    assert ac.exceedance_level(levels, 10) > ac.exceedance_level(levels, 90)


def test_steady_noise_has_low_variability_modulated_has_high():
    rng = np.random.default_rng(1)
    steady = rng.normal(0, 0.05, 4 * FS)
    t = np.arange(4 * FS) / FS
    modulated = steady * (1 + 0.9 * np.sin(2 * np.pi * 0.5 * t))

    def variability(x):
        p = ac.calibrate(x, 94.0)
        lv = ac.time_weighted_levels(ac.apply_weighting(p, FS, "A"), FS)
        return ac.exceedance_level(lv, 10) - ac.exceedance_level(lv, 90)

    assert variability(steady) < 2.0
    assert variability(modulated) > 5.0
