"""Validation against the reference signals that *define* the units.

ARAUS does not distribute its audio, so the extractor cannot be checked
against the precomputed columns it will be paired with. The next best
anchor is the standards' own reference signals, where the correct answer is
1.0 by definition.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative import acoustics as ac
from restorative import psychoacoustics as pa
from restorative.extract import AUDIO_FEATURES, extract

FS = 48000


def test_roughness_reference_signal_is_one_asper():
    """1 asper is defined as a 1 kHz tone at 60 dB, 100% amplitude modulated
    at 70 Hz."""
    t = np.arange(4 * FS) / FS
    am = (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    p = ac.calibrate(am / np.sqrt(np.mean(am ** 2)), 60.0)
    assert pa.roughness(p, FS) == pytest.approx(1.0, abs=0.15)


def test_unmodulated_tone_has_no_roughness():
    t = np.arange(4 * FS) / FS
    tone = np.sin(2 * np.pi * 1000 * t)
    p = ac.calibrate(tone / np.sqrt(np.mean(tone ** 2)), 60.0)
    assert pa.roughness(p, FS) < 0.05


def test_sharpness_reference_signal_is_one_acum():
    """1 acum is defined as narrowband noise centred at 1 kHz with a 160 Hz
    bandwidth at 60 dB (DIN 45692)."""
    n = 4 * FS
    rng = np.random.default_rng(0)
    spec = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1 / FS)
    spec[(freqs < 920) | (freqs > 1080)] = 0
    nb = np.fft.irfft(spec, n)
    p = ac.calibrate(nb / np.sqrt(np.mean(nb ** 2)), 60.0)
    assert pa.sharpness(p, FS) == pytest.approx(1.0, abs=0.15)


def test_tonality_is_not_silently_substituted():
    """ECMA-418-2 tonality has no open implementation. Nothing may quietly
    stand in for it, or the ARAUS-trained model gets a predictor it was
    never fitted on."""
    assert pa.TONALITY_AVAILABLE is False
    assert "Tonality" not in AUDIO_FEATURES


def test_extractor_requires_explicit_calibration():
    x = np.random.default_rng(0).normal(0, 0.1, FS)
    with pytest.raises(ValueError):
        extract(x, FS)                                  # neither given
    with pytest.raises(ValueError):
        extract(x, FS, cal_db=94.0, target_laeq=65.0)   # both given


def test_extractor_hits_the_requested_level():
    x = np.random.default_rng(0).normal(0, 0.1, 3 * FS)
    assert extract(x, FS, target_laeq=63.0).LAeq == pytest.approx(63.0, abs=0.2)


def test_blockwise_averaging_matches_whole_signal():
    """Blocking exists to bound memory; it must not change the answer."""
    t = np.arange(4 * FS) / FS
    am = (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    p = ac.calibrate(am / np.sqrt(np.mean(am ** 2)), 60.0)
    whole = pa.roughness(p, FS, block_seconds=10.0)   # longer than the signal
    blocked = pa.roughness(p, FS, block_seconds=2.0)
    assert blocked == pytest.approx(whole, rel=0.10)
