"""Frequency weighting and level statistics (IEC 61672-1).

Everything here works in pascals. Converting a WAV's arbitrary full-scale
units into pascals needs a calibration constant; see `calibrate`. Getting that
wrong shifts every level by a fixed offset and makes L_Aeq meaningless, so it
is a required argument rather than a default.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig

P_REF = 20e-6  # reference sound pressure, Pa

# IEC 61672-1 pole frequencies
_F1, _F2, _F3, _F4 = 20.598997, 107.65265, 737.86223, 12194.217


def _weighting_zpk(kind: str):
    """Analogue zero-pole-gain of the A or C weighting network."""
    w1, w2, w3, w4 = (2 * np.pi * f for f in (_F1, _F2, _F3, _F4))
    if kind == "A":
        zeros = [0.0, 0.0, 0.0, 0.0]
        poles = [-w1, -w1, -w2, -w3, -w4, -w4]
    elif kind == "C":
        zeros = [0.0, 0.0]
        poles = [-w1, -w1, -w4, -w4]
    else:
        raise ValueError("kind must be 'A' or 'C'")
    return np.array(zeros), np.array(poles), 1.0


def weighting_sos(kind: str, fs: float) -> np.ndarray:
    """Digital weighting filter, normalised to exactly 0 dB at 1 kHz."""
    z, p, k = _weighting_zpk(kind)
    zd, pd, kd = _sig.bilinear_zpk(z, p, k, fs)
    sos = _sig.zpk2sos(zd, pd, kd)
    # normalise the 1 kHz response to unity
    _, h = _sig.sosfreqz(sos, worN=[2 * np.pi * 1000.0 / fs])
    sos[0, :3] /= np.abs(h[0])
    return sos


def apply_weighting(x: np.ndarray, fs: float, kind: str) -> np.ndarray:
    return _sig.sosfilt(weighting_sos(kind, fs), x)


def calibrate(x: np.ndarray, cal_db: float) -> np.ndarray:
    """Full-scale samples to pascals.

    `cal_db` is the offset from dBFS to dB SPL: a signal whose full-scale RMS
    is `r` has level `20*log10(r) + cal_db`. For an uncalibrated recording,
    `calibration_for_target_laeq` can solve for it from a known L_Aeq.
    """
    return np.asarray(x, dtype=float) * (10.0 ** (cal_db / 20.0)) * P_REF


def leq(p: np.ndarray) -> float:
    """Equivalent continuous level of a pressure signal, dB re 20 uPa."""
    p = np.asarray(p, dtype=float)
    ms = float(np.mean(p ** 2))
    if ms <= 0.0:
        return -np.inf
    return 10.0 * np.log10(ms / P_REF ** 2)


def time_weighted_levels(p: np.ndarray, fs: float, tau: float = 0.125,
                         step: float = 0.01) -> np.ndarray:
    """Exponentially time-weighted levels, sampled every `step` seconds.

    `tau` = 0.125 s is IEC 'Fast'. The one-pole smoother is applied to the
    squared pressure, which is what an SLM integrates.
    """
    p = np.asarray(p, dtype=float)
    alpha = float(np.exp(-1.0 / (tau * fs)))
    ms = _sig.lfilter([1.0 - alpha], [1.0, -alpha], p ** 2)
    hop = max(1, int(round(step * fs)))
    ms = ms[::hop]
    ms = np.maximum(ms, 1e-20)
    return 10.0 * np.log10(ms / P_REF ** 2)


def exceedance_level(levels: np.ndarray, n: float) -> float:
    """L_N: the level exceeded for N percent of the time.

    L_10 is therefore the 90th percentile of the level distribution, not the
    10th. ARAUS follows the same convention (its LA10 > LA90 throughout).
    """
    return float(np.percentile(np.asarray(levels, dtype=float), 100.0 - n))


def calibration_for_target_laeq(x: np.ndarray, fs: float, target_laeq: float) -> float:
    """Solve `cal_db` so the signal's L_Aeq equals `target_laeq`.

    For recordings with no calibration tone. The weighting filter is linear, so
    the offset follows in closed form.
    """
    a = apply_weighting(np.asarray(x, dtype=float), fs, "A")
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms <= 0.0:
        raise ValueError("signal is silent - cannot calibrate")
    return target_laeq - 20.0 * np.log10(rms)
