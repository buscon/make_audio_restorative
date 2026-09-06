"""Psychoacoustic metrics, via MoSQITo.

Standards, matched to Versuemer et al. Table II where possible:

    Sharpness   DIN 45631/A1        mosqito.sharpness_din_tv      matches
    Roughness   ECMA-418-2 (Sottek) mosqito.roughness_ecma        matches
    Tonality    ECMA-418-2 (Sottek) NOT IMPLEMENTED ANYWHERE OPEN

MoSQITo's `tonality` subpackage provides ECMA-74 tone-to-noise ratio and
prominence ratio, which are *different quantities on different scales* (dB,
per-tone) from the ECMA-418-2 tonality in tonality units that ARAUS reports.
TNR is exposed below as its own feature, never as a stand-in for Tonality:
substituting it into the Tonality column would feed the ARAUS-trained model a
predictor it was never fitted on.

All inputs are pressure signals in pascals (see `acoustics.calibrate`).

MoSQITo's ECMA-418-2 roughness allocates every analysis frame at once: a 5 s
clip peaks near 3.3 GB, and a 30 s ARAUS clip would need roughly 20 GB. Long
signals are therefore processed in blocks and averaged over time, which is
what the time-aggregated ARAUS columns represent anyway.
"""
from __future__ import annotations

import warnings

import numpy as np

TONALITY_AVAILABLE = False  # ECMA-418-2 tonality; see module docstring

#: seconds per analysis block, chosen to bound peak memory near 1.5 GB
BLOCK_SECONDS = 2.0


def _blockwise(fn, p: np.ndarray, fs: float, block_seconds: float) -> float:
    """Apply `fn` over consecutive blocks and average, energy-independent.

    A trailing block shorter than half a block is folded into the previous one
    so short remainders cannot skew the mean.
    """
    n = int(round(block_seconds * fs))
    if n <= 0 or len(p) <= n:
        return fn(p, fs)

    bounds = list(range(0, len(p), n))
    if len(p) - bounds[-1] < n // 2 and len(bounds) > 1:
        bounds.pop()

    vals = []
    for i, start in enumerate(bounds):
        stop = bounds[i + 1] if i + 1 < len(bounds) else len(p)
        v = fn(p[start:stop], fs)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def sharpness(p: np.ndarray, fs: float,
              block_seconds: float = BLOCK_SECONDS) -> float:
    """Time-averaged sharpness in acum, DIN 45631/A1."""
    return _blockwise(_sharpness_block, p, fs, block_seconds)


def _sharpness_block(p: np.ndarray, fs: float) -> float:
    from mosqito.sq_metrics import sharpness_din_tv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, _ = sharpness_din_tv(p, fs, weighting="din", field_type="free", skip=0.2)
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    return float(np.mean(s)) if s.size else float("nan")


def roughness(p: np.ndarray, fs: float,
              block_seconds: float = BLOCK_SECONDS) -> float:
    """Time-averaged roughness in asper, ECMA-418-2."""
    return _blockwise(_roughness_block, p, fs, block_seconds)


def _roughness_block(p: np.ndarray, fs: float) -> float:
    from mosqito.sq_metrics import roughness_ecma

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = roughness_ecma(p, fs)
    r = out[0] if isinstance(out, tuple) else out
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.mean(r)) if r.size else float("nan")


def tone_to_noise_ratio(p: np.ndarray, fs: float) -> float:
    """Global tone-to-noise ratio in dB, ECMA-74.

    Reported as a diagnostic and as a candidate predictor in its own right.
    It is NOT the ECMA-418-2 tonality of the ARAUS `Tavg_r` column.
    """
    from mosqito.sq_metrics import tnr_ecma_st

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_global, *_ = tnr_ecma_st(p, fs, prominence=True)
    val = np.asarray(t_global, dtype=float).ravel()
    val = val[np.isfinite(val)]
    return float(np.mean(val)) if val.size else 0.0
