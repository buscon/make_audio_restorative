"""Waveform to the predictor's feature vector.

ARAUS reports every psychoacoustic parameter for the right ear of its binaural
recordings (the `_r` columns), so binaural input defaults to the right channel
- the features must be produced the same way they were during training.

Two of the seven indicators of Versuemer et al. cannot be computed here:
Relative Approach (no open implementation, not in ARAUS either) and ECMA-418-2
Tonality (no open implementation; ARAUS ships precomputed values). The
remaining five are what `AUDIO_FEATURES` covers, and the ablation in
`scripts/feature_ablation.py` measures what their absence costs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import acoustics as ac
from . import psychoacoustics as pa

#: features computable from audio, named exactly as in features.ACOUSTIC_FEATURES
AUDIO_FEATURES = ("LAeq", "LA10_LA90", "LCeq_LAeq", "Sharpness", "Roughness")


@dataclass
class AudioFeatures:
    LAeq: float
    LA10_LA90: float
    LCeq_LAeq: float
    Sharpness: float
    Roughness: float
    TNR: float          # diagnostic only, not an ARAUS column
    cal_db: float
    duration_s: float

    def to_row(self) -> dict:
        return {k: getattr(self, k) for k in AUDIO_FEATURES}

    def as_dict(self) -> dict:
        return asdict(self)


def load_wav(path: str | Path, channel: str = "right"):
    """Read a WAV as (samples, fs). `channel` is 'right', 'left' or 'mono'."""
    import soundfile as sf

    x, fs = sf.read(str(path), always_2d=True)
    if x.shape[1] == 1:
        return x[:, 0], fs
    if channel == "right":
        return x[:, 1], fs
    if channel == "left":
        return x[:, 0], fs
    if channel == "mono":
        return x.mean(axis=1), fs
    raise ValueError("channel must be 'right', 'left' or 'mono'")


def extract(
    x: np.ndarray,
    fs: float,
    cal_db: float | None = None,
    target_laeq: float | None = None,
) -> AudioFeatures:
    """Compute the feature vector from a waveform.

    Exactly one of `cal_db` (dBFS to dB SPL offset) or `target_laeq` (solve the
    offset so the clip has this L_Aeq) must be given. There is no default:
    an uncalibrated level would silently corrupt L_Aeq, which the ablation
    shows is the most valuable single predictor.
    """
    x = np.asarray(x, dtype=float)
    if (cal_db is None) == (target_laeq is None):
        raise ValueError("give exactly one of cal_db or target_laeq")
    if target_laeq is not None:
        cal_db = ac.calibration_for_target_laeq(x, fs, target_laeq)

    p = ac.calibrate(x, cal_db)
    p_a = ac.apply_weighting(p, fs, "A")
    p_c = ac.apply_weighting(p, fs, "C")

    levels_a = ac.time_weighted_levels(p_a, fs)
    laeq = ac.leq(p_a)

    return AudioFeatures(
        LAeq=laeq,
        LA10_LA90=ac.exceedance_level(levels_a, 10) - ac.exceedance_level(levels_a, 90),
        LCeq_LAeq=ac.leq(p_c) - laeq,
        Sharpness=pa.sharpness(p, fs),
        Roughness=pa.roughness(p, fs),
        TNR=pa.tone_to_noise_ratio(p, fs),
        cal_db=float(cal_db),
        duration_s=len(x) / fs,
    )


def extract_file(path, channel: str = "right", **kw) -> AudioFeatures:
    x, fs = load_wav(path, channel=channel)
    return extract(x, fs, **kw)
