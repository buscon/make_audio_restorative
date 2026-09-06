#!/usr/bin/env python3
"""Fit and persist the control predictor used by the restoration loop.

Trained on the five indicators that `restorative.extract` computes from a
waveform, so the model can score audio it has never seen. The two-predictor
rule applies: this is the model that *drives* the masker-to-base ratio search.
The model that reports the improvement must be trained separately, on
different data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative.datasets import build
from restorative.extract import AUDIO_FEATURES
from restorative.models import make_model
from restorative.nested_cv import run_nested_cv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="ISOPl", choices=["ISOPl", "ISOEv"])
    ap.add_argument("--model", default="RF", choices=["LR", "RF", "XGBoost"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = build(target=args.target)
    missing = [f for f in AUDIO_FEATURES if f not in data.X.columns]
    if missing:
        raise SystemExit(f"features absent from ARAUS table: {missing}")

    X = data.X[list(AUDIO_FEATURES)]
    print(f"training {args.model} on {len(X):,} ratings x {X.shape[1]} audio features")
    print(f"features: {', '.join(AUDIO_FEATURES)}")

    # honest estimate of what this model does on unseen participants
    sub = data.__class__(X=X, y=data.y, y_iso=data.y_iso,
                         groups=data.groups, frame=data.frame)
    cv = run_nested_cv(sub, args.model, target=args.target,
                       outer_rule="sgkf", inner_rule="gkf", n_trials=0, seed=args.seed)
    print(f"\nnested CV (no leakage): R2 = {cv.r2:.4f} (SE {cv.r2_se:.4f})  "
          f"MSE = {cv.mse:.4f}")

    model = make_model(args.model, seed=args.seed)
    model.fit(X, data.y)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"{args.target}_{args.model}_audio"
    joblib.dump(model, out / f"{stem}.joblib", compress=3)

    meta = {
        "target": args.target, "model": args.model,
        "features": list(AUDIO_FEATURES),
        "target_range": [0.0, 4.0],
        "target_range_note": "ISO [-1,1] transposed to [0,4]; see targets.to_model_range",
        "n_train": int(len(X)), "n_participants": int(len(set(data.groups))),
        "cv_r2": round(cv.r2, 4), "cv_r2_se": round(cv.r2_se, 4),
        "cv_mse": round(cv.mse, 4),
        "training_feature_means": {k: round(float(v), 4) for k, v in X.mean().items()},
        "training_feature_sds": {k: round(float(v), 4) for k, v in X.std().items()},
        "caveat": (
            "Trained on ARAUS precomputed psychoacoustics. Our extractor is "
            "validated against reference signals, not against ARAUS audio, "
            "which is not distributed. Implementation differences between the "
            "two remain unquantified."
        ),
    }
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out / stem}.joblib  (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
