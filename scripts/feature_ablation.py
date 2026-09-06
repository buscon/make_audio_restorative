#!/usr/bin/env python3
"""Leave-one-out feature ablation on ARAUS.

Phase 2 must compute the predictor's inputs from a waveform. Two of the
indicators cannot be reproduced from open implementations:

  - Relative Approach   never available (already absent in Phase 1)
  - Tonality (ECMA-418-2, tu)  MoSQITo ships ECMA-74 tone-to-noise ratio,
                        which is a different quantity on a different scale
                        and cannot be substituted into the same column

This script quantifies what each indicator is worth, so the choice of feature
set for the audio-side predictor is made on evidence rather than convenience.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative.datasets import build
from restorative.nested_cv import run_nested_cv

#: computable from a waveform with open implementations
AUDIO_COMPUTABLE = ["LAeq", "LA10_LA90", "LCeq_LAeq", "Sharpness", "Roughness"]


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "ISOPl"
    model = sys.argv[2] if len(sys.argv) > 2 else "RF"

    data = build(target=target)
    all_feats = list(data.X.columns)
    print(f"\nARAUS v1  n={len(data):,}  target={target}  model={model}")
    print(f"full set: {', '.join(all_feats)}\n")

    def score(cols, label):
        sub = data.__class__(X=data.X[cols], y=data.y, y_iso=data.y_iso,
                             groups=data.groups, frame=data.frame)
        res = run_nested_cv(sub, model, target=target,
                            outer_rule="sgkf", inner_rule="gkf", n_trials=0)
        print(f"  {label:<34s} k={len(cols)}  R2={res.r2:.4f}  MSE={res.mse:.4f}")
        return res.r2

    full = score(all_feats, "full set")

    print("\nleave-one-out (drop in R2 = what the indicator is worth):")
    rows = []
    for f in all_feats:
        cols = [c for c in all_feats if c != f]
        r2 = score(cols, f"without {f}")
        rows.append({"dropped": f, "R2": r2, "loss": full - r2})

    print("\naudio-computable subset:")
    audio = score(AUDIO_COMPUTABLE, "LAeq+var+LCeq-LAeq+Sharp+Rough")

    df = pd.DataFrame(rows).sort_values("loss", ascending=False)
    print("\n" + "=" * 62)
    print(f"indicator value ranking ({target}, {model})")
    print("=" * 62)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nfull set          R2 = {full:.4f}")
    print(f"audio-computable  R2 = {audio:.4f}   cost = {full - audio:.4f}")

    out = Path("reports") / f"ablation_{target}_{model}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
