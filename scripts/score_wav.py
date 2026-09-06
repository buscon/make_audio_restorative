#!/usr/bin/env python3
"""Score a WAV file for predicted soundscape pleasantness.

    python scripts/score_wav.py clip.wav --cal-db 94
    python scripts/score_wav.py clip.wav --target-laeq 63

Calibration is mandatory. L_Aeq is the most valuable single predictor in this
feature set (leave-one-out cost 0.028 R2), so an arbitrary playback gain would
silently move the prediction. Use --cal-db when the recording chain is
calibrated, --target-laeq when you know the level the clip should represent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative.extract import AUDIO_FEATURES, extract_file
from restorative.targets import from_model_range


def load_predictor(path: Path):
    model = joblib.load(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--model", default="models/ISOPl_RF_audio.joblib")
    ap.add_argument("--channel", default="right", choices=["right", "left", "mono"],
                    help="ARAUS reports the right ear; default matches training")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cal-db", type=float, help="dBFS to dB SPL offset")
    g.add_argument("--target-laeq", type=float, help="solve calibration for this L_Aeq")
    ap.add_argument("--out", default=None, help="write a CSV here")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"no predictor at {model_path} - run scripts/train_predictor.py")
    model, meta = load_predictor(model_path)
    target = meta.get("target", "ISOPl")

    rows = []
    for wav in args.wav:
        feats = extract_file(wav, channel=args.channel,
                             cal_db=args.cal_db, target_laeq=args.target_laeq)
        X = pd.DataFrame([feats.to_row()])[list(AUDIO_FEATURES)]
        rows.append({
            "file": Path(wav).name,
            **feats.to_row(),
            "TNR": feats.TNR,
            f"{target}": float(from_model_range(model.predict(X)[0])),
        })

    df = pd.DataFrame(rows)
    print(f"\npredictor: {model_path.name}  (nested-CV R2 {meta.get('cv_r2', '?')}, "
          f"{len(meta.get('features', []))} features)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{target} is on the ISO [-1, 1] scale: higher is more pleasant.")
    print("It ranks candidate mixes of one scene; it is not an absolute")
    print(f"measurement (R2 ~ {meta.get('cv_r2', 0.18)} against human ratings).")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
