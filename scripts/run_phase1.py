#!/usr/bin/env python3
"""Phase 1: reproduce the ARAUS Pleasantness / Eventfulness models.

    python scripts/run_phase1.py --target ISOPl --models LR RF XGBoost --trials 0

`--trials 0` uses library defaults (fast smoke run). The paper's settings are
400 Optuna trials for RF and 300 for XGBoost; use `--trials paper` for that,
and expect it to run for hours.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restorative import reference
from restorative.datasets import build
from restorative.models import MODELS, PAPER_TRIALS
from restorative.nested_cv import run_nested_cv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="ISOPl", choices=["ISOPl", "ISOEv"])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--outer", default="sgkf", choices=["sgkf", "cskf", "gkf"])
    ap.add_argument("--inner", default="gkf", choices=["sgkf", "cskf", "gkf"])
    ap.add_argument("--trials", default="0",
                    help="Optuna trials per inner loop; int, or 'paper'")
    ap.add_argument("--person", action="store_true",
                    help="add the person-related predictors (age, gender, wellbeing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = build(target=args.target, include_person=args.person)
    print(f"\nARAUS v1  n={len(data):,}  participants={len(set(data.groups)):,}  "
          f"features={data.X.shape[1]} ({', '.join(data.X.columns)})")
    print(f"target {args.target}: mean {data.y.mean():.3f}  var {data.y.var():.3f}  "
          f"(modelling range [0, 4])\n")

    rows = []
    for name in args.models:
        n_trials = PAPER_TRIALS[name] if args.trials == "paper" else int(args.trials)
        t0 = time.time()
        res = run_nested_cv(data, name, target=args.target, outer_rule=args.outer,
                            inner_rule=args.inner, n_trials=n_trials, seed=args.seed)
        print(f"{res.summary()}   [{time.time() - t0:.1f}s, {n_trials} trials]")
        rows.append({
            "model": name, "target": args.target,
            "splitting": f"{args.outer}-{args.inner}",
            "MSE": res.mse, "R2": res.r2, "R2_SE": res.r2_se,
            "R2_paper": reference.lookup(reference.R2, args.target, args.outer, args.inner, name),
            "MSE_paper": reference.lookup(reference.MSE, args.target, args.outer, args.inner, name),
            "n_trials": n_trials, "seconds": round(time.time() - t0, 1),
        })

    df = pd.DataFrame(rows)
    df["R2_delta"] = df["R2"] - df["R2_paper"]

    print("\n" + "=" * 78)
    print(f"Phase 1 vs. Versuemer et al. (2025), ARAUSD {args.target}, "
          f"{args.outer}-{args.inner}")
    print("=" * 78)
    print(df[["model", "MSE", "MSE_paper", "R2", "R2_SE", "R2_paper", "R2_delta"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nnote: 6 of 7 indicators (Relative Approach unavailable in ARAUS)")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"phase1_{args.target}_{args.outer}-{args.inner}_trials-{args.trials}"
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out / stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
