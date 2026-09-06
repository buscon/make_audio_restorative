#!/usr/bin/env python3
"""Compare our feature extractor against ARAUS's own published values.

ARAUS computed its psychoacoustic parameters with external commercial
software; nothing in its released code reproduces them. Our extractor is
validated against reference signals (see tests/test_psychoacoustics.py), but
that does not measure the gap between the two implementations - and the
predictor is trained on ARAUS's numbers while being fed ours.

The 293 masker files close that gap: `maskers.csv` carries ARAUS's values for
every indicator we compute, per file, across six sound classes.

Calibration is removed as a confound by solving each file's gain so our L_Aeq
equals ARAUS's `LAavg_m`. What remains is a comparison of the metric
implementations themselves. Two of the four comparisons are entirely
level-independent (LA10-LA90 and LCeq-LAeq), so they hold regardless.

    python scripts/validate_extractor.py --workers 16
    python scripts/validate_extractor.py --report

Resumable: results are appended per file, and a rerun skips what is done.
Memory, not CPU, is usually the limit - each worker peaks near 1.7 GB.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
MASKER_DIR = ROOT / "data" / "external" / "araus_v1" / "maskers"
MASKER_CSV = ROOT / "data" / "external" / "araus_v1" / "maskers.csv"
OUT_CSV = ROOT / "reports" / "extractor_validation.csv"

#: our feature name -> how ARAUS expresses the same quantity
COMPARISONS = {
    "LA10_LA90": ("LA10_m", "LA90_m"),   # difference of two ARAUS columns
    "LCeq_LAeq": ("LCavg_m", "LAavg_m"),
    "Sharpness": ("Savg_m",),
    "Roughness": ("Ravg_m",),
}
FIELDS = ["masker", "class", "araus_LAeq", "ours_LAeq"] + [
    f"{k}_{w}" for k in COMPARISONS for w in ("araus", "ours")
] + ["seconds"]


def _one(job):
    """Extract one file. Runs in a worker process."""
    from restorative.extract import extract_file

    name, cls, ref = job
    t0 = time.time()
    try:
        f = extract_file(MASKER_DIR / name, channel="mono",
                         target_laeq=float(ref["LAavg_m"]))
    except Exception as exc:                       # keep the run going
        return {"masker": name, "class": cls, "error": f"{type(exc).__name__}: {exc}"}

    row = {"masker": name, "class": cls,
           "araus_LAeq": ref["LAavg_m"], "ours_LAeq": round(f.LAeq, 3),
           "seconds": round(time.time() - t0, 1)}
    for feat, cols in COMPARISONS.items():
        araus = float(ref[cols[0]]) - float(ref[cols[1]]) if len(cols) == 2 \
            else float(ref[cols[0]])
        row[f"{feat}_araus"] = round(araus, 4)
        row[f"{feat}_ours"] = round(getattr(f, feat), 4)
    return row


def report(path: Path = OUT_CSV) -> int:
    if not path.exists():
        print(f"nothing at {path} - run the extraction first")
        return 1
    df = pd.read_csv(path)
    df = df[df.get("error").isna()] if "error" in df.columns else df
    print(f"\nextractor vs ARAUS  |  n = {len(df)} masker files")
    if "class" in df.columns:
        print("classes:", ", ".join(f"{k} {v}" for k, v in
                                    df["class"].value_counts().sort_index().items()))

    print(f"\n{'feature':<12}{'unit':<8}{'ARAUS mean':>11}{'ours':>9}"
          f"{'bias':>8}{'MAE':>8}{'r':>7}{'95% limits of agreement':>26}")
    print("-" * 90)
    units = {"LA10_LA90": "dB", "LCeq_LAeq": "dB",
             "Sharpness": "acum", "Roughness": "asper"}
    rows = []
    for feat in COMPARISONS:
        a = df[f"{feat}_araus"].to_numpy(float)
        o = df[f"{feat}_ours"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(o)
        a, o = a[m], o[m]
        if len(a) < 3:
            continue
        d = o - a
        bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
        r = float(np.corrcoef(a, o)[0, 1])
        print(f"{feat:<12}{units[feat]:<8}{a.mean():>11.3f}{o.mean():>9.3f}"
              f"{bias:>8.3f}{np.mean(np.abs(d)):>8.3f}{r:>7.3f}"
              f"{f'[{bias - 1.96 * sd:+.3f}, {bias + 1.96 * sd:+.3f}]':>26}")
        rows.append({"feature": feat, "n": len(a), "araus_mean": a.mean(),
                     "ours_mean": o.mean(), "bias": bias, "mae": np.mean(np.abs(d)),
                     "sd": sd, "pearson_r": r})

    print("\nbias = ours - ARAUS. LA10-LA90 and LCeq-LAeq are level-independent,")
    print("so they test the weighting filters and level statistics alone.")
    print("Sharpness and Roughness are level-dependent and were computed after")
    print("calibrating each file to ARAUS's own L_Aeq.")
    pd.DataFrame(rows).to_csv(path.with_name("extractor_validation_summary.csv"),
                              index=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None, help="stop after N files")
    ap.add_argument("--minutes", type=float, default=None, help="time budget")
    ap.add_argument("--report", action="store_true", help="summarise and exit")
    ap.add_argument("--restart", action="store_true", help="discard previous results")
    args = ap.parse_args()

    if args.report:
        return report()

    if not MASKER_DIR.exists():
        raise SystemExit(f"no maskers at {MASKER_DIR} - see data/README.md")

    meta = pd.read_csv(MASKER_CSV, low_memory=False).set_index("masker")
    OUT_CSV.parent.mkdir(exist_ok=True)
    if args.restart and OUT_CSV.exists():
        OUT_CSV.unlink()

    done = set()
    if OUT_CSV.exists():
        done = set(pd.read_csv(OUT_CSV)["masker"])

    jobs = []
    for name in sorted(meta.index):
        if name in done or not (MASKER_DIR / name).exists():
            continue
        ref = meta.loc[name]
        if any(pd.isna(ref[c]) for cols in COMPARISONS.values() for c in cols):
            continue
        jobs.append((name, ref.get("class", "?"), ref))

    # interleave classes so a partial run is still representative
    by_class: dict = {}
    for j in jobs:
        by_class.setdefault(j[1], []).append(j)
    interleaved = []
    for i in range(max((len(v) for v in by_class.values()), default=0)):
        for cls in sorted(by_class):
            if i < len(by_class[cls]):
                interleaved.append(by_class[cls][i])
    jobs = interleaved[:args.limit] if args.limit else interleaved

    print(f"{len(done)} done, {len(jobs)} to process, {args.workers} workers")
    if not jobs:
        return report()
    print(f"note: each worker peaks near 1.7 GB; {args.workers} workers "
          f"need ~{args.workers * 1.7:.1f} GB\n")

    import multiprocessing as mp

    t0 = time.time()
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["error"], extrasaction="ignore")
        if new:
            w.writeheader()
        with mp.Pool(args.workers, maxtasksperchild=1) as pool:
            for i, row in enumerate(pool.imap_unordered(_one, jobs), 1):
                w.writerow(row)
                fh.flush()
                if i % 10 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i}/{len(jobs)}  {el / 60:.1f} min elapsed, "
                          f"~{el / i * (len(jobs) - i) / 60:.1f} min left", flush=True)
                if args.minutes and (time.time() - t0) > args.minutes * 60:
                    print("time budget reached - rerun to continue")
                    pool.terminate()
                    break

    print(f"\nwrote {OUT_CSV}")
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
