# Progress log

A running record of what has been built, what it produced, and which decisions
are already settled. One section per phase, appended as phases complete.
Phase definitions are in [`README.md`](README.md).

---

## Phase 1 — Perceptual predictor reproduction

**Completed** 2026-09-06 · commit `5a67463` · status: **passed**

### Goal

Rebuild the fixed-effects models of Versümer et al. (2025) on ARAUS v1 and
confirm the published R² range before anything is built on top of them. This
predictor becomes the control signal for the masker-to-base ratio search in
Phase 5, so its accuracy ceiling has to be known and trusted first.

### What was built

```
src/restorative/
  targets.py     ISO/TS 12913-3 projections (Eqs. 1–2), Likert centring,
                 [-1,1] → [0,4] modelling range
  features.py    the seven indicators of Table II, mapped onto ARAUS columns
  datasets.py    ARAUS v1 loading, fold filtering, participants join
  splitting.py   sgkf / gkf / cskf splitting rules + a group-leakage counter
  objective.py   objective function (Eq. 4), out-of-sample R² (Eq. 5)
  models.py      LR / RF / XGBoost, Optuna search spaces from Table X
  nested_cv.py   5×5 nested CV driver
  reference.py   published values, compared automatically on every run
scripts/run_phase1.py
tests/           8 tests, all passing
```

### Data provenance

ARAUS v1 response CSVs, obtained manually. Both the cloud environment and the
local session VM are behind an egress allowlist that refuses `zenodo.org` and
`researchdata.ntu.edu.sg`, so the automated fetch in `scripts/`-land cannot
reach the dataset host; the archive was supplied by hand. See
[`data/README.md`](data/README.md).

Fold accounting, from the 27,255 rows shipped in `responses.csv`:

| `fold_r` | rows | meaning | used |
|---|---|---|---|
| −1 | 1,815 | practice trials + attention checks (605 × 3) | no |
| 0 | 240 | separate test fold, 5 participants × 48 stimuli | no |
| 1–5 | 25,200 | the cross-validation set, 600 participants | **yes** |

25,200 ratings from 600 participants — matching the paper's ARAUSD cohort
exactly.

### Feature mapping

| Versümer indicator | measures | ARAUS column |
|---|---|---|
| `L_Aeq` | loudness | `LAavg_r` |
| `L_A10 − L_A90` | time variability | `LA10_r − LA90_r` |
| Relative Approach | saliency | **not distributed** |
| `L_Ceq − L_Aeq` | low-frequency content | `LCavg_r − LAavg_r` |
| Sharpness (acum) | high-frequency content | `Savg_r` |
| Tonality (tu) | tonal components | `Tavg_r` |
| Roughness (asper) | envelope modulation | `Ravg_r` |

Six of seven. Relative Approach was computed by the authors in ArtemiS SUITE
from the original calibrated stimuli, which ARAUS does not ship. This is
recorded in `features.MISSING_INDICATORS` so it appears in every report rather
than being silently absorbed into the residual.

### Results

Nested 5×5 cross-validation, Stratified-Group-K-Fold outer / Group-K-Fold
inner (groups kept whole, **0 leaked groups** verified per fold), library-default
hyperparameters, no Optuna budget spent.

**Pleasantness (ISOPl)** — target mean 2.054, variance 0.621 on the [0,4] range

| Model | MSE | MSE (paper) | R² | R² SE | R² (paper) | Δ |
|---|---|---|---|---|---|---|
| LR | 0.5251 | 0.5176 | 0.1542 | 0.0082 | 0.1675 | −0.0133 |
| RF | 0.4992 | 0.5026 | **0.1959** | 0.0077 | 0.1914 | **+0.0045** |
| XGBoost | 0.5126 | 0.5052 | 0.1743 | 0.0079 | 0.1872 | −0.0129 |

**Eventfulness (ISOEv)**

| Model | MSE | MSE (paper) | R² | R² SE | R² (paper) | Δ |
|---|---|---|---|---|---|---|
| LR | 0.5079 | 0.5045 | 0.2221 | 0.0076 | 0.2276 | −0.0055 |
| RF | 0.4810 | 0.4718 | **0.2631** | 0.0100 | 0.2775 | −0.0144 |
| XGBoost | 0.4924 | 0.4760 | 0.2456 | 0.0080 | 0.2711 | −0.0255 |

Random forest matches the published figure on Pleasantness while *untuned*.
The paper's Optuna budget — 400 trials for RF and 300 for XGBoost, per inner
loop per outer fold — has not been spent; `--trials paper` does that and runs
for hours.

**Leakage check.** Re-running Pleasantness with `--outer cskf --inner cskf`
spreads every participant across all folds — 3,000 leaked groups (600 × 5) —
and R² rises: LR 0.1542 → 0.1546, RF 0.1959 → 0.2012. The paper's illusory
gain reproduces, which is the direct evidence for the two-predictor rule below.

### Findings that carry forward

1. **All three qualitative results reproduce.** Nonlinear beats linear on both
   targets; Eventfulness models better than Pleasantness; leaky splitting
   inflates R².
2. **The accuracy ceiling is real and low.** R² ≈ 0.20 for Pleasantness from
   acoustics alone. The predictor is a *ranking* signal for comparing candidate
   mixes of the same base scene — not an absolute measurement of how pleasant a
   soundscape is. Phase 5 must be designed around that.
3. **`smr` ∈ {−6, −3, 0, +3, +6} dB.** ARAUS varied the masker-to-base ratio
   over exactly the grid Phase 5 will search, with human ratings attached to
   every point. The control loop can therefore be validated against real
   listener data before any audio is synthesised.
4. **Loudness dominates the feature set**, which is why the Phase 5 search
   needs a hard constraint on output `L_Aeq` — otherwise the optimiser raises
   the predicted score by adding energy.
5. **Two predictors, never one.** The model driving the MBR search must not be
   the model reporting the improvement, or the evaluation measures how well the
   optimiser exploited its own scorer. Control predictor: ARAUS. Evaluation
   predictor: a different dataset (ISD, HSDD) and a different model class.

### Deviations from the paper

- Six of seven indicators (no Relative Approach).
- Hyperparameters untuned so far.
- Mixed-effects variants (MELR, MERF, XGBEM, …) not implemented. They
  generalise better to unknown groups at some cost in performance; not needed
  while the pipeline scores candidate mixes rather than predicting for
  specific listeners.

### Environment

Python 3.10.12 · numpy 1.26.4 · pandas 2.3.3 · scikit-learn 1.7.2 ·
xgboost 3.2.0 · optuna 4.9.0

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/run_phase1.py --target ISOPl --models LR RF XGBoost
./.venv/bin/python -m pytest tests/ -q
```

### Open items

- [ ] Spend the paper's Optuna budget; expect XGBoost to close its 0.013 gap
- [ ] Decide whether to reimplement Relative Approach (Bray 2004) in Phase 2
- [ ] Train the independent evaluation predictor on ISD / HSDD
- [ ] Person-related predictors are wired (`--person`) but not yet evaluated

### Next

**Phase 2 — feature pipeline.** Compute the six indicators from arbitrary WAV
input so the predictor can score audio it has never seen. ARAUS ships
precomputed values only; nothing yet reads a waveform. This is the bridge
between the predictor and the audio path.
