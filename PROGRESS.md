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

---

## Phase 2 — Feature pipeline from audio

**Completed** 2026-09-06 · status: **passed, with one unquantified risk**

### Goal

Compute the predictor's inputs from a waveform, so it can score audio it has
never seen. ARAUS ships precomputed psychoacoustics only; until now nothing in
the repo could read a WAV. This is the bridge between the Phase 1 predictor and
the audio path.

### What was built

```
src/restorative/
  acoustics.py         IEC 61672 A/C weighting, exponential time weighting,
                       exceedance levels, calibration
  psychoacoustics.py   sharpness (DIN 45631/A1) and roughness (ECMA-418-2)
                       via MoSQITo, block-processed
  extract.py           waveform -> feature vector, right channel by default
scripts/
  feature_ablation.py  leave-one-out value of each indicator
  train_predictor.py   fit and persist the control predictor
  score_wav.py         score WAV files end to end
tests/                 30 tests
```

### The tonality problem, and what it cost

MoSQITo has no ECMA-418-2 tonality. Its `tonality` subpackage provides ECMA-74
tone-to-noise ratio and prominence ratio — different quantities, on different
scales (dB per tone, versus tonality units), which cannot be substituted into
ARAUS's `Tavg_r` column without feeding the trained model a predictor it was
never fitted on. TNR is therefore exposed as its own feature and a test asserts
it never enters the ARAUS feature set.

Leave-one-out ablation on ARAUS (RF, ISOPl, sgkf-gkf) measures what each
indicator is worth:

| dropped | R² | loss |
|---|---|---|
| LAeq | 0.1682 | **0.0277** |
| LCeq − LAeq | 0.1738 | 0.0221 |
| Roughness | 0.1752 | 0.0207 |
| Tonality | 0.1761 | 0.0198 |
| LA10 − LA90 | 0.1814 | 0.0145 |
| Sharpness | 0.1828 | 0.0131 |

No single indicator dominates; all six contribute between 0.013 and 0.028.
`LAeq` is the most valuable, independently confirming the Phase 5 rule that the
masker-to-base search needs a hard loudness constraint.

**Decision.** The audio-side predictor uses the five computable indicators:
**R² = 0.1761 (SE 0.0071)** against the full set's 0.1959 — a cost of 0.0198,
still above the paper's linear-regression baseline of 0.1675. Persisted by
`scripts/train_predictor.py` as `models/ISOPl_RF_audio.joblib`
(git-ignored at 184 MB; regenerates in seconds).

### Validation

ARAUS does not distribute its audio, so the extractor cannot be compared
against the columns it will be paired with. The anchor used instead is the
standards' own reference signals, where the correct answer is 1.0 by
definition:

| reference signal | expected | measured |
|---|---|---|
| narrowband noise, 1 kHz centre, 160 Hz BW, 60 dB (DIN 45692) | 1.0 acum | **1.002** |
| 1 kHz tone, 60 dB, 100% AM at 70 Hz | 1.0 asper | **0.995** |
| unmodulated 1 kHz tone | ~0 asper | 0.000 |

A/C weighting is checked against the IEC 61672-1 tolerance table at 31.5 Hz,
125 Hz, 1 kHz, 4 kHz and 8 kHz, and both curves are unity at 1 kHz to within
0.05 dB. The exceedance convention is asserted explicitly (L_10 > L_90, as in
ARAUS), since reversing it would flip the sign of the variability feature.

### Engineering constraints found

- **MoSQITo's ECMA-418-2 roughness allocates every analysis frame at once.**
  A 5 s clip peaks at 3.3 GB; a 30 s ARAUS clip would need roughly 20 GB. It
  is now processed in 2 s blocks and averaged, which bounds peak memory at
  1.65 GB and is what the time-aggregated ARAUS columns represent anyway. A
  test asserts blocking does not change the answer.
- **Throughput is about 45 s per 30 s clip**, single-threaded, dominated by
  MoSQITo. Batch extraction over a corpus should be parallelised across
  processes.
- **Calibration is a required argument**, not a default. `extract()` refuses to
  run without either `cal_db` or `target_laeq`. Given that `LAeq` is the most
  valuable predictor, an arbitrary playback gain would silently move every
  prediction.

### End-to-end smoke test

A synthetic urban base scored `ISOPl = +0.103`; the same base with a synthetic
high-frequency "water" masker scored **−0.008**, i.e. worse. The masker raised
Sharpness from 0.80 to 1.72 acum at constant `L_Aeq`, and the model penalised
it. This is the loop correctly rejecting a bad masker rather than evidence
about masking in general — the "water" was high-passed white noise, not a
recording. It does show the two behaviours the control loop depends on:
variability fell as intended (`LA10−LA90` 6.5 → 3.9 dB, the masker filling the
quiet gaps), and masker *quality* changes the verdict.

### Open risk

**The extractor has never been compared against ARAUS's own values.** The
predictor was trained on ARAUS's precomputed psychoacoustics (commercial
software, calibrated stimuli) and will be fed ours. Reference-signal agreement
constrains the error but does not measure this specific mismatch. Closing it
needs ARAUS audio, which is a separate ~3 GB download from a host the
automated environments cannot reach. The ARAUS masker set (407 files) is part
of that download and is needed for Phase 3 regardless, so the two should be
fetched together. This caveat is recorded in every trained model's metadata.

### Next

**Phase 3 — masking loop with sampled maskers.** Separation, attenuation and
the MBR search using the real ARAUS maskers rather than synthesised ones. This
is a complete working system and the baseline the autoencoder must beat; it
also closes the open risk above, since fetching the maskers brings the audio
needed to validate the extractor.

### Infrastructure note

A remote Linux machine with a strong GPU is available for training. Nothing so
far needs it — Phases 1–3 are CPU-bound signal processing and tree ensembles.
It matters from **Phase 4** (the conditional VAE and the HiFi-GAN vocoder), so
training scripts from here on stay device-agnostic and take their device from
configuration rather than hardcoding CPU.
