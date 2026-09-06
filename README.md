# make_audio_restorative

Transforming recorded soundscapes into more restorative ones, by attenuating
stressors and adding a generatively synthesised natural masker under
perceptual control.

The system deliberately does **not** regenerate the scene with a generative
model. Source separation attenuates the stressors, a conditional autoencoder
supplies only the masker layer, and a psychoacoustic predictor closes the loop
by searching the masker-to-base ratio that maximises predicted pleasantness.

```
 corpus -> masker autoencoder ------------------\
                                                 v
 recording -> separation -> attenuation -> [ mix ] -> restored soundscape
                                             |  ^
                                 candidate   v  |  MBR (dB), masker class
                              psychoacoustics -> pleasantness predictor -> MBR search
```

## Status

Detailed per-phase record: [`PROGRESS.md`](PROGRESS.md)

| Phase | Description | State |
|---|---|---|
| 1 | Perceptual predictor reproduction (ARAUS) | **done** |
| 2 | Feature pipeline from arbitrary WAV input | next |
| 3 | Masking loop with sampled maskers (baseline system) | |
| 4 | Masker autoencoder (cVAE + vocoder) | |
| 5 | Integration | |
| 6 | Listening study | |

## Phase 1 results

Reproduction of Versümer et al. (2025) on ARAUS v1, nested 5x5
cross-validation, Stratified-Group-K-Fold outer / Group-K-Fold inner
(no group leakage), n = 25,200 ratings from 600 participants.
Library-default hyperparameters, no Optuna tuning yet.

**Pleasantness (ISOPl)**

| Model | MSE | MSE (paper) | R² | R² (paper) | Δ |
|---|---|---|---|---|---|
| LR | 0.5251 | 0.5176 | 0.1542 | 0.1675 | −0.013 |
| RF | 0.4992 | 0.5026 | **0.1959** | 0.1914 | +0.005 |
| XGBoost | 0.5126 | 0.5052 | 0.1743 | 0.1872 | −0.013 |

**Eventfulness (ISOEv)**

| Model | MSE | MSE (paper) | R² | R² (paper) | Δ |
|---|---|---|---|---|---|
| LR | 0.5079 | 0.5045 | 0.2221 | 0.2276 | −0.006 |
| RF | 0.4810 | 0.4718 | **0.2631** | 0.2775 | −0.014 |
| XGBoost | 0.4924 | 0.4760 | 0.2456 | 0.2711 | −0.026 |

The paper's central findings reproduce: nonlinear models beat linear
regression on both targets, Eventfulness is easier to model than Pleasantness,
and switching to a leaky splitting rule (`--outer cskf --inner cskf`, which
puts all 600 participants on both sides of every split) inflates R².

We use **six of the paper's seven indicators**. Relative Approach is not
distributed with ARAUS — Versümer et al. computed it with ArtemiS SUITE from
the original calibrated stimuli — which accounts for most of the residual gap.

### Reproducing

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/run_phase1.py --target ISOPl --models LR RF XGBoost
```

`--trials paper` runs the published Optuna budget (400 trials for RF, 300 for
XGBoost, per inner loop, per outer fold) and takes hours. `--trials 0` (default)
uses library defaults and finishes in seconds. `--person` adds the
person-related predictors (age, gender, WHO-5 wellbeing).

Data setup: see [`data/README.md`](data/README.md). ARAUS response CSVs carry
precomputed psychoacoustics, so Phase 1 needs no audio.

## Layout

```
src/restorative/
  targets.py     ISO/TS 12913-3 projections (Eqs. 1-2), [-1,1] -> [0,4]
  features.py    the 7 indicators of Table II, mapped onto ARAUS columns
  datasets.py    ARAUS v1 loading, fold filtering, participant join
  splitting.py   sgkf / gkf / cskf splitting rules
  objective.py   objective function (Eq. 4) and out-of-sample R² (Eq. 5)
  models.py      LR / RF / XGBoost + Optuna search spaces (Table X)
  nested_cv.py   nested CV driver
  reference.py   published values, for automatic comparison
scripts/run_phase1.py
tests/
```

## A note on the two predictors

Phase 5 searches the masker-to-base ratio by maximising *predicted*
pleasantness. The predictor that drives that search must never be the one that
reports the improvement — otherwise the evaluation measures how well the
optimiser exploited its own scorer. The control predictor is trained on ARAUS;
the evaluation predictor will be trained on a different dataset (ISD, HSDD)
with a different model class.

Loudness is the strongest single predictor in this feature set, so the search
also needs a hard constraint on output L_Aeq. Without it the optimiser
discovers that simply adding energy raises the predicted score.

## References

- Versümer, S., Blättermann, P., Rosenthal, F., Weinzierl, S. (2025).
  "A comparison of methods for modeling soundscape dimensions based on
  different datasets." *J. Acoust. Soc. Am.* **157**(1), 234–255.
- Ooi, K. et al. (2023). "ARAUS: A large-scale dataset and baseline models of
  affective responses to augmented urban soundscapes."
  *IEEE Trans. Affective Comput.* **15**, 105–117.
- ISO/TS 12913-2:2018, ISO/TS 12913-3:2019.
