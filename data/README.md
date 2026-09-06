# Data

Nothing in this directory is committed. `data/external/` is git-ignored.

## ARAUS v1 (required for Phase 1)

The response CSVs carry both the participant ratings and precomputed
psychoacoustic parameters, so Phase 1 needs **no audio**.

The dataset host (`researchdata.ntu.edu.sg`) is not reachable from this
project's automated environments, so the archive must be fetched manually:

1. Download `data.zip` (~ tens of MB):
   https://researchdata.ntu.edu.sg/api/access/datafile/99922?gbrecs=true
2. Place it at `data/external/data.zip`
3. Run `python scripts/fetch_araus.py --verify-only`

`scripts/fetch_araus.py` will verify the SHA-512 against the ARAUS manifest and
unpack it to `data/external/data/`.

On a machine with unrestricted network access, `python scripts/fetch_araus.py`
does the download step too.

Expected after unpacking: `data/external/data/responses.csv` (~25,440 rows).

Source: Ooi et al. (2023), ARAUS. https://github.com/ntudsp/araus-dataset-baseline-models

## Audio (required from Phase 3)

Fetch into `data/external/araus_v1/`, then unzip in place:

```bash
wget -O maskers.zip "https://researchdata.ntu.edu.sg/api/access/datafile/89458?gbrecs=true"
unzip -q maskers.zip
```

- `maskers/` — 293 files (82 bird, 82 water, 41 wind, 41 traffic,
  41 construction, 6 silence), ~741 MB. `maskers.csv` carries ARAUS's own
  values for all seven indicators plus per-file calibration gains
  (`gain_46dB` ... `leq_at_gain_XXdB`), which makes this the validation set
  for our extractor.

- `soundscapes/` — note that `soundscapes.zip` (datafile 89459) contains only
  the **6 test-fold recordings**, not the 396 the ARAUS README implies. The
  remaining base soundscapes come from `binaural.zip` on Zenodo, which
  `code/download.py` splits into 30 s segments via `split_usotw_track`.
  Only needed once base scenes are being restored.

Everything here is git-ignored; the audio alone is ~2.8 GB.
