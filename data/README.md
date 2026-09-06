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
