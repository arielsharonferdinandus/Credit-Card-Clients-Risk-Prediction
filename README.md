# Predicting Customer Credit Card Default Risk

Leakage-aware, imbalance-aware, reproducible binary classification study on the UCI
**Default of Credit Card Clients** dataset.

## Project structure

```
project/
├── Main.ipynb                      # Main analytical notebook
├── default-of-credit-card-clients.csv  # Dataset (30,000 rows x 25 columns)
└── README.md                           # This file
```

## Dataset

- **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients*
  (I-Cheng Yeh, 2016), DOI: 10.24432/C55S3E
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Shape:** 30,000 rows x 25 columns
- **Target:** `default payment next month` (0 = non-default, 1 = default)
- **Class balance:** ~22.1% default / ~77.9% non-default (~3.5:1 imbalance)
- The notebook fetches this dataset **live from the UCI Machine Learning Repository** via the
  official `ucimlrepo` client (`fetch_ucirepo(id=350)`), verifies it against the documented
  shape and columns, and caches it locally as `default-of-credit-card-clients.csv`.

## How to run

1. Keep `Main.ipynb` in its own folder (a local `default-of-credit-card-clients.csv`
   fallback copy is optional but recommended in case the live fetch is unavailable — see below).
   All paths used are relative — no personal/machine-specific paths.
2. Restart the kernel and Run All (works in Google Colab or local Jupyter). The notebook is
   designed to execute top-to-bottom with no manual intervention and a fixed random seed
   (`RANDOM_STATE = 42`).
3. **Data loading behavior:** on each run, the notebook first tries to fetch the dataset live
   from `archive.ics.uci.edu` via `ucimlrepo` and validates the result (shape, columns, target).
   If that succeeds, it overwrites the local `default-of-credit-card-clients.csv` with the
   freshly fetched, verified copy. If the live fetch fails for any reason (no internet, UCI API
   downtime, rate limiting), it automatically falls back to the local
   `default-of-credit-card-clients.csv` if present, so the notebook still runs end-to-end
   without a working internet connection.

## Status

Implemented so far:
- Executive project summary (preliminary)
- Project overview and predictive question
- Dataset description and source, with a full data dictionary
- Data loading with live-fetch-and-verified-fallback, plus structural verification
- Data quality assessment: missingness, duplicates, undocumented categories in
  `EDUCATION`/`MARRIAGE`, `PAY_X` sentinel values, negative `BILL_AMT` interpretation, target
  imbalance, and a leakage audit
- Data preparation: ID handling, target verification, undocumented-category recoding, an
  outlier review, and repayment-status (`PAY_X`) recoding

Not yet implemented: exploratory data analysis, ratio/delinquency feature engineering, the
stratified train/validation/test split, the preprocessing pipeline (encoding + scaling fit on
the training fold), the baseline/Logistic Regression/ensemble models, threshold analysis, model
comparison and final selection, the single held-out test evaluation, diagnostics, feature
importance, and the final conclusion.

## Update log
- Feature engineering added: `bill_to_limit_ratio`, `pay_to_bill_ratio`, `max_delay`,
  `total_delinquent_months` (all deterministic row-wise combinations of existing columns, so
  safe to compute before the split).
- Stratified 70/15/15 train/validation/test split implemented and verified (default rate
  matches ~22.1% across all three subsets).
