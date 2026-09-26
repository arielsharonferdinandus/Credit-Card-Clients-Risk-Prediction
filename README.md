# Predicting Customer Credit Card Default Risk

Leakage-aware, imbalance-aware, reproducible binary classification study on the UCI
**Default of Credit Card Clients** dataset.

## Project structure

```
project/
├── Main.ipynb                          # Main analytical notebook
├── default-of-credit-card-clients.csv  # Dataset (30,000 rows x 25 columns)
└── README.md                           # This file
```

## Dataset

* **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients*
(I-Cheng Yeh, 2016), DOI: 10.24432/C55S3E
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
* **Shape:** 30,000 rows x 25 columns
* **Target:** `default payment next month` (0 = non-default, 1 = default)
* **Class balance:** \~22.1% default / \~77.9% non-default (\~3.5:1 imbalance)
* Note: the dataset was originally provided as an `.xls` file; the notebook now fetches the
same dataset **live from the UCI Machine Learning Repository** via the official `ucimlrepo`
client (`fetch\_ucirepo(id=350)`), verifies it against the documented shape/columns, and
caches it locally as `default-of-credit-card-clients.csv`.

## How to run

1. Keep `notebook.ipynb` and `task.txt` in the same folder (a local
`default-of-credit-card-clients.csv` fallback copy is optional but recommended in case the
live fetch is unavailable — see below). All paths used are relative — no
personal/machine-specific paths.
2. Restart the kernel and Run All (works in Google Colab or local Jupyter). The notebook is
designed to execute top-to-bottom with no manual intervention and a fixed random seed
(`RANDOM\_STATE = 42`).
3. **Data loading behavior:** on each run, the notebook first tries to fetch the dataset
live from `archive.ics.uci.edu` via `ucimlrepo` and validates the result (shape, columns,
target). If that succeeds, it overwrites the local
`default-of-credit-card-clients.csv` with the freshly fetched, verified copy. If the live
fetch fails for any reason (no internet, UCI API downtime, rate limiting), it automatically
falls back to the local `default-of-credit-card-clients.csv` if present, so the notebook
still runs end-to-end without a working internet connection.

## Status

This version implements **Sections 1–5** of `task.txt`:

1. Executive Project Summary (preliminary)
2. Project Overview
3. Dataset Description and Source (with full data dictionary)
4. Data Loading and Initial Verification
5. Data Quality Assessment (missingness, duplicates, undocumented categories in
`EDUCATION`/`MARRIAGE`, `PAY\_X` sentinel values, negative `BILL\_AMT` interpretation, target
imbalance, and a leakage audit)

**Not yet implemented:** Data Preparation, EDA, Feature Engineering,
Train/Val/Test Split, Preprocessing Pipeline, Baseline + Logistic Regression +
Ensemble models, Hyperparameter Tuning, Threshold Analysis, Model
Comparison \& Selection, Final Test Evaluation, Diagnostics, Feature
Importance, Final Conclusion, and Executive Summary / interview prep.

