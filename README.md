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

1. Open `Main.ipynb` in Google Colab or local Jupyter — no other files are required to start.
2. Run the first cell (`%pip install -q ucimlrepo`), then Run All. The notebook executes
   top-to-bottom with no manual intervention and a fixed random seed (`RANDOM_STATE = 42`).
3. **Data loading behavior:** the notebook first tries to fetch the dataset live from
   `archive.ics.uci.edu` via `ucimlrepo` and validates the result (shape, columns, target). On
   success it caches a verified copy to `default-of-credit-card-clients.csv`. If the live fetch
   fails (no internet, UCI API downtime, rate limiting), it automatically falls back to that
   local cached copy if present, so the notebook still runs end-to-end offline.

## Status

Implemented so far:
- Executive project summary, project overview, and dataset description with a full data
  dictionary
- Data loading with live-fetch-and-verified-fallback, plus structural verification
- Data quality assessment: missingness, duplicates, undocumented categories in
  `EDUCATION`/`MARRIAGE`, `PAY_X` sentinel values, negative `BILL_AMT` interpretation, target
  imbalance, and a leakage audit
- Data preparation: ID handling, target verification, undocumented-category recoding, an
  outlier review, and repayment-status (`PAY_X`) recoding
- Exploratory data analysis: univariate, categorical default-rate, multivariate, and
  correlation analysis
- Feature engineering: `bill_to_limit_ratio`, `pay_to_bill_ratio`, `max_delay`,
  `total_delinquent_months`
- Stratified 70/15/15 train/validation/test split, verified for class balance
- Preprocessing pipeline: scaling + one-hot encoding, fit on the training fold only
- A majority-class baseline and a Logistic Regression model, with a shared evaluation helper
  and coefficient interpretation
- Two tree-based ensembles (Random Forest and gradient boosting) with an interim comparison
  across all four models fitted so far
- Cross-validated hyperparameter tuning for Logistic Regression, Random Forest, and gradient
  boosting, with tuned models re-evaluated on the validation set
- Per-model threshold analysis, finding each tuned model's F1-maximizing classification
  threshold on the validation set instead of assuming the default 0.5 cutoff
- Model comparison and final selection: Gradient Boosting chosen as the final model based on
  validation F1 and ROC AUC, at its own optimal threshold

Not yet implemented: the single held-out test evaluation, diagnostics, feature importance, and
the final conclusion.
