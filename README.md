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
4. **Runtime note:** hyperparameter tuning uses cross-validated search and takes a few minutes
   on a single CPU core; a multi-core Colab runtime will finish faster.

## Result

The final model is a tuned Gradient Boosting classifier, selected on validation F1/ROC AUC and
evaluated once on the held-out test set:

| Metric | Test Set |
|---|---|
| Accuracy | 0.805 |
| Precision | 0.563 |
| Recall | 0.527 |
| F1 | 0.544 |
| ROC AUC | 0.778 |

Recent repayment status is the dominant predictor, confirmed independently through exploratory
analysis, Logistic Regression coefficients, Random Forest importances, and permutation
importance on the test set. The model has a disclosed blind spot: it struggles to catch
defaults that occur without a prior delinquency trail. See the notebook's Final Conclusion
section for the full discussion of limitations and appropriate use.

## Status: Complete

All sections implemented:
- Executive project summary, project overview, and dataset description with a full data
  dictionary
- Data loading with live-fetch-and-verified-fallback, plus structural verification
- Data quality assessment: missingness, duplicates, undocumented categories, `PAY_X` sentinel
  values, negative `BILL_AMT` interpretation, target imbalance, and a leakage audit
- Data preparation: ID handling, target verification, undocumented-category recoding, an
  outlier review, and repayment-status recoding
- Exploratory data analysis: univariate, categorical default-rate, multivariate, and
  correlation analysis
- Feature engineering: utilization/payment ratios and delinquency-summary indicators
- Stratified 70/15/15 train/validation/test split, verified for class balance
- Preprocessing pipeline: scaling + one-hot encoding, fit on the training fold only
- Baseline, Logistic Regression, Random Forest, and Gradient Boosting models
- Cross-validated hyperparameter tuning and per-model threshold analysis
- Model comparison, final selection, and a single held-out test evaluation
- Diagnostics (ROC, precision-recall, calibration, error analysis) and permutation feature
  importance
- Final conclusion covering results, key drivers, limitations, and suggested use
