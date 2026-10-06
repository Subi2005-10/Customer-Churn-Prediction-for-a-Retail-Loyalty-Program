# FreshBasket Loyalty — Customer Churn Prediction (Assessment 4)

Predicts which FreshBasket loyalty members will churn (no purchase in Apr–Jun 2024), explains why, and
estimates which retention actions pay off. Python, fully reproducible from the raw Excel file.

## Headline results (held-out official label, Apr–Jun 2024)

| Population | Members | Churn rate | Final model PR-AUC | ROC-AUC | Precision / Recall / F1 |
|---|---|---|---|---|---|
| All scored members | 2,368 | 34.3% | 0.988 | 0.991 | 0.95 / 0.95 / 0.95 |
| Active members (bought Jan–Mar 2024) | 1,715 | 9.6% | 0.81 | 0.96 | 0.75 / 0.74 / 0.75 |

* Contacting the riskiest **20%** of active members reaches **91%** of the members who actually churned.
* Targeted coupons retain about the same number of members as a blanket coupon to everyone, at about **one-fifth of the cost**.
* Strongest drivers: falling purchase frequency, low app use, no email opens, few coupons, and recent support tickets.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.13, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, SHAP 0.52.

## Run

```bash
python run_pipeline.py                     # everything: cleaning -> features -> models -> outputs (~30 s)
python notebooks/build_eda_notebook.py     # rebuilds and executes the EDA notebook
```

The raw file must be at `data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx` (included).
Random seed is fixed in `src/config.py`, so reruns give identical results.

## Project structure

```
data/raw/                         source workbook (unchanged)
data/processed/
  customer_model_dataset.csv      FINAL customer-level modelling dataset (features as of Mar-2024 + official label)
  all_snapshots.csv               all out-of-time snapshots used for training/validation/test
  clean_profile|activity|label.csv cleaned tables
notebooks/01_data_quality_eda.ipynb   data-quality & EDA notebook (executed)
src/
  config.py         paths, cutoffs, seed
  data_loading.py   reads the three sheets (header is on the 2nd row)
  data_quality.py   Task 1: audit + cleaning, logs every fix
  features.py       Tasks 2-3: join + leakage-safe snapshot features
  eda.py            Tasks 1, 4: EDA and engagement/support vs churn
  modeling.py       Tasks 5-8: models, thresholds, metrics
  explain.py        Task 9: SHAP drivers, segment risk
  ablation.py       Task 10: feature-group ablation
  scenario.py       Task 11: targeting curve + intervention what-ifs + ROI
  plots.py          shared chart style
run_pipeline.py     orchestrates everything
outputs/            all result tables (CSV), figures/, summary.json
reports/            final report (Word + PDF) and presentation (PDF)
```

## Deliverables map

| Required deliverable | Where |
|---|---|
| Reproducible Python project | `src/`, `run_pipeline.py`, `requirements.txt` |
| Data-quality & EDA notebook | `notebooks/01_data_quality_eda.ipynb`, `outputs/data_quality_log.csv` |
| Final customer-level modelling dataset | `data/processed/customer_model_dataset.csv` |
| Model comparison & validation results | `outputs/model_comparison.csv`, `outputs/validation_design.csv`, `outputs/backtest_expanding_window.csv` |
| Churn prediction file | `outputs/churn_predictions.csv` (ID, probability, predicted label, risk band, top-3 drivers) |
| Driver attribution & feature impact | `outputs/shap_feature_importance.csv`, `outputs/segment_risk.csv`, `outputs/ablation_results.csv` |
| Retention scenario analysis | `outputs/scenario_results.csv`, `outputs/scenario_targeting_curve.csv` |
| Final report | `reports/FreshBasket_Churn_Report.docx` (PDF copy: `reports/FreshBasket_Churn_Report.pdf`) |
| Presentation | `reports/FreshBasket_Churn_Presentation.pdf` |

## Method in brief

**Validation (time-based, no random split).** The churn definition is replayed at earlier cutoffs from the
activity data: features up to the cutoff, label = no purchases in the next 3 months despite earlier purchases.

| Snapshot (features up to) | Label window | Role |
|---|---|---|
| Jun-2023, Sep-2023 | Jul–Sep 2023, Oct–Dec 2023 | Train |
| Dec-2023 | Jan–Mar 2024 | Validation (model choice, decision threshold) |
| Mar-2024 | Apr–Jun 2024 (official `CHURNED`) | Test: the held-out cohort |

The final model is refit on train + validation and scored once on the test cohort. An expanding-window
backtest checks stability across all folds.

**Leakage control.** Features only use months up to the cutoff. `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED`
and `INSUFFICIENT_HISTORY_FLAG` in the label table are computed over the full window (including Apr–Jun 2024)
and are never used.

**Models.** Rule baseline (no purchase for ≥ k months), logistic regression, random forest, LightGBM with
and without class weights. **Final: LightGBM (unweighted) + decision threshold tuned for F1 on validation.**
Logistic regression is statistically tied on ranking; LightGBM was kept because its probabilities are the
best calibrated (Brier 0.027), which the scenario maths depends on.

**Class imbalance.** Handled by threshold tuning on validation (not by resampling); class weighting was tested
and changed PR-AUC by less than 0.005 while making probabilities less calibrated. Precision, recall, F1, PR-AUC, ROC-AUC, accuracy and Brier are all reported.

## Assumptions

* The official `CHURNED` label is the source of truth; 9 records that disagree with the activity data are
  kept as labelled and listed in `outputs/label_mismatches.csv`.
* 232 members had no purchase before April 2024 (120 never purchased; 112 first bought in Apr–Jun). They are
  outside the churn definition and are not scored (`outputs/cold_start_members_not_scored.csv`).
* Spend errors are repaired using `TOTAL_SPEND = TRANSACTIONS × AVG_BASKET_VALUE` (sign errors, zeros, and
  values up to 14.7× the expected amount).
* Months missing *inside* a member's history are treated as lost records; months after a member's last record
  are treated as inactive.
* "Active" = at least one purchase in the last 3 months before the cutoff.
* Scenario economics: 25% gross margin, 12-month value horizon, $5 per coupon, $8 per service call, and only
  50% of the model-implied effect treated as causal. All are in `src/scenario.py` and easy to change.

## Limitations

* Single 18-month synthetic dataset; the snapshots overlap in customers, so the folds are not fully independent.
* The model learns associations. Scenario effects are what-if estimates, not proven causal effects; an A/B
  test (holdout control group) is needed before scaling any campaign.
* 3-month horizon only; members who pause and return later would be counted as churners.
