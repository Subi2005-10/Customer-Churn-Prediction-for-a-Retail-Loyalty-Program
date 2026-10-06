"""Builds and executes notebooks/01_data_quality_eda.ipynb (run from the project root)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# FreshBasket Loyalty Churn — Data Quality & EDA\n"
       "Assessment 4, Tasks 1, 2 and 4. This notebook audits the three tables, shows every cleaning "
       "decision, joins them into the customer-level dataset and explores how engagement and support "
       "relate to churn. All logic lives in `src/`; the notebook calls it so results match the pipeline."),
    code("import sys, pandas as pd, numpy as np\n"
         "from pathlib import Path\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT))\n"
         "from IPython.display import Image, display\n"
         "from src.data_loading import load_raw\n"
         "from src.data_quality import audit_and_clean\n"
         "from src.features import build_snapshot, ALL_FEATURES, FEATURE_GROUPS\n"
         "from src import eda\n"
         "from src.config import TEST_CUTOFF, FIG_DIR\n"
         "pd.set_option('display.max_columns', 50); pd.set_option('display.width', 200)\n"
         "raw = load_raw()\n"
         "{k: v.shape for k, v in raw.items()}"),
    md("## 1. Raw tables"),
    code("for k, v in raw.items():\n    print(f'--- {k}'); display(v.head(3))"),
    md("## 2. Data-quality checks\n### 2.1 Missing values"),
    code("pd.concat({k: v.isna().sum() for k, v in raw.items()}, axis=1).fillna('').loc[lambda d: (d != 0).any(axis=1)]"),
    md("### 2.2 Duplicates"),
    code("p, a, l = raw['profile'], raw['activity'], raw['label']\n"
         "pd.Series({'profile exact dup rows': p.duplicated().sum(),\n"
         "           'activity exact dup rows': a.duplicated().sum(),\n"
         "           'activity dup CUSTOMER_ID+MONTH (after exact dups dropped)': a.drop_duplicates().duplicated(['CUSTOMER_ID','MONTH']).sum(),\n"
         "           'label exact dup rows': l.duplicated().sum()})"),
    code("# The two conflicting customer-month duplicates differ only in the sign of TOTAL_SPEND\n"
         "ad = a.drop_duplicates()\n"
         "ad[ad.duplicated(['CUSTOMER_ID','MONTH'], keep=False)].sort_values(['CUSTOMER_ID','MONTH'])"),
    md("### 2.3 Inconsistent text casing"),
    code("print(p['MEMBERSHIP_TIER'].value_counts().to_dict())\nprint(p['CITY'].value_counts().head(15).to_dict())"),
    md("### 2.4 Spend errors: negative, zero and outlier values\n"
       "`TOTAL_SPEND` should equal `TRANSACTIONS × AVG_BASKET_VALUE`. Checking that identity separates "
       "data-entry errors from genuine high spenders."),
    code("ad = ad.copy()\nad['EXPECTED'] = ad['TRANSACTIONS'] * ad['AVG_BASKET_VALUE']\n"
         "ad['RATIO'] = ad['TOTAL_SPEND'] / ad['EXPECTED'].replace(0, np.nan)\n"
         "print('Negative spend rows:', (ad.TOTAL_SPEND < 0).sum())\n"
         "print('Zero spend with transactions > 0:', ((ad.TOTAL_SPEND == 0) & (ad.TRANSACTIONS > 0)).sum())\n"
         "print('Spend > 1.5x expected (outliers):', (ad.RATIO > 1.5).sum())\n"
         "ad.loc[(ad.TOTAL_SPEND < 0) | (ad.RATIO > 1.5), ['CUSTOMER_ID','MONTH','TRANSACTIONS','AVG_BASKET_VALUE','TOTAL_SPEND','EXPECTED','RATIO']].head(10)"),
    md("### 2.5 Zero-transaction anomalies and missing months"),
    code("z = ad[ad.TRANSACTIONS == 0]\n"
         "print('Zero-transaction months:', len(z))\n"
         "print('...of which report DISTINCT_CATEGORIES > 0 (impossible):', (z.DISTINCT_CATEGORIES > 0).sum())\n"
         "s = ad.sort_values(['CUSTOMER_ID','MONTH'])\n"
         "mi = s.MONTH.dt.year*12 + s.MONTH.dt.month\n"
         "gap = (mi - mi.groupby(s.CUSTOMER_ID).shift() - 1).clip(lower=0)\n"
         "print('Missing months inside a member history:', int(gap.sum()), 'across', s.loc[gap>0,'CUSTOMER_ID'].nunique(), 'members')\n"
         "print('Avg transactions in the month after a gap:', round(s.loc[gap>0,'TRANSACTIONS'].mean(),2), 'vs overall', round(s.TRANSACTIONS.mean(),2))"),
    md("Activity right after a gap is normal, so internal gaps look like **lost records**, not inactivity: "
       "they are left as missing (excluded from averages). Months after a member's *last* record are treated "
       "as inactive, because churners' records simply stop."),
    md("### 2.6 Outlier detection: IQR method vs business-rule check\n"
       "The standard Tukey IQR rule flags values beyond 1.5 x IQR from the quartiles. It is run on the raw activity "
       "table and compared with the spend-identity check used for cleaning."),
    code("iqr_raw = eda.iqr_outliers(raw['activity'].drop_duplicates(), eda.HIST_COLS)\n"
         "iqr_raw.round(2)"),
    code("ra = raw['activity'].drop_duplicates().copy()\n"
         "q1, q3 = ra.TOTAL_SPEND.quantile([.25, .75]); hi = q3 + 1.5 * (q3 - q1)\n"
         "ra['IQR_FLAG'] = (ra.TOTAL_SPEND > hi) | (ra.TOTAL_SPEND < q1 - 1.5 * (q3 - q1))\n"
         "ra['IDENTITY_ERROR'] = (ra.TOTAL_SPEND / (ra.TRANSACTIONS * ra.AVG_BASKET_VALUE)).gt(1.5) | ra.TOTAL_SPEND.lt(0)\n"
         "pd.crosstab(ra.IQR_FLAG, ra.IDENTITY_ERROR, rownames=['IQR flags TOTAL_SPEND'], colnames=['Spend identity broken (error)'])"),
    md("The IQR rule flags about 670 spend values, but 650 of them are **genuine heavy shoppers** (many transactions x "
       "normal basket) whose spend matches the transactions x basket identity, and it still misses 20 of the 37 broken-identity rows, because most negative-sign errors fall inside the fences. "
       "So IQR is used here to **describe** skew and tails, while the identity check decides what is an **error** to fix. "
       "Heavy-but-valid values are kept: tree models are insensitive to them and logistic regression uses scaled, "
       "aggregated features."),
    md("### 2.7 Cleaning log (everything changed, with the action taken)"),
    code("clean, log = audit_and_clean(raw)\nlog"),
    md("### 2.8 Label validation and leakage\n"
       "The official `CHURNED` is re-derived from activity (no transactions Apr–Jun 2024, purchases before). "
       "`LAST_PURCHASE_DATE`, `MONTHS_OBSERVED` and `INSUFFICIENT_HISTORY_FLAG` are computed over the full window "
       "including Apr–Jun 2024, so they leak the target and are **never used as features**."),
    code("lab = clean['label']\n"
         "print(pd.crosstab(lab.CHURNED, lab.LAST_PURCHASE_DATE >= '2024-04-01', rownames=['CHURNED'], colnames=['LAST_PURCHASE_DATE in label window']))\n"
         "lab[lab.LABEL_MISMATCH == 1][['CUSTOMER_ID','CHURNED','LAST_PURCHASE_DATE','PRE_WINDOW_TXN','LABEL_WINDOW_TXN']]"),
    md("`LAST_PURCHASE_DATE` alone would reproduce the label almost perfectly (the 120 members it misses never purchased at all), which is exactly why it must be excluded."),
    md("## 3. Join: customer-level modelling dataset\n"
       "Profile + monthly activity (aggregated using only months up to March 2024) + official label. "
       "Members with no purchase before April 2024 are outside the churn definition (cold start) and are not scored."),
    code("snap = build_snapshot(clean, TEST_CUTOFF)\n"
         "print(snap.shape, '| churn rate', round(snap.CHURNED.mean(), 3))\n"
         "print('Cold-start members not scored:', int(lab.NO_PRE_WINDOW_HISTORY.sum()))\n"
         "{g: len(f) for g, f in FEATURE_GROUPS.items()}"),
    code("snap[['CUSTOMER_ID'] + ALL_FEATURES + ['ACTIVE_AT_CUTOFF','CHURNED']].head()"),
    md("## 4. Exploratory analysis"),
    md("### 4.0 Target balance"),
    code("display(Image(eda.fig_target_balance(snap)))"),
    md("Churn is the minority class (about 1 in 3 overall, under 1 in 10 among active members), so accuracy alone "
       "would be misleading. Models are judged on precision, recall, F1 and PR-AUC."),
    md("### 4.0b Univariate distributions (histograms)"),
    code("display(Image(eda.fig_histograms(clean['activity'])))\n"
         "clean['activity'][eda.HIST_COLS].describe().T.round(2)"),
    md("Spend, app sessions, emails, coupons and support tickets are **right-skewed** with many zeros; "
       "basket value is roughly bell-shaped apart from the spike at 0 from zero-purchase months. Skewed counts are summarised as rolling means and "
       "trends per member, and logistic regression standardises them."),
    md("### 4.1 Churn develops gradually"),
    code("display(Image(eda.fig_monthly_trend(clean['activity'], lab)))\n"
         "path, share = eda.fig_last_purchase(lab, clean['activity'])\ndisplay(Image(path)); print(f'Lapsed before 2024: {share:.1%}')"),
    md("About four in five churners had already stopped buying before 2024. For them, recency alone predicts the "
       "label. The useful retention problem is the **active** members (bought in Jan–Mar 2024), whose churn rate "
       "is about 10%. Results are reported for both groups."),
    code("snap.groupby('ACTIVE_AT_CUTOFF').CHURNED.agg(['size','mean'])"),
    md("### 4.2 Churn by segment (tier, cohort, city)"),
    code("path, tables = eda.fig_segments(snap)\ndisplay(Image(path))\ntables['MEMBERSHIP_TIER']"),
    md("### 4.3 Behaviour profile: churned vs retained (active members)"),
    code("act = snap[snap.ACTIVE_AT_CUTOFF == 1]\n"
         "act.groupby('CHURNED')[['TXN_L1','TXN_L3','TXN_TREND','SPEND_L3','SPEND_TREND_PCT','APP_L3','EMAIL_L3',"
         "'COUPON_L3','PROMO_PCT_L3','TICKETS_L3','COMPLAINTS_L6','TENURE_MONTHS']].mean().T.round(2)"),
    md("### 4.3b Box plots: churned vs retained (active members)"),
    code("display(Image(eda.fig_boxplots_by_churn(snap)))"),
    md("Churners buy less often, spend less, use the app and open emails less, redeem fewer coupons and raise more "
       "support tickets. Basket size, tenure and age overlap heavily between the groups, so they separate churners poorly."),
    md("### 4.4 Engagement, support experience and churn (Task 4)"),
    code("path, rel, corr = eda.engagement_support_relationship(snap)\ndisplay(Image(path))\ncorr.round(3).to_frame('Spearman with churn (active)')"),
    md("### 4.5 Feature correlation (redundancy check)"),
    code("num = [c for c in ALL_FEATURES if c not in ['GENDER','CITY','MEMBERSHIP_TIER','PREFERRED_CATEGORY','HOME_STORE_PRICE_TIER','SIGNUP_YEAR']]\n"
         "display(Image(eda.fig_correlation(snap, num)))"),
    md("## 5. Key EDA findings\n"
       "1. **Data issues fixed:** duplicate rows in all three tables, casing/whitespace in city and tier, "
       "negative/zero/outlier spend (repaired from the transactions × basket identity), impossible category counts in "
       "zero-purchase months, missing months, and missing age/gender/price tier.\n"
       "2. **Leakage:** three label-table columns encode the outcome and are excluded.\n"
       "3. **Two populations:** most churners lapsed long before the label window; active-member churn (~10%) is the "
       "actionable problem.\n"
       "4. **Engagement matters most:** low app use, no email opens and no coupon redemption go with much higher churn.\n"
       "5. **Distributions:** most activity measures are right-skewed with many zeros; IQR describes the tails, but only the spend-identity check identifies errors.\n"
       "6. **Support friction is a warning sign:** churn rises steeply with support tickets in the last quarter "
       "(about 5% with none, about 74% with three or more among active members); formal complaints roughly double risk.\n"
       "7. **Tier and city differences are small**; newer signup cohorts behave differently from 2021–2022 cohorts.\n"
       "These are associations, not causal effects."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).parent / "01_data_quality_eda.ipynb"
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(Path(__file__).parent)}})
nbf.write(nb, out)
print("Wrote", out)
