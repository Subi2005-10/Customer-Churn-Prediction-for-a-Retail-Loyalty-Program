"""Task 1: data-quality audit and cleaning.

`audit_and_clean` returns cleaned tables plus a log of every issue found and the
action taken, so the report and notebook can show exactly what changed.
"""
import numpy as np
import pandas as pd

from .config import HORIZON_MONTHS, TEST_CUTOFF


def _log(log, table, check, n, action):
    log.append({"table": table, "check": check, "rows_affected": int(n), "action": action})


def _month_index(s: pd.Series) -> pd.Series:
    return s.dt.year * 12 + s.dt.month


def clean_profile(p: pd.DataFrame, log: list) -> pd.DataFrame:
    p = p.copy()
    n = p.duplicated().sum()
    _log(log, "profile", "Exact duplicate rows", n, "Dropped")
    p = p.drop_duplicates()
    n = p["CUSTOMER_ID"].duplicated().sum()
    _log(log, "profile", "Duplicate CUSTOMER_ID with conflicting values", n, "Kept first record")
    p = p.drop_duplicates("CUSTOMER_ID")

    for col in ["CITY", "MEMBERSHIP_TIER"]:
        raw = p[col].astype(str)
        fixed = raw.str.strip().str.title()
        _log(log, "profile", f"{col}: inconsistent casing/whitespace", (raw != fixed).sum(),
             "Stripped whitespace, title-cased")
        p[col] = fixed

    _log(log, "profile", "AGE missing", p["AGE"].isna().sum(),
         "Median-imputed at model time + AGE_MISSING flag")
    p["AGE"] = pd.to_numeric(p["AGE"], errors="coerce")
    bad_age = (~p["AGE"].between(16, 100) & p["AGE"].notna()).sum()
    _log(log, "profile", "AGE outside 16-100", bad_age, "Set to missing")
    p.loc[~p["AGE"].between(16, 100), "AGE"] = np.nan

    _log(log, "profile", "GENDER missing", p["GENDER"].isna().sum(), "Labelled 'Unknown'")
    p["GENDER"] = p["GENDER"].fillna("Unknown")
    _log(log, "profile", "HOME_STORE_PRICE_TIER missing", p["HOME_STORE_PRICE_TIER"].isna().sum(),
         "Labelled 'Unknown'")
    p["HOME_STORE_PRICE_TIER"] = p["HOME_STORE_PRICE_TIER"].fillna("Unknown")
    p["MARKETING_OPT_IN"] = p["MARKETING_OPT_IN"].astype(int)
    return p.reset_index(drop=True)


def clean_activity(a: pd.DataFrame, profile_ids, log: list) -> pd.DataFrame:
    a = a.copy()
    n = a.duplicated().sum()
    _log(log, "activity", "Exact duplicate rows", n, "Dropped")
    a = a.drop_duplicates()

    # Spend errors. TOTAL_SPEND should equal TRANSACTIONS x AVG_BASKET_VALUE.
    expected = a["TRANSACTIONS"] * a["AVG_BASKET_VALUE"]
    neg = a["TOTAL_SPEND"] < 0
    _log(log, "activity", "Negative TOTAL_SPEND (sign error: |spend| = txn x basket)", neg.sum(),
         "Replaced with absolute value")
    a.loc[neg, "TOTAL_SPEND"] = a.loc[neg, "TOTAL_SPEND"].abs()

    zero_err = (a["TOTAL_SPEND"] == 0) & (a["TRANSACTIONS"] > 0)
    _log(log, "activity", "Zero TOTAL_SPEND despite transactions > 0", zero_err.sum(),
         "Recomputed as TRANSACTIONS x AVG_BASKET_VALUE")
    a.loc[zero_err, "TOTAL_SPEND"] = expected[zero_err].round(2)

    ratio = a["TOTAL_SPEND"] / expected.replace(0, np.nan)
    outlier = ratio > 1.5
    _log(log, "activity", "Outlier TOTAL_SPEND (> 1.5x txn x basket, up to 14.7x)", outlier.sum(),
         "Recomputed as TRANSACTIONS x AVG_BASKET_VALUE")
    a.loc[outlier, "TOTAL_SPEND"] = expected[outlier].round(2)

    # Duplicate customer-month keys that conflicted only through the spend sign error
    n = a.duplicated(["CUSTOMER_ID", "MONTH"]).sum()
    _log(log, "activity", "Conflicting duplicate CUSTOMER_ID+MONTH (resolved by sign fix)", n,
         "Dropped duplicate key")
    a = a.drop_duplicates(["CUSTOMER_ID", "MONTH"])

    zero_txn_cats = (a["TRANSACTIONS"] == 0) & (a["DISTINCT_CATEGORIES"] > 0)
    _log(log, "activity", "Zero-transaction months reporting DISTINCT_CATEGORIES > 0", zero_txn_cats.sum(),
         "Set DISTINCT_CATEGORIES to 0 (no purchase = no category)")
    a.loc[zero_txn_cats, "DISTINCT_CATEGORIES"] = 0

    _log(log, "activity", "Zero-transaction months (valid inactive months, kept)",
         (a["TRANSACTIONS"] == 0).sum(), "Kept: genuine inactivity is a churn signal")

    bad_pct = ~a["PROMO_TXN_PCT"].between(0, 1)
    _log(log, "activity", "PROMO_TXN_PCT outside 0-1", bad_pct.sum(), "Clipped to 0-1")
    a["PROMO_TXN_PCT"] = a["PROMO_TXN_PCT"].clip(0, 1)

    orphan = ~a["CUSTOMER_ID"].isin(profile_ids)
    _log(log, "activity", "Activity rows with no matching profile", orphan.sum(), "Dropped")
    a = a[~orphan]

    # Missing months inside a member's history (gap between consecutive records)
    a = a.sort_values(["CUSTOMER_ID", "MONTH"])
    mi = _month_index(a["MONTH"])
    gaps = (mi - mi.groupby(a["CUSTOMER_ID"]).shift() - 1).clip(lower=0)
    _log(log, "activity", "Missing months inside a member's history", gaps.sum(),
         "Treated as unrecorded (excluded from averages), counted as MISSING_MONTHS feature")
    return a.reset_index(drop=True)


def clean_label(l: pd.DataFrame, activity: pd.DataFrame, log: list) -> pd.DataFrame:
    l = l.copy()
    n = l.duplicated().sum()
    _log(log, "label", "Exact duplicate rows", n, "Dropped")
    l = l.drop_duplicates().drop_duplicates("CUSTOMER_ID")

    # Re-derive the label from activity and compare with the official one
    cut = pd.Timestamp(TEST_CUTOFF)
    end = cut + pd.DateOffset(months=HORIZON_MONTHS)
    pre = activity[activity["MONTH"] <= cut].groupby("CUSTOMER_ID")["TRANSACTIONS"].sum()
    post = activity[(activity["MONTH"] > cut) & (activity["MONTH"] <= end)].groupby("CUSTOMER_ID")["TRANSACTIONS"].sum()
    l["PRE_WINDOW_TXN"] = l["CUSTOMER_ID"].map(pre).fillna(0)
    l["LABEL_WINDOW_TXN"] = l["CUSTOMER_ID"].map(post).fillna(0)
    derived = ((l["LABEL_WINDOW_TXN"] == 0) & (l["PRE_WINDOW_TXN"] > 0)).astype(int)
    mismatch = derived != l["CHURNED"]
    _log(log, "label", "Official CHURNED disagrees with label re-derived from activity", mismatch.sum(),
         "Kept official label (source of truth); listed in label_mismatches.csv")
    l["LABEL_MISMATCH"] = mismatch.astype(int)

    no_hist = l["PRE_WINDOW_TXN"] == 0
    _log(log, "label", "No purchase history before Apr-2024 (outside churn definition / cold start)",
         no_hist.sum(), "Excluded from model test set; reported separately")
    l["NO_PRE_WINDOW_HISTORY"] = no_hist.astype(int)

    _log(log, "label", "Leaky columns (LAST_PURCHASE_DATE, MONTHS_OBSERVED, INSUFFICIENT_HISTORY_FLAG)",
         len(l), "Never used as features: computed over Apr-Jun 2024")
    return l.reset_index(drop=True)


def audit_and_clean(tables: dict):
    log = []
    profile = clean_profile(tables["profile"], log)
    activity = clean_activity(tables["activity"], set(profile["CUSTOMER_ID"]), log)
    label = clean_label(tables["label"], activity, log)
    missing_label = (~profile["CUSTOMER_ID"].isin(label["CUSTOMER_ID"])).sum()
    _log(log, "join", "Profiles without a churn label", missing_label, "Excluded from modelling")
    return {"profile": profile, "activity": activity, "label": label}, pd.DataFrame(log)
