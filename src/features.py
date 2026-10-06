"""Tasks 2-3: join the three tables and build leakage-safe customer-level features.

`build_snapshot(cutoff)` produces one row per customer using ONLY activity months
<= cutoff. The label is "zero transactions in the 3 months after the cutoff despite
purchase history on or before it" -- the same definition as the official CHURNED
column. For the official cutoff (2024-03) the official label is used instead.

Monthly grid rules
- Months between a member's first and last record with no row = unrecorded (NaN):
  activity around these gaps is normal, so they look like data loss, not inactivity.
- Months after a member's last record up to the cutoff = inactive (0): churners'
  records simply stop, so trailing absence means no activity.
"""
import numpy as np
import pandas as pd

from .config import HORIZON_MONTHS, TEST_CUTOFF

METRICS = ["TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES",
           "PROMO_TXN_PCT", "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED",
           "SUPPORT_TICKETS", "COMPLAINT_FLAG"]

CATEGORICAL = ["GENDER", "CITY", "MEMBERSHIP_TIER", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"]

FEATURE_GROUPS = {
    "demographic": ["AGE", "AGE_MISSING", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
                    "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER", "SIGNUP_YEAR", "TENURE_MONTHS"],
    "behavioural": ["RECENCY_MONTHS", "HISTORY_MONTHS", "ACTIVE_MONTH_RATIO", "MISSING_MONTHS",
                    "TXN_L1", "TXN_L3", "TXN_P3", "TXN_TREND", "TXN_ALL_MEAN",
                    "SPEND_L3", "SPEND_P3", "SPEND_TREND_PCT", "SPEND_ALL_MEAN", "SPEND_CV",
                    "BASKET_L3", "CATEGORIES_L3", "ZERO_TXN_MONTHS_L3"],
    "engagement": ["PROMO_PCT_L3", "PROMO_PCT_ALL", "APP_L3", "APP_TREND", "EMAIL_L3", "EMAIL_TREND",
                   "COUPON_L3", "COUPON_ALL_MEAN"],
    "support": ["TICKETS_L3", "TICKETS_L6", "TICKETS_TREND", "COMPLAINTS_L6", "COMPLAINT_EVER"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _mi(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month


def _wide(activity: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Return {metric: DataFrame[customer x lag]} where lag 0 = cutoff month."""
    a = activity[activity["MONTH"] <= cutoff].copy()
    a["LAG"] = _mi(cutoff) - (a["MONTH"].dt.year * 12 + a["MONTH"].dt.month)
    first = a.groupby("CUSTOMER_ID")["LAG"].max()   # oldest record
    last = a.groupby("CUSTOMER_ID")["LAG"].min()    # most recent record
    lags = range(0, int(first.max()) + 1)
    out = {}
    for m in METRICS:
        w = a.pivot(index="CUSTOMER_ID", columns="LAG", values=m).reindex(columns=lags)
        lag_arr = np.array(list(lags))[None, :]
        trailing = lag_arr < last.reindex(w.index).values[:, None]   # after last record -> inactive
        before_first = lag_arr > first.reindex(w.index).values[:, None]
        vals = w.values.astype(float)
        vals[np.isnan(vals) & trailing] = 0.0
        vals[before_first] = np.nan                                 # not yet a member / no data
        out[m] = pd.DataFrame(vals, index=w.index, columns=list(lags))
    out["_first"], out["_last"] = first, last
    return out


def _win(df, start, end, how="mean"):
    cols = [c for c in df.columns if start <= c <= end]
    sub = df[cols]
    return getattr(sub, how)(axis=1, skipna=True) if how != "sum" else sub.sum(axis=1, min_count=1)


def build_features(activity: pd.DataFrame, profile: pd.DataFrame, cutoff) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    W = _wide(activity, cutoff)
    T, S = W["TRANSACTIONS"], W["TOTAL_SPEND"]
    f = pd.DataFrame(index=T.index)

    bought = T.where(T > 0)
    last_buy = bought.apply(lambda r: r.first_valid_index(), axis=1)   # smallest lag with purchase
    f["RECENCY_MONTHS"] = last_buy
    f["HISTORY_MONTHS"] = W["_first"] + 1
    observed = T.notna().sum(axis=1)
    f["ACTIVE_MONTH_RATIO"] = (T > 0).sum(axis=1) / observed
    f["MISSING_MONTHS"] = f["HISTORY_MONTHS"] - observed

    f["TXN_L1"] = T[0]
    f["TXN_L3"] = _win(T, 0, 2)
    f["TXN_P3"] = _win(T, 3, 5)
    f["TXN_TREND"] = f["TXN_L3"] - f["TXN_P3"]
    f["TXN_ALL_MEAN"] = T.mean(axis=1)
    f["SPEND_L3"] = _win(S, 0, 2)
    f["SPEND_P3"] = _win(S, 3, 5)
    f["SPEND_TREND_PCT"] = (f["SPEND_L3"] - f["SPEND_P3"]) / (f["SPEND_P3"] + 1)
    f["SPEND_ALL_MEAN"] = S.mean(axis=1)
    f["SPEND_CV"] = S.std(axis=1) / (S.mean(axis=1) + 1)
    txn3 = _win(T, 0, 2, "sum")
    f["BASKET_L3"] = (_win(S, 0, 2, "sum") / txn3.replace(0, np.nan))
    f["CATEGORIES_L3"] = _win(W["DISTINCT_CATEGORIES"], 0, 2)
    f["ZERO_TXN_MONTHS_L3"] = (T[[c for c in T.columns if c <= 2]] == 0).sum(axis=1)

    f["PROMO_PCT_L3"] = _win(W["PROMO_TXN_PCT"].where(T > 0), 0, 2)
    f["PROMO_PCT_ALL"] = W["PROMO_TXN_PCT"].where(T > 0).mean(axis=1)
    for name, m in [("APP", "APP_SESSIONS"), ("EMAIL", "EMAILS_OPENED")]:
        f[f"{name}_L3"] = _win(W[m], 0, 2)
        f[f"{name}_TREND"] = f[f"{name}_L3"] - _win(W[m], 3, 5)
    f["COUPON_L3"] = _win(W["COUPONS_REDEEMED"], 0, 2)
    f["COUPON_ALL_MEAN"] = W["COUPONS_REDEEMED"].mean(axis=1)

    tk = W["SUPPORT_TICKETS"]
    f["TICKETS_L3"] = _win(tk, 0, 2, "sum")
    f["TICKETS_L6"] = _win(tk, 0, 5, "sum")
    f["TICKETS_TREND"] = f["TICKETS_L3"] - _win(tk, 3, 5, "sum")
    f["COMPLAINTS_L6"] = _win(W["COMPLAINT_FLAG"], 0, 5, "sum")
    f["COMPLAINT_EVER"] = (W["COMPLAINT_FLAG"].max(axis=1) > 0).astype(int)

    p = profile.set_index("CUSTOMER_ID")
    f = f.join(p[["AGE", "GENDER", "CITY", "MEMBERSHIP_TIER", "MARKETING_OPT_IN",
                  "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER", "SIGNUP_DATE"]], how="inner")
    f["AGE_MISSING"] = f["AGE"].isna().astype(int)
    f["SIGNUP_YEAR"] = f["SIGNUP_DATE"].dt.year
    f["TENURE_MONTHS"] = (_mi(cutoff) - (f["SIGNUP_DATE"].dt.year * 12 + f["SIGNUP_DATE"].dt.month))
    f = f[f["SIGNUP_DATE"] <= cutoff + pd.offsets.MonthEnd(0)]
    # Eligible population: purchase history on or before the cutoff (churn definition)
    f = f[f["RECENCY_MONTHS"].notna()]
    f["ACTIVE_AT_CUTOFF"] = (f["RECENCY_MONTHS"] <= 2).astype(int)
    f.index.name = "CUSTOMER_ID"
    return f


def derive_label(activity: pd.DataFrame, cutoff) -> pd.Series:
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.DateOffset(months=HORIZON_MONTHS)
    win = activity[(activity["MONTH"] > cutoff) & (activity["MONTH"] <= end)]
    return win.groupby("CUSTOMER_ID")["TRANSACTIONS"].sum()


def build_snapshot(clean: dict, cutoff) -> pd.DataFrame:
    """Features at `cutoff` + churn label for the following 3 months."""
    f = build_features(clean["activity"], clean["profile"], cutoff)
    if pd.Timestamp(cutoff) == pd.Timestamp(TEST_CUTOFF):
        lab = clean["label"].set_index("CUSTOMER_ID")
        f = f.join(lab[["CHURNED", "LABEL_MISMATCH"]], how="inner")
    else:
        post = derive_label(clean["activity"], cutoff)
        f["CHURNED"] = (f.index.map(post).fillna(0) == 0).astype(int)
        f["LABEL_MISMATCH"] = 0
    f["SNAPSHOT"] = pd.Timestamp(cutoff).strftime("%Y-%m")
    return f.reset_index()
