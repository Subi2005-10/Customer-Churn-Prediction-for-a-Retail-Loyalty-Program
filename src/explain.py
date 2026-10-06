"""Task 9: churn drivers (SHAP via LightGBM's exact TreeSHAP) and segment risk comparison."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .features import CATEGORICAL
from .plots import BLUE, ORANGE, INK_2, save, hbar_labels

LABELS = {
    "RECENCY_MONTHS": "Months since last purchase", "HISTORY_MONTHS": "Months of history",
    "ACTIVE_MONTH_RATIO": "Share of months with a purchase", "MISSING_MONTHS": "Unrecorded months",
    "TXN_L1": "Transactions, last month", "TXN_L3": "Transactions/month, last 3 mo",
    "TXN_P3": "Transactions/month, prior 3 mo", "TXN_TREND": "Transaction trend (last vs prior qtr)",
    "TXN_ALL_MEAN": "Avg transactions/month (history)", "SPEND_L3": "Spend/month, last 3 mo",
    "SPEND_P3": "Spend/month, prior 3 mo", "SPEND_TREND_PCT": "Spend trend %",
    "SPEND_ALL_MEAN": "Avg spend/month (history)", "SPEND_CV": "Spend volatility",
    "BASKET_L3": "Avg basket, last 3 mo", "CATEGORIES_L3": "Categories bought, last 3 mo",
    "ZERO_TXN_MONTHS_L3": "Zero-purchase months, last 3", "PROMO_PCT_L3": "Promo share, last 3 mo",
    "PROMO_PCT_ALL": "Promo share (history)", "APP_L3": "App sessions/month, last 3 mo",
    "APP_TREND": "App session trend", "EMAIL_L3": "Emails opened/month, last 3 mo",
    "EMAIL_TREND": "Email-open trend", "COUPON_L3": "Coupons redeemed/month, last 3 mo",
    "COUPON_ALL_MEAN": "Coupons/month (history)", "TICKETS_L3": "Support tickets, last 3 mo",
    "TICKETS_L6": "Support tickets, last 6 mo", "TICKETS_TREND": "Support-ticket trend",
    "COMPLAINTS_L6": "Complaints, last 6 mo", "COMPLAINT_EVER": "Ever complained",
    "AGE": "Age", "AGE_MISSING": "Age missing", "GENDER": "Gender", "CITY": "City",
    "MEMBERSHIP_TIER": "Membership tier", "MARKETING_OPT_IN": "Marketing opt-in",
    "PREFERRED_CATEGORY": "Preferred category", "HOME_STORE_PRICE_TIER": "Home-store price tier",
    "SIGNUP_YEAR": "Signup year", "TENURE_MONTHS": "Tenure (months)",
}


def shap_values(lgbm, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions (log-odds) from LightGBM's pred_contrib."""
    Xp = lgbm._prep(X)
    contrib = lgbm.model.booster_.predict(Xp, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=lgbm.features, index=X.index)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "n/a" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return str(v)


def top_drivers(sv: pd.DataFrame, X: pd.DataFrame, k=3) -> pd.DataFrame:
    """Top-k features pushing each member TOWARDS churn (largest positive SHAP)."""
    out = []
    for i in sv.index:
        row = sv.loc[i].sort_values(ascending=False)
        pos = row[row > 0].head(k)
        d = [f"{LABELS.get(f, f)} = {_fmt(X.at[i, f])}" for f in pos.index]
        out.append(d + [""] * (k - len(d)))
    return pd.DataFrame(out, index=sv.index, columns=[f"KEY_DRIVER_{j+1}" for j in range(k)])


def fig_global_importance(sv, title, name, n=15):
    imp = sv.abs().mean().sort_values(ascending=True).tail(n)
    fig, ax = plt.subplots(figsize=(10, 0.42 * n + 1.2))
    bars = ax.barh([LABELS.get(f, f) for f in imp.index], imp.values, color=BLUE, height=0.6)
    hbar_labels(ax, bars, "{:.2f}")
    ax.set_xlabel("Mean |SHAP| (impact on churn log-odds)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return save(fig, name), imp.sort_values(ascending=False)


def fig_beeswarm(sv, X, name, n=12):
    import shap
    order = sv.abs().mean().sort_values(ascending=False).head(n).index
    Xn = X[order].copy()
    for c in Xn.columns:
        if c in CATEGORICAL:
            Xn[c] = Xn[c].astype("category").cat.codes
    shap.summary_plot(sv[order].values, Xn.astype(float), feature_names=[LABELS.get(f, f) for f in order],
                      show=False, max_display=n, plot_size=(10, 7))
    fig = plt.gcf()
    fig.axes[0].set_title("How feature values move churn risk (active members)", loc="left",
                          fontsize=14, fontweight="bold")
    return save(fig, name)


def segment_risk(test: pd.DataFrame, cols=("MEMBERSHIP_TIER", "CITY", "SIGNUP_YEAR")) -> pd.DataFrame:
    rows = []
    for col in cols:
        for scope, df in [("All members", test), ("Active members", test[test["ACTIVE_AT_CUTOFF"] == 1])]:
            g = df.groupby(col).agg(members=("CHURNED", "size"), actual_churn_rate=("CHURNED", "mean"),
                                    mean_predicted_risk=("CHURN_PROBABILITY", "mean"),
                                    high_risk_share=("RISK_BAND", lambda s: (s == "High").mean()))
            g = g.reset_index().rename(columns={col: "segment"})
            g.insert(0, "dimension", col)
            g.insert(1, "scope", scope)
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def fig_segment_risk(seg: pd.DataFrame):
    act = seg[seg["scope"] == "Active members"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 2.2]})
    for ax, dim, title in zip(axes, ["MEMBERSHIP_TIER", "SIGNUP_YEAR", "CITY"],
                              ["Membership tier", "Signup cohort", "City"]):
        d = act[act["dimension"] == dim].copy()
        if dim == "MEMBERSHIP_TIER":
            d["o"] = d["segment"].map({"Silver": 0, "Gold": 1, "Platinum": 2}); d = d.sort_values("o")
        elif dim == "CITY":
            d = d.sort_values("mean_predicted_risk", ascending=False)
        x = np.arange(len(d))
        ax.bar(x - 0.18, d["actual_churn_rate"], 0.34, color=BLUE, label="Actual churn")
        ax.bar(x + 0.18, d["mean_predicted_risk"], 0.34, color=ORANGE, label="Mean predicted risk")
        ax.set_xticks(x, d["segment"].astype(str), rotation=35 if dim == "CITY" else 0)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    axes[0].legend(loc="upper left", fontsize=10, ncol=1)
    axes[0].set_ylabel("Active members")
    fig.suptitle("Predicted vs actual churn by segment (active members, Apr-Jun 2024)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    return save(fig, "segment_risk_active_members")
