"""Tasks 1 & 4: exploratory analysis and the engagement/support vs churn relationship.

All relationship analysis uses the official snapshot (features up to Mar-2024,
label Apr-Jun 2024). Engagement relationships are shown for members still active
in Jan-Mar 2024, because lapsed members have zero activity on every measure and
would make any engagement metric look predictive for trivial reasons.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots import BLUE, ORANGE, INK_2, GRID, save, bar_labels, hbar_labels


def churn_by(df, col, order=None):
    g = df.groupby(col)["CHURNED"].agg(["mean", "size"]).rename(columns={"mean": "churn_rate", "size": "members"})
    if order is not None:
        g = g.reindex(order)
    return g


def fig_monthly_trend(activity, label):
    a = activity.merge(label[["CUSTOMER_ID", "CHURNED"]], on="CUSTOMER_ID")
    g = a.groupby(["MONTH", "CHURNED"])["TRANSACTIONS"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(g.index, g[0], color=BLUE, label="Retained (CHURNED = 0)")
    ax.plot(g.index, g[1], color=ORANGE, label="Churned (CHURNED = 1)")
    ax.axvspan(pd.Timestamp("2024-03-20"), g.index.max() + pd.Timedelta(days=15), color=GRID, alpha=0.6)
    ax.text(pd.Timestamp("2024-04-05"), ax.get_ylim()[1] * 0.92, "Label window\nApr-Jun 2024", color=INK_2, fontsize=10)
    ax.set_title("Churners fade gradually: average transactions per recorded member-month")
    ax.set_ylabel("Avg transactions / month")
    ax.legend(loc="lower left")
    return save(fig, "eda_monthly_transactions_by_outcome")


def fig_last_purchase(label, activity):
    lp = activity[activity["TRANSACTIONS"] > 0].groupby("CUSTOMER_ID")["MONTH"].max()
    ch = label[label["CHURNED"] == 1].copy()
    ch["LAST_Q"] = ch["CUSTOMER_ID"].map(lp).dt.to_period("Q").astype(str)
    s = ch["LAST_Q"].value_counts().sort_index()
    s = s[s.index != "NaT"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(s.index, s.values, color=[ORANGE if q == "2024Q1" else BLUE for q in s.index], width=0.6)
    bar_labels(ax, bars, "{:.0f}")
    ax.set_title("When did churned members make their last purchase?")
    ax.set_ylabel("Churned members")
    share = 1 - s.get("2024Q1", 0) / s.sum()
    ax.set_ylim(0, s.max() * 1.3)
    ax.text(0.01, 0.97, f"{share:.0%} of churners had already stopped buying before 2024 (blue)",
            transform=ax.transAxes, ha="left", va="top", fontsize=12, color=INK_2)
    return save(fig, "eda_churner_last_purchase_quarter"), share


def fig_segments(snap):
    specs = [("MEMBERSHIP_TIER", ["Silver", "Gold", "Platinum"], "Membership tier"),
             ("SIGNUP_YEAR", None, "Signup cohort (year)"),
             ("CITY", None, "City")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 2.2]})
    tables = {}
    for ax, (col, order, title) in zip(axes, specs):
        t = churn_by(snap, col, order)
        if col == "CITY":
            t = t.sort_values("churn_rate", ascending=False)
        tables[col] = t
        bars = ax.bar([str(i) for i in t.index], t["churn_rate"], color=BLUE, width=0.6)
        bar_labels(ax, bars)
        ax.axhline(snap["CHURNED"].mean(), color=INK_2, lw=1, ls="--")
        ax.set_title(title)
        ax.set_ylim(0, max(0.6, t["churn_rate"].max() * 1.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if col == "CITY":
            ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Churn rate (dashed = overall)")
    return save(fig, "eda_churn_by_segment"), tables


def _bucket_rate(df, col, bins, labels):
    b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df.groupby(b, observed=False)["CHURNED"].agg(["mean", "size"])


def engagement_support_relationship(snap):
    act = snap[snap["ACTIVE_AT_CUTOFF"] == 1]
    specs = [
        ("TICKETS_L3", [-0.1, 0, 1, 2, 99], ["0", "1", "2", "3+"], "Support tickets, Jan-Mar"),
        ("COMPLAINTS_L6", [-0.1, 0, 1, 99], ["0", "1", "2+"], "Formal complaints, last 6 mo"),
        ("APP_L3", [-0.1, 0.5, 1.5, 3, 99], ["<0.5", "0.5-1.5", "1.5-3", "3+"], "App sessions / month"),
        ("EMAIL_L3", [-0.1, 0.01, 1, 99], ["0", "0-1", "1+"], "Emails opened / month"),
        ("COUPON_L3", [-0.1, 0.01, 0.67, 99], ["0", "0-0.67", "0.67+"], "Coupons redeemed / month"),
        ("TXN_TREND", [-99, -2, -0.5, 0.5, 99], ["Down >2", "Down", "Flat", "Up"], "Transaction trend vs prior qtr"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    rows = []
    for ax, (col, bins, labels, title) in zip(axes.ravel(), specs):
        t = _bucket_rate(act, col, bins, labels)
        bars = ax.bar(labels, t["mean"].fillna(0), color=BLUE, width=0.6)
        bar_labels(ax, bars)
        ax.axhline(act["CHURNED"].mean(), color=INK_2, lw=1, ls="--")
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, max(0.3, t["mean"].max() * 1.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        for lab, (m, n) in zip(labels, t.values):
            rows.append({"feature": col, "bucket": lab, "churn_rate": m, "members": int(n)})
    fig.suptitle("Active members (bought Jan-Mar 2024): churn rate by engagement & support "
                 "(dashed = active-member average)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    path = save(fig, "eda_engagement_support_vs_churn")
    corr = act[["CHURNED", "TICKETS_L3", "TICKETS_TREND", "COMPLAINTS_L6", "APP_L3", "APP_TREND", "EMAIL_L3",
                "EMAIL_TREND", "COUPON_L3", "PROMO_PCT_L3", "TXN_TREND", "SPEND_TREND_PCT", "MARKETING_OPT_IN"]] \
        .corr(method="spearman")["CHURNED"].drop("CHURNED").sort_values()
    return path, pd.DataFrame(rows), corr


def fig_correlation(snap, cols):
    c = snap[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=9)
    ax.set_yticks(range(len(cols)), cols, fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Feature correlation (official snapshot)")
    return save(fig, "eda_feature_correlation")


# ---------------------------------------------------------------- standard EDA views
def fig_target_balance(snap):
    """Class balance: all scored members vs active members."""
    act = snap[snap["ACTIVE_AT_CUTOFF"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, df, title in [(axes[0], snap, "All scored members"), (axes[1], act, "Active members (bought Jan-Mar 2024)")]:
        counts = df["CHURNED"].value_counts().reindex([0, 1])
        bars = ax.bar(["Retained", "Churned"], counts.values, color=[BLUE, ORANGE], width=0.55)
        for b, n in zip(bars, counts.values):
            ax.annotate(f"{n:,}  ({n / counts.sum():.1%})", (b.get_x() + b.get_width() / 2, n), ha="center",
                        va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_ylabel("Members")
    fig.suptitle("Target balance: churn is a minority class, especially among active members",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    return save(fig, "eda_target_balance")


HIST_COLS = ["TRANSACTIONS", "TOTAL_SPEND", "AVG_BASKET_VALUE", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT",
             "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS"]


def fig_histograms(activity, name="eda_histograms_monthly_activity", title="Distributions of monthly activity (cleaned)"):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.ravel(), HIST_COLS):
        s = activity[col].dropna()
        discrete = s.nunique() <= 20
        if discrete:
            vc = s.value_counts().sort_index()
            ax.bar(vc.index, vc.values, color=BLUE, width=0.8)
        else:
            ax.hist(s, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.axvline(s.median(), color=ORANGE, lw=1.5, ls="--")
        ax.set_title(f"{col}\nmedian {s.median():.2f}, skew {s.skew():.2f}", fontsize=11)
        ax.set_ylabel("Member-months", fontsize=10)
    fig.suptitle(title + " (dashed = median)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    return save(fig, name)


BOX_COLS = [("TXN_L3", "Transactions/month, last 3 mo"), ("SPEND_L3", "Spend/month, last 3 mo"),
            ("BASKET_L3", "Avg basket, last 3 mo"), ("APP_L3", "App sessions/month"),
            ("EMAIL_L3", "Emails opened/month"), ("COUPON_L3", "Coupons redeemed/month"),
            ("TICKETS_L3", "Support tickets, last 3 mo"), ("TENURE_MONTHS", "Tenure (months)"),
            ("AGE", "Age")]


def fig_boxplots_by_churn(snap):
    act = snap[snap["ACTIVE_AT_CUTOFF"] == 1]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (col, label) in zip(axes.ravel(), BOX_COLS):
        data = [act.loc[act["CHURNED"] == k, col].dropna() for k in (0, 1)]
        bp = ax.boxplot(data, tick_labels=["Retained", "Churned"], patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=1.5), flierprops=dict(marker="o", ms=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_title(f"{label}\nmedian {data[0].median():.2f} vs {data[1].median():.2f}", fontsize=11)
    fig.suptitle("Active members: feature distributions by churn outcome (box = IQR, whiskers = 1.5 x IQR)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    return save(fig, "eda_boxplots_by_churn")


def iqr_outliers(df, cols):
    """Tukey IQR rule: flag values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_lo, n_hi = int((s < lo).sum()), int((s > hi).sum())
        rows.append({"column": c, "Q1": q1, "Q3": q3, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
                     "below_fence": n_lo, "above_fence": n_hi, "pct_flagged": (n_lo + n_hi) / len(s), "max": s.max()})
    return pd.DataFrame(rows)
