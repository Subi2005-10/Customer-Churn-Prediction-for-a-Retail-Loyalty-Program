"""Task 11: retention scenarios.

Two complementary estimates:
1. Targeting efficiency (measured, not simulated): if retention budget only covers
   X% of active members, how many actual Apr-Jun churners does model ranking reach
   compared with today's untargeted approach?
2. Intervention what-if (model-implied, correlational): change the features an
   intervention plausibly moves, re-score, and compare expected churners. Because the
   model learned associations, not causes, the model-implied reduction is treated as an
   UPPER BOUND and scaled by an assumed effectiveness rate in the ROI table.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots import BLUE, ORANGE, AQUA, INK_2, save

ASSUMPTIONS = {
    "gross_margin": 0.25,            # retailer gross margin on member spend
    "value_horizon_months": 12,      # months of spend protected by retaining a member
    "coupon_cost": 5.0,              # USD per coupon issued (face value x redemption)
    "outreach_cost": 8.0,            # USD per proactive service call
    "effectiveness": 0.5,            # share of the model-implied reduction assumed causal
}


def targeting_curve(test, prob_col="CHURN_PROBABILITY"):
    act = test[test["ACTIVE_AT_CUTOFF"] == 1].sort_values(prob_col, ascending=False)
    total = act["CHURNED"].sum()
    rows = []
    for frac in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        n = int(round(len(act) * frac))
        hit = act.head(n)["CHURNED"].sum()
        rows.append({"share_contacted": frac, "members_contacted": n, "churners_reached": int(hit),
                     "share_of_churners_reached": hit / total, "precision": hit / n,
                     "random_targeting_reach": frac})
    return pd.DataFrame(rows)


def _apply(df, kind):
    d = df.copy()
    if kind == "outreach":   # proactive service recovery resolves the open support friction
        d["TICKETS_TREND"] = d["TICKETS_TREND"] - d["TICKETS_L3"]
        d["TICKETS_L6"] = d["TICKETS_L6"] - d["TICKETS_L3"]
        d["TICKETS_L3"] = 0
        d["COMPLAINTS_L6"] = 0
    elif kind == "coupon":   # one extra redeemed coupon per month, slightly higher promo share
        d["COUPON_L3"] = d["COUPON_L3"] + 1
        d["COUPON_ALL_MEAN"] = d["COUPON_ALL_MEAN"] + 1 / d["HISTORY_MONTHS"].clip(lower=1) * 3
        d["PROMO_PCT_L3"] = (d["PROMO_PCT_L3"] + 0.10).clip(upper=1)
    return d


def run_scenarios(model, test, threshold):
    act = test[test["ACTIVE_AT_CUTOFF"] == 1].copy()
    p0 = act["CHURN_PROBABILITY"]
    annual_value = act["SPEND_L3"].mean() * ASSUMPTIONS["value_horizon_months"] * ASSUMPTIONS["gross_margin"]
    plans = {
        "A. Service outreach: active members with 1+ ticket in Jan-Mar and risk >= threshold":
            ("outreach", (act["TICKETS_L3"] >= 1) & (p0 >= threshold), ASSUMPTIONS["outreach_cost"]),
        "B. Targeted coupon: top 20% risk among active members":
            ("coupon", p0 >= p0.quantile(0.8), ASSUMPTIONS["coupon_cost"]),
        "C. Blanket coupon: all active members (current practice)":
            ("coupon", pd.Series(True, index=act.index), ASSUMPTIONS["coupon_cost"]),
    }
    rows = []
    for name, (kind, mask, unit_cost) in plans.items():
        tgt = act[mask]
        p1 = model.predict_proba(_apply(tgt, kind))[:, 1]
        base = tgt["CHURN_PROBABILITY"].sum()
        after = p1.sum()
        implied = base - after
        realistic = implied * ASSUMPTIONS["effectiveness"]
        cost = len(tgt) * unit_cost
        rows.append({
            "scenario": name, "members_targeted": len(tgt),
            "actual_churners_in_target": int(tgt["CHURNED"].sum()),
            "expected_churners_before": base, "expected_churners_after_model": after,
            "model_implied_churners_avoided": implied,
            "relative_risk_reduction": implied / base if base else np.nan,
            "churners_avoided_at_assumed_effectiveness": realistic,
            "campaign_cost_usd": cost, "value_protected_usd": realistic * annual_value,
            "net_value_usd": realistic * annual_value - cost,
            "cost_per_member_saved_usd": cost / realistic if realistic > 0 else np.nan,
        })
    return pd.DataFrame(rows), annual_value


def fig_targeting(curve):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [0] + list(curve["share_contacted"])
    ax.plot(x, [0] + list(curve["share_of_churners_reached"]), color=BLUE, marker="o", ms=8,
            label="Ranked by model risk")
    ax.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1.5, label="Untargeted / random")
    r = curve.set_index("share_contacted").loc[0.2]
    ax.annotate(f"Top 20% contacted\nreaches {r['share_of_churners_reached']:.0%} of churners",
                (0.2, r["share_of_churners_reached"]), xytext=(0.32, 0.45), fontsize=12,
                arrowprops=dict(arrowstyle="->", color=INK_2))
    ax.set_xlabel("Share of active members contacted (highest risk first)")
    ax.set_ylabel("Share of actual Apr-Jun churners reached")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Targeting efficiency on held-out active members")
    ax.legend(loc="lower right")
    return save(fig, "scenario_targeting_curve")


def fig_scenarios(res):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    labels = [s.split(":")[0] for s in res["scenario"]]
    y = np.arange(len(res))
    ax.barh(y, res["cost_per_member_saved_usd"], color=[AQUA, BLUE, ORANGE], height=0.55)
    for i, (c, m, n) in enumerate(zip(res["cost_per_member_saved_usd"], res["members_targeted"],
                                       res["churners_avoided_at_assumed_effectiveness"])):
        ax.text(c, i, f"  ${c:,.0f} per member saved  ({m} contacted, ~{n:.0f} saved)", va="center", fontsize=11)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, res["cost_per_member_saved_usd"].max() * 1.9)
    ax.set_xlabel("Campaign cost per member retained (USD, lower is better)")
    ax.set_title("Retention scenarios: cost per member retained")
    ax.grid(axis="y", visible=False)
    return save(fig, "scenario_cost_per_saved_member")
