"""End-to-end pipeline for FreshBasket loyalty churn (Assessment 4).

    python run_pipeline.py

Writes cleaned data, the modelling dataset, model results, predictions, figures and a
summary JSON used by the report. Runtime about 1-2 minutes on a laptop.
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src import ablation, eda, explain, scenario
from src.config import (OUTPUT_DIR, PROCESSED_DIR, TEST_CUTOFF, TRAIN_CUTOFFS, VALID_CUTOFF)
from src.data_loading import load_raw
from src.data_quality import audit_and_clean
from src.features import ALL_FEATURES, CATEGORICAL, build_snapshot
from src.modeling import LGBM, model_zoo, fit_predict, threshold_for, scores, top_k_lift
from src.plots import SERIES, INK_2, save

warnings.filterwarnings("ignore")
FINAL_MODEL = "LightGBM"


def main():
    summary = {}
    # ---------------------------------------------------------------- 1. data quality
    print("1/8 Data quality & cleaning")
    raw = load_raw()
    clean, dq_log = audit_and_clean(raw)
    dq_log.to_csv(OUTPUT_DIR / "data_quality_log.csv", index=False)
    for k, v in clean.items():
        v.to_csv(PROCESSED_DIR / f"clean_{k}.csv", index=False)
    lab = clean["label"]
    lab[lab["LABEL_MISMATCH"] == 1].to_csv(OUTPUT_DIR / "label_mismatches.csv", index=False)
    lab[lab["NO_PRE_WINDOW_HISTORY"] == 1][["CUSTOMER_ID", "CHURNED"]] \
        .to_csv(OUTPUT_DIR / "cold_start_members_not_scored.csv", index=False)
    summary["raw_rows"] = {k: len(v) for k, v in raw.items()}
    summary["clean_rows"] = {k: len(v) for k, v in clean.items()}

    # ---------------------------------------------------------------- 2-3. snapshots / features
    print("2/8 Building leakage-safe snapshots")
    train = pd.concat([build_snapshot(clean, c) for c in TRAIN_CUTOFFS], ignore_index=True)
    valid = build_snapshot(clean, VALID_CUTOFF)
    test = build_snapshot(clean, TEST_CUTOFF)
    test.to_csv(PROCESSED_DIR / "customer_model_dataset.csv", index=False)
    pd.concat([train, valid, test]).to_csv(PROCESSED_DIR / "all_snapshots.csv", index=False)
    design = pd.concat([train, valid, test]).groupby("SNAPSHOT").agg(
        members=("CHURNED", "size"), churn_rate=("CHURNED", "mean"),
        active_members=("ACTIVE_AT_CUTOFF", "sum"))
    for snap, df in pd.concat([train, valid, test]).groupby("SNAPSHOT"):
        design.loc[snap, "active_churn_rate"] = df.loc[df["ACTIVE_AT_CUTOFF"] == 1, "CHURNED"].mean()
    design["role"] = ["Train", "Train", "Validation", "Test (official label)"]
    design.to_csv(OUTPUT_DIR / "validation_design.csv")
    summary["validation_design"] = design.reset_index().to_dict("records")

    # ---------------------------------------------------------------- 4. EDA
    print("3/8 EDA")
    eda.fig_monthly_trend(clean["activity"], lab)
    _, lapsed_share = eda.fig_last_purchase(lab, clean["activity"])
    _, seg_tables = eda.fig_segments(test)
    _, rel, corr = eda.engagement_support_relationship(test)
    rel.to_csv(OUTPUT_DIR / "engagement_support_vs_churn.csv", index=False)
    corr.rename("spearman_with_churn").to_csv(OUTPUT_DIR / "engagement_support_correlations.csv")
    num = [c for c in ALL_FEATURES if c not in CATEGORICAL + ["SIGNUP_YEAR"]]
    eda.fig_correlation(test, num)
    summary["churner_share_lapsed_before_2024"] = lapsed_share
    summary["segment_churn"] = {k: v.round(4).reset_index().to_dict("records") for k, v in seg_tables.items()}

    # ---------------------------------------------------------------- 5-8. models
    print("4/8 Model comparison (out-of-time)")
    F = ALL_FEATURES
    trval = pd.concat([train, valid], ignore_index=True)
    act = (test["ACTIVE_AT_CUTOFF"] == 1).values
    rows, test_probs, models, thresholds = [], {}, {}, {}
    for name, fac in model_zoo(F).items():
        m, pv = fit_predict(fac, train, valid, F)
        thr = threshold_for(m, pv, valid)
        sv = scores(valid["CHURNED"], pv, thr)
        m2, pt = fit_predict(fac, trval, test, F)
        st, sa = scores(test["CHURNED"], pt, thr), scores(test["CHURNED"][act], pt[act], thr)
        test_probs[name], models[name], thresholds[name] = pt, m2, thr
        rows.append({"model": name, "threshold": thr,
                     **{f"valid_{k}": v for k, v in sv.items() if k not in ("threshold", "n", "churn_rate")},
                     **{f"test_{k}": v for k, v in st.items() if k not in ("threshold",)},
                     **{f"active_{k}": v for k, v in sa.items() if k not in ("threshold",)},
                     "active_top10pct_lift": top_k_lift(test["CHURNED"][act].values, pt[act])})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    summary["model_comparison"] = comp.round(4).to_dict("records")

    # Expanding-window backtest for stability
    cuts = TRAIN_CUTOFFS + [VALID_CUTOFF, TEST_CUTOFF]
    snaps = {c: (build_snapshot(clean, c)) for c in cuts}
    bt = []
    for i in range(1, len(cuts)):
        tr = pd.concat([snaps[c] for c in cuts[:i]]); te = snaps[cuts[i]]
        a = (te["ACTIVE_AT_CUTOFF"] == 1).values
        for name in ["Logistic regression (balanced)", "LightGBM"]:
            _, p = fit_predict(model_zoo(F)[name], tr, te, F)
            s, sa_ = scores(te["CHURNED"], p, 0.5), scores(te["CHURNED"][a], p[a], 0.5)
            bt.append({"train_cutoffs": " + ".join(c[:7] for c in cuts[:i]), "test_cutoff": cuts[i][:7],
                       "model": name, "pr_auc_all": s["pr_auc"], "roc_auc_all": s["roc_auc"],
                       "pr_auc_active": sa_["pr_auc"], "roc_auc_active": sa_["roc_auc"]})
    bt = pd.DataFrame(bt)
    bt.to_csv(OUTPUT_DIR / "backtest_expanding_window.csv", index=False)
    summary["backtest"] = bt.round(4).to_dict("records")

    # PR curves (active members) + calibration
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for (name, p), col in zip(test_probs.items(), SERIES):
        pr, rc, _ = precision_recall_curve(test["CHURNED"][act], p[act])
        ax.plot(rc, pr, color=col, label=name, lw=2)
    ax.axhline(test["CHURNED"][act].mean(), color=INK_2, ls="--", lw=1, label="No-skill (base rate)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, active members (held-out Apr-Jun 2024)")
    ax.legend(fontsize=10, loc="lower left")
    save(fig, "model_pr_curves_active")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, col in [("LightGBM", SERIES[0]), ("LightGBM (balanced weights)", SERIES[1]),
                      ("Logistic regression (balanced)", SERIES[2])]:
        fy, fx = calibration_curve(test["CHURNED"], test_probs[name], n_bins=8, strategy="uniform")
        ax.plot(fx, fy, marker="o", ms=7, color=col, label=name)
    ax.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1)
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on held-out label")
    ax.legend(fontsize=10)
    save(fig, "model_calibration")

    # ---------------------------------------------------------------- final scoring
    print("5/8 Scoring held-out cohort with final model")
    final = models[FINAL_MODEL]
    thr = thresholds[FINAL_MODEL]
    p = test_probs[FINAL_MODEL]
    test["CHURN_PROBABILITY"] = p
    test["PREDICTED_CHURN"] = (p >= thr).astype(int)
    test["RISK_BAND"] = pd.cut(p, [-0.01, 0.2, 0.5, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    sv = explain.shap_values(final, test.set_index("CUSTOMER_ID"))
    drivers = explain.top_drivers(sv, test.set_index("CUSTOMER_ID"))
    pred = test.set_index("CUSTOMER_ID")[["CHURN_PROBABILITY", "PREDICTED_CHURN", "RISK_BAND", "ACTIVE_AT_CUTOFF",
                                          "MEMBERSHIP_TIER", "CITY", "SIGNUP_YEAR"]].join(drivers)
    pred["MEMBER_STATUS_AT_CUTOFF"] = np.where(pred.pop("ACTIVE_AT_CUTOFF") == 1, "Active (bought Jan-Mar 2024)",
                                               "Lapsed (no purchase Jan-Mar 2024)")
    pred["ACTUAL_CHURNED"] = test.set_index("CUSTOMER_ID")["CHURNED"]
    pred = pred.reset_index().sort_values("CHURN_PROBABILITY", ascending=False)
    pred["CHURN_PROBABILITY"] = pred["CHURN_PROBABILITY"].round(4)
    pred.to_csv(OUTPUT_DIR / "churn_predictions.csv", index=False)
    summary["final_model"] = FINAL_MODEL
    summary["final_threshold"] = thr
    summary["risk_band_counts"] = pred["RISK_BAND"].value_counts().to_dict()

    # ---------------------------------------------------------------- 9. drivers & segments
    print("6/8 SHAP drivers & segment risk")
    _, imp_all = explain.fig_global_importance(sv, "Top churn drivers - all members", "shap_importance_all")
    sva = sv[act]
    Xa = test.set_index("CUSTOMER_ID")[act]
    _, imp_act = explain.fig_global_importance(sva, "Top churn drivers - active members", "shap_importance_active")
    explain.fig_beeswarm(sva, Xa, "shap_beeswarm_active")
    pd.DataFrame({"mean_abs_shap_all": sv.abs().mean(), "mean_abs_shap_active": sva.abs().mean()}) \
        .sort_values("mean_abs_shap_active", ascending=False).to_csv(OUTPUT_DIR / "shap_feature_importance.csv")
    seg = explain.segment_risk(test)
    seg.to_csv(OUTPUT_DIR / "segment_risk.csv", index=False)
    explain.fig_segment_risk(seg)
    summary["shap_top_all"] = imp_all.head(10).round(4).to_dict()
    summary["shap_top_active"] = imp_act.head(10).round(4).to_dict()

    # ---------------------------------------------------------------- 10. ablation
    print("7/8 Ablation")
    ab = ablation.run_ablation(train, valid, test)
    ab.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    ablation.fig_ablation(ab)
    summary["ablation"] = ab.round(4).to_dict("records")

    # ---------------------------------------------------------------- 11. scenario
    print("8/8 Retention scenarios")
    curve = scenario.targeting_curve(test)
    curve.to_csv(OUTPUT_DIR / "scenario_targeting_curve.csv", index=False)
    scenario.fig_targeting(curve)
    res, annual_value = scenario.run_scenarios(final, test, thr)
    res.to_csv(OUTPUT_DIR / "scenario_results.csv", index=False)
    scenario.fig_scenarios(res)
    summary["targeting_curve"] = curve.round(4).to_dict("records")
    summary["scenarios"] = res.round(2).to_dict("records")
    summary["scenario_assumptions"] = {**scenario.ASSUMPTIONS, "annual_margin_per_member_usd": round(annual_value, 2)}

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Done. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
