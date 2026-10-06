"""Task 10: feature-group ablation. Does engagement / support / behaviour add accuracy over
demographics alone? Same out-of-time protocol as the main model (train <= Dec-2023
snapshots, test on the official Apr-Jun 2024 label)."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURE_GROUPS as G
from .modeling import LGBM, best_f1_threshold, scores
from .plots import BLUE, ORANGE, save

VARIANTS = {
    "Demographics only": G["demographic"],
    "Demographics + engagement": G["demographic"] + G["engagement"],
    "Demographics + support": G["demographic"] + G["support"],
    "Demographics + engagement + support": G["demographic"] + G["engagement"] + G["support"],
    "Demographics + behavioural": G["demographic"] + G["behavioural"],
    "All features": G["demographic"] + G["behavioural"] + G["engagement"] + G["support"],
    "All minus engagement": G["demographic"] + G["behavioural"] + G["support"],
    "All minus support": G["demographic"] + G["behavioural"] + G["engagement"],
}


def run_ablation(train, valid, test):
    rows = []
    act = (test["ACTIVE_AT_CUTOFF"] == 1).values
    for name, feats in VARIANTS.items():
        thr = best_f1_threshold(valid["CHURNED"], LGBM(feats).fit(train, train["CHURNED"]).predict_proba(valid)[:, 1])
        m = LGBM(feats).fit(pd.concat([train, valid]), pd.concat([train, valid])["CHURNED"])
        p = m.predict_proba(test)[:, 1]
        s = scores(test["CHURNED"], p, thr)
        rows.append({"variant": name, "n_features": len(feats),
                     "pr_auc_all": s["pr_auc"], "roc_auc_all": s["roc_auc"], "f1_all": s["f1"],
                     "pr_auc_active": average_precision_score(test["CHURNED"][act], p[act]),
                     "roc_auc_active": roc_auc_score(test["CHURNED"][act], p[act])})
    return pd.DataFrame(rows)


def fig_ablation(df):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = range(len(d))
    ax.barh([i + 0.2 for i in y], d["pr_auc_all"], 0.38, color=BLUE, label="All members")
    ax.barh([i - 0.2 for i in y], d["pr_auc_active"], 0.38, color=ORANGE, label="Active members")
    for i, (a, b) in enumerate(zip(d["pr_auc_all"], d["pr_auc_active"])):
        ax.text(a + 0.01, i + 0.2, f"{a:.2f}", va="center", fontsize=10)
        ax.text(b + 0.01, i - 0.2, f"{b:.2f}", va="center", fontsize=10)
    ax.set_yticks(list(y), d["variant"])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("PR-AUC on held-out Apr-Jun 2024 label (higher is better)")
    ax.set_title("Ablation: what each feature group adds")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", visible=False)
    return save(fig, "ablation_feature_groups")
