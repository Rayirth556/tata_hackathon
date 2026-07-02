"""
tatahack3 / src / train_model.py  (v2 — aligned to Rayirth's repo)

Step 2 (Week 3-4) — Model Training & Tuning

Resolved (previously flagged, now settled by tata_hackathon repo data):
  1. Split: data/train_test_split.json — 5-fold CV over a 50-cell train pool
     + 9-cell held-out secondary test. NOT positional 41/43/40.
  2. Classification threshold: split['classification_threshold'] = 464 cycles
     (not 500 — that was a placeholder from the task doc example).
  3. Feature sets (per Rayirth, 2026-07-01):
       - severson : all 17 features (temp included)
       - hust     : all 17 features, temp columns NaN -> dropped -> 13 used
       - joint    : the 9 cross-dataset-safe features from compatibility_report.md

Run:
    python src/train_model.py --feature_set severson
    python src/train_model.py --feature_set hust
    python src/train_model.py --feature_set joint
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, roc_auc_score

N_ESTIMATORS = 50
MAX_DEPTH = 4
LEARNING_RATE = 0.1
RANDOM_STATE = 42

FEATURE_SETS = {
    "joint": [
        "QD_100", "IR_diff", "dVdQ_var_10", "dVdQ_var_100", "dVdQ_var_diff",
        "I_var_diff", "chargetime_s_mean_2to6", "fade_slope", "fade_intercept",
    ],
    # severson/hust resolved dynamically from the loaded feature matrix columns
    # (all 17 for severson; all non-temp cols for hust, since temp is NaN there)
}


def load_inputs(repo_dir, feature_set):
    stem = "hust_features" if feature_set == "hust" else "severson_features"
    pkl_path = os.path.join(repo_dir, "data", f"{stem}.pkl")
    csv_path = os.path.join(repo_dir, "data", f"{stem}.csv")
    features_path = pkl_path if os.path.exists(pkl_path) else csv_path

    labels_path = os.path.join(repo_dir, "data", "knee_labels.csv")
    split_path = os.path.join(repo_dir, "data", "train_test_split.json")

    for p, desc in [(features_path, "features"), (labels_path, "knee labels"),
                     (split_path, "train/test split")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {desc} file: {p}")

    features_df = (pd.read_pickle(features_path) if features_path.endswith(".pkl")
                   else pd.read_csv(features_path))
    labels_df = pd.read_csv(labels_path)
    with open(split_path) as f:
        split = json.load(f)
    return features_df, labels_df, split


def resolve_feature_cols(feature_set, available_cols):
    if feature_set == "joint":
        cols = FEATURE_SETS["joint"]
        missing = [c for c in cols if c not in available_cols]
        if missing:
            raise ValueError(f"Joint feature set missing columns: {missing}")
        return cols
    if feature_set == "severson":
        return [c for c in available_cols if c not in ("cell_id", "knee_cycle", "knee_early") and not c.startswith("T")]
    if feature_set == "hust":
        # temp cols are NaN for HUST -> drop them explicitly rather than
        # feeding NaNs to XGBoost
        return [c for c in available_cols
                if c not in ("cell_id", "knee_cycle", "knee_early") and not c.startswith("T")]
    raise ValueError(feature_set)


def build_dataset(features_df, labels_df, split, feature_set):
    valid_labels = labels_df[labels_df["has_knee"]]
    data = features_df.merge(valid_labels[["cell_id", "knee_cycle"]], on="cell_id")

    threshold = split["classification_threshold"]  # 464, from repo, not hardcoded 500
    data["knee_early"] = (data["knee_cycle"] < threshold).astype(int)

    feature_cols = resolve_feature_cols(feature_set, data.columns)
    print(f"[{feature_set}] {len(feature_cols)} features: {feature_cols}")
    print(f"[{feature_set}] {len(data)} labeled cells (has_knee=True), "
          f"classification threshold = {threshold} cycles")
    if len(data) == 0:
        raise ValueError(
            f"0 cells matched for feature_set='{feature_set}'. "
            f"knee_labels.csv only contains Severson cell_ids in this repo right now "
            f"(no HUST knee labels yet) — HUST-only training is blocked until "
            f"someone runs knee_labeling.py against HUST's raw cycle data too. "
            f"Flag this to the team before proceeding."
        )
    return data, feature_cols, threshold


def get_cv_folds(data, split):
    """Yield (fold_id, train_df, val_df) using the repo's 5-fold split."""
    for fold_id, fold in split["folds"].items():
        train_df = data[data["cell_id"].isin(fold["train"])]
        val_df = data[data["cell_id"].isin(fold["val"])]
        yield fold_id, train_df, val_df


def get_secondary_test(data, split):
    # secondary test cells = cells not in the train pool at all
    train_pool = set()
    for fold in split["folds"].values():
        train_pool.update(fold["train"])
        train_pool.update(fold["val"])
    return data[~data["cell_id"].isin(train_pool)]


def train_one(X_train, y_reg, y_clf):
    reg = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE,
                            objective="reg:squarederror", random_state=RANDOM_STATE)
    reg.fit(X_train, y_reg)

    clf = xgb.XGBClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                             learning_rate=LEARNING_RATE,
                             objective="binary:logistic", eval_metric="auc",
                             random_state=RANDOM_STATE)
    clf.fit(X_train, y_clf)
    return reg, clf


def main():
    default_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default=default_repo_dir)
    ap.add_argument("--feature_set", choices=["severson", "hust", "joint"], required=True)
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    features_df, labels_df, split = load_inputs(args.repo_dir, args.feature_set)
    data, feature_cols, threshold = build_dataset(features_df, labels_df, split, args.feature_set)

    # ---- 5-fold CV over the train pool (honest generalization estimate) ----
    fold_maes, fold_aurocs = [], []
    for fold_id, train_df, val_df in get_cv_folds(data, split):
        if val_df.empty or train_df.empty:
            continue
        reg, clf = train_one(train_df[feature_cols], train_df["knee_cycle"], train_df["knee_early"])
        mae = mean_absolute_error(val_df["knee_cycle"], reg.predict(val_df[feature_cols]))
        y_val_clf = val_df["knee_early"]
        if y_val_clf.nunique() > 1:
            auroc = roc_auc_score(y_val_clf, clf.predict_proba(val_df[feature_cols])[:, 1])
            fold_aurocs.append(auroc)
        else:
            auroc = float("nan")
        fold_maes.append(mae)
        print(f"[fold {fold_id}] MAE={mae:.2f} cycles  AUROC={auroc:.4f}  "
              f"(train={len(train_df)}, val={len(val_df)})")

    print(f"\n[CV summary] mean MAE={np.mean(fold_maes):.2f} cycles  "
          f"mean AUROC={np.nanmean(fold_aurocs):.4f}")

    # ---- Final model: fit on FULL train pool, held out on secondary test ----
    train_pool_ids = set()
    for fold in split["folds"].values():
        train_pool_ids.update(fold["train"])
        train_pool_ids.update(fold["val"])
    train_pool = data[data["cell_id"].isin(train_pool_ids)]
    secondary_test = get_secondary_test(data, split)

    reg, clf = train_one(train_pool[feature_cols], train_pool["knee_cycle"], train_pool["knee_early"])

    if not secondary_test.empty:
        sec_mae = mean_absolute_error(secondary_test["knee_cycle"],
                                       reg.predict(secondary_test[feature_cols]))
        y_sec = secondary_test["knee_early"]
        sec_auroc = (roc_auc_score(y_sec, clf.predict_proba(secondary_test[feature_cols])[:, 1])
                     if y_sec.nunique() > 1 else float("nan"))
        print(f"\n[secondary_test] n={len(secondary_test)}  MAE={sec_mae:.2f} cycles  "
              f"AUROC={sec_auroc:.4f}")
    else:
        sec_auroc = float("nan")
        print("\n[secondary_test] no held-out cells found in this feature set's data")

    # ---- Plots (PS.txt requirement: scatter + error histogram, not mean alone) ----
    if not secondary_test.empty:
        plot_dir = os.path.join("outputs", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        y_true = secondary_test["knee_cycle"].values
        y_pred = reg.predict(secondary_test[feature_cols])
        errors = y_pred - y_true

        fig, ax = plt.subplots(figsize=(6, 6))
        lims = [min(y_true.min(), y_pred.min()) - 20, max(y_true.max(), y_pred.max()) + 20]
        ax.plot(lims, lims, "k--", linewidth=1, label="45° reference")
        fast = y_true < threshold
        ax.scatter(y_true[fast], y_pred[fast], c="crimson", label="fast degraders (early knee)")
        ax.scatter(y_true[~fast], y_pred[~fast], c="steelblue", label="slow degraders (late knee)")
        ax.set_xlabel("Actual knee cycle"); ax.set_ylabel("Predicted knee cycle")
        ax.set_title(f"Prediction error — {args.feature_set} secondary test (n={len(secondary_test)})")
        ax.legend(fontsize=9); ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        scatter_path = os.path.join(plot_dir, f"scatter_{args.feature_set}.png")
        fig.savefig(scatter_path, dpi=150); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(errors, bins=min(15, len(errors)), color="steelblue", edgecolor="black")
        ax2.axvline(0, color="k", linestyle="--", linewidth=1)
        ax2.set_xlabel("Predicted − Actual (cycles)"); ax2.set_ylabel("Count of cells")
        ax2.set_title(f"Error distribution — {args.feature_set} secondary test")
        fig2.tight_layout()
        hist_path = os.path.join(plot_dir, f"error_hist_{args.feature_set}.png")
        fig2.savefig(hist_path, dpi=150); plt.close(fig2)
        print(f"[PLOTS] saved {scatter_path} and {hist_path}")
    else:
        print("[PLOTS] skipped — no secondary test cells for this feature set")

    fallback_ref_auroc = np.nanmean(fold_aurocs)  # use CV mean as the primary gate metric
    if fallback_ref_auroc < 0.82:
        print(f"\n[FALLBACK TRIGGERED] CV mean AUROC {fallback_ref_auroc:.4f} < 0.82 "
              f"(PS.txt requirement) -> LSTM fallback required. See src/lstm_fallback.py. "
              f"'{args.feature_set}' XGBoost model NOT cleared for handoff yet.")
    else:
        print(f"\n[OK] CV mean AUROC {fallback_ref_auroc:.4f} >= 0.82. Model clears the gate.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Option A: separate raw estimator files for micromlgen.port() on Rayirth's side,
    # plus one combined reference copy with metadata.
    reg_path = os.path.join(args.out_dir, f"trained_model_reg_{args.feature_set}.pkl")
    clf_path = os.path.join(args.out_dir, f"trained_model_clf_{args.feature_set}.pkl")
    out_path = os.path.join(args.out_dir, f"trained_model_{args.feature_set}.pkl")

    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)
    joblib.dump({
        "regressor": reg,
        "classifier": clf,
        "feature_cols": feature_cols,
        "feature_set": args.feature_set,
        "classification_threshold": threshold,
        "cv_mean_mae": float(np.mean(fold_maes)),
        "cv_mean_auroc": float(fallback_ref_auroc),
        "secondary_test_auroc": float(sec_auroc),
        "xgboost_version": xgb.__version__,
    }, out_path)
    print(f"\n[SAVED] {reg_path}  (raw XGBRegressor -> micromlgen.port(reg))")
    print(f"[SAVED] {clf_path}  (raw XGBClassifier -> micromlgen.port(clf))")
    print(f"[SAVED] {out_path}  (combined reference copy, not for direct porting)")


if __name__ == "__main__":
    main()
