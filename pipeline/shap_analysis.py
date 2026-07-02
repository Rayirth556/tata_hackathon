"""
tatahack3 / src / shap_analysis.py

Point 3 — Global beeswarm + per-prediction waterfall SHAP.
Generalized version of the repo's shap_analysis.py: works for any of
severson / hust / joint feature sets, and lets you pick the demo cell
explicitly (should be the HUST demo cell Rayirth runs on the ESP32, per
task doc — only usable once Triya's HUST knee labels land).

Run:
    python src/shap_analysis.py --feature_set hust --demo_cell hust_1-1 \
        --repo_dir /path/to/tata_hackathon
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from train_model import load_inputs, build_dataset  # reuse the same data logic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default="/home/claude/tata_hackathon")
    ap.add_argument("--feature_set", choices=["severson", "hust", "joint"], required=True)
    ap.add_argument("--demo_cell", default=None,
                     help="cell_id for the waterfall plot. Should be the HUST demo "
                          "cell for the ESP32 demo. Defaults to first test cell.")
    ap.add_argument("--demo_feature_source", choices=["severson", "hust"], default=None,
                     help="Use this if the demo cell lives in a different dataset than "
                          "--feature_set trains on (e.g. --feature_set joint "
                          "--demo_feature_source hust, since 'joint' trains on Severson "
                          "only but the ESP32 demo cell is a HUST cell). Only works if "
                          "all of feature_cols exist in that dataset's features file too "
                          "(true for the 9-feature 'joint' set).")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--plot_dir", default="outputs/plots")
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, f"trained_model_{args.feature_set}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found — run train_model.py --feature_set {args.feature_set} first."
        )
    bundle = joblib.load(model_path)
    reg = bundle["regressor"]
    feature_cols = bundle["feature_cols"]

    features_df, labels_df, split = load_inputs(args.repo_dir, args.feature_set)
    data, feature_cols, threshold = build_dataset(features_df, labels_df, split, args.feature_set)

    train_pool_ids = set()
    for fold in split["folds"].values():
        train_pool_ids.update(fold["train"])
        train_pool_ids.update(fold["val"])
    test_df = data[~data["cell_id"].isin(train_pool_ids)]
    if test_df.empty:
        print("[WARN] no held-out secondary test cells for this feature_set — "
              "using train pool for SHAP (fine for explainability demo, "
              "just don't quote these as generalization numbers).")
        test_df = data

    X_test = test_df[feature_cols]

    # ---- Global SHAP (TreeExplainer, per task doc — not feature_importances_) ----
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer(X_test)

    os.makedirs(args.plot_dir, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(args.plot_dir, f"shap_beeswarm_{args.feature_set}.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"[SAVED] {beeswarm_path}")

    # sanity check flagged in the task doc: if fade_slope dominates instead of
    # dV/dQ variance, flag Triya's extraction as possibly flawed
    mean_abs_shap = dict(zip(feature_cols, abs(shap_values.values).mean(axis=0)))
    top_feature = max(mean_abs_shap, key=mean_abs_shap.get)
    print(f"[INFO] top SHAP feature: {top_feature}")
    if "fade_slope" in top_feature:
        print("[FLAG] fade_slope dominates global SHAP — per task doc, this is "
              "NOT expected (dV/dQ variance should dominate). Flag to Triya.")

    # ---- Per-prediction waterfall for the demo cell ----
    demo_cell = args.demo_cell

    if args.demo_feature_source and args.demo_feature_source != args.feature_set:
        # Demo cell lives in a different dataset than what the model trained on
        # (e.g. joint model trained on Severson, but ESP32 demo cell is HUST).
        # Score it directly using the trained regressor + this feature source's columns.
        demo_features_df, _, _ = load_inputs(args.repo_dir, args.demo_feature_source)
        if demo_cell is None:
            demo_cell = demo_features_df["cell_id"].iloc[0]
            print(f"[INFO] no --demo_cell given, defaulting to {demo_cell} "
                  f"from {args.demo_feature_source}")
        row = demo_features_df[demo_features_df["cell_id"] == demo_cell]
        if row.empty:
            raise ValueError(f"demo_cell '{demo_cell}' not found in "
                              f"{args.demo_feature_source}_features.pkl")
        missing = [c for c in feature_cols if c not in row.columns]
        if missing:
            raise ValueError(f"demo_feature_source '{args.demo_feature_source}' is "
                              f"missing columns the model needs: {missing}")
        X_demo = row[feature_cols]
        demo_shap = explainer(X_demo)
        plt.figure()
        shap.plots.waterfall(demo_shap[0], show=False)
        plt.tight_layout()
        waterfall_path = os.path.join(
            args.plot_dir, f"shap_waterfall_{args.feature_set}_{demo_cell}.png")
        plt.savefig(waterfall_path, dpi=150)
        plt.close()
        print(f"[SAVED] {waterfall_path}  (demo cell scored from "
              f"{args.demo_feature_source}_features.pkl, model trained on {args.feature_set})")
        return

    if demo_cell is None:
        demo_cell = test_df["cell_id"].iloc[0]
        print(f"[INFO] no --demo_cell given, defaulting to {demo_cell}")

    if demo_cell not in test_df["cell_id"].values:
        raise ValueError(f"demo_cell '{demo_cell}' not found in {args.feature_set} test set")

    demo_idx = test_df.reset_index(drop=True).index[
        test_df.reset_index(drop=True)["cell_id"] == demo_cell
    ][0]

    plt.figure()
    shap.plots.waterfall(shap_values[demo_idx], show=False)
    plt.tight_layout()
    waterfall_path = os.path.join(args.plot_dir, f"shap_waterfall_{args.feature_set}_{demo_cell}.png")
    plt.savefig(waterfall_path, dpi=150)
    plt.close()
    print(f"[SAVED] {waterfall_path}")


if __name__ == "__main__":
    main()
