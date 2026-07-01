"""
shap_analysis.py — Compute SHAP attributions for model explainability.

Generates:
  1. Global beeswarm plot of feature importances (plots/shap_beeswarm.png).
  2. Per-prediction waterfall plot for a demo cell (plots/shap_waterfall_demo.png).

Allows specifying the cell ID via command-line arguments.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import shap
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
SEVERSON_FEATURES_PATH = os.path.join(WORKSPACE, 'data/severson_features.pkl')
KNEE_LABELS_PATH = os.path.join(WORKSPACE, 'data/knee_labels.csv')
SPLIT_PATH = os.path.join(WORKSPACE, 'data/train_test_split.json')
MODEL_REG_PATH = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
PLOT_DIR = os.path.join(WORKSPACE, 'plots')


def main():
    # ── Parse command-line args ────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="SHAP explainability analysis.")
    parser.add_argument('--cell_id', type=str, default=None,
                        help="Cell ID for the demo waterfall plot. Default: first cell in secondary test set.")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(SEVERSON_FEATURES_PATH):
        print(f"⚠️ Features file {SEVERSON_FEATURES_PATH} does not exist yet. Cannot run SHAP analysis.")
        return
    if not os.path.exists(MODEL_REG_PATH):
        print(f"⚠️ Model file {MODEL_REG_PATH} does not exist yet. Run train_model.py first.")
        return

    features_df = pd.read_pickle(SEVERSON_FEATURES_PATH)
    labels_df = pd.read_csv(KNEE_LABELS_PATH)
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)

    # Exclude cells that have no knee detected (has_knee is False)
    valid_labels = labels_df[labels_df['has_knee']]
    data = features_df.merge(valid_labels[['cell_id', 'knee_cycle']], on='cell_id')

    all_features = [col for col in data.columns if col not in ['cell_id', 'knee_cycle']]
    no_temp_features = [col for col in all_features if not col.startswith('T')]

    # Finalize train pool and secondary test set cells
    train_pool_cells = []
    for fold_id, fold_data in split['folds'].items():
        train_pool_cells.extend(fold_data['train'])
        train_pool_cells.extend(fold_data['val'])
    train_pool_cells = list(set(train_pool_cells))
    secondary_test_cells = split['secondary_test']

    # Load trained regressor model
    reg_model = joblib.load(MODEL_REG_PATH)
    
    # ── SHAP calculations ──────────────────────────────────────────────────
    print("Initializing SHAP TreeExplainer...")
    X_train = data[data['cell_id'].isin(train_pool_cells)][no_temp_features]
    X_test = data[data['cell_id'].isin(secondary_test_cells)][no_temp_features]
    
    if X_test.empty:
        print("  ⚠️ Test set is empty, using whole dataset for SHAP check.")
        X_test = data[no_temp_features]
        test_ids = data['cell_id'].tolist()
    else:
        test_ids = data[data['cell_id'].isin(secondary_test_cells)]['cell_id'].tolist()

    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer(X_test)

    # ── Global Explainability Plot ─────────────────────────────────────────
    print("Generating global SHAP beeswarm plot...")
    plt.figure(figsize=(12, 8))
    # We must call shap.plots.beeswarm on the shap_values object
    # Turn off show so we can modify titles and save via matplotlib
    shap.plots.beeswarm(shap_values, show=False)
    plt.title('Global Feature Importance (SHAP values)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    beeswarm_path = os.path.join(PLOT_DIR, 'shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"  Saved beeswarm plot to {beeswarm_path}")

    # ── Per-Prediction Waterfall Plot ──────────────────────────────────────
    # Determine the cell to analyze
    target_cell = args.cell_id
    if target_cell is None:
        target_cell = test_ids[0]
        print(f"No cell_id provided. Defaulting to first test cell: {target_cell}")
    else:
        if target_cell not in test_ids:
            if target_cell in data['cell_id'].tolist():
                # Re-calculate SHAP specifically for this cell
                X_target = data[data['cell_id'] == target_cell][no_temp_features]
                shap_values_target = explainer(X_target)
                shap_values = shap_values_target
                test_ids = [target_cell]
            else:
                raise ValueError(f"Cell ID '{target_cell}' not found in dataset.")

    target_idx = test_ids.index(target_cell)
    print(f"Generating per-prediction SHAP waterfall plot for cell {target_cell}...")
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[target_idx], show=False)
    plt.title(f'Feature Attribution for Demo Cell: {target_cell}', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    waterfall_path = os.path.join(PLOT_DIR, 'shap_waterfall_demo.png')
    plt.savefig(waterfall_path, dpi=150)
    plt.close()
    print(f"  Saved waterfall plot to {waterfall_path}")


if __name__ == '__main__':
    main()
