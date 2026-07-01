"""
generate_plots.py — Generate presentation-ready performance and error plots.

Generates:
  1. Predicted vs. Actual scatter plot (plots/pred_vs_actual.png).
  2. Error distribution histogram (plots/error_histogram.png).
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
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
    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(SEVERSON_FEATURES_PATH):
        print(f"⚠️ Features file {SEVERSON_FEATURES_PATH} does not exist yet. Cannot run generate_plots.py.")
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
    
    threshold = split['classification_threshold']
    data['knee_early'] = (data['knee_cycle'] < threshold).astype(int)

    all_features = [col for col in data.columns if col not in ['cell_id', 'knee_cycle', 'knee_early']]
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
    
    # Predict values
    X_train = data[data['cell_id'].isin(train_pool_cells)][no_temp_features]
    y_train = data[data['cell_id'].isin(train_pool_cells)]['knee_cycle']
    
    X_test = data[data['cell_id'].isin(secondary_test_cells)][no_temp_features]
    y_test = data[data['cell_id'].isin(secondary_test_cells)]['knee_cycle']
    
    if X_test.empty:
        print("  ⚠️ Test set is empty, generating plots on train pool data.")
        X_test = X_train
        y_test = y_train
        test_early = data[data['cell_id'].isin(train_pool_cells)]['knee_early']
    else:
        test_early = data[data['cell_id'].isin(secondary_test_cells)]['knee_early']

    preds = reg_model.predict(X_test)
    errors = preds - y_test

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 1. Predicted vs Actual scatter plot ────────────────────────────────
    print("Generating Predicted vs. Actual scatter plot...")
    plt.figure(figsize=(9, 8))
    
    # Color-code by early vs late degraders
    early_mask = test_early == 1
    late_mask = test_early == 0
    
    plt.scatter(y_test[early_mask], preds[early_mask], color='#E53935', s=60, label=f'Early Knee (<{threshold} cycles)', alpha=0.8, edgecolors='k')
    plt.scatter(y_test[late_mask], preds[late_mask], color='#43A047', s=60, label=f'Late Knee (≥{threshold} cycles)', alpha=0.8, edgecolors='k')
    
    # 45-degree reference line
    min_val = min(y_test.min(), preds.min()) - 20
    max_val = max(y_test.max(), preds.max()) + 20
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, label='Perfect Prediction')
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel('Actual Knee Point Cycle', fontsize=12)
    plt.ylabel('Predicted Knee Point Cycle', fontsize=12)
    plt.title('Predicted vs. Actual Knee Point (Secondary Test Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_path = os.path.join(PLOT_DIR, 'pred_vs_actual.png')
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"  Saved scatter plot to {scatter_path}")

    # ── 2. Error Distribution Histogram ────────────────────────────────────
    print("Generating Error Distribution Histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=12, color='#4CAF50', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='Zero Error')
    
    plt.xlabel('Prediction Error (Predicted - Actual) in Cycles', fontsize=12)
    plt.ylabel('Count of Cells', fontsize=12)
    plt.title('Prediction Error Distribution (Secondary Test Set)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    hist_path = os.path.join(PLOT_DIR, 'error_histogram.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Saved error histogram to {hist_path}")


if __name__ == '__main__':
    main()
