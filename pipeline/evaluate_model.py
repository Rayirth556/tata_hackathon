"""
evaluate_model.py — Perform 5-fold cross-validation and secondary test evaluation.

Evaluates the no-temperature model:
  1. 5-fold stratified cross-validation on Batch 1 + Batch 2 train pool.
  2. Generalization evaluation on Batch 3 secondary test set.
  3. Generates error metrics, classification AUROC, and print outputs.
  4. Saves cross-validation and evaluation metrics to modeling_report.md.
"""

import pandas as pd
import numpy as np
import json
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
SEVERSON_FEATURES_PATH = os.path.join(WORKSPACE, 'data/severson_features.pkl')
KNEE_LABELS_PATH = os.path.join(WORKSPACE, 'data/knee_labels.csv')
SPLIT_PATH = os.path.join(WORKSPACE, 'data/train_test_split.json')
REPORT_PATH = os.path.join(WORKSPACE, 'modeling_report.md')
PLOT_DIR = os.path.join(WORKSPACE, 'plots')

# Constrained hyperparameters
N_ESTIMATORS = 50
MAX_DEPTH = 4
LEARNING_RATE = 0.1
RANDOM_STATE = 42


def evaluate_folds(data, split, no_temp_features):
    """Run cross-validation over the 5 folds defined in build_split.py."""
    fold_maes = []
    fold_aucs = []
    all_preds_reg = []
    all_preds_clf = []
    all_y_reg = []
    all_y_clf = []
    all_cell_ids = []

    print("\nExecuting stratified 5-fold cross-validation...")
    for fold_id in sorted(split['folds'].keys()):
        fold_data = split['folds'][fold_id]
        train_cells = fold_data['train']
        val_cells = fold_data['val']

        train_subset = data[data['cell_id'].isin(train_cells)]
        val_subset = data[data['cell_id'].isin(val_cells)]

        X_train = train_subset[no_temp_features]
        y_train_reg = train_subset['knee_cycle']
        y_train_clf = train_subset['knee_early']

        X_val = val_subset[no_temp_features]
        y_val_reg = val_subset['knee_cycle']
        y_val_clf = val_subset['knee_early']

        # Train regressor
        reg_model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            objective='reg:squarederror',
            random_state=RANDOM_STATE
        )
        reg_model.fit(X_train, y_train_reg)
        val_preds_reg = reg_model.predict(X_val)

        # Train classifier
        clf_model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=RANDOM_STATE
        )
        clf_model.fit(X_train, y_train_clf)
        val_preds_clf = clf_model.predict_proba(X_val)[:, 1]

        # Calculate fold metrics
        fold_mae = mean_absolute_error(y_val_reg, val_preds_reg)
        try:
            fold_auc = roc_auc_score(y_val_clf, val_preds_clf)
        except ValueError:
            # Handle edge case where one class is missing in val set (unlikely with stratified split)
            fold_auc = np.nan

        fold_maes.append(fold_mae)
        fold_aucs.append(fold_auc)

        # Accumulate out-of-fold predictions
        all_preds_reg.extend(val_preds_reg)
        all_preds_clf.extend(val_preds_clf)
        all_y_reg.extend(y_val_reg)
        all_y_clf.extend(y_val_clf)
        all_cell_ids.extend(val_subset['cell_id'].tolist())

        print(f"  Fold {fold_id}: MAE = {fold_mae:.2f} cycles, AUROC = {fold_auc:.4f} "
              f"(val cells={len(val_cells)})")

    cv_mae_mean = np.mean(fold_maes)
    cv_mae_std = np.std(fold_maes)
    cv_auc_mean = np.nanmean(fold_aucs)
    cv_auc_std = np.nanstd(fold_aucs)

    print(f"\nCV Aggregated Performance:")
    print(f"  Mean MAE:   {cv_mae_mean:.2f} ± {cv_mae_std:.2f} cycles")
    print(f"  Mean AUROC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    oof_df = pd.DataFrame({
        'cell_id': all_cell_ids,
        'y_reg': all_y_reg,
        'y_clf': all_y_clf,
        'pred_reg': all_preds_reg,
        'pred_clf': all_preds_clf
    })

    return cv_mae_mean, cv_mae_std, cv_auc_mean, cv_auc_std, oof_df


def evaluate_secondary(data, split, no_temp_features):
    """Evaluate on the Batch 3 secondary test set."""
    test_cells = split['secondary_test']
    test_subset = data[data['cell_id'].isin(test_cells)]
    
    if test_subset.empty:
        print("\n⚠️ No cells from Batch 3 secondary test set found in valid dataset.")
        return np.nan, np.nan, None

    # Retrieve all B1+B2 cells for training the final model
    train_pool_cells = []
    for fold_id, fold_data in split['folds'].items():
        train_pool_cells.extend(fold_data['train'])
        train_pool_cells.extend(fold_data['val'])
    train_pool_cells = list(set(train_pool_cells))
    train_subset = data[data['cell_id'].isin(train_pool_cells)]

    X_train = train_subset[no_temp_features]
    y_train_reg = train_subset['knee_cycle']
    y_train_clf = train_subset['knee_early']

    X_test = test_subset[no_temp_features]
    y_test_reg = test_subset['knee_cycle']
    y_test_clf = test_subset['knee_early']

    # Final regressor
    reg_model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    )
    reg_model.fit(X_train, y_train_reg)
    test_preds_reg = reg_model.predict(X_test)

    # Final classifier
    clf_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE
    )
    clf_model.fit(X_train, y_train_clf)
    test_preds_clf = clf_model.predict_proba(X_test)[:, 1]

    test_mae = mean_absolute_error(y_test_reg, test_preds_reg)
    try:
        test_auc = roc_auc_score(y_test_clf, test_preds_clf)
    except ValueError:
        test_auc = np.nan

    print(f"\nSecondary Test (Batch 3) Performance (n={len(test_subset)}):")
    print(f"  MAE:   {test_mae:.2f} cycles")
    print(f"  AUROC: {test_auc:.4f}")

    test_df = pd.DataFrame({
        'cell_id': test_subset['cell_id'].tolist(),
        'y_reg': y_test_reg.tolist(),
        'y_clf': y_test_clf.tolist(),
        'pred_reg': test_preds_reg.tolist(),
        'pred_clf': test_preds_clf.tolist()
    })

    return test_mae, test_auc, test_df


def generate_report(cv_mae, cv_mae_std, cv_auc, cv_auc_std, test_mae, test_auc, split):
    """Write metrics report to modeling_report.md."""
    report_content = f"""# VitalEdge Modeling Performance Report

Generated: 2026-06-28 (Automatic Evaluation)

## 📋 Configuration Summary
- **Classification Threshold:** Cycle {split['classification_threshold']}
- **XGBoost Hyperparameters:** 
  - `n_estimators`: {N_ESTIMATORS}
  - `max_depth`: {MAX_DEPTH}
  - `learning_rate`: {LEARNING_RATE}
- **Features Used:** 13 (No-temperature)

## 📊 Cross-Validation Performance (Batch 1 + Batch 2)
Evaluated using stratified 5-fold cross-validation on {split['train_pool_size']} cells.

| Metric | Fold Average | Standard Deviation | Target Status |
|--------|--------------|--------------------|---------------|
| **Regression MAE** | {cv_mae:.2f} cycles | ±{cv_mae_std:.2f} cycles | — |
| **Classification AUROC** | {cv_auc:.4f} | ±{cv_auc_std:.4f} | {'✅ Passed (>= 0.82)' if cv_auc >= 0.82 else '❌ Failed (< 0.82) - Fallback required'} |

## 🧪 Generalization Performance (Batch 3 Secondary Test)
Evaluated on the {split['secondary_test_size']} surviving Batch 3 cells.

| Metric | Secondary Test Value |
|--------|----------------------|
| **Regression MAE** | {test_mae:.2f} cycles |
| **Classification AUROC** | {test_auc:.4f} |

## ⚠️ LSTM Fallback Decision
- **Status:** {'XGBoost AUROC matches target. No LSTM fallback needed.' if cv_auc >= 0.82 else 'XGBoost AUROC is below 0.82. LSTM Fallback MUST be evaluated.'}
"""

    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)
    print(f"\nWritten performance report to {REPORT_PATH}")


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(SEVERSON_FEATURES_PATH):
        print(f"⚠️ Features file {SEVERSON_FEATURES_PATH} does not exist yet. Cannot run actual evaluation.")
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

    # Run fold CV
    cv_mae, cv_mae_std, cv_auc, cv_auc_std, oof_df = evaluate_folds(data, split, no_temp_features)

    # Run secondary test evaluation
    test_mae, test_auc, test_df = evaluate_secondary(data, split, no_temp_features)

    # Generate metrics report
    generate_report(cv_mae, cv_mae_std, cv_auc, cv_auc_std, test_mae, test_auc, split)

    # Plot prediction error distributions (predicted - actual)
    os.makedirs(PLOT_DIR, exist_ok=True)
    if oof_df is not None:
        errors = oof_df['pred_reg'] - oof_df['y_reg']
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=15, color='#4CAF50', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='Zero Error')
        plt.xlabel('Prediction Error (Predicted - Actual) in Cycles', fontsize=12)
        plt.ylabel('Count of Cells', fontsize=12)
        plt.title('Out-of-Fold Prediction Error Distribution (Severson B1+B2)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        hist_path = os.path.join(PLOT_DIR, 'error_histogram.png')
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Saved prediction error histogram to {hist_path}")


if __name__ == '__main__':
    main()
