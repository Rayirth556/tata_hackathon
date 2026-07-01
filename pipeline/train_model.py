"""
train_model.py — Train constrained XGBoost models for knee-point prediction.

Loads features, merges with knee labels, loads split indexes, and trains:
  1. Regression model (predicts knee cycle number)
  2. Classification model (predicts early vs. late degradation)

Hyperparameters are constrained for ESP32 compilation safety (max_depth=4, n_estimators=50).
Saves models as trained_model_reg.pkl, trained_model_clf.pkl, and trained_model.pkl (primary regressor).
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
SEVERSON_FEATURES_PATH = os.path.join(WORKSPACE, 'data/severson_features.pkl')
KNEE_LABELS_PATH = os.path.join(WORKSPACE, 'data/knee_labels.csv')
SPLIT_PATH = os.path.join(WORKSPACE, 'data/train_test_split.json')

MODEL_REG_PATH = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
MODEL_CLF_PATH = os.path.join(WORKSPACE, 'models/trained_model_clf.pkl')
PRIMARY_MODEL_PATH = os.path.join(WORKSPACE, 'models/trained_model.pkl')

# Constrained hyperparameters for ESP32 compile safety
N_ESTIMATORS = 50
MAX_DEPTH = 4
LEARNING_RATE = 0.1
RANDOM_STATE = 42


def main():
    # ── Load Data ──────────────────────────────────────────────────────────
    print("Loading data...")
    if not os.path.exists(SEVERSON_FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature file {SEVERSON_FEATURES_PATH}. Has Triya delivered it?")
    if not os.path.exists(KNEE_LABELS_PATH):
        raise FileNotFoundError(f"Missing labels file {KNEE_LABELS_PATH}. Run knee_labeling.py first.")
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Missing split file {SPLIT_PATH}. Run build_split.py first.")

    features_df = pd.read_pickle(SEVERSON_FEATURES_PATH)
    labels_df = pd.read_csv(KNEE_LABELS_PATH)
    
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)

    # ── Merge Features and Labels ──────────────────────────────────────────
    # Exclude cells that have no knee detected (has_knee is False)
    valid_labels = labels_df[labels_df['has_knee']]
    data = features_df.merge(valid_labels[['cell_id', 'knee_cycle']], on='cell_id')
    
    threshold = split['classification_threshold']
    data['knee_early'] = (data['knee_cycle'] < threshold).astype(int)
    
    print(f"Merged dataset size: {data.shape[0]} cells with valid knees.")

    # ── Prepare Feature Sets ───────────────────────────────────────────────
    # Temperature features can only be used on Severson. We build two feature lists:
    # 1. Full features (17 features)
    # 2. No-temp features (13 features) — this is the model that can run on HUST
    all_features = [col for col in data.columns if col not in ['cell_id', 'knee_cycle', 'knee_early']]
    no_temp_features = [col for col in all_features if not col.startswith('T')]

    print(f"Full feature list ({len(all_features)}): {all_features}")
    print(f"No-temperature feature list ({len(no_temp_features)}): {no_temp_features}")

    # Gather all train pool cell IDs from split folds
    train_pool_cells = []
    for fold_id, fold_data in split['folds'].items():
        train_pool_cells.extend(fold_data['train'])
        train_pool_cells.extend(fold_data['val'])
    train_pool_cells = list(set(train_pool_cells))
    
    # Filter dataset for training pool (Batch 1 + Batch 2 survivors with knees)
    train_data = data[data['cell_id'].isin(train_pool_cells)].copy()
    print(f"Training pool size (B1+B2): {train_data.shape[0]} cells")

    X_train_full = train_data[all_features]
    X_train_notemp = train_data[no_temp_features]
    y_reg = train_data['knee_cycle']
    y_clf = train_data['knee_early']

    # ── Model Initialization ───────────────────────────────────────────────
    # Constrain trees/depth because micromlgen C headers exceed compile memory limits on ESP32 above ~40-50KB
    print("\nTraining constrained XGBoost models on full training pool...")
    
    # Regression model (Primary model)
    # We train the version WITHOUT temperature so it is compatible with HUST.
    # If full Severson-only model is needed, we can train that, but the primary target
    # is deploying a model on ESP32 that runs HUST replay telemetry.
    reg_model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    )
    reg_model.fit(X_train_notemp, y_reg)
    
    # Classification model (For AUROC metrics)
    clf_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE
    )
    clf_model.fit(X_train_notemp, y_clf)

    # ── Evaluate on Train Pool (Self-check) ────────────────────────────────
    train_reg_preds = reg_model.predict(X_train_notemp)
    train_clf_preds = clf_model.predict_proba(X_train_notemp)[:, 1]
    
    train_mae = mean_absolute_error(y_reg, train_reg_preds)
    train_auc = roc_auc_score(y_clf, train_clf_preds)
    
    print(f"Self-evaluation on entire Train Pool:")
    print(f"  Regression MAE:       {train_mae:.2f} cycles")
    print(f"  Classification AUROC: {train_auc:.4f}")

    # Save models
    joblib.dump(reg_model, MODEL_REG_PATH)
    joblib.dump(clf_model, MODEL_CLF_PATH)
    # trained_model.pkl is the primary model for Rayirth's C export
    joblib.dump(reg_model, PRIMARY_MODEL_PATH)
    
    print(f"\nModels successfully saved:")
    print(f"  {MODEL_REG_PATH}")
    print(f"  {MODEL_CLF_PATH}")
    print(f"  {PRIMARY_MODEL_PATH} (Rayirth Hand-off)")


if __name__ == '__main__':
    main()
