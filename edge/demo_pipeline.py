import os
import sys
import pickle
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
WORKSPACE = '/home/godkiller/Documents/tata'
sys.path.append(WORKSPACE)

from edge.hust_replay import hust_replay
from edge.feature_extractor import extract_features_from_cycles, get_feature_dataframe, NO_TEMP_FEATURES

def main():
    parser = argparse.ArgumentParser(description="VitalEdge Demo Pipeline")
    parser.add_argument('cell_id', type=str, nargs='?', default="1-1",
                        help="HUST cell ID to stream (e.g. 1-1, 3-5). Default: 1-1")
    args = parser.parse_args()
    
    cell_id = args.cell_id
    clean_pkl = os.path.join(WORKSPACE, 'data/hust_clean.pkl')
    model_reg_path = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
    model_clf_path = os.path.join(WORKSPACE, 'models/trained_model_clf.pkl')
    c_bin_path = os.path.join(WORKSPACE, 'edge', 'host_test', 'host_test_bin')
    
    # ── Verify dependencies ──────────────────────────────────────────────────
    if not os.path.exists(clean_pkl):
        print(f"Error: Missing HUST clean dataset at {clean_pkl}.")
        sys.exit(1)
    if not os.path.exists(model_reg_path) or not os.path.exists(model_clf_path):
        print("Error: Missing trained Python models. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(c_bin_path):
        print("Error: Missing compiled host C binary. Run make in host_test first.")
        sys.exit(1)

    print("=" * 80)
    print(f"VITALEDGE DEMO: Cell {cell_id} HUST Replay Telemetry -> Inference -> SHAP")
    print("=" * 80)
    
    # ── 1. Replay Telemetry ─────────────────────────────────────────────────
    print(f"\n[1/5] Streaming raw telemetry for cell {cell_id} (cycles 1 to 100)...")
    cycles = []
    try:
        for cycle_data in hust_replay(clean_pkl, cell_id=cell_id, start_cycle=1, end_cycle=100):
            cycles.append(cycle_data)
    except Exception as e:
        print(f"Error replaying cell {cell_id}: {e}")
        sys.exit(1)
    print(f"      Successfully streamed {len(cycles)} cycles.")
    
    # ── 2. Feature Extraction ───────────────────────────────────────────────
    print("\n[2/5] Extracting 13 features from telemetry stream...")
    features_dict = extract_features_from_cycles(cycles)
    features_df = get_feature_dataframe(features_dict)
    
    # Display features
    print("\nExtracted features vector:")
    for col in NO_TEMP_FEATURES:
        print(f"  {col:<25}: {features_dict[col]:.6e}")
        
    # ── 3. Python Inference ──────────────────────────────────────────────────
    print("\n[3/5] Running inference on Python XGBoost models...")
    reg_model = joblib.load(model_reg_path)
    clf_model = joblib.load(model_clf_path)
    
    py_reg_pred = float(reg_model.predict(features_df)[0])
    py_clf_pred_prob = clf_model.predict_proba(features_df)[0]
    
    print(f"      Python Regression RUL Knee-Cycle: {py_reg_pred:.4f}")
    print(f"      Python Classification early degradation prob: {py_clf_pred_prob[1]:.4f} (Class 0: {py_clf_pred_prob[0]:.4f})")
    
    # ── 4. Host C Inference ──────────────────────────────────────────────────
    print("\n[4/5] Running inference on compiled Host C binary...")
    # Prepare inputs for C stdin
    input_str = " ".join([f"{features_dict[col]:.17g}" for col in NO_TEMP_FEATURES])
    
    try:
        proc = subprocess.run([c_bin_path], input=input_str, text=True, capture_output=True, check=True)
        c_output = proc.stdout.strip().split('\n')
    except Exception as e:
        print(f"Error running Host C binary: {e}")
        sys.exit(1)
        
    c_results = {}
    for line in c_output:
        if ':' in line:
            k, v = line.split(':')
            c_results[k.strip()] = float(v.strip())
            
    c_reg_pred = c_results["regression_prediction"]
    c_clf_pred_prob_1 = c_results["classification_prediction_prob_1"]
    c_clf_pred_prob_0 = c_results["classification_prediction_prob_0"]
    reg_latency_us = c_results["reg_latency_us"]
    clf_latency_us = c_results["clf_latency_us"]
    
    print(f"      C Regression RUL Knee-Cycle: {c_reg_pred:.4f}")
    print(f"      C Classification early degradation prob: {c_clf_pred_prob_1:.4f} (Class 0: {c_clf_pred_prob_0:.4f})")
    print(f"      C Latency (Regression)    : {reg_latency_us:.4f} us (average over {1000000} runs)")
    print(f"      C Latency (Classification): {clf_latency_us:.4f} us (average over {1000000} runs)")
    
    # ── Validation ──────────────────────────────────────────────────────────
    reg_diff = abs(py_reg_pred - c_reg_pred)
    clf_diff = abs(py_clf_pred_prob[1] - c_clf_pred_prob_1)
    
    print("\n      --- Numeric Alignment Verification ---")
    if reg_diff < 1e-4:
        print(f"      ✅ Regression prediction aligned (diff={reg_diff:.2e})")
    else:
        print(f"      ❌ Regression prediction MISALIGNED (diff={reg_diff:.2e})")
        sys.exit(1)
        
    if clf_diff < 1e-4:
        print(f"      ✅ Classification prediction aligned (diff={clf_diff:.2e})")
    else:
        print(f"      ❌ Classification prediction MISALIGNED (diff={clf_diff:.2e})")
        sys.exit(1)
        
    # ── 5. SHAP Explainability ──────────────────────────────────────────────
    print("\n[5/5] Generating SHAP waterfall explanation...")
    # Initialize explainer on training data to establish base values
    # Find all train cells from split folds
    split_path = os.path.join(WORKSPACE, 'data/train_test_split.json')
    with open(split_path, 'r') as f:
        split = json.load(f)
    
    train_pool_cells = []
    for fold_id, fold_data in split['folds'].items():
        train_pool_cells.extend(fold_data['train'])
        train_pool_cells.extend(fold_data['val'])
    train_pool_cells = list(set(train_pool_cells))
    
    sev_features = pd.read_pickle(os.path.join(WORKSPACE, 'data/severson_features.pkl'))
    labels_df = pd.read_csv(os.path.join(WORKSPACE, 'data/knee_labels.csv'))
    valid_labels = labels_df[labels_df['has_knee']]
    train_data = sev_features.merge(valid_labels[['cell_id', 'knee_cycle']], on='cell_id')
    X_train = train_data[train_data['cell_id'].isin(train_pool_cells)][NO_TEMP_FEATURES]
    
    # Compute SHAP
    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer(features_df)
    
    # Plot SHAP waterfall
    os.makedirs(os.path.join(WORKSPACE, 'plots'), exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title(f'VitalEdge SHAP Attribution: HUST Cell {cell_id}', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    shap_plot_path = os.path.join(WORKSPACE, 'plots', f'shap_waterfall_{cell_id.replace("-", "_")}.png')
    plt.savefig(shap_plot_path, dpi=150)
    plt.close()
    print(f"      SHAP waterfall plot saved to {shap_plot_path}")
    print("\n" + "=" * 80)
    print("VITALEDGE DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
