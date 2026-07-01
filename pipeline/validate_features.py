"""
validate_features.py — Validate feature matrices delivered by Triya.

Checks that files exist, columns match, datatypes are correct, there are no
unexpected NaNs, and values are physically plausible.
"""

import pandas as pd
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
SEVERSON_FEATURES_PATH = os.path.join(WORKSPACE, 'data/severson_features.pkl')
HUST_FEATURES_PATH = os.path.join(WORKSPACE, 'data/hust_features.pkl')

EXPECTED_COLS_ALL = [
    'cell_id', 'QD_100', 'IR_cycle2', 'IR_cycle100', 'IR_diff',
    'dVdQ_var_10', 'dVdQ_var_100', 'dVdQ_var_diff',
    'Tavg_mean', 'Tmax_max', 'Tmin_min', 'Tavg_100',
    'I_var_10', 'I_var_100', 'chargetime_s_mean_2to6',
    'fade_slope', 'fade_intercept'
]

# HUST columns should not contain temperature columns
EXPECTED_COLS_NO_TEMP = [col for col in EXPECTED_COLS_ALL if not col.startswith('T')]


def check_ranges(df, name):
    """Check if feature values are in plausible ranges."""
    issues = []
    
    # 1. QD_100 (Discharge capacity at cycle 100)
    if 'QD_100' in df.columns:
        bad_qd = df[(df['QD_100'] < 0.5) | (df['QD_100'] > 1.3)]
        if not bad_qd.empty:
            issues.append(f"QD_100 out of bounds [0.5, 1.3] Ah for cells: {bad_qd['cell_id'].tolist()}")

    # 2. IR values (should be > 0 and typically < 0.1 Ohm)
    for col in ['IR_cycle2', 'IR_cycle100']:
        if col in df.columns:
            bad_ir = df[(df[col] <= 0) | (df[col] > 0.5)]
            if not bad_ir.empty:
                issues.append(f"{col} out of bounds (0, 0.5] Ohm for cells: {bad_ir['cell_id'].tolist()}")

    # 3. fade_slope (typically negative, representing fade)
    if 'fade_slope' in df.columns:
        positive_fade = df[df['fade_slope'] > 0.01]  # allow very tiny positive slope due to measurement noise
        if not positive_fade.empty:
            issues.append(f"Positive fade_slope > 0.01 for cells: {positive_fade['cell_id'].tolist()}")

    # 4. chargetime (should be positive, e.g. 500s to 5000s)
    if 'chargetime_s_mean_2to6' in df.columns:
        bad_ct = df[(df['chargetime_s_mean_2to6'] < 100) | (df['chargetime_s_mean_2to6'] > 10000)]
        if not bad_ct.empty:
            issues.append(f"chargetime_s_mean_2to6 out of bounds [100, 10000] seconds for cells: {bad_ct['cell_id'].tolist()}")

    return issues


def validate_dataset(filepath, name, expected_cols, allow_temp_nan=False):
    """Validate a single feature matrix file."""
    print(f"\nValidating {name} features from {filepath}...")
    if not os.path.exists(filepath):
        print(f"  ❌ File does not exist!")
        return False

    try:
        if filepath.endswith('.pkl'):
            df = pd.read_pickle(filepath)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  ❌ Failed to load file: {e}")
        return False

    print(f"  Shape: {df.shape}")
    
    # Check for cell_id
    if 'cell_id' not in df.columns:
        print("  ❌ 'cell_id' column is missing!")
        return False

    # Check columns presence
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"  ❌ Missing expected columns: {missing_cols}")
        return False

    # Check for unexpected NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        if allow_temp_nan:
            # For Severson, allow temp NaNs only if specifically documented
            non_temp_nans = [col for col in nan_cols if not col.startswith('T')]
            if non_temp_nans:
                print(f"  ❌ Unexpected NaNs in non-temperature columns: {non_temp_nans}")
                return False
            else:
                print(f"  ⚠️ Note: Temperature columns contain NaNs: {[col for col in nan_cols if col.startswith('T')]}")
        else:
            print(f"  ❌ Unexpected NaNs in columns: {nan_cols}")
            return False

    # Check column types (should be float64 except cell_id)
    non_numeric = []
    for col in df.columns:
        if col == 'cell_id':
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            non_numeric.append(col)
    if non_numeric:
        print(f"  ❌ Non-numeric data types in columns: {non_numeric}")
        return False

    # Check value ranges
    range_issues = check_ranges(df, name)
    if range_issues:
        for issue in range_issues:
            print(f"  ❌ Range warning: {issue}")
        return False

    print(f"  ✅ {name} features are VALID!")
    return True


def main():
    severson_ok = validate_dataset(SEVERSON_FEATURES_PATH, 'Severson', EXPECTED_COLS_ALL, allow_temp_nan=True)
    hust_ok = validate_dataset(HUST_FEATURES_PATH, 'HUST', EXPECTED_COLS_NO_TEMP, allow_temp_nan=False)

    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"  Severson features: {'PASS' if severson_ok else 'FAIL'}")
    print(f"  HUST features:     {'PASS' if hust_ok else 'FAIL'}")
    print("="*50)


if __name__ == '__main__':
    main()
