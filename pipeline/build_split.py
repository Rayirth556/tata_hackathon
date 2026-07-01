"""
build_split.py — Define train/test split and classification threshold.

Reads knee_labels.csv and assigns cells into:
  - Train+CV pool: Batch 1 + Batch 2 cells with detected knees (5-fold stratified CV)
  - Secondary test: Batch 3 survivors with detected knees
  - Excluded: cells with no detectable knee

Classification threshold is derived from the median knee cycle of all labeled cells.

Output: train_test_split.json
"""

import json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
INPUT_CSV = os.path.join(WORKSPACE, 'data/knee_labels.csv')
OUTPUT_JSON = os.path.join(WORKSPACE, 'data/train_test_split.json')
N_FOLDS = 5
RANDOM_STATE = 42


def main():
    # ── Load knee labels ───────────────────────────────────────────────────
    print("Loading knee labels...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Total cells: {len(df)}")
    print(f"  Cells with knee: {df['has_knee'].sum()}")
    print(f"  Cells without knee: {(~df['has_knee']).sum()}")

    # ── Derive classification threshold from median ────────────────────────
    knee_df = df[df['has_knee']].copy()
    median_knee = knee_df['knee_cycle'].median()
    threshold = int(median_knee)
    print(f"\n  Classification threshold (median knee cycle): {threshold}")

    # ── Assign early/late labels ───────────────────────────────────────────
    knee_df['knee_early'] = (knee_df['knee_cycle'] < threshold).astype(int)
    n_early = knee_df['knee_early'].sum()
    n_late = len(knee_df) - n_early
    print(f"  Early knee (<{threshold}): {n_early} cells")
    print(f"  Late knee  (≥{threshold}): {n_late} cells")

    # ── Split by batch ─────────────────────────────────────────────────────
    # Batch is encoded in cell_id prefix: b1c*, b2c*, b3c*
    knee_df['batch'] = knee_df['cell_id'].str.extract(r'b(\d+)c').astype(int)

    train_pool = knee_df[knee_df['batch'].isin([1, 2])].copy()
    secondary_test = knee_df[knee_df['batch'] == 3].copy()
    excluded = df[~df['has_knee']]['cell_id'].tolist()

    print(f"\n  Train+CV pool (B1+B2): {len(train_pool)} cells")
    print(f"  Secondary test (B3):   {len(secondary_test)} cells")
    print(f"  Excluded (no knee):    {len(excluded)} cells")

    # ── Stratified K-Fold on train pool ────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X = train_pool['cell_id'].values
    y = train_pool['knee_early'].values

    folds = {}
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        train_cells = X[train_indices].tolist()
        val_cells = X[val_indices].tolist()
        folds[str(fold_idx)] = {
            'train': train_cells,
            'val': val_cells,
            'train_early': int(y[train_indices].sum()),
            'train_late': int(len(train_indices) - y[train_indices].sum()),
            'val_early': int(y[val_indices].sum()),
            'val_late': int(len(val_indices) - y[val_indices].sum()),
        }

    # ── Build output structure ─────────────────────────────────────────────
    output = {
        'classification_threshold': threshold,
        'n_folds': N_FOLDS,
        'random_state': RANDOM_STATE,
        'total_cells_with_knee': len(knee_df),
        'train_pool_size': len(train_pool),
        'secondary_test_size': len(secondary_test),
        'folds': folds,
        'secondary_test': secondary_test['cell_id'].tolist(),
        'cells_excluded_no_knee': excluded,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Split saved to {OUTPUT_JSON}")

    # ── Print fold summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FOLD SUMMARY")
    print(f"{'='*60}")
    for fold_id, fold_data in folds.items():
        print(f"  Fold {fold_id}: train={len(fold_data['train'])} "
              f"(early={fold_data['train_early']}, late={fold_data['train_late']}) | "
              f"val={len(fold_data['val'])} "
              f"(early={fold_data['val_early']}, late={fold_data['val_late']})")

    print(f"\n  Secondary test: {secondary_test['cell_id'].tolist()}")
    print(f"  Excluded (no knee): {excluded}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
