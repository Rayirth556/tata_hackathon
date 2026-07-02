"""
tatahack3 / src / lstm_fallback.py

Point 2 — LSTM fallback (PS.txt mandates this since joint-model XGBoost
AUROC = 0.818 < 0.82).

INPUT: cycles 1-100 discharge-capacity (QD) sequence per cell.
  - Source: severson_full_curves.pkl -> {cell_id: {'cycles': arr, 'QD': arr, 'total_cycles': int}}
  - Truncated/padded to exactly 100 timesteps (NO look-ahead past cycle 100,
    same constraint as the XGBoost model).
  - Normalized per-cell: QD / QD[cycle=1], so the LSTM sees relative fade,
    not absolute capacity (removes cell-to-cell nominal-capacity offset).
  - Shape fed to model: (n_cells, 100, 1)

LABELS: same knee_labels.csv + train_test_split.json as XGBoost (Kneedle-
primary labels, per team decision — see conversation w/ Triya/lead).

Run (on a machine that has severson_full_curves.pkl — NOT this sandbox,
raw .mat files are gitignored / too large):
    python src/lstm_fallback.py --repo_dir /path/to/tata_hackathon
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, roc_auc_score

N_TIMESTEPS = 100  # cycles 1-100 only, no look-ahead


def load_sequences(full_curves_path, n_timesteps=N_TIMESTEPS):
    with open(full_curves_path, "rb") as f:
        curves = pickle.load(f)

    cell_ids, sequences = [], []
    for cell_id, d in curves.items():
        cycles, qd = np.asarray(d["cycles"]), np.asarray(d["QD"])
        order = np.argsort(cycles)
        cycles, qd = cycles[order], qd[order]

        mask = cycles <= n_timesteps
        seq = qd[mask].astype(float)
        if len(seq) < 5:
            continue

        # Don't drop a cell just because cycle-1 is NaN (common formation-cycle
        # noise) — interpolate internal NaNs, normalize against first VALID reading.
        valid_mask = ~np.isnan(seq)
        if valid_mask.sum() < 5:
            continue
        first_valid_idx = np.argmax(valid_mask)
        idx = np.arange(len(seq))
        seq = np.interp(idx, idx[valid_mask], seq[valid_mask])
        base = seq[first_valid_idx]
        if base == 0:
            continue

        seq = seq / base
        seq = (seq - 1.0) * 100.0  # % degradation from baseline — wider gradient range

        if len(seq) >= n_timesteps:
            seq = seq[:n_timesteps]
        else:
            pad = np.full(n_timesteps - len(seq), seq[-1])
            seq = np.concatenate([seq, pad])

        cell_ids.append(cell_id)
        sequences.append(seq)

    X = np.array(sequences).reshape(-1, n_timesteps, 1)
    return cell_ids, X


def build_lstm(n_timesteps, n_features, task="regression"):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    out_activation = "linear" if task == "regression" else "sigmoid"
    loss = "mse" if task == "regression" else "binary_crossentropy"
    metrics = ["mae"] if task == "regression" else ["AUC"]

    model = models.Sequential([
        layers.Input(shape=(n_timesteps, n_features)),
        layers.LSTM(64, return_sequences=True, unroll=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False, unroll=True),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation=out_activation),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


def run_cv(cell_ids, X, labels_df, split, epochs=60, batch_size=8):
    threshold = split["classification_threshold"]
    id_to_idx = {c: i for i, c in enumerate(cell_ids)}

    valid = labels_df[labels_df["has_knee"]].copy()
    valid = valid[valid["cell_id"].isin(id_to_idx)]
    valid["knee_early"] = (valid["knee_cycle"] < threshold).astype(int)

    fold_maes, fold_aurocs = [], []
    for fold_id, fold in split["folds"].items():
        train_ids = [c for c in fold["train"] if c in id_to_idx and c in set(valid["cell_id"])]
        val_ids = [c for c in fold["val"] if c in id_to_idx and c in set(valid["cell_id"])]
        if not train_ids or not val_ids:
            continue

        train_idx = [id_to_idx[c] for c in train_ids]
        val_idx = [id_to_idx[c] for c in val_ids]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train_reg = valid.set_index("cell_id").loc[train_ids, "knee_cycle"].values
        y_val_reg = valid.set_index("cell_id").loc[val_ids, "knee_cycle"].values
        y_train_clf = valid.set_index("cell_id").loc[train_ids, "knee_early"].values
        y_val_clf = valid.set_index("cell_id").loc[val_ids, "knee_early"].values

        # Scale regression target (raw cycle counts, ~100-1200) to help the
        # LSTM converge on this small dataset. Unscale predictions before MAE.
        y_scale = max(y_train_reg.max(), 1.0)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=8, restore_best_weights=True)

        reg = build_lstm(N_TIMESTEPS, 1, task="regression")
        reg.fit(X_train, y_train_reg / y_scale, epochs=epochs, batch_size=batch_size,
                verbose=0, callbacks=[early_stop])
        pred_reg = reg.predict(X_val, verbose=0).ravel() * y_scale
        mae = mean_absolute_error(y_val_reg, pred_reg)

        auroc = float("nan")
        if len(set(y_train_clf)) > 1 and len(set(y_val_clf)) > 1:
            clf = build_lstm(N_TIMESTEPS, 1, task="classification")
            clf.fit(X_train, y_train_clf, epochs=epochs, batch_size=batch_size,
                    verbose=0, callbacks=[early_stop])
            proba = clf.predict(X_val, verbose=0).ravel()
            auroc = roc_auc_score(y_val_clf, proba)
            fold_aurocs.append(auroc)

        fold_maes.append(mae)
        print(f"[fold {fold_id}] MAE={mae:.2f} cycles  AUROC={auroc:.4f}  "
              f"(train={len(train_ids)}, val={len(val_ids)})")

    print(f"\n[LSTM CV summary] mean MAE={np.mean(fold_maes):.2f} cycles  "
          f"mean AUROC={np.nanmean(fold_aurocs):.4f}")
    return np.mean(fold_maes), np.nanmean(fold_aurocs)


def main():
    default_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default=default_repo_dir)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    full_curves_path = os.path.join(args.repo_dir, "data", "severson_full_curves.pkl")
    labels_path = os.path.join(args.repo_dir, "data", "knee_labels.csv")
    split_path = os.path.join(args.repo_dir, "data", "train_test_split.json")

    for p, desc in [(full_curves_path, "full curves"), (labels_path, "knee labels"),
                     (split_path, "split")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {desc}: {p}. "
                f"{'Run pipeline/extract_full_curves.py first (needs raw .mat files).' if 'curves' in desc else ''}"
            )

    cell_ids, X = load_sequences(full_curves_path)
    print(f"[INFO] Loaded {len(cell_ids)} cell sequences, shape {X.shape}")

    labels_df = pd.read_csv(labels_path)
    with open(split_path) as f:
        split = json.load(f)

    mean_mae, mean_auroc = run_cv(cell_ids, X, labels_df, split, epochs=args.epochs)

    if mean_auroc >= 0.82:
        print(f"\n[OK] LSTM fallback clears the 0.82 AUROC gate ({mean_auroc:.4f}).")
    else:
        print(f"\n[STILL BELOW GATE] LSTM AUROC {mean_auroc:.4f} < 0.82 — "
              f"report both XGBoost and LSTM numbers to the team, this is now "
              f"a data/label problem, not a model-choice problem.")

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Final fit on the full train pool + export (.h5 + .tflite for Rayirth) ----
    import tensorflow as tf

    threshold = split["classification_threshold"]
    id_to_idx = {c: i for i, c in enumerate(cell_ids)}
    valid = labels_df[labels_df["has_knee"]].copy()
    valid = valid[valid["cell_id"].isin(id_to_idx)]
    valid["knee_early"] = (valid["knee_cycle"] < threshold).astype(int)

    pool_ids = set()
    for fold in split["folds"].values():
        pool_ids.update(fold["train"]); pool_ids.update(fold["val"])
    pool_ids = [c for c in pool_ids if c in id_to_idx and c in set(valid["cell_id"])]
    pool_idx = [id_to_idx[c] for c in pool_ids]
    X_pool = X[pool_idx]
    y_reg = valid.set_index("cell_id").loc[pool_ids, "knee_cycle"].values
    y_clf = valid.set_index("cell_id").loc[pool_ids, "knee_early"].values

    final_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=8, restore_best_weights=True)
    final_reg = build_lstm(N_TIMESTEPS, 1, task="regression")
    final_reg.fit(X_pool, y_reg, epochs=args.epochs, batch_size=8, verbose=0,
                  callbacks=[final_early_stop])
    final_clf = build_lstm(N_TIMESTEPS, 1, task="classification")
    final_clf.fit(X_pool, y_clf, epochs=args.epochs, batch_size=8, verbose=0,
                  callbacks=[final_early_stop])

    reg_h5 = os.path.join(args.out_dir, "lstm_reg.h5")
    clf_h5 = os.path.join(args.out_dir, "lstm_clf.h5")
    final_reg.save(reg_h5)
    final_clf.save(clf_h5)
    print(f"[SAVED] {reg_h5}")
    print(f"[SAVED] {clf_h5}")

    for name, model in [("lstm_reg", final_reg), ("lstm_clf", final_clf)]:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        tflite_path = os.path.join(args.out_dir, f"{name}.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"[SAVED] {tflite_path}")

    print("\n[NOTE] These are trained on the FULL train pool (not CV-held-out), "
          "for deployment only. The CV numbers above are the ones that go in "
          "the presentation/report — not a metric on these exact export files.")


if __name__ == "__main__":
    main()
