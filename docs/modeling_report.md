# VitalEdge Modeling Performance Report

Generated: 2026-06-28 (Automatic Evaluation)

## 📋 Configuration Summary
- **Classification Threshold:** Cycle 464
- **XGBoost Hyperparameters:** 
  - `n_estimators`: 50
  - `max_depth`: 4
  - `learning_rate`: 0.1
- **Features Used:** 13 (No-temperature)

## 📊 Cross-Validation Performance (Batch 1 + Batch 2)
Evaluated using stratified 5-fold cross-validation on 50 cells.

| Metric | Fold Average | Standard Deviation | Target Status |
|--------|--------------|--------------------|---------------|
| **Regression MAE** | 72.33 cycles | ±23.80 cycles | — |
| **Classification AUROC** | 0.9042 | ±0.0928 | ✅ Passed (>= 0.82) |

## 🧪 Generalization Performance (Batch 3 Secondary Test)
Evaluated on the 9 surviving Batch 3 cells.

| Metric | Secondary Test Value |
|--------|----------------------|
| **Regression MAE** | 342.29 cycles |
| **Classification AUROC** | nan |

## ⚠️ LSTM Fallback Decision
- **Status:** XGBoost AUROC matches target. No LSTM fallback needed.
