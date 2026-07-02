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

## ⚠️ LSTM Fallback Decision & Performance Analysis

### 1. Fallback Trigger Status
* **Severson Feature Set Model (17 features):** Mean CV AUROC = **0.9083** (Clears the $0.82$ gate; no fallback needed).
* **Joint Feature Set Model (9 features):** Mean CV AUROC = **0.8180** (Just under the $0.82$ gate; triggers the fallback to [lstm_fallback.py](file:///home/godkiller/Documents/tata/pipeline/lstm_fallback.py) because temperature features are dropped).

### 2. LSTM Performance Analysis & Data Starvation
When the fallback is triggered, an LSTM is trained on cycles 1-100 of the discharge capacity sequence. However, evaluation yields poor validation AUROC ($0.65 \text{ to } 0.73$) with high variance:
* **The Data Size Problem:** Deep learning models like LSTMs have thousands of parameters. In our 5-fold cross-validation, each fold has only **40 training cell sequences** (from the Severson dataset). This extremely small sample size is insufficient for a complex neural network to generalize, leading to severe overfitting.
* **Why XGBoost Wins:** XGBoost trains on domain-specific hand-crafted features (like `IR_diff` and `dVdQ_var_diff`), which drastically reduces the dimensionality and prevents overfitting on small datasets ($N \approx 50$).

### 3. Keras Version Workaround (Warning)
Loading the trained LSTM fallback models (like `lstm_reg.h5`) can trigger version mismatch errors due to differences in optimizer configuration in Keras. Always load the model with optimizer compilation disabled:
```python
import tensorflow as tf
# Load model without compiling optimizer state to prevent environment mismatches
model = tf.keras.models.load_model("lstm_reg.h5", compile=False)
```

### 4. Future Roadmap: Relevance of Autoencoders
* **How they could help:** In future iterations, an **Unsupervised Sequence Autoencoder** (e.g., LSTM or 1D-CNN Autoencoder) could be trained on raw telemetry curves (voltage/current/temperature) to automatically extract low-dimensional latent features, replacing hand-crafted feature engineering.
* **Why they are not used now:** Like the LSTM fallback, Autoencoders are deep neural networks and would overfit on our current small dataset. Additionally, running autoencoder inference on an ESP32 micro-controller introduces significant memory (RAM/Flash) and computation overhead.

