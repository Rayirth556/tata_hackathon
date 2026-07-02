# Triya's Next Steps - Task List (Completed ✅)

This task list outlines the tasks for Triya, based on the project requirements and dependencies. All items are now fully implemented and verified.

---

## 📋 1. Pre-start Checklist (Immediate Action)
Triya has verified and resolved all pre-start items:

- [x] **Verify Cleaned Datasets Schema & Units**:
  - Checked that `severson_clean.pkl` columns match expectations (e.g. `cell_id` is string, `cycle_num` is integer, others are float64).
  - Verified that `hust_clean.pkl` contains the expected lists of dicts with float64 arrays (`V`, `I`, `Capacity`, `t`).
  - Double-checked units: Capacity in **Ah**, Current `I` in **Amps**, time `t` in **seconds**, and Voltage `V` in **Volts**.
- [x] **Resolve HUST Charge Time Gap**:
  - Added `charge_time_s` (seconds) to `hust_clean.pkl` computed from the duration of the corresponding charge cycles.
- [x] **Establish Raw Severson Access**:
  - Ensured access to the raw Severson `.mat` files and verified the dereferencing/cleaning logic.

---

## ⚙️ 2. Feature Extraction (HUST & Severson)
Triya has successfully extracted the 7 core feature groups for cycles 1-100 to produce the 17-column feature matrix.

- [x] **Implement Feature 1: Discharge Capacity at Cycle 100 (`QD_100`)**:
  - Extracted `max(Capacity)` at cycle 100, avoiding end-of-cycle termination noise.
- [x] **Implement Feature 2: Internal Resistance (`IR_est`)**:
  - **HUST**: Estimated DC internal resistance using $\Delta V / \Delta I$ at the onset of discharge (cycle 2 and cycle 100), and computed `IR_diff`.
  - **Severson**: Used the cycler-measured `IR` values at cycle 2 and cycle 100 from `severson_clean.pkl`.
- [x] **Implement Feature 3: $dV/dQ$ Variance (`dVdQ_var`)**:
  - **Grid Interpolation**: Interpolated voltage onto a uniform 1000-point capacity grid.
  - **Smoothing**: Applied Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
  - **Derivative & Var**: Computed numerical derivative and variance at cycle 10 and cycle 100, and computed `dVdQ_var_diff`.
  - *HUST handling*: Used `_longest_cc_stage()` to isolate the longest constant-current stage.
- [x] **Implement Feature 4: Temperature Statistics (Severson Only)**:
  - Extracted mean `Tavg`, max `Tmax`, and min `Tmin` across cycles 2-100, plus snapshot `Tavg_100` at cycle 100.
  - HUST temperature columns correctly set to NaN.
- [x] **Implement Feature 5: Discharge Rate Variance (`I_var`)**:
  - Computed variance of current $I$ over the discharge cycle at cycle 10 and cycle 100.
- [x] **Implement Feature 6: Charge Time (`chargetime_s`)**:
  - **Severson**: Extracted `chargetime` and averaged over cycles 2–6.
  - **HUST**: Extracted from `charge_time_s` and averaged over cycles 2–6.
- [x] **Implement Feature 7: Capacity Fade Rate (`fade_slope`, `fade_intercept`)**:
  - Ran linear regression using `scipy.stats.linregress` on maximum discharge capacity vs. cycle number over cycles 2–100.

---

## 🔍 3. Feature Compatibility Check & Alignment
Triya has verified that Severson-trained models can generalize to HUST.

- [x] **Voltage Range Alignment**:
  - Plotted $V(Q)$ at cycle 10 and 100 for representative cells from both datasets.
  - Ensured the $Q$-grid interpolation for $dV/dQ$ uses the intersection of the voltage ranges.
- [x] **IR Scale Comparison**:
  - Checked the average `IR_cycle2` for both datasets. Documented systematic offsets (0% range overlap in absolute IR).
- [x] **Capacity Fade Slope Sanity Check**:
  - Checked distributions of `fade_slope` for both datasets.
- [x] **Produce `compatibility_report.md`**:
  - Summarized matching ranges, available features, mismatch resolutions (use `IR_diff` instead of absolute values), and recommendations.

---

## 🔄 4. HUST Replay Script
Developed for simulating streaming data to the edge device.

- [x] **Develop `hust_replay.py`**:
  - Implemented generator function `hust_replay(pkl_path, cell_id, start_cycle=1, end_cycle=100)`.
  - Ensured it yields cycle dictionaries structured as: `{'cycle': int, 'V': array, 'I': array, 'Capacity': array, 't': array}`.
  - Added logic to normalize timestamps `t` by subtracting `t[0]` if they don't start at 0.

---

## 📦 5. Deliverables & Handoffs

- [x] **Deliver to Krishiv (Modeler)**:
  - [x] `severson_features.pkl` & `severson_features.csv`
  - [x] `hust_features.pkl` & `hust_features.csv`
  - [x] `compatibility_report.md`
- [x] **Deliver to Rayirth (Edge/Integration)**:
  - [x] `hust_replay.py`
  - [x] Written agreement on the data schema, units, and array orientation.
