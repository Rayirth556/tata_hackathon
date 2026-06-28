# Triya's Next Steps - Task List

This task list outlines the immediate and upcoming tasks for Triya, based on the project requirements and dependencies detailed in the task documents.

---

## 📋 1. Pre-start Checklist (Immediate Action)
Before starting feature extraction, Triya must verify the following items:

- [ ] **Verify Cleaned Datasets Schema & Units**:
  - Check that `severson_clean.pkl` columns match expectations (e.g. `cell_id` is string, `cycle_num` is integer, others are float64).
  - Verify that `hust_clean.pkl` contains the expected lists of dicts with float64 arrays (`V`, `I`, `Capacity`, `t`).
  - Double-check units: Capacity in **Ah** (not mAh), Current `I` in **Amps** (not mA), time `t` in **seconds**, and Voltage `V` in **Volts**.
- [ ] **Resolve HUST Charge Time Gap**:
  - Request Gauri to add `charge_time_s` (seconds) to `hust_clean.pkl` computed from the duration of the corresponding charge cycles.
- [ ] **Establish Raw Severson Access**:
  - Ensure access to the three raw Severson `.mat` files (`2017-05-12_...`, `2017-06-30_...`, `2018-04-12_...`) on local disk.

---

## ⚙️ 2. Feature Extraction (HUST & Severson)
Triya must extract the 7 core feature groups for cycles 1-100 to produce a 17-column feature matrix.

- [ ] **Implement Feature 1: Discharge Capacity at Cycle 100 (`QD_100`)**:
  - Extract `max(Capacity)` at cycle 100. *Warning: Avoid using the last element of the capacity array due to termination artifacts.*
- [ ] **Implement Feature 2: Internal Resistance (`IR_est`)**:
  - **HUST**: Estimate DC internal resistance using $\Delta V / \Delta I$ at the onset of discharge (cycle 2 and cycle 100), then compute `IR_diff`.
  - **Severson**: Use the cycler-measured `IR` values at cycle 2 and cycle 100 from `severson_clean.pkl`.
- [ ] **Implement Feature 3: $dV/dQ$ Variance (`dVdQ_var`)**:
  - **Grid Interpolation**: Interpolate voltage onto a uniform 1000-point capacity grid.
  - **Smoothing**: Apply Savitzky-Golay smoothing (`window_length=11`, `polyorder=3`).
  - **Derivative & Var**: Compute numerical derivative and variance at cycle 10 and cycle 100, then compute `dVdQ_var_diff`.
  - *Note: For HUST, decide how to handle stage transitions (either full curve or longest constant-current stage).*
- [ ] **Implement Feature 4: Temperature Statistics (Severson Only)**:
  - Extract mean `Tavg`, max `Tmax`, and min `Tmin` across cycles 2-100, plus snapshot `Tavg_100` at cycle 100.
  - *Reminder: HUST has no temperature data; these columns will remain NaN for HUST.*
- [ ] **Implement Feature 5: Discharge Rate Variance (`I_var`)**:
  - Compute variance of current $I$ over the discharge cycle at cycle 10 and cycle 100.
  - *Note: Severson will have $I\_var = 0$ since it has a constant 4C discharge.*
- [ ] **Implement Feature 6: Charge Time (`chargetime_s`)**:
  - **Severson**: Extract `chargetime` (convert minutes to seconds) and average over cycles 2–6.
  - **HUST**: Extract from `charge_time_s` (if provided by Gauri) and average over cycles 2–6.
- [ ] **Implement Feature 7: Capacity Fade Rate (`fade_slope`, `fade_intercept`)**:
  - Run linear regression using `scipy.stats.linregress` on maximum discharge capacity vs. cycle number over cycles 2–100.

---

## 🔍 3. Feature Compatibility Check & Alignment
Triya must verify that Severson-trained models can generalize to HUST.

- [ ] **Voltage Range Alignment**:
  - Plot $V(Q)$ at cycle 10 and 100 for representative cells from both datasets.
  - Ensure the $Q$-grid interpolation for $dV/dQ$ uses the intersection of the voltage ranges.
- [ ] **IR Scale Comparison**:
  - Check the average `IR_cycle2` for both datasets. Document any systematic offsets if they exceed 30%.
- [ ] **Capacity Fade Slope Sanity Check**:
  - Plot distributions of `fade_slope` for both datasets as overlapping histograms to identify any domain shifts.
- [ ] **Produce `compatibility_report.md`**:
  - Summarize matching ranges, available features, mismatch resolutions, and recommendations.

---

## 🔄 4. HUST Replay Script
Rayirth needs this script to simulate streaming data to the edge device.

- [ ] **Develop `hust_replay.py`**:
  - Implement a generator function `hust_replay(pkl_path, cell_id, start_cycle=1, end_cycle=100)`.
  - Ensure it yields cycle dictionaries structured as: `{'cycle': int, 'V': array, 'I': array, 'Capacity': array, 't': array}`.
  - Add logic to normalize timestamps `t` by subtracting `t[0]` if they don't start near 0.

---

## 📦 5. Deliverables & Handoffs

- [ ] **Deliver to Krishiv (Modeler)**:
  - [ ] `severson_features.pkl` & `severson_features.csv`
  - [ ] `hust_features.pkl` & `hust_features.csv`
  - *Note: Provide two variants: with and without temperature columns.*
  - [ ] `compatibility_report.md`
- [ ] **Deliver to Rayirth (Edge/Integration)**:
  - [ ] `hust_replay.py`
  - [ ] Written agreement on the data schema, units, and array orientation.
