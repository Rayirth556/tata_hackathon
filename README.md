# VitalEdge — Accelerated Battery Degradation & Edge Lifetime Prediction

VitalEdge is an end-to-end data cleaning, feature extraction, and machine learning pipeline designed to predict the remaining useful life (RUL) of lithium-ion battery cells using early-life telemetry. The project processes and aligns two major datasets: the **MIT Severson et al. (2019)** dataset and the **HUST** dataset, preparing them for deployment on resource-constrained edge devices.

---

## 🚀 Pipeline Features

### 1. Robust Data Cleaning & Preprocessing
* **MATLAB & H5PY Integration**: Extracts raw, cycle-level summary statistics from large MATLAB v7.3 batch files.
* **Outlier Mitigation**: Detects and interpolates capacity measurement spikes (such as the cycle 40 glitch in cell `b1c18`) using linear interpolation rather than throwing away viable cells.
* **Zero-to-NaN Conversion**: Replaces zero-filled missing measurements (such as Cycle 1 parameters and mid-cycle internal resistance zeroes in Batch 1) with `NaN` to prevent downstream features (like capacity fade slopes) from being artificially skewed towards zero.
* **HUST Lifecycle Segmentation**: Automates the segmentation of multi-stage charge/discharge cycles, normalizes timestamps, and extracts charge durations (`charge_time_s`) from raw telemetry files.
* **Unit Standardization**: Converts current and capacity to standard units (Amps and Amp-hours) across both datasets.

### 2. Feature Extraction (17 Core Features)
Extracts early-life features over cycles 2–100 to feed downstream prediction models:
* **Discharge Capacity**: Capacity values at cycle 100 (`QD_100`).
* **Ohmic Growth**: Internal resistance values at cycle 2, cycle 100, and their differences (`IR_diff`).
* **Electrochemical Phase Transitions**: Variance of the differential voltage curve ($dV/dQ$) at cycle 10, cycle 100, and the difference (`dVdQ_var_diff`).
* **Thermal Stress**: Surface temperature averages, maximums, and minimums over the first 100 cycles (*Severson only*).
* **Current Dynamics**: Discharge rate variance over cycles.
* **Charging Rates**: CC charge time averaged over early cycles (`chargetime_s_mean_2to6`).
* **Linear Fade Trends**: Slope and intercept of early-life capacity fade.

### 3. Edge Telemetry Replay Simulator
* Implements a BMS (Battery Management System) telemetry replay loop to stream cycles one-by-one, validating edge inference latency and resource constraints in real-time.

---

## 📂 Repository Structure

* [clean_severson.py](clean_severson.py) — Cleans the raw Severson `.mat` batch files, applies outlier interpolation, converts zero-filled placeholders to NaNs, and exports `severson_clean.pkl`.
* [clean_hust.py](clean_hust.py) — Segment HUST `.pkl` cells, computes CC charge times, standardizes units, and exports `hust_clean.pkl`.
* [validate_pickles.py](validate_pickles.py) — Automates verification of schema, column types, shapes, and capacity/IR value ranges.
* [data_cleaning_log.md](data_cleaning_log.md) — Detailed scientific documentation of the 39 dropped cells, EOL capacity metrics, NaN modifications, and dataset size details.
* [triya_todo_list.md](triya_todo_list.md) — Detailed checklist for the next step of the pipeline (Feature Extraction and Alignment).

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have python 3.8+ and `uv` package manager installed.

### Installation
1. Initialize the virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install numpy pandas scipy openpyxl h5py
   ```

### Running the Data Cleaning Pipeline
1. Place raw Severson `.mat` files and HUST `our_data` folder in the project root.
2. Run Severson cleaning:
   ```bash
   python clean_severson.py
   ```
3. Run HUST cleaning:
   ```bash
   python clean_hust.py
   ```
4. Validate the resulting clean pickle files:
   ```bash
   python validate_pickles.py
   ```
