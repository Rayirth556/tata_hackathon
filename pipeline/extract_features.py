"""
extract_features.py
====================
Extracts 17-feature matrices (+ cell_id) from the cleaned Severson and HUST datasets.

Outputs (in /home/godkiller/Documents/tata/):
  severson_features.pkl / .csv         -- 101 cells × 18 columns (17 features + cell_id)
  severson_features_no_temp.pkl / .csv -- 101 cells × 14 columns (no temperature)
  hust_features.pkl / .csv             -- 77 cells  × 18 columns (temp columns = NaN)
  hust_features_no_temp.pkl / .csv     -- 77 cells  × 14 columns

Feature columns (in order):
  cell_id, QD_100, IR_cycle2, IR_cycle100, IR_diff,
  dVdQ_var_10, dVdQ_var_100, dVdQ_var_diff,
  Tavg_mean, Tmax_max, Tmin_min, Tavg_100,
  I_var_10, I_var_100, I_var_diff,
  chargetime_s_mean_2to6, fade_slope, fade_intercept
"""

import os
import pickle
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

WORKSPACE = "/home/godkiller/Documents/tata"
SEV_CLEAN  = os.path.join(WORKSPACE, "data/severson_clean.pkl")
HUST_CLEAN = os.path.join(WORKSPACE, "data/hust_clean.pkl")

BATCH_FILES = {
    1: os.path.join(WORKSPACE, "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    2: os.path.join(WORKSPACE, "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
    3: os.path.join(WORKSPACE, "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
}

EXCLUSIONS = {
    1: [0, 2, 3],
    2: [],
    3: [3] + list(range(11, 46)),
}

FEATURE_COLS = [
    "cell_id", "QD_100",
    "IR_cycle2", "IR_cycle100", "IR_diff",
    "dVdQ_var_10", "dVdQ_var_100", "dVdQ_var_diff",
    "Tavg_mean", "Tmax_max", "Tmin_min", "Tavg_100",
    "I_var_10", "I_var_100", "I_var_diff",
    "chargetime_s_mean_2to6",
    "fade_slope", "fade_intercept",
]

TEMP_COLS   = ["Tavg_mean", "Tmax_max", "Tmin_min", "Tavg_100"]
NO_TEMP_COLS = [c for c in FEATURE_COLS if c not in TEMP_COLS]


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def compute_dvdq_var(V, Q, n_grid=1000):
    """Return variance of dV/dQ for one discharge curve (V vs Q)."""
    order = np.argsort(Q)
    Q_s, V_s = Q[order], V[order]
    _, unique_idx = np.unique(Q_s, return_index=True)
    Q_u, V_u = Q_s[unique_idx], V_s[unique_idx]

    if len(Q_u) < 10:
        return np.nan

    Q_grid = np.linspace(Q_u[0], Q_u[-1], n_grid)
    try:
        interp_fn = interp1d(Q_u, V_u, kind="linear", bounds_error=False, fill_value="extrapolate")
        V_interp  = interp_fn(Q_grid)
    except Exception:
        return np.nan

    # Savitzky-Golay — window must be odd and >= polyorder+2
    win = min(11, len(V_interp))
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    try:
        V_smooth = savgol_filter(V_interp, window_length=win, polyorder=3)
    except Exception:
        V_smooth = V_interp

    dVdQ = np.gradient(V_smooth, Q_grid)
    return float(np.var(dVdQ))


# ─────────────────────────────────────────────────────────────────
# SEVERSON
# ─────────────────────────────────────────────────────────────────

def _read_cycle_arrays(f, cell_group, cycle_idx):
    """
    Dereference HDF5 object-reference arrays for a given cycle index.
    Returns (V, Qd, I) as float64 numpy arrays, or (None, None, None) on error.
    Cycle index is 0-based (cycle N → index N-1).
    """
    try:
        V_ref  = cell_group["V"][cycle_idx, 0]
        Qd_ref = cell_group["Qd"][cycle_idx, 0]
        I_ref  = cell_group["I"][cycle_idx, 0]

        V  = np.squeeze(np.array(f[V_ref],  dtype=np.float64))
        Qd = np.squeeze(np.array(f[Qd_ref], dtype=np.float64))
        I  = np.squeeze(np.array(f[I_ref],  dtype=np.float64))

        # Remove NaN / inf rows
        mask = np.isfinite(V) & np.isfinite(Qd) & np.isfinite(I)
        return V[mask], Qd[mask], I[mask]
    except Exception:
        return None, None, None


def extract_severson_features():
    print("=" * 60)
    print("Extracting Severson features...")
    print("=" * 60)

    sev_df = pd.read_pickle(SEV_CLEAN)
    rows   = []

    for b_id, mat_path in BATCH_FILES.items():
        print(f"\n  Batch {b_id} — {os.path.basename(mat_path)}")
        with h5py.File(mat_path, "r") as f:
            batch     = f["batch"]
            num_cells = batch["summary"].shape[0]

            for i in range(num_cells):
                if i in EXCLUSIONS[b_id]:
                    continue

                cell_id  = f"b{b_id}c{i}"
                cell_df  = sev_df[sev_df["cell_id"] == cell_id].sort_values("cycle_num")
                if len(cell_df) < 100:
                    print(f"    SKIP {cell_id}: only {len(cell_df)} cycles in summary")
                    continue

                row = {"cell_id": cell_id}

                # ── QD at cycle 100 ──────────────────────────────
                qd_100_rows = cell_df[cell_df["cycle_num"] == 100]["QD"].values
                row["QD_100"] = float(qd_100_rows[0]) if len(qd_100_rows) else np.nan

                # ── IR ───────────────────────────────────────────
                def _ir(cn):
                    v = cell_df[cell_df["cycle_num"] == cn]["IR"].values
                    return float(v[0]) if len(v) and np.isfinite(v[0]) else np.nan

                row["IR_cycle2"]   = _ir(2)
                row["IR_cycle100"] = _ir(100)
                row["IR_diff"]     = (row["IR_cycle100"] - row["IR_cycle2"]
                                      if np.isfinite(row["IR_cycle100"]) and np.isfinite(row["IR_cycle2"])
                                      else np.nan)

                # ── Charge time (minutes → seconds) ─────────────
                ct = cell_df[cell_df["cycle_num"].between(2, 6)]["chargetime"].dropna()
                row["chargetime_s_mean_2to6"] = float(ct.mean() * 60.0) if len(ct) else np.nan

                # ── Temperature ──────────────────────────────────
                t_df = cell_df[cell_df["cycle_num"].between(2, 100)]
                row["Tavg_mean"] = float(t_df["Tavg"].mean())
                row["Tmax_max"]  = float(t_df["Tmax"].max())
                row["Tmin_min"]  = float(t_df["Tmin"].min())
                tavg_c100 = cell_df[cell_df["cycle_num"] == 100]["Tavg"].values
                row["Tavg_100"] = float(tavg_c100[0]) if len(tavg_c100) else np.nan

                # ── Fade slope (cycles 2–100) ────────────────────
                f_df = cell_df[cell_df["cycle_num"].between(2, 100)].dropna(subset=["QD"])
                if len(f_df) >= 2:
                    sl, ic, *_ = linregress(f_df["cycle_num"].values, f_df["QD"].values)
                    row["fade_slope"]     = float(sl)
                    row["fade_intercept"] = float(ic)
                else:
                    row["fade_slope"]     = np.nan
                    row["fade_intercept"] = np.nan

                # ── Timeseries from raw .mat ─────────────────────
                cell_ref   = batch["cycles"][i, 0]
                cell_group = f[cell_ref]
                n_mat      = cell_group["Qd"].shape[0]  # total cycles stored

                dVdQ_var, I_var = {}, {}
                for target in [10, 100]:
                    idx = target - 1  # 0-based
                    if idx >= n_mat:
                        dVdQ_var[target] = np.nan
                        I_var[target]    = np.nan
                        continue

                    V, Qd, I = _read_cycle_arrays(f, cell_group, idx)
                    if V is None or len(V) < 10:
                        dVdQ_var[target] = np.nan
                        I_var[target]    = np.nan
                        continue

                    dVdQ_var[target] = compute_dvdq_var(V, Qd)
                    # Severson: constant 4C discharge → I is negative constant; var ≈ 0
                    I_var[target] = float(np.var(I))

                row["dVdQ_var_10"]   = dVdQ_var[10]
                row["dVdQ_var_100"]  = dVdQ_var[100]
                row["dVdQ_var_diff"] = (dVdQ_var[100] - dVdQ_var[10]
                                        if np.isfinite(dVdQ_var[100]) and np.isfinite(dVdQ_var[10])
                                        else np.nan)

                row["I_var_10"]   = I_var[10]
                row["I_var_100"]  = I_var[100]
                row["I_var_diff"] = (I_var[100] - I_var[10]
                                     if np.isfinite(I_var[100]) and np.isfinite(I_var[10])
                                     else np.nan)

                rows.append(row)
                print(f"    ✓ {cell_id}  QD_100={row['QD_100']:.4f}  "
                      f"IR_diff={row['IR_diff']}  "
                      f"dVdQ_var_diff={row['dVdQ_var_diff']:.6f}  "
                      f"fade_slope={row['fade_slope']:.2e}")

    return pd.DataFrame(rows, columns=FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────
# HUST helpers
# ─────────────────────────────────────────────────────────────────

# NOTE: HUST IR is computed in clean_hust.py using the last CV-charge row
# as the OCV proxy and the first discharge row as the loaded voltage:
#   IR = (V_cv_last - V_dis_first) / (I_dis_first - I_cv_last)
# This is stored per cycle as 'ir_est' (Ohm) in hust_clean.pkl.


def _longest_cc_stage(V, I, Q, tol_frac=0.05):
    """
    Return (V_seg, Q_seg) for the longest constant-current discharge stage.
    Handles multi-stage HUST discharge protocols.
    """
    if len(I) < 10:
        return V, Q

    abs_I = np.abs(I)
    valid = abs_I > 0.05
    if valid.sum() < 5:
        return V, Q

    # Detect large current steps (stage transitions)
    dI = np.abs(np.diff(abs_I))
    thr = tol_frac * np.median(abs_I[valid])
    breaks = list(np.where(dI > thr)[0] + 1)
    bounds = [0] + breaks + [len(I)]

    # Find segment with maximum voltage span
    best_span, best_s, best_e = -1.0, 0, len(I)
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s >= 5:
            span = np.max(V[s:e]) - np.min(V[s:e])
            if span > best_span:
                best_span, best_s, best_e = span, s, e

    return V[best_s:best_e], Q[best_s:best_e]


# ─────────────────────────────────────────────────────────────────
# HUST
# ─────────────────────────────────────────────────────────────────

def extract_hust_features():
    print("\n" + "=" * 60)
    print("Extracting HUST features...")
    print("=" * 60)

    with open(HUST_CLEAN, "rb") as fh:
        hust_data = pickle.load(fh)

    rows = []

    for cell_id, cycles_list in sorted(hust_data.items()):
        row = {"cell_id": f"hust_{cell_id}"}
        cycle_map = {c["cycle"]: c for c in cycles_list}
        cycle_nums = sorted(cycle_map.keys())

        # ── QD_100 ─────────────────────────────────────────────
        if 100 in cycle_map:
            row["QD_100"] = float(np.max(cycle_map[100]["Capacity"]))
        else:
            row["QD_100"] = np.nan

        # ── IR: read pre-computed ir_est from clean pickle ─────
        # Computed in clean_hust.py via CV→discharge voltage step:
        #   IR = ΔV / ΔI at the charge/discharge phase boundary.
        def _ir_hust(cn):
            if cn not in cycle_map:
                return np.nan
            return float(cycle_map[cn].get("ir_est", float("nan")))

        row["IR_cycle2"]   = _ir_hust(2)
        row["IR_cycle100"] = _ir_hust(100)
        row["IR_diff"]     = (row["IR_cycle100"] - row["IR_cycle2"]
                              if np.isfinite(row["IR_cycle100"]) and np.isfinite(row["IR_cycle2"])
                              else np.nan)

        # ── dV/dQ variance ─────────────────────────────────────
        def _dvdq_hust(cn):
            if cn not in cycle_map:
                return np.nan
            c = cycle_map[cn]
            V_seg, Q_seg = _longest_cc_stage(c["V"], c["I"], c["Capacity"])
            if len(V_seg) < 10:
                return np.nan
            return compute_dvdq_var(V_seg, Q_seg)

        row["dVdQ_var_10"]   = _dvdq_hust(10)
        row["dVdQ_var_100"]  = _dvdq_hust(100)
        row["dVdQ_var_diff"] = (row["dVdQ_var_100"] - row["dVdQ_var_10"]
                                if np.isfinite(row["dVdQ_var_100"]) and np.isfinite(row["dVdQ_var_10"])
                                else np.nan)

        # ── Temperature — not available in HUST ────────────────
        row["Tavg_mean"] = np.nan
        row["Tmax_max"]  = np.nan
        row["Tmin_min"]  = np.nan
        row["Tavg_100"]  = np.nan

        # ── I_var ───────────────────────────────────────────────
        def _ivar_hust(cn):
            if cn not in cycle_map:
                return np.nan
            return float(np.var(cycle_map[cn]["I"]))

        row["I_var_10"]   = _ivar_hust(10)
        row["I_var_100"]  = _ivar_hust(100)
        row["I_var_diff"] = (row["I_var_100"] - row["I_var_10"]
                             if np.isfinite(row["I_var_100"]) and np.isfinite(row["I_var_10"])
                             else np.nan)

        # ── Charge time (cycles 2–6) ───────────────────────────
        ct_vals = [cycle_map[n]["charge_time_s"] for n in range(2, 7) if n in cycle_map]
        row["chargetime_s_mean_2to6"] = float(np.mean(ct_vals)) if ct_vals else np.nan

        # ── Fade slope (cycles 2–100) ──────────────────────────
        valid_cns = [n for n in cycle_nums if 2 <= n <= 100]
        if len(valid_cns) >= 2:
            qd_vals   = np.array([np.max(cycle_map[n]["Capacity"]) for n in valid_cns])
            c_arr     = np.array(valid_cns)
            sl, ic, *_ = linregress(c_arr, qd_vals)
            row["fade_slope"]     = float(sl)
            row["fade_intercept"] = float(ic)
        else:
            row["fade_slope"]     = np.nan
            row["fade_intercept"] = np.nan

        rows.append(row)
        print(f"  ✓ {cell_id}  QD_100={row['QD_100']:.4f}  "
              f"IR_diff={row['IR_diff']}  "
              f"dVdQ_var_diff={row['dVdQ_var_diff']}  "
              f"fade_slope={row['fade_slope']:.2e}")

    return pd.DataFrame(rows, columns=FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────

def save_features(df, prefix):
    paths = {
        "with_temp_pkl":    os.path.join(WORKSPACE, f"{prefix}_features.pkl"),
        "with_temp_csv":    os.path.join(WORKSPACE, f"{prefix}_features.csv"),
        "no_temp_pkl":      os.path.join(WORKSPACE, f"{prefix}_features_no_temp.pkl"),
        "no_temp_csv":      os.path.join(WORKSPACE, f"{prefix}_features_no_temp.csv"),
    }
    df.to_pickle(paths["with_temp_pkl"])
    df.to_csv(paths["with_temp_csv"], index=False)

    df_nt = df[NO_TEMP_COLS]
    df_nt.to_pickle(paths["no_temp_pkl"])
    df_nt.to_csv(paths["no_temp_csv"], index=False)

    print(f"\n  → {paths['with_temp_pkl']}")
    print(f"  → {paths['with_temp_csv']}")
    print(f"  → {paths['no_temp_pkl']}")
    print(f"  → {paths['no_temp_csv']}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sev_df = extract_severson_features()
    print(f"\nSeverson feature matrix: {sev_df.shape}")
    print(sev_df.drop("cell_id", axis=1).describe().round(5).to_string())
    save_features(sev_df, "severson")

    hust_df = extract_hust_features()
    print(f"\nHUST feature matrix: {hust_df.shape}")
    print(hust_df.drop(["cell_id"] + TEMP_COLS, axis=1).describe().round(5).to_string())
    save_features(hust_df, "hust")

    print("\n✅ All feature matrices extracted and saved!")
