import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d

# Define the exact 13 features expected by the no-temperature model
NO_TEMP_FEATURES = [
    "QD_100", "IR_cycle2", "IR_cycle100", "IR_diff",
    "dVdQ_var_10", "dVdQ_var_100", "dVdQ_var_diff",
    "I_var_10", "I_var_100", "I_var_diff",
    "chargetime_s_mean_2to6",
    "fade_slope", "fade_intercept"
]

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

def extract_features_from_cycles(cycles_list):
    """
    Extracts the 13 no-temperature features from a list of cycle dictionaries (cycles 1 to 100).
    
    Parameters
    ----------
    cycles_list : list of dicts
        Each dict must contain keys: cycle, V, I, Capacity, t, charge_time_s, ir_est.
        
    Returns
    -------
    dict
        Feature name to value mapping.
    """
    cycle_map = {c["cycle"]: c for c in cycles_list}
    row = {}

    # 1. QD_100 (Discharge capacity at cycle 100)
    if 100 in cycle_map:
        row["QD_100"] = float(np.max(cycle_map[100]["Capacity"]))
    else:
        row["QD_100"] = np.nan

    # 2. IR features (IR_cycle2, IR_cycle100, IR_diff)
    row["IR_cycle2"]   = float(cycle_map[2]["ir_est"]) if 2 in cycle_map else np.nan
    row["IR_cycle100"] = float(cycle_map[100]["ir_est"]) if 100 in cycle_map else np.nan
    
    if np.isfinite(row["IR_cycle2"]) and np.isfinite(row["IR_cycle100"]):
        row["IR_diff"] = row["IR_cycle100"] - row["IR_cycle2"]
    else:
        row["IR_diff"] = np.nan

    # 3. dV/dQ variance features (dVdQ_var_10, dVdQ_var_100, dVdQ_var_diff)
    def _dvdq_var(cn):
        if cn not in cycle_map:
            return np.nan
        c = cycle_map[cn]
        V_seg, Q_seg = _longest_cc_stage(c["V"], c["I"], c["Capacity"])
        return compute_dvdq_var(V_seg, Q_seg)

    row["dVdQ_var_10"] = _dvdq_var(10)
    row["dVdQ_var_100"] = _dvdq_var(100)
    
    if np.isfinite(row["dVdQ_var_10"]) and np.isfinite(row["dVdQ_var_100"]):
        row["dVdQ_var_diff"] = row["dVdQ_var_100"] - row["dVdQ_var_10"]
    else:
        row["dVdQ_var_diff"] = np.nan

    # 4. Current variance features (I_var_10, I_var_100, I_var_diff)
    row["I_var_10"]  = float(np.var(cycle_map[10]["I"])) if 10 in cycle_map else np.nan
    row["I_var_100"] = float(np.var(cycle_map[100]["I"])) if 100 in cycle_map else np.nan
    
    if np.isfinite(row["I_var_10"]) and np.isfinite(row["I_var_100"]):
        row["I_var_diff"] = row["I_var_100"] - row["I_var_10"]
    else:
        row["I_var_diff"] = np.nan

    # 5. Charge time mean cycles 2–6
    ct_vals = [cycle_map[n]["charge_time_s"] for n in range(2, 7) if n in cycle_map]
    row["chargetime_s_mean_2to6"] = float(np.mean(ct_vals)) if ct_vals else np.nan

    # 6. Capacity fade slope & intercept cycles 2–100
    valid_cns = [n for n in sorted(cycle_map.keys()) if 2 <= n <= 100]
    if len(valid_cns) >= 2:
        qd_vals = np.array([np.max(cycle_map[n]["Capacity"]) for n in valid_cns])
        c_arr   = np.array(valid_cns)
        sl, ic, *_ = linregress(c_arr, qd_vals)
        row["fade_slope"]     = float(sl)
        row["fade_intercept"] = float(ic)
    else:
        row["fade_slope"]     = np.nan
        row["fade_intercept"] = np.nan

    return row

def get_feature_dataframe(row_dict):
    """Converts the feature dictionary to a pandas DataFrame with features in the correct order."""
    df = pd.DataFrame([row_dict])
    return df[NO_TEMP_FEATURES]
