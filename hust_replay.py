"""
hust_replay.py
===============
BMS telemetry replay simulator for the HUST dataset.

Provides a generator function that streams HUST cycle dictionaries
one-by-one to simulate a real-time edge-device BMS data feed.

Usage:
    from hust_replay import hust_replay

    for cycle_data in hust_replay("hust_clean.pkl", cell_id="1-1", start_cycle=1, end_cycle=100):
        V   = cycle_data["V"]          # Voltage array (V)
        I   = cycle_data["I"]          # Current array (A)
        Q   = cycle_data["Capacity"]   # Cumulative capacity removed (Ah)
        t   = cycle_data["t"]          # Time from cycle start (s)
        cyc = cycle_data["cycle"]      # Cycle number
        ct  = cycle_data["charge_time_s"]  # CC charge duration (s)
        # … run inference here …
"""

import pickle
import numpy as np
from typing import Generator, Dict


def hust_replay(
    pkl_path: str,
    cell_id: str,
    start_cycle: int = 1,
    end_cycle: int = 100,
) -> Generator[Dict, None, None]:
    """
    Generator that yields HUST cycle dictionaries in cycle-number order.

    Parameters
    ----------
    pkl_path : str
        Absolute path to hust_clean.pkl.
    cell_id : str
        HUST cell identifier, e.g. '1-1', '3-5'.
    start_cycle : int
        First cycle to yield (inclusive). Default: 1.
    end_cycle : int
        Last cycle to yield (inclusive). Default: 100.

    Yields
    ------
    dict with keys:
        cycle          : int    — cycle number
        V              : ndarray[float64] — voltage (V)
        I              : ndarray[float64] — current (A, negative = discharge)
        Capacity       : ndarray[float64] — cumulative capacity removed (Ah)
        t              : ndarray[float64] — time from cycle start (s)
        charge_time_s  : float  — CC charge phase duration (s)

    Raises
    ------
    KeyError
        If cell_id is not found in the pickle file.
    ValueError
        If no cycles exist in the requested range.
    """
    with open(pkl_path, "rb") as fh:
        hust_data = pickle.load(fh)

    if cell_id not in hust_data:
        available = sorted(hust_data.keys())
        raise KeyError(
            f"Cell '{cell_id}' not found. Available cell IDs: {available[:10]}..."
        )

    cycles_list = hust_data[cell_id]

    # Filter and sort by cycle number
    filtered = sorted(
        [c for c in cycles_list if start_cycle <= c["cycle"] <= end_cycle],
        key=lambda x: x["cycle"],
    )

    if not filtered:
        raise ValueError(
            f"No cycles found for cell '{cell_id}' in range [{start_cycle}, {end_cycle}]. "
            f"Available cycles: {sorted(c['cycle'] for c in cycles_list)[:10]}..."
        )

    for cyc in filtered:
        # Ensure timestamps start at 0 (relative to cycle start)
        t = cyc["t"].copy()
        if len(t) > 0 and t[0] != 0.0:
            t = t - t[0]

        yield {
            "cycle":         int(cyc["cycle"]),
            "V":             cyc["V"].astype(np.float64),
            "I":             cyc["I"].astype(np.float64),
            "Capacity":      cyc["Capacity"].astype(np.float64),
            "t":             t.astype(np.float64),
            "charge_time_s": float(cyc["charge_time_s"]),
        }


# ─────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    PKL_PATH = os.path.join(os.path.dirname(__file__), "hust_clean.pkl")
    DEMO_CELL = "1-1"

    print(f"HUST Replay Demo — cell '{DEMO_CELL}', cycles 1–10\n")
    print(f"{'Cycle':>6}  {'Samples':>8}  {'t_end (s)':>10}  "
          f"{'V_min':>7}  {'V_max':>7}  {'Q_max (Ah)':>11}  {'charge_t (s)':>13}")
    print("-" * 75)

    for cyc_data in hust_replay(PKL_PATH, DEMO_CELL, start_cycle=1, end_cycle=10):
        print(
            f"{cyc_data['cycle']:>6}  "
            f"{len(cyc_data['V']):>8}  "
            f"{cyc_data['t'][-1]:>10.1f}  "
            f"{cyc_data['V'].min():>7.4f}  "
            f"{cyc_data['V'].max():>7.4f}  "
            f"{cyc_data['Capacity'].max():>11.4f}  "
            f"{cyc_data['charge_time_s']:>13.1f}"
        )
