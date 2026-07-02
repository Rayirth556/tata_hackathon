"""
tatahack3 / pipeline / extract_full_curves.py

Builds data/severson_full_curves.pkl from the 3 raw Severson .mat batch files,
for lstm_fallback.py's input:
    {cell_id: {"cycles": np.array, "QD": np.array, "total_cycles": int}}

The 3 raw files (place in data/raw/):
    2017-05-12_batchdata_updated_struct_errorcorrect.mat   -> batch 1 -> b1c*
    2017-06-30_batchdata_updated_struct_errorcorrect.mat   -> batch 2 -> b2c*
    2018-04-12_batchdata_updated_struct_errorcorrect.mat   -> batch 3 -> b3c*

Each file is HDF5-based (MATLAB v7.3), read with h5py. Every cell is a
top-level key like "cell_1_0", "cell_2_5", "cell_3_10", containing
"cycle" and "QDischarge" arrays of shape (1, n_cycles).

Run:
    python pipeline/extract_full_curves.py --raw_dir data/raw --out data/severson_full_curves.pkl
"""

import argparse
import os
import pickle

import h5py
import numpy as np

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def extract_batch(mat_path, batch_num):
    """Returns {cell_id: {'cycles': arr, 'QD': arr, 'total_cycles': int}} for one batch file."""
    out = {}
    with h5py.File(mat_path, "r") as f:
        cell_keys = [k for k in f.keys() if k.startswith(f"cell_{batch_num}_")]
        for key in cell_keys:
            cell_idx = key.split("_")[-1]
            cell_id = f"b{batch_num}c{cell_idx}"

            cycles = np.asarray(f[key]["cycle"][()]).flatten()
            qd = np.asarray(f[key]["QDischarge"][()]).flatten()

            # drop NaNs / keep aligned
            mask = ~(np.isnan(cycles) | np.isnan(qd))
            cycles, qd = cycles[mask], qd[mask]

            if len(cycles) == 0:
                print(f"[WARN] {cell_id}: no valid cycle/QD data, skipping")
                continue

            order = np.argsort(cycles)
            cycles, qd = cycles[order], qd[order]

            out[cell_id] = {
                "cycles": cycles,
                "QD": qd,
                "total_cycles": int(cycles.max()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw",
                     help="folder containing the 3 raw .mat files")
    ap.add_argument("--out", default="data/severson_full_curves.pkl")
    args = ap.parse_args()

    all_curves = {}
    for batch_num, fname in BATCH_FILES.items():
        mat_path = os.path.join(args.raw_dir, fname)
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing raw file: {mat_path}")
        print(f"[batch {batch_num}] reading {mat_path} ...")
        batch_curves = extract_batch(mat_path, batch_num)
        print(f"[batch {batch_num}] extracted {len(batch_curves)} cells")
        all_curves.update(batch_curves)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(all_curves, f)

    n_short = sum(1 for c in all_curves.values() if c["total_cycles"] < 100)
    print(f"\n[DONE] {len(all_curves)} total cells written to {args.out}")
    print(f"[NOTE] {n_short} cells have fewer than 100 observed cycles "
          f"(lstm_fallback.py will pad these with their last value).")


if __name__ == "__main__":
    main()
