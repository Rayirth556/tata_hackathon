"""
extract_full_curves.py — Extract full lifetime QD curves from raw Severson .mat files.

Reads the 3 raw Severson batch .mat files via h5py and extracts the COMPLETE
QDischarge summary array for each of the 101 surviving cells (all cycles, not
just 1-100). Applies the same exclusion list as clean_severson.py.

Output: severson_full_curves.pkl — dict keyed by cell_id, each value is a dict
with keys 'cycles' (np.array), 'QD' (np.array), 'total_cycles' (int).
"""

import h5py
import numpy as np
import pickle
import os

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'

BATCHES = {
    1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}

# Same exclusion list as clean_severson.py
EXCLUSIONS = {
    1: [0, 2, 3],
    2: [],
    3: [3] + list(range(11, 46)),
}

OUTPUT_FILE = os.path.join(WORKSPACE, 'data/severson_full_curves.pkl')


def extract_full_curves():
    """Extract full lifetime QD capacity curves for all surviving Severson cells."""
    results = {}
    cells_processed = 0
    cells_dropped = 0

    for b_id, filename in BATCHES.items():
        path = os.path.join(WORKSPACE, filename)
        print(f"Loading Batch {b_id} from {filename}...")

        with h5py.File(path, 'r') as f:
            batch = f['batch']
            num_cells = batch['summary'].shape[0]

            for i in range(num_cells):
                cell_id = f"b{b_id}c{i}"

                if i in EXCLUSIONS[b_id]:
                    cells_dropped += 1
                    continue

                # Read the full summary arrays
                summary_group = f[batch['summary'][i, 0]]
                cycles = np.squeeze(np.array(summary_group['cycle'])).astype(int)
                qd = np.squeeze(np.array(summary_group['QDischarge'])).astype(np.float64).copy()

                total_cycles = len(cycles)

                # ── Zero-to-NaN conversion ─────────────────────────────────
                # Zero values represent missing measurements (e.g. cycle 1 in
                # Batch 1). Replace with NaN so they don't corrupt the curve.
                qd[qd == 0.0] = np.nan

                # ── Outlier interpolation over entire curve ────────────────
                # Same logic as clean_severson.py: if QD is outside [0.5, 1.3]
                # (and not NaN), interpolate from nearest valid neighbours.
                for c_idx in range(1, len(qd)):
                    if np.isnan(qd[c_idx]):
                        continue
                    if qd[c_idx] < 0.5 or qd[c_idx] > 1.3:
                        # Find previous valid value
                        prev_idx = c_idx - 1
                        while prev_idx >= 0 and (np.isnan(qd[prev_idx]) or qd[prev_idx] < 0.5 or qd[prev_idx] > 1.3):
                            prev_idx -= 1
                        # Find next valid value
                        next_idx = c_idx + 1
                        while next_idx < len(qd) and (np.isnan(qd[next_idx]) or qd[next_idx] < 0.5 or qd[next_idx] > 1.3):
                            next_idx += 1

                        if prev_idx >= 0 and next_idx < len(qd):
                            qd[c_idx] = qd[prev_idx] + (qd[next_idx] - qd[prev_idx]) * \
                                        (c_idx - prev_idx) / (next_idx - prev_idx)
                        elif prev_idx >= 0:
                            qd[c_idx] = qd[prev_idx]
                        # else: leave as is (no valid reference)

                results[cell_id] = {
                    'cycles': cycles,
                    'QD': qd,
                    'total_cycles': total_cycles,
                }
                cells_processed += 1

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*60}")
    print(f"Full curve extraction complete.")
    print(f"  Cells processed: {cells_processed}")
    print(f"  Cells dropped:   {cells_dropped}")
    print(f"  Output:          {OUTPUT_FILE}")
    print(f"  File size:       {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    # Print a summary table
    print(f"\n{'Cell':<10} {'Total Cycles':>13} {'QD Start':>10} {'QD End':>10} {'QD Min':>10}")
    print('-' * 55)
    for cid in sorted(results.keys()):
        d = results[cid]
        qd_valid = d['QD'][~np.isnan(d['QD'])]
        print(f"{cid:<10} {d['total_cycles']:>13} {qd_valid[0]:>10.4f} {qd_valid[-1]:>10.4f} {qd_valid.min():>10.4f}")


if __name__ == '__main__':
    extract_full_curves()
