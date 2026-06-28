import h5py
import numpy as np
import pandas as pd
import os

batches = {
    1: '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    2: '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    3: '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
}

workspace_dir = '/home/godkiller/Documents/tata'
output_file = os.path.join(workspace_dir, 'severson_clean.pkl')

print("Starting Severson Data Cleaning (with outlier interpolation & zero-to-NaN conversion)...")

all_data = []

# Exclusions mapping (batch_id -> list of cell indices to exclude)
exclusions = {
    1: [0, 2, 3],
    2: [],
    3: [3] + list(range(11, 46))
}

cells_processed = 0
cells_dropped = 0

for b_id, filename in batches.items():
    path = os.path.join(workspace_dir, filename)
    print(f"Loading Batch {b_id} from {filename}...")
    
    with h5py.File(path, 'r') as f:
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        
        for i in range(num_cells):
            cell_id = f"b{b_id}c{i}"
            
            # Check exclusions
            if i in exclusions[b_id]:
                print(f"  Excluding cell {cell_id}")
                cells_dropped += 1
                continue
                
            # Read cell summary data
            summary_group = f[batch['summary'][i, 0]]
            
            cycles = np.squeeze(np.array(summary_group['cycle']))
            qd = np.squeeze(np.array(summary_group['QDischarge'])).copy()
            qc = np.squeeze(np.array(summary_group['QCharge'])).copy()
            ir = np.squeeze(np.array(summary_group['IR'])).copy()
            tavg = np.squeeze(np.array(summary_group['Tavg']))
            tmin = np.squeeze(np.array(summary_group['Tmin']))
            tmax = np.squeeze(np.array(summary_group['Tmax']))
            chargetime = np.squeeze(np.array(summary_group['chargetime'])).copy()
            
            # Interpolate outliers for cycles 2 to 100 (indices 1 to 99)
            # We do this first so that spikes are corrected before zero-to-NaN logic
            for c_idx in range(1, 100):
                # Interpolate QD if outside sane range (excluding 0.0 which represents a missing value)
                if qd[c_idx] != 0.0 and (qd[c_idx] < 0.5 or qd[c_idx] > 1.3):
                    prev_idx = c_idx - 1
                    while prev_idx >= 1 and (qd[prev_idx] == 0.0 or qd[prev_idx] < 0.5 or qd[prev_idx] > 1.3):
                        prev_idx -= 1
                    next_idx = c_idx + 1
                    while next_idx < len(qd) and (qd[next_idx] == 0.0 or qd[next_idx] < 0.5 or qd[next_idx] > 1.3):
                        next_idx += 1
                    
                    if next_idx < len(qd):
                        qd[c_idx] = qd[prev_idx] + (qd[next_idx] - qd[prev_idx]) * (c_idx - prev_idx) / (next_idx - prev_idx)
                    else:
                        qd[c_idx] = qd[prev_idx]
                    print(f"  [Interpolated QD] Cell {cell_id} cycle {c_idx+1}: set to {qd[c_idx]:.4f} Ah")
                
                # Interpolate QC if outside sane range (excluding 0.0)
                if qc[c_idx] != 0.0 and (qc[c_idx] < 0.5 or qc[c_idx] > 1.3):
                    prev_idx = c_idx - 1
                    while prev_idx >= 1 and (qc[prev_idx] == 0.0 or qc[prev_idx] < 0.5 or qc[prev_idx] > 1.3):
                        prev_idx -= 1
                    next_idx = c_idx + 1
                    while next_idx < len(qc) and (qc[next_idx] == 0.0 or qc[next_idx] < 0.5 or qc[next_idx] > 1.3):
                        next_idx += 1
                        
                    if next_idx < len(qc):
                        qc[c_idx] = qc[prev_idx] + (qc[next_idx] - qc[prev_idx]) * (c_idx - prev_idx) / (next_idx - prev_idx)
                    else:
                        qc[c_idx] = qc[prev_idx]
                    print(f"  [Interpolated QC] Cell {cell_id} cycle {c_idx+1}: set to {qc[c_idx]:.4f} Ah")
            
            # Extract first 100 cycles
            for c_idx in range(100):
                cycle_num = int(cycles[c_idx])
                
                # Zero-to-NaN conversion for missing measurements
                val_qd = float(qd[c_idx]) if qd[c_idx] != 0.0 else np.nan
                val_qc = float(qc[c_idx]) if qc[c_idx] != 0.0 else np.nan
                val_ir = float(ir[c_idx]) if ir[c_idx] != 0.0 else np.nan
                val_chargetime = float(chargetime[c_idx]) if chargetime[c_idx] != 0.0 else np.nan
                
                all_data.append({
                    'cell_id': cell_id,
                    'cycle_num': cycle_num,
                    'QD': val_qd,
                    'QC': val_qc,
                    'IR': val_ir,
                    'Tavg': float(tavg[c_idx]),
                    'Tmin': float(tmin[c_idx]),
                    'Tmax': float(tmax[c_idx]),
                    'chargetime': val_chargetime
                })
                
            cells_processed += 1

# Create DataFrame
df = pd.DataFrame(all_data)

# Ensure correct column dtypes
df['cell_id'] = df['cell_id'].astype(str)
df['cycle_num'] = df['cycle_num'].astype(np.int64)
for col in ['QD', 'QC', 'IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']:
    df[col] = df[col].astype(np.float64)

print(f"\nProcessing Complete!")
print(f"Processed Cells: {cells_processed}")
print(f"Dropped Cells: {cells_dropped}")
print(f"DataFrame Shape: {df.shape}")

# Save to pickle
df.to_pickle(output_file)
print(f"Saved cleaned Severson data to {output_file}")
