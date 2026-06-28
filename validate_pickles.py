import pickle
import pandas as pd
import numpy as np

severson_path = '/home/godkiller/Documents/tata/severson_clean.pkl'
hust_path = '/home/godkiller/Documents/tata/hust_clean.pkl'

print("=== Validating Severson Clean Pickle ===")
try:
    df = pd.read_pickle(severson_path)
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", list(df.columns))
    print("Surviving cells count:", df['cell_id'].nunique())
    print("Cycles per cell:", df.groupby('cell_id').size().unique())
    print("Any NaN values?", df.isna().any().any())
    print("Dtypes:\n", df.dtypes)
    print("QD min/max:", df['QD'].min(), df['QD'].max())
    print("QC min/max:", df['QC'].min(), df['QC'].max())
    print("IR min/max:", df['IR'].min(), df['IR'].max())
    print("chargetime min/max:", df['chargetime'].min(), df['chargetime'].max())
except Exception as e:
    print("Error validating Severson:", e)

print("\n=== Validating HUST Clean Pickle ===")
try:
    with open(hust_path, 'rb') as f:
        hust_data = pickle.load(f)
    print("Total cells in HUST dictionary:", len(hust_data.keys()))
    
    first_cell_id = list(hust_data.keys())[0]
    first_cell = hust_data[first_cell_id]
    print(f"Sample cell ID: {first_cell_id}")
    print(f"Number of cycles for sample cell: {len(first_cell)}")
    
    # Check fields in first cycle
    first_cycle = first_cell[0]
    print("First cycle keys:", list(first_cycle.keys()))
    for key in ['V', 'I', 'Capacity', 't']:
        val = first_cycle[key]
        print(f"  Field '{key}': type={type(val)}, shape={val.shape}, dtype={val.dtype}")
        
    print(f"  Field 'charge_time_s': type={type(first_cycle['charge_time_s'])}, value={first_cycle['charge_time_s']}")
    
    # Check overall limits and ranges
    nan_found = False
    cycle_count = 0
    cap_ranges = []
    
    for cid, cycles in hust_data.items():
        cycle_count += len(cycles)
        for cyc in cycles:
            for key in ['V', 'I', 'Capacity', 't']:
                if np.isnan(cyc[key]).any():
                    nan_found = True
            cap_ranges.append(cyc['Capacity'].max())
            
    print("Total cycles across HUST cells:", cycle_count)
    print("NaN values found in any array?", nan_found)
    print("Capacity (Ah) max range across all cycles:", min(cap_ranges), "to", max(cap_ranges))
    
except Exception as e:
    print("Error validating HUST:", e)
