import os
import pickle
import numpy as np
import pandas as pd

our_data_dir = '/home/godkiller/Documents/tata/our_data'
output_file = '/home/godkiller/Documents/tata/hust_clean.pkl'

files = sorted([f for f in os.listdir(our_data_dir) if f.endswith('.pkl')])

print(f"Starting HUST Data Cleaning on {len(files)} cells...")

hust_clean = {}

for idx, f_name in enumerate(files):
    cell_id = f_name.replace('.pkl', '')
    path = os.path.join(our_data_dir, f_name)
    print(f"[{idx+1}/{len(files)}] Processing cell {cell_id}...")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    cell_data = data[cell_id]
    cycles_dict = cell_data['data']
    
    cycle_list = []
    
    for c_num in sorted(cycles_dict.keys()):
        df = cycles_dict[c_num]
        
        # Calculate charge time from CC charge phase
        cc_charge = df[df['Status'] == 'Constant current charge']
        if len(cc_charge) > 0:
            charge_time_s = float(cc_charge['Time (s)'].max() - cc_charge['Time (s)'].min())
        else:
            charge_time_s = 0.0
            
        # Extract discharge phase
        discharge = df[df['Status'].str.startswith('Constant current discharge')].copy()
        if len(discharge) == 0:
            continue
            
        discharge = discharge.sort_values('Time (s)')
        
        V = discharge['Voltage (V)'].values.astype(np.float64)
        I = (discharge['Current (mA)'].values / 1000.0).astype(np.float64)
        
        # Cumulative capacity removed in Ah
        start_cap = discharge['Capacity (mAh)'].iloc[0]
        Capacity = ((start_cap - discharge['Capacity (mAh)'].values) / 1000.0).astype(np.float64)
        
        t = discharge['Time (s)'].values.astype(np.float64)
        t = t - t[0]  # Normalize time to start at 0
        
        cycle_list.append({
            'cycle': int(c_num),
            'V': V,
            'I': I,
            'Capacity': Capacity,
            't': t,
            'charge_time_s': charge_time_s
        })
        
    hust_clean[cell_id] = cycle_list

print("All cells processed. Saving to pickle file...")
with open(output_file, 'wb') as f:
    pickle.dump(hust_clean, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"HUST clean pickle saved successfully to {output_file}!")
