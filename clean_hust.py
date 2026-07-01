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
        
        # ── CC Charge time ─────────────────────────────────────────
        cc_charge = df[df['Status'] == 'Constant current charge']
        if len(cc_charge) > 0:
            charge_time_s = float(cc_charge['Time (s)'].max() - cc_charge['Time (s)'].min())
        else:
            charge_time_s = 0.0
            
        # ── IR estimation: last CV charge row → first discharge row ─
        # Use the last row of the CV charge phase as the OCV proxy.
        # The voltage just before discharge load is applied is approximately OCV
        # since the CV phase tapers the current to near-zero.
        cv_charge = df[df['Status'] == 'Constant current-constant voltage charge']
        discharge = df[df['Status'].str.startswith('Constant current discharge')].copy()
        
        if len(cv_charge) > 0 and len(discharge) > 0:
            v_ocv  = float(cv_charge['Voltage (V)'].iloc[-1])
            i_ocv  = float(cv_charge['Current (mA)'].iloc[-1]) / 1000.0  # A (small, tapering)
            v_load = float(discharge.sort_values('Time (s)')['Voltage (V)'].iloc[0])
            i_load = float(discharge.sort_values('Time (s)')['Current (mA)'].iloc[0]) / 1000.0  # A (large, negative)

            delta_V = v_ocv - v_load            # voltage drop (positive)
            delta_I = abs(i_load - i_ocv)       # current step (positive, in A)

            if delta_I > 0.01 and delta_V > 0:
                ir_est = delta_V / delta_I       # Ohm
                # Sanity check: reasonable IR range for LFP cells (0.01–0.15 Ω)
                ir_est = float(ir_est) if 0.005 < ir_est < 0.15 else float('nan')
            else:
                ir_est = float('nan')
        else:
            ir_est = float('nan')
            
        # ── Discharge timeseries ───────────────────────────────────
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
            'charge_time_s': charge_time_s,
            'ir_est': ir_est          # DC IR from CV→discharge voltage step (Ω)
        })
        
    hust_clean[cell_id] = cycle_list

print("All cells processed. Saving to pickle file...")
with open(output_file, 'wb') as f:
    pickle.dump(hust_clean, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"HUST clean pickle saved successfully to {output_file}!")
