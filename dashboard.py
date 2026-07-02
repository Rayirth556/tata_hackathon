import os
import sys
import json
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import serial
import serial.tools.list_ports

# Add project root to path
WORKSPACE = '/home/godkiller/Documents/tata'
sys.path.append(WORKSPACE)

from edge.hust_replay import hust_replay
from edge.feature_extractor import extract_features_from_cycles, get_feature_dataframe, NO_TEMP_FEATURES

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="VitalEdge Edge AI Control Room",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        color: #202124;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #e43f5a 0%, #1f4068 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(228,63,90,0.4);
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        color: #202124;
    }
    .metric-title {
        font-size: 14px;
        color: #5f6368;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
        color: #202124;
    }
    .metric-status {
        font-size: 14px;
        font-weight: bold;
        border-radius: 20px;
        padding: 4px 12px;
        display: inline-block;
    }
    .status-red { background-color: rgba(228, 63, 90, 0.1); color: #d93025; border: 1px solid #d93025; }
    .status-yellow { background-color: rgba(245, 166, 35, 0.1); color: #f29900; border: 1px solid #f29900; }
    .status-green { background-color: rgba(76, 175, 80, 0.1); color: #1e8e3e; border: 1px solid #1e8e3e; }
</style>
""", unsafe_allow_html=True)

# ── Caching Resources for Quick UI Reloads ────────────────────────────────────

def get_esp32_footprints():
    """Dynamically extracts flash size from .bin and DRAM size from ELF using size utility."""
    bin_path = os.path.join(WORKSPACE, 'edge/esp32_project/build/vitaledge_inference.bin')
    elf_path = os.path.join(WORKSPACE, 'edge/esp32_project/build/vitaledge_inference.elf')
    
    flash_str = "169.5 KB (fallback)"
    dram_str = "12.9 KB (fallback)"
    
    if os.path.exists(bin_path):
        flash_str = f"{os.path.getsize(bin_path) / 1024.0:.1f} KB"
        
    if os.path.exists(elf_path):
        try:
            proc = subprocess.run(['size', elf_path], text=True, capture_output=True, check=True)
            lines = proc.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                data_sz = int(parts[1])
                bss_sz = int(parts[2])
                dram_str = f"{(data_sz + bss_sz) / 1024.0:.1f} KB"
        except Exception:
            pass
            
    return flash_str, dram_str

@st.cache_resource
def load_models_and_explainer():
    """Loads XGBoost models and pre-builds the SHAP TreeExplainer."""
    model_reg_path = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
    model_clf_path = os.path.join(WORKSPACE, 'models/trained_model_clf.pkl')
    
    if not os.path.exists(model_reg_path) or not os.path.exists(model_clf_path):
        st.error("Missing trained models. Run model training first!")
        st.stop()
        
    reg_model = joblib.load(model_reg_path)
    clf_model = joblib.load(model_clf_path)
    
    # Initialize explainer using path-dependent perturbation (avoids categorical split issues)
    explainer = shap.TreeExplainer(reg_model)
    
    return reg_model, clf_model, explainer


@st.cache_data
def get_hust_cells(clean_pkl_path):
    """Loads available HUST cells from pickle."""
    with open(clean_pkl_path, "rb") as fh:
        hust_data = pickle.load(fh)
    return sorted(hust_data.keys())

# Initialize data paths
clean_pkl = os.path.join(WORKSPACE, 'data/hust_clean.pkl')
hust_cells = get_hust_cells(clean_pkl)

# Load resources
reg_model, clf_model, shap_explainer = load_models_and_explainer()

# ── Sidebar UI (Control Room) ──────────────────────────────────────────────────

st.sidebar.markdown(
    "<div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>"
    "<h2 style='color: white; margin: 0;'>🔋 VitalEdge</h2>"
    "<p style='color: #9aa0a6; font-size: 12px; margin: 5px 0 0 0;'>EDGE AI LIFETIME PREDICTION</p>"
    "</div>", 
    unsafe_allow_html=True
)

st.sidebar.header("🕹️ Control Panel")

# Initialize session state for connection mode fallback and warning
if 'conn_mode' not in st.session_state:
    st.session_state.conn_mode = "Host Simulation (Local C Binary)"
if 'warning_msg' not in st.session_state:
    st.session_state.warning_msg = None

# Select cell
selected_cell = st.sidebar.selectbox("Select Battery Cell ID", hust_cells, index=0)

# Connect to serial or run local host simulation
conn_mode_selection = st.sidebar.radio(
    "Target Hardware Connection",
    ["Host Simulation (Local C Binary)", "Live ESP32 Hardware (Serial UART)"],
    index=0 if st.session_state.conn_mode == "Host Simulation (Local C Binary)" else 1,
    key="conn_mode_widget"
)
st.session_state.conn_mode = conn_mode_selection
conn_mode = st.session_state.conn_mode

# Auto-detect serial ports
available_ports = [p.device for p in serial.tools.list_ports.comports()]
default_port = available_ports[0] if available_ports else "/dev/ttyUSB0"

serial_port = st.sidebar.text_input("Serial Port Device", value=default_port, 
                                    help="Path to ESP32 serial port (e.g. COM3 or /dev/ttyUSB0)")

run_inference = st.sidebar.button("Run Telemetry & Edge Inference")

# ── Main Page Header ──────────────────────────────────────────────────────────

st.markdown(
    "<div style='background: linear-gradient(135deg, #162447 0%, #1f4068 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.4);'>"
    "<h1 style='color: white; margin: 0;'>VitalEdge Edge AI Control Room</h1>"
    "<p style='color: #9aa0a6; font-size: 14px; margin: 5px 0 0 0;'>Lithium-ion Battery Prognostics, RUL Estimation, and Hardware Benchmarks</p>"
    "</div>", 
    unsafe_allow_html=True
)

if st.session_state.warning_msg:
    st.warning(st.session_state.warning_msg)
    st.session_state.warning_msg = None

# ── Actions on Trigger ────────────────────────────────────────────────────────

if run_inference:
    # ── 1. Telemetry Replay & Feature Extraction ──────────────────────────────
    with st.spinner(f"Replaying raw telemetry for cell {selected_cell} (cycles 1 to 100)..."):
        cycles = []
        try:
            for cycle_data in hust_replay(clean_pkl, cell_id=selected_cell, start_cycle=1, end_cycle=100):
                cycles.append(cycle_data)
        except Exception as e:
            st.error(f"Error streaming HUST cell: {e}")
            st.stop()
            
        # Extract features
        features_dict = extract_features_from_cycles(cycles)
        features_df = get_feature_dataframe(features_dict)
        
    # ── 2. Run Inference ──────────────────────────────────────────────────────
    knee_cycle = 0.0
    early_prob = 0.0
    reg_latency_us = 0.0
    clf_latency_us = 0.0
    precision_alignment = "❌ Disconnected"
    inference_source = ""

    if conn_mode == "Live ESP32 Hardware (Serial UART)":
        with st.spinner("Connecting to ESP32 board over Serial..."):
            try:
                # Format features into comma-separated string
                csv_payload = ",".join([f"{features_dict[col]:.17g}" for col in NO_TEMP_FEATURES])
                
                # Initialize serial object
                ser = serial.Serial()
                ser.port = serial_port
                ser.baudrate = 115200
                ser.timeout = 3.0
                
                response_found = False
                knee_cycle, early_prob, reg_latency_us, clf_latency_us = 0.0, 0.0, 0.0, 0.0
                
                # Attempt communication up to 3 times to handle potential noise/dropped chars
                for attempt in range(3):
                    try:
                        if not ser.is_open:
                            ser.open()
                            ser.setDTR(False)
                            ser.setRTS(False)
                            time.sleep(2.0) # Wait for ESP32 boot/reset to finish
                        
                        ser.reset_input_buffer()
                        ser.reset_output_buffer()
                        
                        # Send payload
                        ser.write(f"{csv_payload}\n".encode())
                        
                        start_read_time = time.time()
                        while time.time() - start_read_time < 3.0:
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            if line.startswith('{') and line.endswith('}'):
                                try:
                                    res_data = json.loads(line)
                                    knee_cycle = res_data["knee_cycle"]
                                    early_prob = res_data["early_prob"]
                                    reg_latency_us = res_data["reg_latency_us"]
                                    clf_latency_us = res_data["clf_latency_us"]
                                    response_found = True
                                    break
                                except (json.JSONDecodeError, KeyError):
                                    continue
                        
                        if response_found:
                            break
                    except Exception:
                        pass
                    finally:
                        if ser.is_open:
                            ser.close()
                        time.sleep(0.5)
                
                if not response_found:
                    raise ValueError("Failed to receive a valid JSON telemetry payload after 3 attempts.")
                
                # Perform live Python ↔ C alignment check!
                py_reg_pred = reg_model.predict(features_df)[0]
                py_clf_pred = clf_model.predict_proba(features_df)[0][1]
                
                reg_diff = abs(py_reg_pred - knee_cycle)
                clf_diff = abs(py_clf_pred - early_prob)
                
                if reg_diff < 1e-4 and clf_diff < 1e-4:
                    precision_alignment = "✅ Pass (Numerically Exact)"
                else:
                    precision_alignment = f"⚠️ Mismatch (Reg diff={reg_diff:.5f})"
                
                inference_source = "ESP32 Hardware"
                st.success(f"Successfully received inference payload from ESP32 on {serial_port}!")
                
            except Exception as e:
                st.session_state.warning_msg = f"Failed to communicate with ESP32 on {serial_port}: {e}. Falling back to Local Host C Binary Simulation."
                st.session_state.conn_mode = "Host Simulation (Local C Binary)"
                st.rerun()

    # Fallback / Local Simulation mode
    if conn_mode == "Host Simulation (Local C Binary)":
        c_bin_path = os.path.join(WORKSPACE, 'edge', 'host_test', 'host_test_bin')
        if not os.path.exists(c_bin_path):
            st.error("Missing compiled C host binary. Compile it in edge/host_test/ first!")
            st.stop()
            
        with st.spinner("Running inference using compiled C binary..."):
            input_str = ",".join([f"{features_dict[col]:.17g}" for col in NO_TEMP_FEATURES])
            try:
                proc = subprocess.run([c_bin_path], input=input_str + "\n", text=True, capture_output=True, check=True)
                c_output = proc.stdout.strip().split('\n')
                
                c_results = {}
                for line in c_output:
                    if ':' in line:
                        k, v = line.split(':')
                        c_results[k.strip()] = float(v.strip())
                
                knee_cycle = c_results["regression_prediction"]
                early_prob = c_results["classification_prediction_prob_1"]
                
                # Fetch host C simulation latency
                reg_latency_us = c_results.get("reg_latency_us", 0.0)
                clf_latency_us = c_results.get("clf_latency_us", 0.0)
                
                # Perform live Python ↔ C alignment check!
                py_reg_pred = reg_model.predict(features_df)[0]
                py_clf_pred = clf_model.predict_proba(features_df)[0][1]
                
                reg_diff = abs(py_reg_pred - knee_cycle)
                clf_diff = abs(py_clf_pred - early_prob)
                
                if reg_diff < 1e-4 and clf_diff < 1e-4:
                    precision_alignment = "✅ Pass (Numerically Exact)"
                else:
                    precision_alignment = f"⚠️ Mismatch (Reg diff={reg_diff:.5f})"
                
                inference_source = "Host C Simulation"
                
            except Exception as e:
                st.error(f"Error running host C simulator: {e}")
                st.stop()

    # Calculate metrics
    rul_cycles = max(0.0, knee_cycle - 100.0)
    
    # Determine dynamic status text and colors based on early risk probability and knee predictions
    if early_prob > 0.8:
        card_a_status_class = "status-red"
        card_a_status_text = "Accelerated Aging"
        card_a_value_color = "#d32f2f" # Red
        
        card_b_status_class = "status-red"
        card_b_status_text = f"Early Knee (<464 Threshold)"
        card_b_value_color = "#d32f2f" # Red
        
        status_class = "status-red"
        status_text = "HIGH RISK - Flag for Maintenance"
    elif early_prob > 0.2:
        card_a_status_class = "status-yellow"
        card_a_status_text = "Moderate Wear"
        card_a_value_color = "#f57c00" # Orange/Yellow
        
        card_b_status_class = "status-yellow"
        card_b_status_text = "Moderate Degradation"
        card_b_value_color = "#f57c00" # Orange/Yellow
        
        status_class = "status-yellow"
        status_text = "MODERATE RISK - Monitor Cell"
    else:
        card_a_status_class = "status-green"
        card_a_status_text = "Healthy Operation"
        card_a_value_color = "#388e3c" # Green
        
        card_b_status_class = "status-green"
        card_b_status_text = "Normal Degradation (≥464)"
        card_b_value_color = "#388e3c" # Green
        
        status_class = "status-green"
        status_text = "LOW RISK - Normal Degradation"

    col1, col2, col3 = st.columns(3)
    
    # Card A: RUL
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Remaining Useful Life (RUL)</div>
            <div class="metric-value" style="color: {card_a_value_color};">{rul_cycles:.1f} Cycles</div>
            <div class="metric-status {card_a_status_class}">{card_a_status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Card B: Knee Point
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Knee Point</div>
            <div class="metric-value" style="color: {card_b_value_color};">Cycle {knee_cycle:.1f}</div>
            <div class="metric-status {card_b_status_class}">{card_b_status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Card C: Early Degradation Risk
    with col3:
        risk_pct = early_prob * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Early Degradation Risk</div>
            <div class="metric-value">{risk_pct:.1f}%</div>
            <div class="metric-status {status_class}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Row 2: Telemetry Plot & Hardware Metrics ──────────────────────────────
    
    col_left, col_right = st.columns([2, 1.2])
    
    with col_left:
        st.subheader("📈 Early-Life Telemetry (Discharge Capacity)")
        # Plot capacity fade over the first 100 cycles
        cap_history = [np.max(c["Capacity"]) for c in cycles]
        cycle_numbers = [c["cycle"] for c in cycles]
        
        # Make the graph smaller (6x3 instead of 8x4) and adjust theme colors
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fa')
        
        ax.plot(cycle_numbers, cap_history, color='#e43f5a', linewidth=2.0, label='Capacity Decay')
        ax.scatter([100], [cap_history[-1]], color='#1e8e3e', s=80, zorder=5, label='Current State (C100)')
        
        ax.set_xlabel("Cycle Number", color='#202124', fontsize=10)
        ax.set_ylabel("Capacity (Ah)", color='#202124', fontsize=10)
        ax.tick_params(colors='#202124', labelsize=9)
        ax.grid(color='#e0e0e0', linestyle='--', alpha=0.7)
        ax.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#202124', fontsize=9)
        plt.title(f"HUST Cell {selected_cell} Telemetry Curve", color='#202124', fontsize=11, fontweight='bold')
        fig.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig) # Explicitly close figure to prevent memory leak
        
    with col_right:
        st.subheader("⚡ Edge Performance Footprint")
        
        # Fetch dynamic footprints from the compiled binary files
        flash_size, dram_size = get_esp32_footprints()
        
        # Display specs in tabular format
        metrics_table = {
            "Spec / Measurement": [
                "Target Hardware",
                "Inference Engine",
                "Flash Binary Footprint",
                "DRAM Static Footprint",
                "Inference Latency (Regression)",
                "Inference Latency (Classification)",
                "Python ↔ C Alignment"
            ],
            "Value": [
                "ESP32 (240 MHz)",
                "micromlgen C compiled",
                flash_size,
                dram_size,
                f"{reg_latency_us:.1f} µs" if conn_mode == "Live ESP32 Hardware (Serial UART)" else "68.1 µs (ESP32 Ref)",
                f"{clf_latency_us:.1f} µs" if conn_mode == "Live ESP32 Hardware (Serial UART)" else "90.8 µs (ESP32 Ref)",
                precision_alignment
            ]
        }
        df_metrics = pd.DataFrame(metrics_table)
        st.table(df_metrics.set_index("Spec / Measurement"))
        st.caption(f"Inference Source: {inference_source} (Runs completely disconnected on board)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Explainability (SHAP Waterfall) ────────────────────────────────
    
    st.subheader("🧠 Explainable AI: SHAP Prediction Attribution")
    with st.spinner("Generating SHAP Waterfall explanation..."):
        # Run SHAP TreeExplainer
        shap_values = shap_explainer(features_df)
        
        # Adjust SHAP figure size to make it compact
        fig = plt.figure(figsize=(6, 3.5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"VitalEdge SHAP Attribution: Cell {selected_cell}", fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        
        # Save temp file
        temp_plot_path = os.path.join(WORKSPACE, 'plots', f'temp_shap_{selected_cell.replace("-", "_")}.png')
        plt.savefig(temp_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Display image centered and smaller
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(temp_plot_path, use_container_width=True)
        # Clean up temp file
        try:
            os.remove(temp_plot_path)
        except Exception:
            pass

else:
    # Landing view instructions
    st.info("👈 Select a battery cell and execution mode in the left panel, then click 'Run Telemetry & Edge Inference' to stream data.")
    
    # Showcase pre-compiled benchmarks
    st.subheader("📊 ESP32 System Performance Benchmarks")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regression Latency", "68.1 µs", "6200x faster than real-time")
    with col2:
        st.metric("Classification Latency", "90.8 µs", "4600x faster than real-time")
    with col3:
        st.metric("ESP32 Static RAM Footprint", "12.9 KB", "2.5% of total ESP32 SRAM")
