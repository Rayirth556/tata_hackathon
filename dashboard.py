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
    .metric-card-compare {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        color: #202124;
    }
    .metric-title-compare {
        font-size: 11px;
        color: #5f6368;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 0.5px;
    }
    .metric-value-compare {
        font-size: 20px;
        font-weight: bold;
        margin: 5px 0;
        color: #202124;
    }
    .metric-status-compare {
        font-size: 10px;
        font-weight: bold;
        border-radius: 15px;
        padding: 2px 8px;
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
def load_models_and_explainers():
    """Loads XGBoost models and pre-builds the SHAP TreeExplainers."""
    model_reg_path = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
    model_clf_path = os.path.join(WORKSPACE, 'models/trained_model_clf.pkl')
    
    if not os.path.exists(model_reg_path) or not os.path.exists(model_clf_path):
        st.error("Missing trained models. Run model training first!")
        st.stop()
        
    reg_model = joblib.load(model_reg_path)
    clf_model = joblib.load(model_clf_path)
    
    # Initialize explainers using path-dependent perturbation (avoids categorical split issues)
    reg_explainer = shap.TreeExplainer(reg_model)
    clf_explainer = shap.TreeExplainer(clf_model)
    
    return reg_model, clf_model, reg_explainer, clf_explainer


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
reg_model, clf_model, shap_explainer_reg, shap_explainer_clf = load_models_and_explainers()

# ── Helper function to run the full inference pipeline ───────────────────────────

def run_inference_pipeline(cell_id, conn_mode, serial_port):
    """Runs the full telemetry replay, feature extraction, and model inference for a single cell."""
    cycles = []
    try:
        for cycle_data in hust_replay(clean_pkl, cell_id=cell_id, start_cycle=1, end_cycle=100):
            cycles.append(cycle_data)
    except Exception as e:
        return {"error": f"Error streaming HUST cell {cell_id}: {e}"}
        
    # Extract features
    features_dict = extract_features_from_cycles(cycles)
    features_df = get_feature_dataframe(features_dict)
    
    knee_cycle = 0.0
    early_prob = 0.0
    reg_latency_us = 0.0
    clf_latency_us = 0.0
    precision_alignment = "❌ Disconnected"
    inference_source = ""
    warning_msg = None

    if conn_mode == "Live ESP32 Hardware (Serial UART)":
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
            
        except Exception as e:
            warning_msg = f"Failed to communicate with ESP32 on {serial_port}: {e}. Falling back to Local Host C Binary Simulation."
            conn_mode = "Host Simulation (Local C Binary)"

    # Fallback / Local Simulation mode
    if conn_mode == "Host Simulation (Local C Binary)":
        c_bin_path = os.path.join(WORKSPACE, 'edge', 'host_test', 'host_test_bin')
        input_str = ",".join([f"{features_dict[col]:.17g}" for col in NO_TEMP_FEATURES])
        c_run_success = False
        
        # Check if binary exists and can be run (depends on OS compatibility)
        if os.path.exists(c_bin_path):
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
                c_run_success = True
            except Exception:
                pass
        
        if not c_run_success:
            # If C binary is missing, compiled for a different OS, or crashes, fall back to Python models
            knee_cycle = reg_model.predict(features_df)[0]
            early_prob = clf_model.predict_proba(features_df)[0][1]
            reg_latency_us = 0.0
            clf_latency_us = 0.0
            precision_alignment = "⚠️ Unverified (Pure Python Run)"
            inference_source = "Python Fallback (C Binary Incompatible)"

    # Calculate metrics
    rul_cycles = max(0.0, knee_cycle - 100.0)
    
    # Determine dynamic status text and colors based on early risk probability and knee predictions
    if early_prob > 0.8:
        card_a_status_class = "status-red"
        card_a_status_text = "Accelerated Aging"
        card_a_value_color = "#d32f2f" # Red
        
        card_b_status_class = "status-red"
        card_b_status_text = "Early Knee (<464)"
        card_b_value_color = "#d32f2f" # Red
        
        status_class = "status-red"
        status_text = "HIGH RISK - Flag Cell"
    elif early_prob > 0.2:
        card_a_status_class = "status-yellow"
        card_a_status_text = "Moderate Wear"
        card_a_value_color = "#f57c00" # Orange/Yellow
        
        card_b_status_class = "status-yellow"
        card_b_status_text = "Moderate Degradation"
        card_b_value_color = "#f57c00" # Orange/Yellow
        
        status_class = "status-yellow"
        status_text = "MODERATE RISK - Monitor"
    else:
        card_a_status_class = "status-green"
        card_a_status_text = "Healthy Operation"
        card_a_value_color = "#388e3c" # Green
        
        card_b_status_class = "status-green"
        card_b_status_text = "Normal (≥464)"
        card_b_value_color = "#388e3c" # Green
        
        status_class = "status-green"
        status_text = "LOW RISK - Normal"

    return {
        "cell_id": cell_id,
        "cycles": cycles,
        "features_dict": features_dict,
        "features_df": features_df,
        "knee_cycle": knee_cycle,
        "early_prob": early_prob,
        "reg_latency_us": reg_latency_us,
        "clf_latency_us": clf_latency_us,
        "precision_alignment": precision_alignment,
        "inference_source": inference_source,
        "warning_msg": warning_msg,
        "rul_cycles": rul_cycles,
        "card_a_status_class": card_a_status_class,
        "card_a_status_text": card_a_status_text,
        "card_a_value_color": card_a_value_color,
        "card_b_status_class": card_b_status_class,
        "card_b_status_text": card_b_status_text,
        "card_b_value_color": card_b_value_color,
        "status_class": status_class,
        "status_text": status_text,
    }


# ── Helper functions for Rendering ──────────────────────────────────────────────

def render_metrics_cards(res, is_compare=False):
    """Renders the three metrics cards for a cell."""
    card_class = "metric-card-compare" if is_compare else "metric-card"
    title_class = "metric-title-compare" if is_compare else "metric-title"
    val_class = "metric-value-compare" if is_compare else "metric-value"
    status_class_lbl = "metric-status-compare" if is_compare else "metric-status"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="{card_class}">
            <div class="{title_class}">Remaining Useful Life (RUL)</div>
            <div class="{val_class}" style="color: {res['card_a_value_color']};">{res['rul_cycles']:.1f} Cycles</div>
            <div class="{status_class_lbl} {res['card_a_status_class']}">{res['card_a_status_text']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="{card_class}">
            <div class="{title_class}">Predicted Knee Point</div>
            <div class="{val_class}" style="color: {res['card_b_value_color']};">Cycle {res['knee_cycle']:.1f}</div>
            <div class="{status_class_lbl} {res['card_b_status_class']}">{res['card_b_status_text']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        risk_pct = res['early_prob'] * 100
        st.markdown(f"""
        <div class="{card_class}">
            <div class="{title_class}">Early Degradation Risk</div>
            <div class="{val_class}">{risk_pct:.1f}%</div>
            <div class="{status_class_lbl} {res['status_class']}">{res['status_text']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_telemetry_plot(res, is_compare=False):
    """Renders the dual-panel feature evolution plot (Capacity Decay & IR Growth)."""
    cell_id = res["cell_id"]
    cycles = res["cycles"]
    
    cap_history = [np.max(c["Capacity"]) for c in cycles]
    ir_history = [c["ir_est"] for c in cycles]
    cycle_numbers = [c["cycle"] for c in cycles]
    
    figsize = (5.5, 4.5) if is_compare else (6, 5)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#f8f9fa')
    ax2.set_facecolor('#f8f9fa')
    
    # Panel 1: Capacity Decay
    ax1.plot(cycle_numbers, cap_history, color='#e43f5a', linewidth=2.0, label='Capacity Decay')
    ax1.scatter([100], [cap_history[-1]], color='#1e8e3e', s=80, zorder=5, label='Current (C100)')
    ax1.set_ylabel("Capacity (Ah)", color='#202124', fontsize=9)
    ax1.tick_params(colors='#202124', labelsize=8)
    ax1.grid(color='#e0e0e0', linestyle='--', alpha=0.7)
    ax1.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#202124', fontsize=8, loc='upper right')
    ax1.set_title(f"HUST Cell {cell_id} Feature Evolution", color='#202124', fontsize=10, fontweight='bold')
    
    # Panel 2: IR Growth
    valid_indices = [i for i, val in enumerate(ir_history) if np.isfinite(val)]
    valid_cycles = [cycle_numbers[i] for i in valid_indices]
    valid_ir = [ir_history[i] for i in valid_indices]
    
    ax2.plot(valid_cycles, valid_ir, color='#1f4068', linewidth=2.0, label='IR Growth')
    if valid_ir:
        ax2.scatter([100], [valid_ir[-1]], color='#1e8e3e', s=80, zorder=5, label='Current (C100)')
    ax2.set_xlabel("Cycle Number", color='#202124', fontsize=9)
    ax2.set_ylabel("IR (Ohms)", color='#202124', fontsize=9)
    ax2.tick_params(colors='#202124', labelsize=8)
    ax2.grid(color='#e0e0e0', linestyle='--', alpha=0.7)
    ax2.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#202124', fontsize=8, loc='upper left')
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_shap_plot(res, shap_mode, is_compare=False):
    """Renders the SHAP Waterfall plot for a cell using the selected prediction mode."""
    cell_id = res["cell_id"]
    features_df = res["features_df"]
    
    # Select corresponding explainer
    explainer = shap_explainer_reg if "Regressor" in shap_mode else shap_explainer_clf
    
    figsize = (5.5, 3.2) if is_compare else (6, 3.5)
    
    shap_values = explainer(features_df)
    
    fig = plt.figure(figsize=figsize)
    shap.plots.waterfall(shap_values[0], show=False)
    
    title_label = "Remaining Useful Life" if "Regressor" in shap_mode else "Early Degradation Risk"
    plt.title(f"SHAP Waterfall ({title_label}): Cell {cell_id}", fontsize=10, fontweight='bold', pad=10)
    plt.tight_layout()
    
    # Save to temp file
    temp_plot_path = os.path.join(WORKSPACE, 'plots', f'temp_shap_{cell_id.replace("-", "_")}.png')
    plt.savefig(temp_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    st.image(temp_plot_path, use_container_width=True)
    
    try:
        os.remove(temp_plot_path)
    except Exception:
        pass


# ── Sidebar UI (Control Room) ──────────────────────────────────────────────────

st.sidebar.markdown(
    "<div style='background: linear-gradient(135deg, #1f4068 0%, #162447 100%); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>"
    "<h2 style='color: white; margin: 0;'>🔋 VitalEdge</h2>"
    "<p style='color: #9aa0a6; font-size: 12px; margin: 5px 0 0 0;'>EDGE AI LIFETIME PREDICTION</p>"
    "</div>", 
    unsafe_allow_html=True
)

st.sidebar.header("🕹️ Control Panel")

# Initialize session state variables if they do not exist
if 'conn_mode' not in st.session_state:
    st.session_state.conn_mode = "Host Simulation (Local C Binary)"
if 'warning_msg' not in st.session_state:
    st.session_state.warning_msg = None
if 'inference_results' not in st.session_state:
    st.session_state.inference_results = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = None

# Toggle Compare Mode
compare_mode = st.sidebar.checkbox("Compare Mode (Side-by-Side)", value=False)

if not compare_mode:
    selected_cell = st.sidebar.selectbox("Select Battery Cell ID", hust_cells, index=0)
    selected_cell_a = None
    selected_cell_b = None
else:
    selected_cell = None
    selected_cell_a = st.sidebar.selectbox("Select Battery Cell A", hust_cells, index=0)
    selected_cell_b = st.sidebar.selectbox("Select Battery Cell B", hust_cells, index=min(1, len(hust_cells)-1))

# Target Hardware Connection
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

# ── Actions on Trigger ────────────────────────────────────────────────────────

if run_inference:
    if compare_mode:
        with st.spinner(f"Streaming data & running inference for Cell {selected_cell_a}..."):
            res_a = run_inference_pipeline(selected_cell_a, conn_mode, serial_port)
            if "error" in res_a:
                st.error(res_a["error"])
                st.stop()
            if res_a.get("warning_msg"):
                st.session_state.warning_msg = res_a["warning_msg"]
                st.session_state.conn_mode = "Host Simulation (Local C Binary)"
                
        with st.spinner(f"Streaming data & running inference for Cell {selected_cell_b}..."):
            current_conn_mode = st.session_state.conn_mode
            res_b = run_inference_pipeline(selected_cell_b, current_conn_mode, serial_port)
            if "error" in res_b:
                st.error(res_b["error"])
                st.stop()
            if res_b.get("warning_msg"):
                st.session_state.warning_msg = res_b["warning_msg"]
                st.session_state.conn_mode = "Host Simulation (Local C Binary)"
                
        st.session_state.inference_results = {
            "mode": "compare",
            "res_a": res_a,
            "res_b": res_b
        }
        st.session_state.last_run_params = {
            "compare_mode": compare_mode,
            "cell_a": selected_cell_a,
            "cell_b": selected_cell_b,
            "conn_mode": st.session_state.conn_mode
        }
    else:
        with st.spinner(f"Streaming data & running inference for Cell {selected_cell}..."):
            res = run_inference_pipeline(selected_cell, conn_mode, serial_port)
            if "error" in res:
                st.error(res["error"])
                st.stop()
            if res.get("warning_msg"):
                st.session_state.warning_msg = res["warning_msg"]
                st.session_state.conn_mode = "Host Simulation (Local C Binary)"
                
        st.session_state.inference_results = {
            "mode": "single",
            "res": res
        }
        st.session_state.last_run_params = {
            "compare_mode": compare_mode,
            "cell": selected_cell,
            "conn_mode": st.session_state.conn_mode
        }
    
    st.rerun()

# ── Render Page Content ───────────────────────────────────────────────────────

# Show warnings if any
if st.session_state.warning_msg:
    st.warning(st.session_state.warning_msg)
    st.session_state.warning_msg = None

# Show parameter mismatch warning if settings changed after running inference
if st.session_state.inference_results is not None:
    last_params = st.session_state.last_run_params
    current_params = {
        "compare_mode": compare_mode,
        "cell_a": selected_cell_a,
        "cell_b": selected_cell_b,
        "conn_mode": conn_mode
    } if compare_mode else {
        "compare_mode": compare_mode,
        "cell": selected_cell,
        "conn_mode": conn_mode
    }
    
    mismatch = False
    for k, v in current_params.items():
        if last_params.get(k) != v:
            mismatch = True
            break
            
    if mismatch:
        st.info("🔄 Configuration changed. Click 'Run Telemetry & Edge Inference' in the sidebar to refresh results.")

# Draw results page
if st.session_state.inference_results is not None:
    results_mode = st.session_state.inference_results["mode"]
    last_params = st.session_state.last_run_params
    
    if results_mode == "single":
        res = st.session_state.inference_results["res"]
        
        # Row 1: Metrics Cards
        render_metrics_cards(res, is_compare=False)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Telemetry Plot & Edge Footprint
        col_left, col_right = st.columns([2, 1.2])
        with col_left:
            st.subheader("📈 Early-Life Telemetry & Feature Evolution")
            render_telemetry_plot(res, is_compare=False)
            
        with col_right:
            st.subheader("⚡ Edge Performance Footprint")
            flash_size, dram_size = get_esp32_footprints()
            
            # Specs Table
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
                    f"{res['reg_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "68.1 µs (ESP32 Ref)",
                    f"{res['clf_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "90.8 µs (ESP32 Ref)",
                    res["precision_alignment"]
                ]
            }
            df_metrics = pd.DataFrame(metrics_table)
            st.table(df_metrics.set_index("Spec / Measurement"))
            st.caption(f"Inference Source: {res['inference_source']} (Runs completely disconnected on board)")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 3: Explainability (SHAP)
        st.subheader("🧠 Explainable AI: SHAP Prediction Attribution")
        shap_mode = st.radio(
            "Select Prediction to Explain",
            ["Remaining Useful Life (Regressor)", "Early Degradation Risk (Classifier)"],
            horizontal=True,
            key="shap_mode_single"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_shap_plot(res, shap_mode, is_compare=False)
            
    else: # Compare mode
        res_a = st.session_state.inference_results["res_a"]
        res_b = st.session_state.inference_results["res_b"]
        
        # Row 1: Cell Metrics side-by-side
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<h3 style='text-align: center; color: #1f4068;'>Cell {res_a['cell_id']}</h3>", unsafe_allow_html=True)
            render_metrics_cards(res_a, is_compare=True)
        with col_b:
            st.markdown(f"<h3 style='text-align: center; color: #1f4068;'>Cell {res_b['cell_id']}</h3>", unsafe_allow_html=True)
            render_metrics_cards(res_b, is_compare=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Telemetry Plots side-by-side
        col_plot_a, col_plot_b = st.columns(2)
        with col_plot_a:
            render_telemetry_plot(res_a, is_compare=True)
        with col_plot_b:
            render_telemetry_plot(res_b, is_compare=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 3: Combined Specs Table
        st.subheader("⚡ Edge Performance Footprint Comparison")
        flash_size, dram_size = get_esp32_footprints()
        
        metrics_table = {
            "Spec / Measurement": [
                "Target Hardware",
                "Inference Engine",
                "Flash Binary Footprint",
                "DRAM Static Footprint",
                "Python ↔ C Alignment",
                "Inference Latency (Regression)",
                "Inference Latency (Classification)",
                "Inference Source"
            ],
            f"Cell {res_a['cell_id']}": [
                "ESP32 (240 MHz)",
                "micromlgen C compiled",
                flash_size,
                dram_size,
                res_a["precision_alignment"],
                f"{res_a['reg_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "68.1 µs (ESP32 Ref)",
                f"{res_a['clf_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "90.8 µs (ESP32 Ref)",
                res_a["inference_source"]
            ],
            f"Cell {res_b['cell_id']}": [
                "ESP32 (240 MHz)",
                "micromlgen C compiled",
                flash_size,
                dram_size,
                res_b["precision_alignment"],
                f"{res_b['reg_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "68.1 µs (ESP32 Ref)",
                f"{res_b['clf_latency_us']:.1f} µs" if last_params.get("conn_mode") == "Live ESP32 Hardware (Serial UART)" else "90.8 µs (ESP32 Ref)",
                res_b["inference_source"]
            ]
        }
        df_metrics = pd.DataFrame(metrics_table)
        st.table(df_metrics.set_index("Spec / Measurement"))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 4: SHAP Explainability side-by-side
        st.subheader("🧠 Explainable AI: SHAP Prediction Attribution")
        shap_mode = st.radio(
            "Select Prediction to Explain",
            ["Remaining Useful Life (Regressor)", "Early Degradation Risk (Classifier)"],
            horizontal=True,
            key="shap_mode_compare"
        )
        
        col_shap_a, col_shap_b = st.columns(2)
        with col_shap_a:
            render_shap_plot(res_a, shap_mode, is_compare=True)
        with col_shap_b:
            render_shap_plot(res_b, shap_mode, is_compare=True)

else:
    # Landing view instructions
    st.info("👈 Select battery cell(s) and execution mode in the left panel, then click 'Run Telemetry & Edge Inference' to stream data.")
    
    # Showcase pre-compiled benchmarks
    st.subheader("📊 ESP32 System Performance Benchmarks")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Regression Latency", "68.1 µs", "6200x faster than real-time")
    with col2:
        st.metric("Classification Latency", "90.8 µs", "4600x faster than real-time")
    with col3:
        st.metric("ESP32 Static RAM Footprint", "12.9 KB", "2.5% of total ESP32 SRAM")
