"""
knee_labeling.py — Detect knee points on full Severson capacity curves.

Strategy:
  The PS.txt threshold of 0.003 Ah/cycle² is too strict for most Severson LFP
  cells — only 3/101 cells cross it. However, Kneedle detects physically
  plausible knees in nearly every cell. The approach here:

  1. Primary label: Kneedle algorithm (detects the transition point reliably)
  2. Confirmation: S-G second derivative peak magnitude at the Kneedle location
     (reported for transparency, NOT used as a gate)
  3. Fallback exclusion: cells where Kneedle returns None OR where the knee is
     within the last 10% of the cell's lifetime (boundary artifact)

Produces:
  - knee_labels.csv       — per-cell labels
  - plots/knee_verify/    — verification plots for ≥15 cells
  - Console summary with classification threshold (median knee cycle)
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from kneed import KneeLocator

# ── Configuration ──────────────────────────────────────────────────────────────
WORKSPACE = '/home/godkiller/Documents/tata'
INPUT_FILE = os.path.join(WORKSPACE, 'data/severson_full_curves.pkl')
OUTPUT_CSV = os.path.join(WORKSPACE, 'data/knee_labels.csv')
PLOT_DIR = os.path.join(WORKSPACE, 'plots', 'knee_verify')

# PS.txt parameters for S-G smoothing
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3
SG_THRESHOLD = 0.003  # Original PS.txt threshold — kept for reporting


def compute_sg_derivative(cycles, qd):
    """
    Compute S-G smoothed second derivative and return the peak value and cycle.
    """
    valid_mask = ~np.isnan(qd)
    cycles_clean = cycles[valid_mask]
    qd_clean = qd[valid_mask]

    if len(qd_clean) < SAVGOL_WINDOW + 2:
        return None, None, None, None

    qd_smooth = savgol_filter(qd_clean, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)
    d1 = np.gradient(qd_smooth, cycles_clean)
    d2 = np.gradient(d1, cycles_clean)

    max_d2 = d2.max()
    max_d2_idx = np.argmax(d2)
    max_d2_cycle = int(cycles_clean[max_d2_idx])

    return max_d2, max_d2_cycle, cycles_clean, d2


def detect_knee_kneedle(cycles, qd):
    """
    Detect knee point using the Kneedle algorithm.
    """
    valid_mask = ~np.isnan(qd)
    cycles_clean = cycles[valid_mask]
    qd_clean = qd[valid_mask]

    if len(qd_clean) < 10:
        return None

    try:
        kl = KneeLocator(
            cycles_clean, qd_clean,
            S=1.0,
            curve="concave",
            direction="decreasing",
        )
        return int(kl.knee) if kl.knee is not None else None
    except Exception:
        return None


def generate_verification_plot(cell_id, cycles, qd, knee_kneedle, max_d2_cycle,
                                d2, cycles_d2, save_path, has_knee):
    """
    Generate a verification plot showing the capacity curve with knee detection.
    """
    valid_mask = ~np.isnan(qd)
    cycles_clean = cycles[valid_mask]
    qd_clean = qd[valid_mask]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Capacity curve
    ax1.plot(cycles_clean, qd_clean, color='#2196F3', linewidth=1.5,
             label='Discharge Capacity', zorder=2)
    ax1.set_xlabel('Cycle Number', fontsize=13)
    ax1.set_ylabel('Discharge Capacity (Ah)', fontsize=13, color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    # Knee markers
    if knee_kneedle is not None:
        ax1.axvline(x=knee_kneedle, color='#E53935', linewidth=2.5, linestyle='--',
                     label=f'Kneedle Knee: cycle {knee_kneedle}', zorder=3)
    if max_d2_cycle is not None:
        ax1.axvline(x=max_d2_cycle, color='#43A047', linewidth=2, linestyle=':',
                     label=f'S-G d² peak: cycle {max_d2_cycle}', zorder=3)

    # Second derivative on twin axis
    ax2 = ax1.twinx()
    if d2 is not None and cycles_d2 is not None:
        ax2.plot(cycles_d2, d2, color='#FF9800', linewidth=1, alpha=0.7, label='d²QD/dcycle²')
        ax2.axhline(y=SG_THRESHOLD, color='#FF9800', linewidth=1, linestyle='-.',
                     alpha=0.5, label=f'PS.txt threshold ({SG_THRESHOLD})')
        ax2.set_ylabel('d²QD/dcycle² (Ah/cycle²)', fontsize=13, color='#FF9800')
        ax2.tick_params(axis='y', labelcolor='#FF9800', labelsize=11)

    # Title
    if has_knee:
        status = f"✓ Knee at cycle {knee_kneedle}"
    else:
        status = "✗ No reliable knee"
    ax1.set_title(f'{cell_id} — {status}', fontsize=15, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    # ── Load full curves ───────────────────────────────────────────────────
    print("Loading full lifetime curves from severson_full_curves.pkl...")
    with open(INPUT_FILE, 'rb') as f:
        curves = pickle.load(f)
    print(f"  Loaded {len(curves)} cells.\n")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Process each cell ──────────────────────────────────────────────────
    results = []
    no_knee_cells = []

    for cell_id in sorted(curves.keys()):
        data = curves[cell_id]
        cycles = data['cycles']
        qd = data['QD']
        total_cycles = data['total_cycles']

        # Kneedle detection (primary label)
        knee_kneedle = detect_knee_kneedle(cycles, qd)

        # S-G second derivative (confirmation signal)
        max_d2, max_d2_cycle, cycles_d2, d2 = compute_sg_derivative(cycles, qd)

        # Determine if the knee is reliable:
        # 1. Kneedle found something
        # 2. Knee is NOT in the last 10% of the cell's life (boundary artifact)
        # 3. Cell has enough cycles to be meaningful (>50)
        has_knee = False
        if knee_kneedle is not None and total_cycles > 50:
            # Check if knee is a boundary artifact (last 10% of life)
            if knee_kneedle <= total_cycles * 0.90:
                has_knee = True

        sg_crosses = max_d2 is not None and max_d2 > SG_THRESHOLD

        results.append({
            'cell_id': cell_id,
            'knee_cycle': knee_kneedle if has_knee else None,
            'knee_kneedle_raw': knee_kneedle,
            'sg_max_d2': max_d2,
            'sg_max_d2_cycle': max_d2_cycle,
            'sg_crosses_threshold': sg_crosses,
            'has_knee': has_knee,
            'total_cycles': total_cycles,
        })

        if not has_knee:
            no_knee_cells.append(cell_id)

        # Status
        knee_str = f"knee={knee_kneedle}" if has_knee else f"raw_kn={knee_kneedle}"
        d2_str = f"d2_max={max_d2:.6f}" if max_d2 else "d2_max=N/A"
        flag = "✓" if has_knee else "✗"
        print(f"  {flag} {cell_id:<8}  {knee_str:<16}  {d2_str:<22}  cycles={total_cycles}")

    # ── Generate verification plots ───────────────────────────────────────
    # Plot ALL cells to enable thorough visual verification
    plot_cells = set()

    # All no-knee cells
    plot_cells.update(no_knee_cells)

    # Sample of cells with knees — spread across the knee range
    knee_results = [r for r in results if r['has_knee']]
    if knee_results:
        knee_sorted = sorted(knee_results, key=lambda r: r['knee_cycle'])
        n_sample = max(15, len(knee_sorted))  # at least 15
        indices = np.linspace(0, len(knee_sorted) - 1, min(n_sample, len(knee_sorted)), dtype=int)
        for idx in indices:
            plot_cells.add(knee_sorted[idx]['cell_id'])

    # Fill to at least 15
    for r in results:
        if len(plot_cells) >= 15:
            break
        plot_cells.add(r['cell_id'])

    print(f"\n  Generating verification plots for {len(plot_cells)} cells...")
    for cell_id in sorted(plot_cells):
        data = curves[cell_id]
        row = next(r for r in results if r['cell_id'] == cell_id)
        _, _, cycles_d2, d2 = compute_sg_derivative(data['cycles'], data['QD'])
        save_path = os.path.join(PLOT_DIR, f'cell_{cell_id}.png')
        generate_verification_plot(
            cell_id, data['cycles'], data['QD'],
            row['knee_kneedle_raw'], row['sg_max_d2_cycle'],
            d2, cycles_d2, save_path, row['has_knee']
        )
    print(f"  Plots saved to {PLOT_DIR}/")

    # ── Save labels ────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Labels saved to {OUTPUT_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────
    knee_df = df[df['has_knee']]
    no_knee_count = (~df['has_knee']).sum()

    print(f"\n{'='*70}")
    print(f"KNEE LABELING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total cells:              {len(df)}")
    print(f"  Cells WITH reliable knee: {len(knee_df)}")
    print(f"  Cells WITHOUT knee:       {no_knee_count}")
    print(f"  S-G threshold crosses:    {df['sg_crosses_threshold'].sum()} / {len(df)}")

    if len(knee_df) > 0:
        median_knee = knee_df['knee_cycle'].median()
        mean_knee = knee_df['knee_cycle'].mean()
        min_knee = knee_df['knee_cycle'].min()
        max_knee = knee_df['knee_cycle'].max()
        std_knee = knee_df['knee_cycle'].std()

        print(f"\n  Knee cycle statistics (Kneedle method, reliable knees only):")
        print(f"    Min:    {min_knee:.0f}")
        print(f"    Max:    {max_knee:.0f}")
        print(f"    Mean:   {mean_knee:.1f}")
        print(f"    Median: {median_knee:.0f}")
        print(f"    Std:    {std_knee:.1f}")

        threshold = int(median_knee)
        n_early = (knee_df['knee_cycle'] < threshold).sum()
        n_late = (knee_df['knee_cycle'] >= threshold).sum()

        print(f"\n  ► CLASSIFICATION THRESHOLD (median): cycle {threshold}")
        print(f"    Early knee (<{threshold}):  {n_early} cells")
        print(f"    Late knee (≥{threshold}):  {n_late} cells")

    if no_knee_cells:
        print(f"\n  Cells with no reliable knee ({len(no_knee_cells)}):")
        for cid in no_knee_cells:
            row = next(r for r in results if r['cell_id'] == cid)
            reason = ""
            if row['knee_kneedle_raw'] is None:
                reason = "Kneedle returned None"
            elif row['total_cycles'] <= 50:
                reason = "Too few cycles"
            else:
                reason = f"Knee at cycle {row['knee_kneedle_raw']} is in last 10% of {row['total_cycles']} cycles"
            print(f"    {cid} ({row['total_cycles']} cycles) — {reason}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
