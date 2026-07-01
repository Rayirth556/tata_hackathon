# VitalEdge — Output Schema Agreement

Generated: 2026-06-20 15:51

**For:** Triya (feature extraction) and Krishiv (knee point labeling)

## Severson — `severson_clean.pkl`

**Format:** pandas DataFrame, one row per (cell_id, cycle_num)

**Cycles:** 1–100 only

| Column | Unit | dtype | Description |
|--------|------|-------|-------------|
| `cell_id` | — | str | Unique cell identifier (e.g. `b1c0`) |
| `cycle_num` | — | int64 | Cycle number (1–100) |
| `QD` | Ah | float64 | Discharge capacity |
| `QC` | Ah | float64 | Charge capacity |
| `IR` | Ohm | float64 | Internal resistance |
| `Tavg` | °C | float64 | Average temperature |
| `Tmin` | °C | float64 | Minimum temperature |
| `Tmax` | °C | float64 | Maximum temperature |
| `chargetime` | min | float64 | Charge time |

## HUST — `hust_clean.pkl`

**Format:** Nested dict — `{cell_id: [cycle_dicts]}`, loaded with `pickle.load()`

| Field | Unit | dtype | Description |
|-------|------|-------|-------------|
| `cycle` | — | int | Cycle number |
| `V` | V | np.float64 array | Voltage time series |
| `I` | A | np.float64 array | Current time series (converted from mA) |
| `Capacity` | Ah | np.float64 array | Capacity time series (converted from mAh) |
| `t` | s | np.float64 array | Time stamps |
| `charge_time_s` | s | float | Duration of CC charge phase |
| `ir_est` | Ω | float | DC internal resistance, estimated from CV→discharge voltage step: `(V_cv_last − V_dis_first) / (I_dis_first − I_cv_last)` |

## ⚠️ CRITICAL: HUST Temperature Gap

> **HUST has NO temperature data.** The cells were tested in a constant 30°C chamber,
> but temperature was NOT logged per-cycle or per-timestep. There is no `T` field in the
> HUST output structure. Do NOT attempt to extract temperature features from HUST data.

> **Impact on Triya's feature extraction:** Temperature-based features (mean T, max T)
> can only be computed for Severson, not HUST. Triya must handle this asymmetry in her code.

> **Impact on Krishiv's modeling:** If temperature features are used in the model,
> a separate branch or imputation strategy is needed for HUST. Discuss with team before Week 2 ends.