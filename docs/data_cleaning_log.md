# VitalEdge — Data Cleaning Log

Generated: 2026-06-28 12:59

## Severson Dataset

**Original cells:** Batch 1 (46) + Batch 2 (48) + Batch 3 (46) = 140

**Cells dropped:** 39

**Cells surviving:** 101

### Drop Log

| Cell ID | Step | Reason |
|---------|------|--------|
| `b1c0` | B | Glitch in Arbin cycler software (incorrect currents, capacity spike to 1.5391 Ah) |
| `b1c2` | B | Glitch in Arbin cycler software (incorrect charge/discharge currents during early cycles) |
| `b1c3` | B | Glitch in Arbin cycler software (incorrect charge/discharge currents during early cycles) |
| `b3c3` | A | Noisy current data |
| `b3c11` | A | End capacity 0.8811 Ah > 80% initial capacity (0.8545 Ah, did not reach EOL) |
| `b3c12` | A | End capacity 0.8808 Ah > 80% initial capacity (0.8593 Ah, did not reach EOL) |
| `b3c13` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8524 Ah, did not reach EOL) |
| `b3c14` | A | End capacity 0.8807 Ah > 80% initial capacity (0.8582 Ah, did not reach EOL) |
| `b3c15` | A | End capacity 0.8806 Ah > 80% initial capacity (0.8547 Ah, did not reach EOL) |
| `b3c16` | A | End capacity 0.8801 Ah > 80% initial capacity (0.8488 Ah, did not reach EOL) |
| `b3c17` | A | End capacity 0.8804 Ah > 80% initial capacity (0.8532 Ah, did not reach EOL) |
| `b3c18` | A | End capacity 0.8800 Ah > 80% initial capacity (0.8544 Ah, did not reach EOL) |
| `b3c19` | A | End capacity 0.8801 Ah > 80% initial capacity (0.8575 Ah, did not reach EOL) |
| `b3c20` | A | End capacity 0.8806 Ah > 80% initial capacity (0.8520 Ah, did not reach EOL) |
| `b3c21` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8543 Ah, did not reach EOL) |
| `b3c22` | A | End capacity 0.8809 Ah > 80% initial capacity (0.8450 Ah, did not reach EOL) |
| `b3c23` | A | End capacity 0.9370 Ah > 0.885 Ah (stopped early, did not reach EOL) |
| `b3c24` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8370 Ah, did not reach EOL) |
| `b3c25` | A | End capacity 0.8817 Ah > 80% initial capacity (0.8489 Ah, did not reach EOL) |
| `b3c26` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8576 Ah, did not reach EOL) |
| `b3c27` | A | End capacity 0.8800 Ah > 80% initial capacity (0.8483 Ah, did not reach EOL) |
| `b3c28` | A | End capacity 0.8801 Ah > 80% initial capacity (0.8415 Ah, did not reach EOL) |
| `b3c29` | A | End capacity 0.8801 Ah > 80% initial capacity (0.8531 Ah, did not reach EOL) |
| `b3c30` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8550 Ah, did not reach EOL) |
| `b3c31` | A | End capacity 0.8803 Ah > 80% initial capacity (0.8545 Ah, did not reach EOL) |
| `b3c32` | A | End capacity 0.9739 Ah > 0.885 Ah (stopped early, did not reach EOL) |
| `b3c33` | A | End capacity 0.8808 Ah > 80% initial capacity (0.8482 Ah, did not reach EOL) |
| `b3c34` | A | End capacity 0.8802 Ah > 80% initial capacity (0.8492 Ah, did not reach EOL) |
| `b3c35` | A | End capacity 0.8801 Ah > 80% initial capacity (0.8507 Ah, did not reach EOL) |
| `b3c36` | A | End capacity 0.8806 Ah > 80% initial capacity (0.8460 Ah, did not reach EOL) |
| `b3c37` | A | Channel 46 — data collection failure |
| `b3c38` | A | End capacity 0.8800 Ah > 80% initial capacity (0.8561 Ah, did not reach EOL) |
| `b3c39` | A | End capacity 0.8804 Ah > 80% initial capacity (0.8379 Ah, did not reach EOL) |
| `b3c40` | A | Noisy current data |
| `b3c41` | A | Noisy current data |
| `b3c42` | A | End capacity 0.8804 Ah > 80% initial capacity (0.8535 Ah, did not reach EOL) |
| `b3c43` | A | End capacity 0.8806 Ah > 80% initial capacity (0.8561 Ah, did not reach EOL) |
| `b3c44` | A | End capacity 0.8805 Ah > 80% initial capacity (0.8558 Ah, did not reach EOL) |
| `b3c45` | A | End capacity 0.8805 Ah > 80% initial capacity (0.8556 Ah, did not reach EOL) |

### Zero-Filled Missing Value Fixes (Converted to NaN)
The raw MATLAB structures contained zero values in place of missing measurements, which have been converted to `NaN` in `severson_clean.pkl` to prevent model corruption:

- **Cycle 1 QD, QC, chargetime**: Converted to `NaN` for all 43 surviving Batch 1 cells (due to cycle 1 diagnostic protocol).
- **IR**: Converted to `NaN` for all 43 Batch 1 cells at cycle 1, and at cycle 12/13 for 9 specific Batch 1 cells (`b1c1`, `b1c4`, `b1c6`, `b1c7`, `b1c10`, `b1c11`, `b1c16`, `b1c20`, `b1c21`) where they were zero-filled.

### ⚠️ Note on Cell Counts (140 vs 124)
> **Important for Modeling**: The original study by Severson et al. (2019) reported on **124 cells** (excluding 16 cells in Batch 3 that did not reach 80% EOL).
> Our pipeline starts from all **140 raw cells** across the 3 batch files. By dropping the 3 Batch 1 cells and all 36 Batch 3 EOL-incomplete/noisy cells, we are training on a clean subset of **101 cells**.
> This subset contains only cells that fully degraded to EOL. Krishiv (modeling) and Triya (feature extraction) should note that they are working with this clean **101-cell subset**, not the exact 124-cell paper configuration.

## HUST Dataset

**Cells processed:** 77

**Total clean cycles:** 146122

**Total cycles dropped:** 0

No cycles were dropped from any HUST cell.

## Final Counts

| Dataset | Clean Cells | Clean Rows/Cycles |
|---------|------------|-------------------|
| Severson | 101 | 10100 rows (100 cycles each) |
| HUST | 77 | 146122 total cycles |
