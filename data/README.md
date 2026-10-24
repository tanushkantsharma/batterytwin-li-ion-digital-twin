# Data

| Path | Content | In Git |
|---|---|---|
| `raw/nasa/B00xx.mat` | NASA PCoE raw files | no (license not specified) |
| `processed/` | CSV(.gz) produced by `scripts/prepare_nasa_data.py` | no |
| `nasa_sha256.json` | SHA-256 of the raw files used for the published results | yes |
| `params/reference_ocv_chen2020.csv` | Model-derived OCV table (PyBaMM Chen2020, C/50 SPM) | yes |
| `params/reference_cell_ecm.json` | Synthetic reference cell (ASSUMED R/C) | yes |
| `params/nasa_B0005_ecm_1RC.json`, `_2RC.json` | ECM fitted to B0005 cycles 1–2 (with approx. 1-σ log-parameter std) | yes |
| `params/nasa_B0005_thermal.json` | Lumped thermal parameters fitted to B0005 discharge 1 | yes |

Prepare: `python scripts/prepare_nasa_data.py` (or `--raw-dir DIR` for files you downloaded from
NASA). The script reports whether each file's hash matches `nasa_sha256.json`.
Sources, units, sampling and preprocessing: `../docs/data_sources.md`.
