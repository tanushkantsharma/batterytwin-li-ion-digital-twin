# BatteryTwin

A reproducible, computational lithium-ion battery digital twin: an equivalent-circuit model (ECM)
with lumped thermal dynamics, bounded least-squares parameter identification, coulomb-counting and
EKF state-of-charge estimation, PyBaMM electrochemical aging simulations, and charge-feature SOH
learning on measured NASA aging data.

> **Scope.** No laboratory work. Results are either *simulations* (synthetic reference cell, PyBaMM)
> or analyses of *measured* public data (NASA PCoE cells B0005/6/7/18). Each result states which.
> See [`docs/assumptions_and_limitations.md`](docs/assumptions_and_limitations.md).

## Setup and reproduction

```bash
python -m pip install -r requirements.txt
python scripts/prepare_nasa_data.py       # download + SHA-256 check + preprocess (~56 MB raw)
python scripts/make_reference_ocv.py      # PyBaMM Chen2020 → data/params/reference_ocv_chen2020.csv
python -m pytest                          # 16 tests
python scripts/run_simulations.py         # A–E, numerical checks, H (≈ 2 min incl. 4×500 PyBaMM cycles)
python scripts/run_evaluation.py          # G1–G4, thermal fit, F, SOH, RUL (≈ 2.5 min)
```
Tested with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1,
Matplotlib 3.11.2, PyBaMM 26.10.1.1 on a 2-core Linux machine. All figures go to
`results/figures/`, all numbers to `results/metrics/`, fitted parameters to `data/params/`.

## Methods
| Layer | Implementation |
|---|---|
| ECM | OCV(SOC) + R0 + 1 or 2 RC branches, exact ZOH discretisation, voltage/SOC limits ([equations](docs/model_equations.md)) |
| Thermal | Lumped `m·cp dT/dt = Q_gen − hA(T − T_amb)`, Joule + polarisation heat |
| Identification | `scipy.optimize.least_squares`, log-parameters, box bounds, multi-start; joint OCV identification (Plett combined model) when no OCV test exists |
| State estimation | Coulomb counting vs EKF with SOC-bound projection |
| Aging (physics) | PyBaMM SPMe + SEI (solvent-diffusion limited), `OKane2022` |
| SOH (data) | Charge-phase features → next-discharge capacity; linear baseline vs ridge / RF / GBR; leave-one-cell-out |
| RUL (data) | Linear and exponential capacity extrapolation to the 1.4 Ah EOL criterion |

## Results (all numbers produced by the scripts above)

**Forward model (simulation).** Energy-audit residuals ≤ 2.3·10⁻⁸ (relative) on all runs; the
electro-thermal coupling changes voltage by ≤ 0.79 mV between Δt = 0.1 s and 10 s. Delivered
capacity of the 5.15 Ah reference cell: 5.13 / 5.10 / 5.01 Ah at 0.5C / 1C / 2C.
Sensitivity (±20 %, US06): capacity dominates (25–38 mV RMSE), then R0 (8.1 mV), RC terms ≤ 2.2 mV.

**G1 – parameter recovery (synthetic, known truth, 5 noise seeds, train US06 / validate UDDS).**
| σ_V | R0 err | R1 err | C1 err | R2 err | C2 err | held-out V RMSE vs truth |
|---|---|---|---|---|---|---|
| 1 mV | −0.05 % | 0.21 % | −0.56 % | −0.12 % | 0.04 % | 0.05 mV |
| 5 mV | −0.22 % | 1.36 % | −2.50 % | −1.15 % | 1.61 % | 0.22 mV |
(mean errors; standard deviations in `G1_synthetic_param_recovery_summary.csv`)

**G2 – ECM vs PyBaMM SPMe (model-to-model).** Fitted on US06, held-out UDDS RMSE 5.6 mV (1RC) and
4.8 mV (2RC); max error ≈ 51 mV.

**G3/G4 – NASA measured data.** 1RC ECM fitted on B0005 charge+discharge 1–2:
| Held-out trajectory | Charge RMSE | Discharge RMSE |
|---|---|---|
| B0005 cycles 3 / 10 / 50 | 17 / 17 / 24 mV | 56 / 66 / 51 mV |
| B0005 cycles 100 / 150 (aged, no refit) | 62 / 93 mV | 117 / 148 mV |
| B0006 / B0007 / B0018 cycle 1 | 27 / 28 / 17 mV | 67 / 103 / 68 mV |

Discharge errors are dominated by the end-of-discharge knee (MAE 26–40 mV on early cycles).
Re-fitting R0, R1, C1 every 10 cycles tracks resistance growth: ECM R0 rises from 99 to 173 mΩ and
correlates with independent EIS (r = 0.87 with Re + Rct, 0.88 with Re).
Lumped thermal model fitted on one discharge: held-out surface-temperature RMSE 0.79–0.84 K on
other cells, 1.86 K on aged B0005 cycle 100.

**F/I – SOC estimation.** Synthetic (5 seeds, US06 × 25):
| Scenario | Coulomb counting RMSE | EKF RMSE |
|---|---|---|
| nominal (σ_V 5 mV, σ_I 20 mA) | 0.01 % | 0.28 % |
| initial SOC error −0.2 | 20.0 % | 0.68 % |
| current bias +50 mA | 2.34 % | 0.80 % |
| capacity −10 % | 4.39 % | 1.66 % |
| R0 +30 % / OCV +10 mV | 0.01 % / 0.01 % | 0.88 % / 1.16 % |
| combined (init −0.2, bias, R0 +30 %) | 22.1 % | 1.32 % |

Coulomb counting is best when the initial SOC and current sensor are exact; the EKF trades a small
model-error penalty for robustness to initial-state, bias and capacity errors.
On NASA B0005 (estimators started mid-discharge with ±0.15–0.2 SOC error), the EKF reduced the error
on cycles 1 and 50 (RMSE 0.06–0.09 vs 0.15–0.20) but not on aged cycles 100/150 with cycle-1
parameters — parameter adaptation is required (`F_soc_estimation_nasa.csv`).

**H – PyBaMM aging (simulation, 500 cycles, SEI only).**
| Ambient, discharge rate | Capacity fade | 10-s DC resistance growth |
|---|---|---|
| 10 °C, 1C | 0.31 % | 1.5 % |
| 25 °C, 1C | 0.48 % | 3.4 % |
| 45 °C, 1C | 0.71 % | 7.7 % |
| 25 °C, 2C | 0.56 % | 3.2 % |

**SOH (NASA, leave-one-cell-out, 627 cycles).**
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear baseline (CC time only) | 0.024 | 0.030 | 0.908 |
| Ridge (3 charge features) | **0.012** | **0.017** | **0.969** |
| Random forest | 0.020 | 0.027 | 0.924 |
| Gradient boosting | 0.019 | 0.025 | 0.933 |

With four cells the regularised linear model generalises best; tree ensembles do not beat it.

**RUL (NASA, empirical, 3 cells reaching 1.4 Ah).** Mean absolute EOL error over start cycles
40/60/80: exponential 28 cycles, linear 50 cycles; both improve as the start cycle approaches EOL.
B0007 never reaches 1.4 Ah and is excluded. No ML RUL model is claimed (n = 3).

## Repository layout
```
src/            battery_ecm, thermal_model, parameter_estimation, state_estimation, degradation,
                soh_ml, metrics, reference_cell, nasa_data, profiles, plotting
scripts/        prepare_nasa_data, make_reference_ocv, run_simulations, run_evaluation
optimization/   identification bounds / initial guesses
tests/          pytest suite
data/           README, SHA-256 of raw files, params/ (fitted & reference parameters)
results/        figures/, metrics/
report/         technical_report.md
docs/           model_equations, data_sources, assumptions_and_limitations
```

## Data and licensing
Code: MIT (see `LICENSE`). NASA raw and processed data are not redistributed (license not specified
by NASA); `scripts/prepare_nasa_data.py` downloads them. See [`docs/data_sources.md`](docs/data_sources.md).
