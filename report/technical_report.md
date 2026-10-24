# BatteryTwin — Technical Report

*Computational study; no laboratory measurements were made. Generated results: `results/`.*

## 1. Objective
Build a lithium-ion digital twin that (i) simulates terminal voltage, SOC and temperature,
(ii) identifies its parameters from data, (iii) estimates SOC online, (iv) simulates degradation with
an electrochemical model, and (v) estimates SOH / RUL from measured aging data — while keeping
simulated and measured evidence strictly separate.

## 2. Data
* **Measured:** NASA PCoE cells B0005, B0006, B0007, B0018 (18650, 2 Ah rated, 24 °C ambient,
  ≈1.5 A CC-CV charge, 2 A CC discharge, periodic EIS). Files obtained from a third-party GitHub
  redistribution because the official archive was unreachable from the build environment; SHA-256
  recorded (`data/nasa_sha256.json`), not verified against NASA's archive.
* **Simulated:** a 5.15 Ah reference cell — OCV from PyBaMM's Chen2020 parameter set (C/50 SPM bulk
  OCV), assumed R/C/thermal values — and PyBaMM SPMe/OKane2022 aging runs.
Details: `docs/data_sources.md`.

## 3. Models
ECM with OCV + R0 + 1–2 RC, exact ZOH discretisation; lumped thermal model with irreversible heat;
bounded least-squares identification; coulomb counting and EKF; PyBaMM SPMe + SEI; ridge / tree SOH
regressors on charge-phase features. Equations: `docs/model_equations.md`.

## 4. Verification of the implementation
16 pytest tests pass: analytic CC/RC response (error < 10⁻¹²), step-size invariance (< 10⁻⁹ V),
energy audit (< 10⁻⁴ relative in the test; ≤ 2.3·10⁻⁸ in experiments A/B), voltage-limit
termination, exact first-order thermal response, thermal steady state, noise-free recovery of all five
R/C parameters to < 1 %, EKF convergence from a 0.3 SOC error, and processed-data sanity checks.

## 5. Experiments and results

| ID | Experiment | Basis | Key result |
|---|---|---|---|
| A | CC 0.5/1/2C discharge, 0.5/1C CC charge | sim | 5.13 / 5.10 / 5.01 Ah delivered; CC charge reaches 4.2 V at SOC 0.86 / 0.72 |
| B | US06 × 20 from SOC 0.9 with thermal model | sim | SOC 0.90 → 0.36, V 3.45–4.16 V, T_max 27.0 °C (25 °C ambient) |
| C | Initial SOC 0.9 / 0.6 / 0.3, repeated US06 to V_min | sim | 328 / 217 / 107 min to cut-off |
| D | Ambient 0 / 25 / 45 °C at 1C, with and without assumed Arrhenius R(T) | sim | ΔT 20.4 K (Ea = 0); 23.1 / 15.0 / 11.2 K with Ea = 30 kJ/mol |
| E | ±20 % parameter perturbations | sim | V RMSE: Q 25–38 mV, R0 8.1 mV, R1/R2 ≈ 2 mV, C1/C2 < 1 mV |
| G1 | Parameter recovery, train US06 / validate UDDS | sim, truth known | all parameters within 2.5 % (mean, σ_V = 5 mV) |
| G2 | ECM fitted to PyBaMM SPMe | model-to-model | held-out UDDS RMSE 5.6 mV (1RC), 4.8 mV (2RC) |
| G3 | ECM on NASA B0005, held-out cycles/cells | measured | discharge RMSE 51–66 mV early cycles, 67–103 mV other cells; 2RC no better than 1RC |
| G4 | Per-cycle R tracking vs EIS | measured | R0 99 → 173 mΩ; r = 0.87 with EIS Re + Rct |
| TH | Thermal identification | measured | m·cp = 44 J/K, hA = 0.034 W/K; held-out T RMSE 0.79–0.84 K (other cells) |
| F/I | Coulomb counting vs EKF | sim + measured | see §5.1 |
| H | PyBaMM SEI aging, 4 conditions × 500 cycles | sim | fade 0.31–0.71 %, R_10s growth 1.5–7.7 %, both increasing with temperature |
| SOH | Leave-one-cell-out | measured | ridge MAE 0.012 SOH vs 0.024 for linear baseline |
| RUL | Empirical extrapolation (3 cells) | measured | mean |EOL error| 28 cycles (exponential), 50 (linear) |

### 5.1 State estimation
On synthetic data the EKF keeps SOC RMSE between 0.26 % and 1.7 % under every single fault and 1.3 %
under the combined fault, whereas coulomb counting retains initial-state errors (20 %) and integrates
current bias (2.3 %) and capacity error (4.4 %). With a perfect initial state and unbiased current,
coulomb counting is more accurate (0.01 % vs 0.28 %). On measured NASA discharges the EKF with
cycle-1 parameters reduced initial errors of −0.2/+0.15 to RMSE 0.06–0.09 on cycles 1 and 50 but
gave no improvement on cycles 100 and 150, where identified R0 had increased by 45–65 % (G4); updating the
capacity from the SOH model alone did not fix this, which points to joint resistance adaptation as the
necessary next step.

## 6. Discussion
* The twin is internally consistent (energy audit, analytic tests) and parameter identification is
  well-posed under dynamic excitation (G1), but real-data identification is limited by the excitation:
  NASA cycles provide only two current levels and coarse sampling, so OCV and polarisation are
  confounded and the end-of-discharge knee is poorly represented.
* Tracking parameters over life (G4) is physically meaningful: identified R0 follows independent EIS.
* The PyBaMM SEI-only model predicts small, temperature-accelerated fade; it is not calibrated to and
  cannot explain the NASA cells' ~30 % fade.
* Data-driven SOH from charge timing generalises across the four cells; model complexity beyond a
  regularised linear model was not rewarded.

## 7. Limitations and extensions
See `docs/assumptions_and_limitations.md`. Priority extensions: dual (state + resistance) estimation,
OCV-hysteresis modelling, temperature-dependent parameters from multi-temperature data, plating and
LAM mechanisms in PyBaMM, and validation on additional datasets (e.g. CALCE).

## References
1. B. Saha, K. Goebel, *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames (2007).
2. G. L. Plett, Extended Kalman filtering for battery management systems of LiPB-based HEV battery
   packs, Parts 2–3, *J. Power Sources* 134 (2004) 262–292.
3. V. Sulzer et al., Python Battery Mathematical Modelling (PyBaMM), *J. Open Res. Softw.* 9 (2021) 14.
4. C.-H. Chen et al., Development of experimental techniques for parameterization of multi-scale
   lithium-ion battery models, *J. Electrochem. Soc.* 167 (2020) 080534.
5. S. E. J. O'Kane et al., Lithium-ion battery degradation: how to model it,
   *Phys. Chem. Chem. Phys.* 24 (2022) 7909.
