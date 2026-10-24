# Model equations

Sign convention: current `I > 0` is discharge. SI units unless noted (capacity in Ah).

## 1. Equivalent-circuit model (`src/battery_ecm.py`)

States `x = [z, v1, v2]` (SOC and RC-branch voltages). Continuous time:

```
dz/dt   = -η(I) I / (3600 Q)          η = 1 for discharge, η_charge for charge
dv_i/dt = -v_i / (R_i C_i) + I / C_i    i = 1, 2   (R2 = 0 → single-RC model)
V       = OCV(z) - v1 - v2 - R0 I
```

Discretisation is the exact zero-order-hold solution for current held constant on `[t_k, t_k+1)`:

```
z[k+1]   = z[k] - η I[k] Δt / (3600 Q)
v_i[k+1] = a_i v_i[k] + R_i (1 - a_i) I[k],      a_i = exp(-Δt / (R_i C_i))
```

It is therefore unconditionally stable and step-size independent for piecewise-constant current
(verified in `tests/test_ecm.py::test_step_size_invariance`).

Limits: a run with `stop_at_limits=True` terminates at the first sample with `V < V_min`
(discharging), `V > V_max` (charging), or `z ∉ [0, 1]`. CC-CV charging is not implemented in the
forward simulator (constant-current charge stops at `V_max`).

### OCV models
* **Table** (`OCVTable`): linear interpolation; slope by central difference. Used for the synthetic
  reference cell, built from PyBaMM's Chen2020 parameter set (bulk OCV of a C/50 SPM discharge).
* **Combined model** (`OCVCombined`, Plett 2004): `OCV(z) = K0 − K1/z − K2 z + K3 ln z + K4 ln(1−z)`,
  `z` clipped to `[0.01, 0.99]`. Used for NASA cells, where no OCV test exists and OCV is
  identified jointly with R/C.

### Energy audit
`E_chem = ∫ OCV·I dt` must equal `E_terminal + ∫R0 I² dt + Σ∫ v_i²/R_i dt + Δ(½ C_i v_i²)`.
`energy_balance()` evaluates each term with the left rule for current and the interval midpoint
for `v_i`; the residual checks that the discrete RC update is consistent with dissipation plus
stored energy (it is a consistency check of the implementation, not an independent physical
validation).

## 2. Lumped thermal model (`src/thermal_model.py`)

```
m c_p dT/dt = Q_gen − hA (T − T_amb)
Q_gen       = R0 I² + v1²/R1 + v2²/R2        (irreversible heat only)
```

Integrated exactly over each step for constant `Q_gen`. Optional, **assumed** Arrhenius scaling of
`R0, R1`: `R(T) = R(T_ref) exp[Ea/R_g (1/T − 1/T_ref)]`, explicit coupling (R updated from the
temperature at the start of each step). Reversible (entropic) heat is omitted.

## 3. Parameter estimation (`src/parameter_estimation.py`)

Minimise `Σ_traj Σ_k (V_model,k(θ) − V_meas,k)²` with `scipy.optimize.least_squares`
(trust-region reflective, box bounds from `optimization/bounds.json`). `R`, `C` are optimised as
`log` values; OCV coefficients linearly, initialised by linear least squares on `V + R0·I`.
Multi-start (random log-uniform restarts) guards against local minima. A Gauss–Newton covariance
`(JᵀJ)⁻¹ σ̂²` gives approximate 1-σ log-parameter uncertainties (stored in the parameter JSON).

## 4. State estimation (`src/state_estimation.py`)

* Coulomb counting: `ẑ[k] = ẑ0 − Σ η I_meas Δt / (3600 Q)`.
* EKF on the ECM: linear process `x[k+1] = F x[k] + B I[k]`, `F = diag(1, a1, a2)`;
  output Jacobian `H = [dOCV/dz, −1, −1]`. Process noise: `Q_z = (σ_I Δt / 3600Q)²`,
  `Q_vi = (R_i (1−a_i) σ_I)² + q_floor` (current-sensor noise propagated through each input gain).
  After each update the SOC estimate is projected onto `[0, 1]`.

## 5. Degradation (`src/degradation.py`)

* **Electrochemical (simulation):** PyBaMM SPMe with SEI growth `"solvent-diffusion limited"`,
  parameter set `OKane2022`, isothermal at ambient. Each cycle: 0.5C CC to 4.2 V, CV to C/20,
  20 min rest, 10 s 1C pulse, CC discharge to 2.5 V, 20 min rest. Reported: discharge capacity per
  cycle, 10-s DC resistance `(V_rest − V_10s)/I_1C`, loss of lithium inventory.
* **Empirical (measured NASA capacity):** linear `q = c0 + c1 n` and exponential
  `q = a + b e^{c n}` fitted to cycles `≤ n_start`, extrapolated to the 1.4 Ah EOL threshold.

## 6. SOH learning (`src/soh_ml.py`)

`SOH = Q_discharge / 2.0 Ah`. Features from the preceding *charge* only: CC duration, CV duration,
time for V to rise 3.9→4.1 V. Models: one-feature linear baseline, ridge (standardised, α = 1),
random forest, gradient boosting. Leave-one-cell-out cross-validation.
