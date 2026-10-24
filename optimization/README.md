# Optimization configuration

`bounds.json` holds the box bounds, initial guesses and multi-start count used by
`src/parameter_estimation.fit` in `scripts/run_evaluation.py`:

* `synthetic` — bounds for G1/G2 (synthetic reference cell and PyBaMM comparison).
* `nasa`, `nasa_ocv`, `nasa_initial` — bounds, combined-OCV coefficient bounds and starting points
  for the NASA fits (G3/G4).

Solver: `scipy.optimize.least_squares`, trust-region reflective, `x_scale="jac"`, ≤ 3000 function
evaluations per start. Parameters that finish on a bound are flagged in
`results/metrics/G4_nasa_parameter_tracking.csv` (`params_at_bound`).
