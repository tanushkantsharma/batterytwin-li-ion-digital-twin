"""Electrochemical aging simulations with PyBaMM (physics-model outputs, NOT measured data),
plus small empirical capacity-fade models for measured aging trajectories (NASA data).

PyBaMM setup: SPMe + SEI growth ("solvent-diffusion limited"), parameter set "OKane2022"
(LG M50, built on Chen2020; O'Kane et al., PCCP 24 (2022) 7909).  Isothermal at the ambient
temperature; SEI growth is Arrhenius-temperature dependent through the parameter set.
Only the SEI mechanism is enabled - lithium plating, particle cracking and LAM are not.
"""
from __future__ import annotations
import time
import numpy as np


def run_pybamm_aging(T_amb_C=25.0, c_dis=1.0, n_cycles=100, c_chg=0.5, model_name="SPMe"):
    import pybamm
    pv = pybamm.ParameterValues("OKane2022")
    pv.update({"Ambient temperature [K]": 273.15 + T_amb_C, "Initial temperature [K]": 273.15 + T_amb_C})
    opts = {"SEI": "solvent-diffusion limited"}
    model = getattr(pybamm.lithium_ion, model_name)(opts)
    cycle = (f"Charge at {c_chg}C until 4.2 V", "Hold at 4.2 V until C/20", "Rest for 20 minutes",
             "Discharge at 1C for 10 seconds",                 # DC-resistance pulse at full charge
             f"Discharge at {c_dis}C until 2.5 V", "Rest for 20 minutes")
    exp = pybamm.Experiment([cycle] * n_cycles)
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=exp)
    t0 = time.perf_counter()
    sol = sim.solve(initial_soc=0.0)
    runtime = time.perf_counter() - t0
    I1C = pv["Nominal cell capacity [A.h]"]
    rows = []
    for k, cyc in enumerate(sol.cycles):
        if cyc is None or len(cyc.steps) < 6:
            break
        rest, pulse, dis = cyc.steps[2], cyc.steps[3], cyc.steps[4]
        V_rest = rest["Terminal voltage [V]"].entries[-1]
        V_pulse = pulse["Terminal voltage [V]"].entries[-1]
        q = (pulse["Discharge capacity [A.h]"].entries[-1] - pulse["Discharge capacity [A.h]"].entries[0]
             + dis["Discharge capacity [A.h]"].entries[-1] - dis["Discharge capacity [A.h]"].entries[0])
        rows.append(dict(cycle=k + 1, capacity_Ah=q, R_dc10s_ohm=(V_rest - V_pulse) / I1C,
                         lli_pct=float(cyc["Loss of lithium inventory [%]"].entries[-1])))
    return rows, runtime, pybamm.__version__


# ---------------- empirical capacity-fade models for measured data (RUL baselines)
def fit_exponential(n, q):
    """q(n) = a + b * exp(c n)  (c > 0 allows accelerating fade); fallback linear."""
    from scipy.optimize import curve_fit
    f = lambda n, a, b, c: a + b * np.exp(c * n)  # noqa: E731
    try:
        p, _ = curve_fit(f, n, q, p0=(q[0] + 0.05, -0.05, 0.01), maxfev=20000)
        return lambda x: f(np.asarray(x, float), *p), p
    except Exception:  # noqa: BLE001
        return fit_linear(n, q)


def fit_linear(n, q):
    c = np.polyfit(n, q, 1)
    return (lambda x: np.polyval(c, np.asarray(x, float))), c


def first_crossing(n, q, threshold):
    idx = np.nonzero(np.asarray(q) <= threshold)[0]
    return int(n[idx[0]]) if len(idx) else None


def predict_eol(model, threshold, n_start, n_max=2000):
    grid = np.arange(n_start, n_max + 1)
    return first_crossing(grid, model(grid), threshold)
