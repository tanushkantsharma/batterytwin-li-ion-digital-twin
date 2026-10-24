"""Forward-simulation experiments A-E, numerical checks, and H (PyBaMM aging).

All results here are SIMULATIONS of the synthetic reference cell (src/reference_cell.py) or of
PyBaMM's OKane2022 parameter set; none are measurements.
Usage: python scripts/run_simulations.py [--aging-cycles N] [--skip-aging]
"""
from __future__ import annotations
import argparse, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_ecm import simulate, energy_balance  # noqa: E402
from src.reference_cell import reference_params, reference_thermal  # noqa: E402
from src.profiles import drive_cycle, constant_current  # noqa: E402
from src.metrics import rmse, max_abs  # noqa: E402
from src.plotting import plt, save, SERIES  # noqa: E402

FIG, MET = ROOT / "results" / "figures", ROOT / "results" / "metrics"
EA_ASSUMED = 30e3   # J/mol, ASSUMED activation energy for R(T) in experiment D (sensitivity only)


def timed(f, *a, **k):
    t0 = time.perf_counter(); r = f(*a, **k); return r, time.perf_counter() - t0


def exp_A(p):
    rows, fig, ax = [], *plt.subplots(1, 2, figsize=(10, 3.6))
    for j, c in enumerate([0.5, 1.0, 2.0]):
        t, I = constant_current(c * p.Q_Ah, 3.0 * 3600 / c)
        r, rt = timed(simulate, p, t, I, 1.0, stop_at_limits=True)
        eb = energy_balance(p, r)
        rows.append(dict(case=f"discharge {c}C", soc0=1.0, delivered_Ah=float((1 - r.soc[-1]) * p.Q_Ah),
                         energy_Wh=eb["E_terminal_J"] / 3600, end_soc=r.soc[-1], stop=r.stop_reason,
                         energy_residual_rel=eb["residual_rel"], runtime_s=rt))
        ax[0].plot((1 - r.soc) * p.Q_Ah, r.V, color=SERIES[j], label=f"{c}C")
    for j, c in enumerate([0.5, 1.0]):
        t, I = constant_current(-c * p.Q_Ah, 3.0 * 3600 / c)
        r, rt = timed(simulate, p, t, I, 0.0, stop_at_limits=True)
        eb = energy_balance(p, r)
        rows.append(dict(case=f"CC charge {c}C", soc0=0.0, delivered_Ah=float(r.soc[-1] * p.Q_Ah),
                         energy_Wh=-eb["E_terminal_J"] / 3600, end_soc=r.soc[-1], stop=r.stop_reason,
                         energy_residual_rel=eb["residual_rel"], runtime_s=rt))
        ax[1].plot(r.soc * p.Q_Ah, r.V, color=SERIES[j], label=f"{c}C (CC only)")
    ax[0].set(title="A. Constant-current discharge", xlabel="Discharged [Ah]", ylabel="Terminal voltage [V]")
    ax[1].set(title="A. Constant-current charge", xlabel="Charged [Ah]", ylabel="Terminal voltage [V]")
    for a in ax: a.legend()
    save(fig, FIG / "A_constant_current.png")
    return pd.DataFrame(rows)


def exp_B(p, tp):
    t, I = drive_cycle("US06", 20)
    r, rt = timed(simulate, p, t, I, 0.9, thermal=tp, T_amb=25.0)
    eb = energy_balance(p, r)
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(t / 60, I, lw=0.6, color=SERIES[0]); ax[0].set(ylabel="Current [A]\n(+ = discharge)",
                                                             title="B. US06-derived dynamic profile, synthetic reference cell")
    ax[1].plot(t / 60, r.V, lw=0.6, color=SERIES[0], label="terminal V"); ax[1].plot(t / 60, r.ocv, color=SERIES[1], label="OCV")
    ax[1].set(ylabel="Voltage [V]"); ax[1].legend(loc="lower left")
    ax[2].plot(t / 60, r.T, color=SERIES[0]); ax[2].set(ylabel="Temperature [°C]", xlabel="Time [min]")
    save(fig, FIG / "B_dynamic_profile.png")
    return pd.DataFrame([dict(case="US06 x20, SOC0=0.9, T_amb=25C", duration_s=t[-1], soc_end=r.soc[-1],
                              V_min=r.V.min(), V_max=r.V.max(), T_max_C=r.T.max(), I_rms_A=np.sqrt(np.mean(I**2)),
                              energy_residual_rel=eb["residual_rel"], runtime_s=rt)])


def exp_C(p):
    t, I = drive_cycle("US06", 60)
    rows, (fig, ax) = [], plt.subplots(figsize=(10, 3.6))
    for j, s0 in enumerate([0.9, 0.6, 0.3]):
        r = simulate(p, t, I, s0, stop_at_limits=True)
        rows.append(dict(soc0=s0, time_to_cutoff_min=r.t[-1] / 60, stop=r.stop_reason or "end of profile",
                         soc_at_cutoff=r.soc[-1], Ah_delivered=(s0 - r.soc[-1]) * p.Q_Ah))
        ax.plot(r.t / 60, r.V, lw=0.5, color=SERIES[j], label=f"SOC0 = {s0}")
    ax.axhline(p.V_min, color="#6b6a63", lw=0.8, ls="--"); ax.text(1, p.V_min + 0.03, "V_min", color="#6b6a63", fontsize=8)
    ax.set(title="C. Repeated US06 from different initial SOC (run stops at V_min)", xlabel="Time [min]", ylabel="Terminal voltage [V]")
    ax.legend(); save(fig, FIG / "C_initial_soc.png")
    return pd.DataFrame(rows)


def exp_D(p):
    rows, (fig, ax) = [], plt.subplots(1, 2, figsize=(10, 3.6))
    t, I = constant_current(p.Q_Ah, 4000)
    for j, Ta in enumerate([0.0, 25.0, 45.0]):
        for Ea, ls in [(0.0, ":"), (EA_ASSUMED, "-")]:
            tp = reference_thermal(Ea=Ea)
            r = simulate(p, t, I, 1.0, stop_at_limits=True, thermal=tp, T_amb=Ta)
            rows.append(dict(T_amb_C=Ta, Ea_J_per_mol=Ea, Ah_to_cutoff=(1 - r.soc[-1]) * p.Q_Ah,
                             T_max_C=r.T.max(), dT_C=r.T.max() - Ta, mean_V=r.V.mean()))
            lab = f"{Ta:.0f} °C" + (" (Ea=0)" if Ea == 0 else "")
            ax[0].plot((1 - r.soc) * p.Q_Ah, r.V, ls=ls, color=SERIES[j], label=lab)
            ax[1].plot(r.t / 60, r.T - Ta, ls=ls, color=SERIES[j], label=lab)
    ax[0].set(title="D. 1C discharge vs ambient (solid: assumed Arrhenius R)", xlabel="Discharged [Ah]", ylabel="Terminal voltage [V]")
    ax[1].set(title="D. Temperature rise above ambient", xlabel="Time [min]", ylabel="T - T_amb [K]")
    ax[0].legend(fontsize=7); save(fig, FIG / "D_temperature.png")
    return pd.DataFrame(rows)


def exp_E(p):
    t, I = drive_cycle("US06", 10)
    ref = simulate(p, t, I, 0.8)
    rows = []
    for name in ["Q_Ah", "R0", "R1", "C1", "R2", "C2"]:
        for f in (0.8, 1.2):
            r = simulate(replace(p, **{name: getattr(p, name) * f}), t, I, 0.8)
            rows.append(dict(parameter=name, factor=f, V_rmse_mV=1e3 * rmse(ref.V, r.V), V_maxdev_mV=1e3 * max_abs(ref.V, r.V),
                             soc_end_shift=r.soc[-1] - ref.soc[-1]))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    piv = df.pivot(index="parameter", columns="factor", values="V_rmse_mV").sort_values(1.2)
    y = np.arange(len(piv))
    ax.barh(y - 0.18, piv[0.8], height=0.34, color=SERIES[0], label="-20 %")
    ax.barh(y + 0.18, piv[1.2], height=0.34, color=SERIES[1], label="+20 %")
    ax.set_yticks(y, piv.index); ax.set(xlabel="Voltage RMSE vs nominal [mV]", title="E. Sensitivity of terminal voltage (US06 x10, SOC0 0.8)")
    ax.legend(); save(fig, FIG / "E_sensitivity.png")
    return df


def numerical_checks(p):
    rows = []
    # 1) step-size convergence of the electro-thermal coupling (explicit R(T) update)
    tp = reference_thermal(Ea=EA_ASSUMED)
    ref = None
    for dt in [0.1, 1.0, 10.0]:
        t, I = constant_current(p.Q_Ah, 3000, dt)
        r = simulate(p, t, I, 1.0, thermal=tp, T_amb=0.0)
        idx = np.searchsorted(t, np.arange(0, 3001, 10.0))
        if ref is None:
            ref = r, idx
        rows.append(dict(check="electro-thermal dt convergence (1C, 0C amb, Ea assumed)", dt_s=dt,
                         max_dV_mV_vs_dt0p1=1e3 * max_abs(ref[0].V[ref[1]], r.V[idx]),
                         max_dT_K_vs_dt0p1=max_abs(ref[0].T[ref[1]], r.T[idx])))
    # 2) energy balance with a long dynamic profile at several dt
    for dt in [0.5, 1.0]:
        t, I = drive_cycle("US06", 5)
        tt = np.arange(0, t[-1], dt); II = np.interp(np.floor(tt), t, I)
        eb = energy_balance(p, simulate(p, tt, II, 0.8))
        rows.append(dict(check="energy balance US06 x5", dt_s=dt, energy_residual_rel=eb["residual_rel"],
                         E_terminal_Wh=eb["E_terminal_J"] / 3600, E_loss_Wh=(eb["E_ohmic_J"] + eb["E_rc_J"]) / 3600))
    return pd.DataFrame(rows)


def exp_H(n_cycles):
    from src.degradation import run_pybamm_aging
    conds = [(10.0, 1.0), (25.0, 1.0), (45.0, 1.0), (25.0, 2.0)]
    allrows, summ = [], []
    for T, c in conds:
        rows, rt, ver = run_pybamm_aging(T, c, n_cycles)
        for r in rows:
            r.update(T_amb_C=T, c_dis=c)
        allrows += rows
        q0, q1 = rows[0]["capacity_Ah"], rows[-1]["capacity_Ah"]
        r0, r1 = rows[0]["R_dc10s_ohm"], rows[-1]["R_dc10s_ohm"]
        summ.append(dict(T_amb_C=T, c_dis=c, cycles=len(rows), cap_first_Ah=q0, cap_last_Ah=q1,
                         cap_fade_pct=100 * (1 - q1 / q0), R10s_first_mohm=1e3 * r0, R10s_last_mohm=1e3 * r1,
                         R_growth_pct=100 * (r1 / r0 - 1), LLI_last_pct=rows[-1]["lli_pct"], runtime_s=rt, pybamm=ver))
        print(f"  aging {T}C {c}C: {len(rows)} cycles, fade {summ[-1]['cap_fade_pct']:.2f} %, {rt:.0f} s")
    df = pd.DataFrame(allrows)
    df.to_csv(MET / "H_pybamm_aging_per_cycle.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (T, c) in enumerate(conds):
        g = df[(df.T_amb_C == T) & (df.c_dis == c)]
        lab = f"{T:.0f} °C, {c:.0f}C dis"
        ax[0].plot(g.cycle, 100 * g.capacity_Ah / g.capacity_Ah.iloc[0], color=SERIES[j], label=lab)
        ax[1].plot(g.cycle, 100 * (g.R_dc10s_ohm / g.R_dc10s_ohm.iloc[0] - 1), color=SERIES[j], label=lab)
    ax[0].set(title="H. PyBaMM SPMe + SEI: capacity retention", xlabel="Cycle", ylabel="Capacity / cycle-1 capacity [%]")
    ax[1].set(title="H. 10-s DC resistance growth", xlabel="Cycle", ylabel="Change vs cycle 1 [%]")
    ax[0].legend(); save(fig, FIG / "H_pybamm_aging.png")
    return pd.DataFrame(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging-cycles", type=int, default=500)
    ap.add_argument("--skip-aging", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); MET.mkdir(parents=True, exist_ok=True)
    p = reference_params()
    p.to_json(ROOT / "data" / "params" / "reference_cell_ecm.json",
              extra={"note": "ASSUMED R/C ground truth; OCV from PyBaMM Chen2020 C/50 SPM (model-derived)"})
    for name, fn in [("A_constant_current", lambda: exp_A(p)), ("B_dynamic", lambda: exp_B(p, reference_thermal())),
                     ("C_initial_soc", lambda: exp_C(p)), ("D_temperature", lambda: exp_D(p)),
                     ("E_sensitivity", lambda: exp_E(p)), ("numerical_checks", lambda: numerical_checks(p))]:
        df = fn(); df.to_csv(MET / f"{name}.csv", index=False)
        print(f"== {name}\n{df.to_string(index=False, float_format=lambda x: f'{x:.5g}')}")
    if not a.skip_aging:
        df = exp_H(a.aging_cycles); df.to_csv(MET / "H_pybamm_aging_summary.csv", index=False)
        print(f"== H_pybamm_aging\n{df.to_string(index=False, float_format=lambda x: f'{x:.5g}')}")


if __name__ == "__main__":
    main()
