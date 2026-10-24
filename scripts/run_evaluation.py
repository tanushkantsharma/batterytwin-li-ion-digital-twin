"""Estimation & learning experiments: F, G, I, thermal identification, SOH-ML, RUL.

  G1  synthetic parameter recovery (known ground truth), held-out drive cycle
  G2  ECM fitted to PyBaMM SPMe output (structural-mismatch test, model-to-model)
  G3  NASA B0005: joint OCV + R/C identification, held-out cycles and cells (MEASURED data)
  G4  NASA B0005: parameter tracking over aging vs independent EIS Re / Rct
  TH  NASA: lumped thermal identification and held-out temperature prediction
  F/I SOC estimation: coulomb counting vs EKF (synthetic truth; NASA with reference SOC)
  SOH leave-one-cell-out ML vs linear baseline (NASA)
  RUL empirical extrapolation baselines (NASA cells that reach the 1.4 Ah EOL criterion)
Usage: python scripts/run_evaluation.py [--quick]
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_ecm import ECMParams, simulate  # noqa: E402
from src.reference_cell import reference_params, TRUE_RC  # noqa: E402
from src.parameter_estimation import fit, Trajectory, init_combined_ocv, predict, RC_NAMES  # noqa: E402
from src.state_estimation import coulomb_counting, ekf_soc  # noqa: E402
from src.thermal_model import ThermalParams, simulate_temperature  # noqa: E402
from src.profiles import drive_cycle  # noqa: E402
from src import nasa_data as nd  # noqa: E402
from src.soh_ml import build_dataset, leave_one_cell_out, models, FEATURES, RATED_AH  # noqa: E402
from src.degradation import fit_exponential, fit_linear, first_crossing, predict_eol  # noqa: E402
from src.metrics import rmse, mae, max_abs, r2  # noqa: E402
from src.plotting import plt, save, SERIES, MUTED  # noqa: E402

FIG, MET, PAR = ROOT / "results" / "figures", ROOT / "results" / "metrics", ROOT / "data" / "params"
CFG = json.load(open(ROOT / "optimization" / "bounds.json"))
LOG = {}


def out(name, df):
    df.to_csv(MET / f"{name}.csv", index=False)
    print(f"== {name}\n{df.to_string(index=False, float_format=lambda x: f'{x:.5g}')}")


# --------------------------------------------------------------------------- G1
def g1_synthetic_recovery(seeds):
    p = reference_params()
    tt, It = drive_cycle("US06", 4)
    tv, Iv = drive_cycle("UDDS", 3)
    Vt, Vv = simulate(p, tt, It, 0.8).V, simulate(p, tv, Iv, 0.6).V
    guess = reference_params(R0=0.03, R1=0.005, C1=6000, R2=0.03, C2=1e4)
    rows = []
    for sv in [0.001, 0.005]:
        for s in range(seeds):
            rng = np.random.default_rng(s)
            tr = Trajectory(tt, It + rng.normal(0, 0.01, len(tt)), Vt + rng.normal(0, sv, len(tt)), 0.8)
            est, info = fit([tr], guess, RC_NAMES, CFG["synthetic"])
            row = dict(sigma_V_mV=1e3 * sv, sigma_I_A=0.01, seed=s, runtime_s=info["runtime_s"],
                       train_rmse_mV=1e3 * info["rmse_train"],
                       val_rmse_mV_vs_truth=1e3 * rmse(Vv, simulate(est, tv, Iv, 0.6).V))
            row.update({f"err_{n}_pct": 100 * (getattr(est, n) / TRUE_RC[n] - 1) for n in RC_NAMES})
            rows.append(row)
    df = pd.DataFrame(rows)
    out("G1_synthetic_param_recovery_runs", df)
    agg = df.drop(columns="seed").groupby(["sigma_V_mV", "sigma_I_A"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    out("G1_synthetic_param_recovery_summary", agg.reset_index())


# --------------------------------------------------------------------------- G2
def g2_pybamm_mismatch():
    import pybamm
    p = reference_params()

    def run(name, reps, soc0):
        t, I = drive_cycle(name, reps)
        pv = pybamm.ParameterValues("Chen2020")
        pv["Current function [A]"] = pybamm.Interpolant(t, I, pybamm.t, interpolator="linear")
        sim = pybamm.Simulation(pybamm.lithium_ion.SPMe(), parameter_values=pv)
        t0 = time.perf_counter()
        sol = sim.solve(t_eval=t, initial_soc=soc0)
        return t, I, sol["Voltage [V]"](t), time.perf_counter() - t0

    tt, It, Vt, rt1 = run("US06", 4, 0.8)
    tv, Iv, Vv, rt2 = run("UDDS", 3, 0.6)
    # PyBaMM interpolates current linearly; resample the ECM input at the same instants (ZOH)
    rows, preds = [], {}
    for names, base in [(["R0", "R1", "C1"], replace(p, R2=0.0)), (RC_NAMES, p)]:
        est, info = fit([Trajectory(tt, It, Vt, 0.8)], base, names, CFG["synthetic"], n_starts=3)
        Vh = simulate(est, tv, Iv, 0.6).V
        lab = f"{len(names) // 2}RC"
        preds[lab] = Vh
        rows.append(dict(model=lab, **{n: getattr(est, n) for n in names}, train_rmse_mV=1e3 * info["rmse_train"],
                         val_rmse_mV=1e3 * rmse(Vv, Vh), val_mae_mV=1e3 * mae(Vv, Vh), val_max_mV=1e3 * max_abs(Vv, Vh),
                         fit_runtime_s=info["runtime_s"], pybamm_runtime_s=rt1 + rt2))
    out("G2_ecm_vs_pybamm_spme", pd.DataFrame(rows))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(tv / 60, Vv, color=SERIES[0], lw=0.8, label="PyBaMM SPMe (Chen2020)")
    ax[0].plot(tv / 60, preds["2RC"], color=SERIES[1], lw=0.8, label="ECM 2RC fitted on US06")
    ax[0].set(ylabel="Voltage [V]", title="G2. Held-out UDDS: ECM vs electrochemical model (both simulated)"); ax[0].legend()
    ax[1].plot(tv / 60, 1e3 * (preds["2RC"] - Vv), color=SERIES[1], lw=0.6, label="2RC")
    ax[1].plot(tv / 60, 1e3 * (preds["1RC"] - Vv), color=SERIES[0], lw=0.6, label="1RC")
    ax[1].set(ylabel="Residual [mV]", xlabel="Time [min]"); ax[1].legend()
    save(fig, FIG / "G2_ecm_vs_pybamm.png")


# --------------------------------------------------------------------------- G3
TRAIN = [("B0005", 1), ("B0005", 2)]
HELD = [("B0005", 3), ("B0005", 10), ("B0005", 50), ("B0005", 100), ("B0005", 150),
        ("B0006", 1), ("B0007", 1), ("B0018", 1)]


def g3_nasa_fit():
    trajs = [x for c, i in TRAIN for x in nd.pair(c, i)]
    K0 = init_combined_ocv(trajs, [t.Q_Ah for t in trajs], 0.1)
    rows, fitted = [], {}
    for lab, names in [("1RC", ["R0", "R1", "C1"]), ("2RC", RC_NAMES)]:
        base = ECMParams(Q_Ah=2.0, V_min=2.0, V_max=4.2, **CFG["nasa_initial"][lab])
        est, info = fit(trajs, base, names, CFG["nasa"], fit_ocv=True, ocv_K0=K0, ocv_bounds=CFG["nasa_ocv"],
                        n_starts=CFG["n_starts"])
        fitted[lab] = est
        est.to_json(PAR / f"nasa_B0005_ecm_{lab}.json", extra={
            "fitted_on": [f"{c} charge+discharge {i}" for c, i in TRAIN],
            "note": "OCV identified jointly (no OCV test in dataset); Q_Ah overridden per cycle by measured capacity",
            "train_rmse_V": info["rmse_train"], "log_param_std": info.get("log_param_std")})
        k = len(est.ocv.K) + len(names)
        n = info["n_samples"]
        aic = n * np.log(2 * info["cost"] / n) + 2 * k
        for c, i in [("train", 0)] + HELD:
            tset = trajs if c == "train" else nd.pair(c, i)
            for tr in tset:
                Vh = predict(est, tr)
                rows.append(dict(model=lab, split="train" if c == "train" else "held-out", trajectory=tr.name,
                                 rmse_mV=1e3 * rmse(tr.V, Vh), mae_mV=1e3 * mae(tr.V, Vh), max_mV=1e3 * max_abs(tr.V, Vh),
                                 Q_Ah=tr.Q_Ah, AIC_train=aic, fit_runtime_s=info["runtime_s"]))
        LOG[f"nasa_{lab}"] = {n: getattr(est, n) for n in names}
    df = pd.DataFrame(rows)
    out("G3_nasa_ecm_fit", df)
    # independent check: fitted R0 vs EIS Re around the training cycles
    imp = nd.impedance().query("cell == 'B0005'")
    near = imp.iloc[(imp.discharges_before - 2).abs().argsort()[:1]]
    out("G3_R0_vs_EIS", pd.DataFrame([dict(model=m, R0_fit_ohm=fitted[m].R0, EIS_Re_ohm=float(near.Re_ohm.iloc[0]),
                                           EIS_Re_plus_Rct_ohm=float(near.Re_ohm.iloc[0] + near.Rct_ohm.iloc[0]),
                                           EIS_after_discharges=int(near.discharges_before.iloc[0])) for m in fitted]))
    # residual figure (1RC): one training + two held-out discharges
    p = fitted["1RC"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for j, (c, i, tag) in enumerate([("B0005", 1, "train"), ("B0005", 100, "held-out cycle"), ("B0007", 1, "held-out cell")]):
        tr = nd.pair(c, i)[1]
        Vh = predict(p, tr)
        ax[0, j].plot(tr.t / 60, tr.V, "o", ms=2.5, color=SERIES[0], label="measured")
        ax[0, j].plot(tr.t / 60, Vh, color=SERIES[1], label="ECM 1RC")
        ax[0, j].set(title=f"{c} discharge {i} ({tag})"); ax[1, j].set(xlabel="Time [min]")
        ax[1, j].plot(tr.t / 60, 1e3 * (Vh - tr.V), color=SERIES[1], lw=1)
        ax[1, j].axhline(0, color=MUTED, lw=0.6)
    ax[0, 0].set_ylabel("Voltage [V]"); ax[1, 0].set_ylabel("Residual [mV]"); ax[0, 0].legend()
    fig.suptitle("G3. NASA PCoE measured discharge vs fitted ECM (2 A constant current)", fontsize=10, y=1.03)
    save(fig, FIG / "G3_nasa_fit_residuals.png")
    return fitted["1RC"]


# --------------------------------------------------------------------------- G4
def g4_tracking(p):
    imp = nd.impedance().query("cell == 'B0005'")
    rows = []
    for i in range(1, 168, 10):
        tc, td = nd.pair("B0005", i)
        before = rmse(td.V, predict(p, td))
        est, info = fit([tc, td], p, ["R0", "R1", "C1"], CFG["nasa"])
        e = imp.iloc[(imp.discharges_before - i).abs().argsort()[:1]]
        at_bound = [n for n in ["R0", "R1", "C1"] if np.isclose(getattr(est, n), CFG["nasa"][n], rtol=1e-3).any()]
        rows.append(dict(discharge_idx=i, capacity_Ah=td.Q_Ah, R0_ohm=est.R0, R1_ohm=est.R1, C1_F=est.C1,
                         params_at_bound=",".join(at_bound),
                         dis_rmse_fixed_params_mV=1e3 * before, dis_rmse_refit_mV=1e3 * rmse(td.V, predict(est, td)),
                         EIS_Re_ohm=float(e.Re_ohm.iloc[0]), EIS_Rct_ohm=float(e.Rct_ohm.iloc[0]),
                         EIS_discharges_before=int(e.discharges_before.iloc[0])))
    df = pd.DataFrame(rows)
    out("G4_nasa_parameter_tracking", df)
    c = np.corrcoef(df.R0_ohm, df.EIS_Re_ohm + df.EIS_Rct_ohm)[0, 1]
    LOG["corr_R0_vs_EIS_Re_plus_Rct"] = float(c)
    LOG["corr_R0_vs_EIS_Re"] = float(np.corrcoef(df.R0_ohm, df.EIS_Re_ohm)[0, 1])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(df.discharge_idx, 1e3 * df.R0_ohm, "o-", ms=4, color=SERIES[0], label="ECM R0 (refit per cycle)")
    ax.plot(df.discharge_idx, 1e3 * (df.EIS_Re_ohm + df.EIS_Rct_ohm), "s-", ms=4, color=SERIES[1], label="EIS Re + Rct (independent)")
    ax.plot(df.discharge_idx, 1e3 * df.EIS_Re_ohm, "^-", ms=4, color=SERIES[2], label="EIS Re (independent)")
    ax.set(xlabel="Discharge index", ylabel="Resistance [mΩ]", title="G4. B0005 resistance growth: ECM tracking vs EIS")
    ax.legend(); save(fig, FIG / "G4_resistance_tracking.png")


# --------------------------------------------------------------------------- thermal
def thermal_fit(p):
    def q_and_T(c, i):
        tr = nd.pair(c, i)[1]
        g = nd.discharge_frame(c, i)
        r = simulate(replace(p, Q_Ah=tr.Q_Ah), tr.t, tr.I, 1.0)
        q = p.R0 * tr.I**2 + r.v1**2 / p.R1
        return tr.t, q, g.T_C.values

    T_AMB = 24.0   # documented ambient for these cells (data field ambient_temperature)
    t, q, T = q_and_T("B0005", 1)

    def resid(th):
        tp = ThermalParams(m_cp=np.exp(th[0]), hA=np.exp(th[1]))
        return simulate_temperature(tp, t, q, T[0], T_AMB) - T

    r = least_squares(resid, np.log([40.0, 0.05]), bounds=(np.log([1, 1e-3]), np.log([1e3, 5])))
    tp = ThermalParams(m_cp=float(np.exp(r.x[0])), hA=float(np.exp(r.x[1])))
    json.dump({"m_cp_J_per_K": tp.m_cp, "hA_W_per_K": tp.hA, "tau_s": tp.tau, "T_amb_C": T_AMB,
               "fitted_on": "B0005 discharge 1 (measured surface temperature)",
               "note": "irreversible heat only; entropic heat neglected"}, open(PAR / "nasa_B0005_thermal.json", "w"), indent=2)
    rows, fig = [], plt.figure(figsize=(8, 3.6))
    ax = fig.gca()
    for j, (c, i) in enumerate([("B0005", 1), ("B0005", 100), ("B0006", 1), ("B0007", 1), ("B0018", 1)]):
        t, q, T = q_and_T(c, i)
        Th = simulate_temperature(tp, t, q, T[0], T_AMB)
        rows.append(dict(trajectory=f"{c}-d{i}", split="train" if (c, i) == ("B0005", 1) else "held-out",
                         T_rmse_K=rmse(T, Th), T_max_abs_K=max_abs(T, Th), T_peak_meas_C=T.max(), T_peak_pred_C=Th.max(),
                         m_cp=tp.m_cp, hA=tp.hA))
        if j < 3:
            ax.plot(t / 60, T, "o", ms=2.5, color=SERIES[j], label=f"{c}-d{i} measured")
            ax.plot(t / 60, Th, color=SERIES[j], lw=1.2, label=f"{c}-d{i} model")
    ax.set(xlabel="Time [min]", ylabel="Surface temperature [°C]", title="Thermal model vs NASA measured temperature (2 A discharge)")
    ax.legend(fontsize=7, ncol=2); save(fig, FIG / "TH_nasa_thermal.png")
    out("TH_nasa_thermal_fit", pd.DataFrame(rows))


# --------------------------------------------------------------------------- F / I
def f_soc_synthetic(seeds):
    p = reference_params()
    t, I = drive_cycle("US06", 25)
    truth = simulate(p, t, I, 0.9)
    scen = {
        "nominal (sV=5mV, sI=20mA)": dict(),
        "initial SOC error -0.2": dict(soc_guess=0.7),
        "voltage noise 20 mV": dict(sV=0.02),
        "current bias +50 mA": dict(bias=0.05),
        "R0 mismatch +30 %": dict(model=dict(R0=p.R0 * 1.3)),
        "capacity mismatch -10 %": dict(model=dict(Q_Ah=p.Q_Ah * 0.9)),
        "OCV offset +10 mV": dict(ocv_off=0.01),
        "combined: init -0.2, bias, R0 +30 %": dict(soc_guess=0.7, bias=0.05, model=dict(R0=p.R0 * 1.3)),
    }
    rows, trace = [], None
    for name, s in scen.items():
        for seed in range(seeds):
            rng = np.random.default_rng(100 + seed)
            sV = s.get("sV", 0.005)
            Vm = truth.V + rng.normal(0, sV, len(t))
            Im = I + rng.normal(0, 0.02, len(t)) + s.get("bias", 0.0)
            pm = replace(p, **s.get("model", {}))
            if "ocv_off" in s:
                base_ocv = p.ocv
                class Off:  # noqa: E306
                    def __call__(self, z): return base_ocv(z) + s["ocv_off"]
                    def deriv(self, z): return base_ocv.deriv(z)
                pm = replace(pm, ocv=Off())
            g = s.get("soc_guess", 0.9)
            cc = coulomb_counting(t, Im, g, pm.Q_Ah)
            t0 = time.perf_counter()
            ek = ekf_soc(pm, t, Im, Vm, g, sigma_v=sV, sigma_i=0.02)["soc"]
            rt = time.perf_counter() - t0
            conv = t >= 600   # metrics after a 10-min convergence window are reported separately
            for m, est in [("coulomb counting", cc), ("EKF", ek)]:
                rows.append(dict(scenario=name, method=m, seed=seed, rmse=rmse(truth.soc, est),
                                 max_abs=max_abs(truth.soc, est), rmse_after_10min=rmse(truth.soc[conv], est[conv]),
                                 final_abs=abs(est[-1] - truth.soc[-1]), runtime_s=rt if m == "EKF" else np.nan))
            if name.startswith("combined") and seed == 0:
                trace = (t, truth.soc, cc, ek)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "F_soc_estimation_synthetic_runs.csv", index=False)
    agg = df.groupby(["scenario", "method"], sort=False)[["rmse", "max_abs", "rmse_after_10min", "final_abs", "runtime_s"]].mean().reset_index()
    out("F_soc_estimation_synthetic_summary", agg)
    t, s, cc, ek = trace
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"wspace": 0.75})
    ax[0].plot(t / 60, s, color=SERIES[0], label="true SOC (simulated)")
    ax[0].plot(t / 60, cc, color=SERIES[1], label="coulomb counting")
    ax[0].plot(t / 60, ek, color=SERIES[2], label="EKF")
    ax[0].set(xlabel="Time [min]", ylabel="SOC [-]", title="F. Combined fault: init -0.2, +50 mA bias, R0 +30 %"); ax[0].legend()
    piv = agg.pivot(index="scenario", columns="method", values="rmse").reindex(list(scen))
    y = np.arange(len(piv))
    ax[1].barh(y - 0.18, 100 * piv["coulomb counting"], 0.34, color=SERIES[1], label="coulomb counting")
    ax[1].barh(y + 0.18, 100 * piv["EKF"], 0.34, color=SERIES[2], label="EKF")
    ax[1].set_yticks(y, piv.index, fontsize=7.5); ax[1].invert_yaxis()
    ax[1].set(xlabel="SOC RMSE [%]", title=f"F. Mean over {seeds} noise seeds"); ax[1].legend()
    save(fig, FIG / "F_soc_estimation.png")


def f_soc_nasa(p, soh_model):
    """EKF on measured discharges. Reference SOC = coulomb count from SOC 1 with the cycle's measured
    capacity (this is a reference definition, not an independent ground truth)."""
    feats, mdl = soh_model
    rows = []
    sohdf, _ = build_dataset(nd.PROC, cells=("B0005",))
    # Estimators start mid-discharge (first sample with reference SOC <= 0.8, under load, RC state
    # unknown) so that the [0,1] SOC projection cannot hide the initial error at the full-charge bound.
    # sigma_v = 50 mV reflects the held-out ECM voltage RMSE on these cells (G3), not sensor noise.
    for i in [1, 50, 100, 150]:
        tc, td = nd.pair("B0005", i)
        ref_full = coulomb_counting(td.t, td.I, 1.0, td.Q_Ah)
        k0 = int(np.argmax(ref_full <= 0.8))
        t, I, V, ref = td.t[k0:] - td.t[k0], td.I[k0:], td.V[k0:], ref_full[k0:]
        on = I > 0.5
        frow = sohdf.query("discharge_idx == @i")
        Q_ml = float(mdl.predict(frow[feats].values)[0]) * RATED_AH if len(frow) else np.nan
        for qname, Q in [("Q fixed (cycle-1 fit)", nd.pair("B0005", 1)[1].Q_Ah), ("Q from SOH-ML", Q_ml),
                         ("Q measured (oracle)", td.Q_Ah)]:
            if np.isnan(Q):
                continue
            pm = replace(p, Q_Ah=Q)
            for d0 in (-0.2, +0.15):
                g = ref[0] + d0
                ek = ekf_soc(pm, t, I, V, g, P0_soc=0.3**2, sigma_v=0.05, sigma_i=0.01)["soc"]
                cc = coulomb_counting(t, I, g, Q)
                for m, est in [("coulomb counting", cc), ("EKF", ek)]:
                    n2 = on.sum() // 2
                    rows.append(dict(discharge_idx=i, Q_true_Ah=td.Q_Ah, Q_variant=qname, Q_used_Ah=Q,
                                     init_error=d0, method=m, rmse=rmse(ref[on], est[on]),
                                     max_abs=max_abs(ref[on], est[on]),
                                     rmse_last_half=rmse(ref[on][n2:], est[on][n2:])))
    df = pd.DataFrame(rows)
    out("F_soc_estimation_nasa", df)
    out("F_soc_estimation_nasa_summary", df.groupby(["Q_variant", "method"])[["rmse", "max_abs", "rmse_last_half"]]
        .mean().reset_index())


# --------------------------------------------------------------------------- SOH / RUL
def soh_ml():
    df, dropped = build_dataset(nd.PROC)
    df.to_csv(MET / "SOH_dataset_features.csv", index=False)
    dropped.to_csv(MET / "SOH_dropped_records.csv", index=False)
    pred = leave_one_cell_out(df)
    pred.to_csv(MET / "SOH_loco_predictions.csv", index=False)
    rows = []
    for (m, c), g in pred.groupby(["model", "test_cell"]):
        rows.append(dict(model=m, test_cell=c, n=len(g), mae=mae(g.soh_true, g.soh_pred),
                         rmse=rmse(g.soh_true, g.soh_pred), r2=r2(g.soh_true, g.soh_pred)))
    for m, g in pred.groupby("model"):
        rows.append(dict(model=m, test_cell="ALL (pooled LOCO)", n=len(g), mae=mae(g.soh_true, g.soh_pred),
                         rmse=rmse(g.soh_true, g.soh_pred), r2=r2(g.soh_true, g.soh_pred)))
    out("SOH_loco_metrics", pd.DataFrame(rows))
    cells = sorted(df.cell.unique())
    fig, ax = plt.subplots(1, len(cells), figsize=(13, 3.4), sharey=True)
    for j, c in enumerate(cells):
        for k, m in enumerate(["baseline_linear_t_cc", "ridge_all", "random_forest"]):
            g = pred.query("model == @m and test_cell == @c")
            if k == 0:
                ax[j].plot(g.discharge_idx, g.soh_true, "o", ms=2.2, color="#6b6a63", label="measured")
            ax[j].plot(g.discharge_idx, g.soh_pred, color=SERIES[k], lw=1.1, label=m)
        ax[j].set(title=f"test cell {c}", xlabel="Discharge index")
    ax[0].set_ylabel("SOH = capacity / 2 Ah"); ax[0].legend(fontsize=7)
    fig.suptitle("SOH leave-one-cell-out: charge-phase features -> next-discharge capacity (NASA, measured)", fontsize=10, y=1.03)
    save(fig, FIG / "SOH_loco.png")
    # model for downstream EKF on B0005: trained WITHOUT B0005
    feats, mdl = models()["ridge_all"]
    tr = df[df.cell != "B0005"]
    mdl.fit(tr[feats].values, tr.soh.values)
    return feats, mdl


def rul():
    cyc = nd.cycles()
    EOL = 1.4   # Ah: 30 % fade of the 2 Ah rating, the EOL criterion stated by NASA for this dataset
    rows = []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    j = 0
    for c in ["B0005", "B0006", "B0007", "B0018"]:
        g = cyc[cyc.cell == c]
        n, q = g.discharge_idx.values + 1, g.capacity_Ah.values
        true_eol = first_crossing(n, q, EOL)
        if true_eol is None:
            rows.append(dict(cell=c, note=f"never reaches {EOL} Ah (min {q.min():.3f}); excluded"))
            continue
        for start in [40, 60, 80]:
            if start >= true_eol:
                continue
            m = n <= start
            for name, fitter in [("linear", fit_linear), ("exponential", fit_exponential)]:
                f, _ = fitter(n[m], q[m])
                pe = predict_eol(f, EOL, start)
                rows.append(dict(cell=c, start_cycle=start, model=name, true_eol=true_eol, pred_eol=pe,
                                 true_rul=true_eol - start, pred_rul=(pe - start) if pe else np.nan,
                                 rul_error_cycles=(pe - true_eol) if pe else np.nan))
        f_lin, _ = fit_linear(n[n <= 60], q[n <= 60]); f_exp, _ = fit_exponential(n[n <= 60], q[n <= 60])
        grid = np.arange(1, n.max() + 40)
        ax[j].plot(n, q, "o", ms=2.2, color="#6b6a63", label="measured")
        ax[j].plot(grid, f_lin(grid), color=SERIES[0], label="linear fit (≤60)")
        ax[j].plot(grid, f_exp(grid), color=SERIES[1], label="exponential fit (≤60)")
        ax[j].axhline(EOL, color=MUTED, ls="--", lw=0.8); ax[j].axvline(60, color=MUTED, ls=":", lw=0.8)
        ax[j].set_ylim(1.1, 2.1)
        ax[j].set(title=f"{c} (true EOL cycle {true_eol})", xlabel="Discharge cycle"); j += 1
    ax[0].set_ylabel("Capacity [Ah]"); ax[0].legend(fontsize=7)
    fig.suptitle("RUL: empirical extrapolation to 1.4 Ah (NASA, measured capacity)", fontsize=10, y=1.03)
    save(fig, FIG / "RUL_extrapolation.png")
    out("RUL_empirical", pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer noise seeds")
    a = ap.parse_args()
    seeds = 2 if a.quick else 5
    for d in (FIG, MET, PAR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    g1_synthetic_recovery(seeds)
    g2_pybamm_mismatch()
    p = g3_nasa_fit()
    g4_tracking(p)
    thermal_fit(p)
    f_soc_synthetic(seeds)
    soh_model = soh_ml()
    f_soc_nasa(p, soh_model)
    rul()
    LOG["total_runtime_s"] = time.perf_counter() - t0
    json.dump(LOG, open(MET / "evaluation_log.json", "w"), indent=2, default=float)
    print(json.dumps(LOG, indent=2, default=float))


if __name__ == "__main__":
    main()
