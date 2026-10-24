"""Bounded nonlinear least-squares identification of ECM parameters.

Positive physical parameters (R0, R_i, C_i) are optimised in log-space; OCV coefficients of the
Plett combined model are optimised linearly.  `scipy.optimize.least_squares` (trust-region
reflective) enforces box bounds.  Multiple trajectories can be fitted jointly.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import numpy as np
from scipy.optimize import least_squares

from .battery_ecm import ECMParams, OCVCombined, simulate

RC_NAMES = ["R0", "R1", "C1", "R2", "C2"]


@dataclass
class Trajectory:
    t: np.ndarray
    I: np.ndarray
    V: np.ndarray
    soc0: float
    Q_Ah: float | None = None     # per-trajectory capacity override (e.g. measured cycle capacity)
    name: str = ""
    eta_charge: float | None = None  # per-trajectory coulombic efficiency override


def _predict(p: ECMParams, tr: Trajectory):
    pp = replace(p, Q_Ah=tr.Q_Ah or p.Q_Ah, eta_charge=tr.eta_charge or p.eta_charge)
    return simulate(pp, tr.t, tr.I, tr.soc0).V


def predict(p: ECMParams, tr: Trajectory):
    return _predict(p, tr)


def _unpack(theta, names, base: ECMParams, n_ocv):
    kw = {}
    for i, n in enumerate(names):
        kw[n] = float(np.exp(theta[i]))
    p = replace(base, **kw)
    if n_ocv:
        p = replace(p, ocv=OCVCombined(theta[len(names):]))
    return p


def fit(trajs, base: ECMParams, names, bounds, fit_ocv=False, ocv_K0=None, ocv_bounds=None,
        n_starts=1, seed=0, x0_override=None):
    """Fit `names` (subset of RC_NAMES) and optionally the 5 combined-OCV coefficients.

    bounds: dict name -> (lo, hi) in physical units.  Returns (params, info)."""
    rng = np.random.default_rng(seed)
    lo = np.log([bounds[n][0] for n in names])
    hi = np.log([bounds[n][1] for n in names])
    if fit_ocv:
        olo, ohi = (np.array(b, float) for b in ocv_bounds)
        lo, hi = np.r_[lo, olo], np.r_[hi, ohi]
    n_ocv = 5 if fit_ocv else 0

    def resid(theta):
        p = _unpack(theta, names, base, n_ocv)
        return np.concatenate([_predict(p, tr) - tr.V for tr in trajs])

    best, t0 = None, time.perf_counter()
    for s in range(n_starts):
        if s == 0:
            x0 = np.log([getattr(base, n) for n in names]) if x0_override is None else np.log(x0_override)
        else:
            x0 = rng.uniform(lo[:len(names)], hi[:len(names)])
        if fit_ocv:
            x0 = np.r_[x0, ocv_K0]
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    p = _unpack(best.x, names, base, n_ocv)
    info = {"cost": float(best.cost), "nfev": int(best.nfev), "success": bool(best.success),
            "runtime_s": time.perf_counter() - t0, "n_samples": int(sum(len(tr.t) for tr in trajs)),
            "rmse_train": float(np.sqrt(2 * best.cost / sum(len(tr.t) for tr in trajs)))}
    # approximate 1-sigma uncertainty of log-parameters from the Gauss-Newton Hessian
    try:
        J = best.jac
        dof = max(len(best.fun) - len(best.x), 1)
        s2 = 2 * best.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        info["log_param_std"] = {n: float(np.sqrt(cov[i, i])) for i, n in enumerate(names)}
    except Exception:  # noqa: BLE001
        pass
    return p, info


def init_combined_ocv(trajs, Q_list, R0_guess):
    """Linear least-squares initial guess for the combined OCV model from V + R0*I (ignores RC)."""
    rows, ys = [], []
    for tr, Q in zip(trajs, Q_list):
        z = tr.soc0 - np.concatenate([[0], np.cumsum(tr.I[:-1] * np.diff(tr.t))]) / (3600 * Q)
        z = np.clip(z, OCVCombined.eps, 1 - OCVCombined.eps)
        rows.append(np.c_[np.ones_like(z), -1 / z, -z, np.log(z), np.log(1 - z)])
        ys.append(tr.V + R0_guess * tr.I)
    K, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(ys), rcond=None)
    return K
