"""SOC estimators: coulomb counting and an extended Kalman filter (EKF) on the ECM.

EKF state x = [SOC, v1(, v2)], linear process model (exact ZOH discretisation), nonlinear output
    y = OCV(SOC) - sum(v_i) - R0 I,   H = [dOCV/dSOC, -1(, -1)].
Process noise on SOC and on the RC voltages is derived from the current-sensor standard deviation
(the noise each state receives through its input gain), plus an optional floor q_rc [V^2/step].
"""
from __future__ import annotations
import numpy as np
from .battery_ecm import ECMParams


def coulomb_counting(t, I_meas, soc0, Q_Ah, eta_charge=1.0):
    dt = np.diff(t)
    eta = np.where(I_meas[:-1] < 0, eta_charge, 1.0)
    return soc0 - np.concatenate([[0.0], np.cumsum(eta * I_meas[:-1] * dt)]) / (3600.0 * Q_Ah)


def ekf_soc(p: ECMParams, t, I_meas, V_meas, soc0, P0_soc=0.1**2, sigma_v=0.01, sigma_i=0.05,
            q_rc=1e-10, P0_rc=1e-4, clip_soc=True):
    """Returns dict with soc estimate, its 1-sigma standard deviation, innovations and voltage prediction."""
    n_rc = p.n_rc
    nx = 1 + n_rc
    x = np.zeros(nx)
    x[0] = soc0
    P = np.diag([P0_soc] + [P0_rc] * n_rc)
    Rk = sigma_v**2
    N = len(t)
    soc = np.zeros(N); sig = np.zeros(N); innov = np.zeros(N); vhat = np.zeros(N)
    Rs = [p.R1, p.R2][:n_rc]
    Cs = [p.C1, p.C2][:n_rc]
    for k in range(N):
        # ---- measurement update at t_k
        ocv = p.ocv(x[0])
        yhat = ocv - x[1:].sum() - p.R0 * I_meas[k]
        H = np.r_[p.ocv.deriv(np.clip(x[0], 0, 1)), -np.ones(n_rc)]
        S = H @ P @ H + Rk
        K = P @ H / S
        e = V_meas[k] - yhat
        x = x + K * e
        P = (np.eye(nx) - np.outer(K, H)) @ P
        P = 0.5 * (P + P.T)
        if clip_soc:   # state-constraint projection: SOC is physically bounded to [0, 1]
            x[0] = min(max(x[0], 0.0), 1.0)
        soc[k], sig[k], innov[k], vhat[k] = x[0], np.sqrt(max(P[0, 0], 0)), e, yhat
        # ---- time update to t_{k+1}
        if k < N - 1:
            dt = t[k + 1] - t[k]
            a = np.array([np.exp(-dt / (R * C)) for R, C in zip(Rs, Cs)])
            eta = p.eta_charge if I_meas[k] < 0 else 1.0
            F = np.diag(np.r_[1.0, a])
            Bu = np.r_[-eta * dt / (3600 * p.Q_Ah), np.array(Rs) * (1 - a)] * I_meas[k]
            x = F @ x + Bu
            Qk = np.diag(np.r_[(sigma_i * dt / (3600 * p.Q_Ah))**2, (np.array(Rs) * (1 - a) * sigma_i)**2 + q_rc])
            P = F @ P @ F.T + Qk
    return {"soc": soc, "sigma": sig, "innovation": innov, "v_pred": vhat}
