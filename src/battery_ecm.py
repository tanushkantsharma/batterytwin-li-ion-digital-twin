"""Equivalent-circuit model (ECM): OCV(SOC) + R0 + up to two RC branches.

Sign convention: current I > 0 is DISCHARGE, I < 0 is charge.  Units: A, V, s, Ohm, F, Ah.

States x = [SOC, v1, v2] evolve with the exact zero-order-hold discretisation
    SOC[k+1] = SOC[k] - eta(I) * I[k] * dt / (3600 Q)
    v_i[k+1] = a_i v_i[k] + R_i (1 - a_i) I[k],  a_i = exp(-dt / (R_i C_i))
Output  V[k] = OCV(SOC[k]) - v1[k] - v2[k] - R0 I[k]
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import json
import numpy as np


# ---------------------------------------------------------------- OCV models
class OCVTable:
    """Monotone OCV(SOC) lookup with linear interpolation and finite-difference slope."""

    def __init__(self, soc, ocv):
        soc = np.asarray(soc, float)
        ocv = np.asarray(ocv, float)
        order = np.argsort(soc)
        self.soc, self.ocv = soc[order], ocv[order]

    def __call__(self, z):
        return np.interp(z, self.soc, self.ocv)

    def deriv(self, z, h=1e-3):
        return (self(np.minimum(z + h, 1.0)) - self(np.maximum(z - h, 0.0))) / (
            np.minimum(z + h, 1.0) - np.maximum(z - h, 0.0))

    @classmethod
    def from_csv(cls, path):
        d = np.genfromtxt(path, delimiter=",", names=True)
        return cls(d["soc"], d["ocv_V"])

    def to_dict(self):
        return {"type": "table", "soc": self.soc.tolist(), "ocv": self.ocv.tolist()}


class OCVCombined:
    """Plett 'combined' parametric OCV model (Plett, 2004, J. Power Sources 134):
        OCV(z) = K0 - K1/z - K2 z + K3 ln z + K4 ln(1 - z),  z clipped to [eps, 1-eps].
    Used when no measured OCV test is available, so OCV is identified jointly with R/C."""

    eps = 0.01

    def __init__(self, K):
        self.K = np.asarray(K, float)

    def _z(self, z):
        return np.clip(z, self.eps, 1 - self.eps)

    def __call__(self, z):
        K, z = self.K, self._z(np.asarray(z, float))
        return K[0] - K[1] / z - K[2] * z + K[3] * np.log(z) + K[4] * np.log(1 - z)

    def deriv(self, z):
        K, z = self.K, self._z(np.asarray(z, float))
        return K[1] / z**2 - K[2] + K[3] / z - K[4] / (1 - z)

    def to_dict(self):
        return {"type": "combined", "K": self.K.tolist()}


# ---------------------------------------------------------------- parameters
@dataclass
class ECMParams:
    Q_Ah: float                 # capacity used for SOC definition
    R0: float                   # ohmic resistance
    R1: float
    C1: float
    R2: float = 0.0             # R2 = 0 -> single-RC model
    C2: float = 1.0
    eta_charge: float = 1.0     # coulombic efficiency applied to charge current
    V_min: float = 2.5
    V_max: float = 4.2
    ocv: Optional[Callable] = field(default=None, repr=False)

    @property
    def n_rc(self):
        return 2 if self.R2 > 0 else 1

    def to_json(self, path, extra=None):
        d = {k: v for k, v in asdict(self).items() if k != "ocv"}
        d["ocv"] = self.ocv.to_dict() if hasattr(self.ocv, "to_dict") else None
        if extra:
            d.update(extra)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        o = d.pop("ocv")
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        p = cls(**d)
        p.ocv = OCVTable(o["soc"], o["ocv"]) if o["type"] == "table" else OCVCombined(o["K"])
        return p


# ---------------------------------------------------------------- simulation
@dataclass
class SimResult:
    t: np.ndarray
    I: np.ndarray
    V: np.ndarray
    soc: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    ocv: np.ndarray
    T: Optional[np.ndarray] = None
    stopped_at: Optional[int] = None   # index where a limit terminated the run
    stop_reason: str = ""

    def truncate(self):
        n = len(self.t) if self.stopped_at is None else self.stopped_at + 1
        for k in ("t", "I", "V", "soc", "v1", "v2", "ocv", "T"):
            a = getattr(self, k)
            if a is not None:
                setattr(self, k, a[:n])
        return self


def rc_coeffs(p: ECMParams, dt):
    a1 = np.exp(-dt / (p.R1 * p.C1))
    a2 = np.exp(-dt / (p.R2 * p.C2)) if p.R2 > 0 else 0.0
    return a1, a2


def step(p: ECMParams, x, I, dt, R0=None):
    """One ZOH step of the state x=[soc,v1,v2] with constant current I over dt."""
    soc, v1, v2 = x
    a1, a2 = rc_coeffs(p, dt)
    eta = p.eta_charge if I < 0 else 1.0
    soc_n = soc - eta * I * dt / (3600.0 * p.Q_Ah)
    v1_n = a1 * v1 + p.R1 * (1 - a1) * I
    v2_n = a2 * v2 + p.R2 * (1 - a2) * I if p.R2 > 0 else 0.0
    return np.array([soc_n, v1_n, v2_n])


def terminal_voltage(p: ECMParams, x, I, R0=None):
    R0 = p.R0 if R0 is None else R0
    return p.ocv(x[0]) - x[1] - x[2] - R0 * I


def simulate(p: ECMParams, t, I, soc0, v0=(0.0, 0.0), stop_at_limits=False,
             thermal=None, T0=None, T_amb=None):
    """Simulate ECM on time grid t (s) with current samples I (held constant over [t_k, t_k+1)).

    thermal: optional src.thermal_model.ThermalParams; if given, temperature is integrated and
    R0/R1 are scaled by its (optional, assumed) Arrhenius factor.
    """
    t = np.asarray(t, float)
    I = np.asarray(I, float)
    n = len(t)
    soc, v1, v2, V, ocv = (np.zeros(n) for _ in range(5))
    T = np.zeros(n) if thermal is not None else None
    x = np.array([soc0, v0[0], v0[1]], float)
    Tk = T0 if T0 is not None else (T_amb if T_amb is not None else 25.0)
    res = SimResult(t, I, V, soc, v1, v2, ocv, T)
    for k in range(n):
        if thermal is not None:
            f = thermal.resistance_factor(Tk)
            pk = ECMParams(**{**{kk: getattr(p, kk) for kk in p.__dataclass_fields__},
                              "R0": p.R0 * f, "R1": p.R1 * f})
            T[k] = Tk
        else:
            pk = p
        soc[k], v1[k], v2[k] = x
        ocv[k] = p.ocv(x[0])
        V[k] = terminal_voltage(pk, x, I[k])
        if stop_at_limits and k > 0:
            reason = ""
            if V[k] < p.V_min and I[k] > 0:
                reason = "V_min"
            elif V[k] > p.V_max and I[k] < 0:
                reason = "V_max"
            elif not (0.0 <= x[0] <= 1.0):
                reason = "SOC bound"
            if reason:
                res.stopped_at, res.stop_reason = k, reason
                return res.truncate()
        if k < n - 1:
            dt = t[k + 1] - t[k]
            if thermal is not None:
                q = thermal.heat_generation(I[k], x[1], x[2], pk)
                Tk = thermal.step(Tk, q, T_amb, dt)
            x = step(pk, x, I[k], dt)
    return res


def energy_balance(p: ECMParams, res: SimResult):
    """Discrete energy audit (J).  Continuous-time identity:
        int OCV*I dt = int V*I dt + int (R0 I^2 + v1^2/R1 + v2^2/R2) dt + dE_cap,
    E_cap = 0.5 C1 v1^2 + 0.5 C2 v2^2.  Rectangle (left) rule matches the ZOH current."""
    dt = np.diff(res.t)
    I, V, ocv, v1, v2 = (a[:-1] for a in (res.I, res.V, res.ocv, res.v1, res.v2))
    # use midpoint of RC voltage inside each interval for dissipation/charge terms
    v1m = 0.5 * (res.v1[:-1] + res.v1[1:])
    v2m = 0.5 * (res.v2[:-1] + res.v2[1:])
    E_chem = np.sum(ocv * I * dt)                    # left rule; OCV varies slowly
    E_term = np.sum((ocv - v1m - v2m - p.R0 * I) * I * dt)
    E_ohm = np.sum(p.R0 * I**2 * dt)
    E_rc = np.sum(v1m**2 / p.R1 * dt) + (np.sum(v2m**2 / p.R2 * dt) if p.R2 > 0 else 0.0)
    dEc = 0.5 * p.C1 * (res.v1[-1]**2 - res.v1[0]**2)
    if p.R2 > 0:
        dEc += 0.5 * p.C2 * (res.v2[-1]**2 - res.v2[0]**2)
    resid = E_chem - (E_term + E_ohm + E_rc + dEc)
    return {"E_chem_J": E_chem, "E_terminal_J": E_term, "E_ohmic_J": E_ohm,
            "E_rc_J": E_rc, "dE_cap_J": dEc, "residual_J": resid,
            "residual_rel": resid / max(abs(E_chem), 1e-12)}
