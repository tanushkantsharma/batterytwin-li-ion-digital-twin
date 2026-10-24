"""Lumped (single-node) thermal model coupled to the ECM.

    m cp dT/dt = Q_gen - hA (T - T_amb)
    Q_gen = R0 I^2 + v1^2/R1 + v2^2/R2        (irreversible Joule + polarisation heat)

Reversible (entropic) heat is NOT modelled (no defensible dU/dT data for the cells used).
Temperature dependence of resistance is OPTIONAL and uses an ASSUMED Arrhenius law
    R(T) = R(T_ref) * exp(Ea/Rg (1/T - 1/T_ref)),
which is disabled (Ea = 0) unless explicitly set.  See docs/assumptions_and_limitations.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

RG = 8.314


@dataclass
class ThermalParams:
    m_cp: float          # heat capacity [J/K]
    hA: float            # heat-transfer conductance to ambient [W/K]
    Ea: float = 0.0      # activation energy for R(T) [J/mol]; 0 disables
    T_ref_C: float = 25.0

    @property
    def tau(self):
        return self.m_cp / self.hA

    def resistance_factor(self, T_C):
        if self.Ea == 0.0:
            return 1.0
        return float(np.exp(self.Ea / RG * (1 / (T_C + 273.15) - 1 / (self.T_ref_C + 273.15))))

    def heat_generation(self, I, v1, v2, p):
        q = p.R0 * I**2 + v1**2 / p.R1
        if p.R2 > 0:
            q += v2**2 / p.R2
        return q

    def step(self, T, q, T_amb, dt):
        """Exact solution over dt for constant q and T_amb."""
        T_ss = T_amb + q / self.hA
        return T_ss + (T - T_ss) * np.exp(-dt / self.tau)


def simulate_temperature(tp: ThermalParams, t, q, T0, T_amb):
    """Integrate temperature for a given heat-generation trace q(t) (W)."""
    T = np.empty(len(t))
    T[0] = T0
    for k in range(len(t) - 1):
        T[k + 1] = tp.step(T[k], q[k], T_amb, t[k + 1] - t[k])
    return T
