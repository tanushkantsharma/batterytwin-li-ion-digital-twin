"""Synthetic reference cell used where ground truth is needed.

OCV(SOC): PyBaMM 'Chen2020' parameter set (LG M50 21700, NMC811 / graphite-SiOx; Chen et al.,
J. Electrochem. Soc. 167 (2020) 080534), extracted as the bulk open-circuit voltage of a C/50
SPM discharge (scripts/make_reference_ocv.py).  This is a MODEL-DERIVED OCV, not a measurement.

R0, RC and thermal values below are ASSUMED (order-of-magnitude typical for a 5 Ah 21700 cell)
and serve only as known ground truth for synthetic estimation experiments.
"""
from pathlib import Path
from .battery_ecm import ECMParams, OCVTable
from .thermal_model import ThermalParams

ROOT = Path(__file__).resolve().parents[1]
OCV_CSV = ROOT / "data" / "params" / "reference_ocv_chen2020.csv"

TRUE_RC = dict(R0=0.020, R1=0.010, C1=3000.0, R2=0.015, C2=40000.0)   # ASSUMED ground truth


def reference_params(**overrides) -> ECMParams:
    # Q_Ah = C/50 capacity of the same PyBaMM run (5.1495 Ah), so SOC 0..1 spans the OCV table
    p = dict(Q_Ah=5.15, V_min=2.5, V_max=4.2, **TRUE_RC)
    p.update(overrides)
    return ECMParams(**p, ocv=OCVTable.from_csv(OCV_CSV))


def reference_thermal(Ea=0.0) -> ThermalParams:
    # m_cp: ~69 g cell x ~1000 J/(kg K) (assumed); hA: h~10 W/(m2 K) x 0.0053 m2 (natural convection, assumed)
    return ThermalParams(m_cp=69.0, hA=0.053, Ea=Ea)
