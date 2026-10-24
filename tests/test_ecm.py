import numpy as np
import pytest
from src.battery_ecm import simulate, energy_balance, ECMParams, OCVTable, OCVCombined
from src.reference_cell import reference_params, reference_thermal
from src.thermal_model import ThermalParams


@pytest.fixture
def p():
    return reference_params()


def test_rest_keeps_state(p):
    t = np.arange(0, 600, 1.0)
    r = simulate(p, t, np.zeros_like(t), 0.5)
    assert np.allclose(r.soc, 0.5) and np.allclose(r.V, p.ocv(0.5))


def test_coulomb_and_rc_analytic(p):
    I = 5.0
    t = np.arange(0, 1201, 1.0)
    r = simulate(p, t, np.full_like(t, I), 0.9)
    assert np.allclose(r.soc, 0.9 - I * t / (3600 * p.Q_Ah), atol=1e-12)
    v1 = p.R1 * I * (1 - np.exp(-t / (p.R1 * p.C1)))
    v2 = p.R2 * I * (1 - np.exp(-t / (p.R2 * p.C2)))
    assert np.allclose(r.v1, v1, atol=1e-12) and np.allclose(r.v2, v2, atol=1e-12)
    assert np.allclose(r.V, p.ocv(r.soc) - v1 - v2 - p.R0 * I, atol=1e-12)


def test_step_size_invariance(p):
    """Exact ZOH discretisation: piecewise-constant current gives identical states at shared times."""
    tf = np.arange(0, 3601, 1.0)
    I = np.where((tf // 300) % 2 == 0, 4.0, -2.0)
    tc = tf[::10]
    rf = simulate(p, tf, I, 0.7)
    rc = simulate(p, tc, I[::10], 0.7)
    assert np.max(np.abs(rf.V[::10] - rc.V)) < 1e-9


def test_energy_balance(p):
    t = np.arange(0, 3000, 0.5)
    I = 5 * np.sin(2 * np.pi * t / 400) + 2
    r = simulate(p, t, I, 0.8)
    eb = energy_balance(p, r)
    assert abs(eb["residual_rel"]) < 1e-4
    assert eb["E_ohmic_J"] > 0 and eb["E_rc_J"] > 0


def test_voltage_limit_stops(p):
    t = np.arange(0, 7200, 1.0)
    r = simulate(p, t, np.full_like(t, 5.15), 1.0, stop_at_limits=True)
    assert r.stop_reason == "V_min" and r.V[-1] < p.V_min and r.V[-2] >= p.V_min
    r = simulate(p, t, np.full_like(t, -5.15), 0.2, stop_at_limits=True)
    assert r.stop_reason == "V_max"


def test_thermal_steady_state_and_rest(p):
    tp = reference_thermal()
    t = np.arange(0, 20 * tp.tau, 5.0)
    I = np.full_like(t, 2.0)
    pp = ECMParams(Q_Ah=1e6, R0=p.R0, R1=p.R1, C1=p.C1, ocv=p.ocv)  # huge Q: SOC ~ const
    r = simulate(pp, t, I, 0.5, thermal=tp, T_amb=25.0)
    q = p.R0 * 4 + p.R1 * 4  # v1 -> R1 I in steady state, v1^2/R1 = R1 I^2
    assert abs(r.T[-1] - (25 + q / tp.hA)) < 1e-3
    r0 = simulate(p, t[:100], np.zeros(100), 0.5, thermal=tp, T_amb=25.0)
    assert np.allclose(r0.T, 25.0)


def test_arrhenius_direction():
    tp = ThermalParams(m_cp=1, hA=1, Ea=30000)
    assert tp.resistance_factor(0) > 1 > tp.resistance_factor(45)
    assert tp.resistance_factor(25) == pytest.approx(1.0)


def test_json_roundtrip(tmp_path, p):
    p.to_json(tmp_path / "p.json")
    q = ECMParams.from_json(tmp_path / "p.json")
    assert q.R1 == p.R1 and np.isclose(q.ocv(0.37), p.ocv(0.37))
    c = ECMParams(Q_Ah=2, R0=.1, R1=.1, C1=1, ocv=OCVCombined([3.8, 0.01, 0.1, 0.1, -0.05]))
    c.to_json(tmp_path / "c.json")
    assert np.isclose(ECMParams.from_json(tmp_path / "c.json").ocv(0.5), c.ocv(0.5))


def test_ocv_derivative(p):
    z = np.linspace(0.05, 0.95, 19)
    c = OCVCombined([3.8, 0.01, 0.1, 0.1, -0.05])
    assert np.allclose(c.deriv(z), (c(z + 1e-6) - c(z - 1e-6)) / 2e-6, rtol=1e-4)
    assert np.all(p.ocv.deriv(z) > 0)
