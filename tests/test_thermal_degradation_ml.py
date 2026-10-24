import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.thermal_model import ThermalParams, simulate_temperature
from src.degradation import fit_linear, fit_exponential, first_crossing, predict_eol
from src.soh_ml import charge_features

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_thermal_exact_first_order():
    tp = ThermalParams(m_cp=50.0, hA=0.05)
    t = np.linspace(0, 5000, 501)
    T = simulate_temperature(tp, t, np.full_like(t, 0.5), 25.0, 25.0)
    assert np.allclose(T, 25 + 10 * (1 - np.exp(-t / tp.tau)), atol=1e-10)


def test_eol_helpers():
    n = np.arange(1, 101)
    q = 2.0 - 0.005 * n
    assert first_crossing(n, q, 1.6) == 80
    f, _ = fit_linear(n[:40], q[:40])
    assert predict_eol(f, 1.6, 40) == 80
    f, _ = fit_exponential(n[:60], 1.5 + 0.5 * np.exp(-0.0 * n[:60]) - 0.01 * np.exp(0.03 * n[:60]))
    assert np.isfinite(f(np.array([70.0]))).all()


def test_charge_features_synthetic_record():
    t = np.arange(0, 6000, 10.0)
    V = np.where(t < 2000, 3.5 + 0.7 * t / 2000, 4.2)
    I = np.where(t < 2000, -1.5, -1.5 * np.exp(-(t - 2000) / 600))
    f = charge_features(pd.DataFrame({"t_s": t, "V": V, "I": I, "T_C": 25.0}))
    assert f["t_cc_s"] == pytest.approx(1990, abs=10)
    assert f["t_39_41_s"] == pytest.approx(2000 * 0.2 / 0.7, abs=20)
    assert f["t_cv_s"] > 0
    g = pd.DataFrame({"t_s": t, "V": V + 0.5, "I": I, "T_C": 25.0})
    assert isinstance(charge_features(g), str)     # starts above 3.9 V -> rejected


@pytest.mark.skipif(not (PROC / "nasa_cycles.csv").exists(), reason="NASA data not prepared")
def test_nasa_processed_sanity():
    cyc = pd.read_csv(PROC / "nasa_cycles.csv")
    assert set(cyc.cell) == {"B0005", "B0006", "B0007", "B0018"}
    assert cyc.groupby("cell").size().to_dict() == {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
    assert cyc.capacity_Ah.between(1.0, 2.2).all()
    d = pd.read_csv(PROC / "nasa_B0005_discharge.csv.gz")
    on = d[d.I > 0.5]
    assert on.I.median() == pytest.approx(2.0, abs=0.05)       # 2 A constant-current discharge
