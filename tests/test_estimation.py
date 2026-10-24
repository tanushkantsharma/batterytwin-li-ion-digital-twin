import numpy as np
from src.battery_ecm import simulate
from src.reference_cell import reference_params, TRUE_RC
from src.parameter_estimation import fit, Trajectory, RC_NAMES
from src.state_estimation import coulomb_counting, ekf_soc
from src.profiles import pulse_train, drive_cycle

BOUNDS = dict(R0=(1e-3, 0.2), R1=(1e-4, 0.2), C1=(10, 2e4), R2=(1e-4, 0.2), C2=(1e3, 5e5))


def test_noise_free_parameter_recovery():
    p = reference_params()
    t, I = pulse_train(I=5.0, on=120, off=600, n=6)
    V = simulate(p, t, I, 0.8).V
    guess = reference_params(R0=0.03, R1=0.02, C1=1000, R2=0.03, C2=1e4)
    est, info = fit([Trajectory(t, I, V, 0.8)], guess, RC_NAMES, BOUNDS)
    for n in RC_NAMES:
        assert abs(getattr(est, n) / TRUE_RC[n] - 1) < 0.01, n
    assert info["rmse_train"] < 1e-5


def test_coulomb_counting_exact():
    p = reference_params()
    t, I = drive_cycle("US06", 2)
    r = simulate(p, t, I, 0.7)
    assert np.allclose(coulomb_counting(t, I, 0.7, p.Q_Ah), r.soc, atol=1e-12)


def test_ekf_corrects_initial_soc_error():
    rng = np.random.default_rng(1)
    p = reference_params()
    t, I = drive_cycle("US06", 6)
    r = simulate(p, t, I, 0.8)
    Vm = r.V + rng.normal(0, 0.005, len(t))
    out = ekf_soc(p, t, I, Vm, soc0=0.5, sigma_v=0.005, sigma_i=0.01)
    err_end = np.abs(out["soc"][-600:] - r.soc[-600:])
    assert err_end.max() < 0.02
    cc = coulomb_counting(t, I, 0.5, p.Q_Ah)
    assert np.abs(cc[-1] - r.soc[-1]) > 0.29     # coulomb counting keeps the initial error
