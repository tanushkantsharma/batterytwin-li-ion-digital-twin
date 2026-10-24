"""Current profiles. Drive-cycle current traces are the US06 and UDDS files distributed with
PyBaMM (pybamm-data v1.0.2, fetched via pybamm.DataLoader); they are current-vs-time traces
(max 8.1 A), used here as SYNTHETIC excitation for the reference cell. I > 0 = discharge."""
import numpy as np


def drive_cycle(name="US06", repeats=1):
    import pybamm
    path = pybamm.DataLoader().get_data(f"{name}.csv")
    d = np.loadtxt(path, delimiter=",", comments="#")
    t1, I1 = d[:, 0], d[:, 1]
    period = t1[-1] + (t1[1] - t1[0])
    t = np.concatenate([t1 + k * period for k in range(repeats)])
    I = np.tile(I1, repeats)
    return t, I


def constant_current(I, duration_s, dt=1.0):
    t = np.arange(0, duration_s + dt, dt)
    return t, np.full_like(t, float(I))


def pulse_train(I=5.0, on=60, off=240, n=10, dt=1.0):
    t = np.arange(0, n * (on + off), dt)
    return t, np.where((t % (on + off)) < on, I, 0.0)
