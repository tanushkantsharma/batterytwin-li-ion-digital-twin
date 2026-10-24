"""Loaders turning preprocessed NASA PCoE records into estimation trajectories."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from .parameter_estimation import Trajectory

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


@lru_cache(None)
def cycles():
    return pd.read_csv(PROC / "nasa_cycles.csv")


@lru_cache(None)
def _ts(cell, kind):
    return dict(tuple(pd.read_csv(PROC / f"nasa_{cell}_{kind}.csv.gz").groupby(f"{kind}_idx")))


def impedance():
    return pd.read_csv(PROC / "nasa_impedance.csv")


def discharge_frame(cell, i):
    return _ts(cell, "discharge")[i]


def pair(cell, i):
    """(charge trajectory, discharge trajectory) for discharge i and its preceding charge.

    SOC convention (stated assumption): the discharge starts at SOC=1 after CC-CV charge and its
    measured capacity Q (to the protocol cut-off at 2 A) defines SOC=0.  The preceding charge starts
    at SOC=0 and ends at SOC=1, so its coulombic efficiency is eta = Q / Ah_charged."""
    r = cycles().query("cell == @cell and discharge_idx == @i").iloc[0]
    g = discharge_frame(cell, i)
    h = _ts(cell, "charge")[int(r.prev_charge_idx)]
    ah = -np.sum(h.I.values[:-1] * np.diff(h.t_s.values)) / 3600
    tc = Trajectory(h.t_s.values, h.I.values, h.V.values, 0.0, r.capacity_Ah, f"{cell}-c{i}",
                    eta_charge=r.capacity_Ah / ah)
    td = Trajectory(g.t_s.values, g.I.values, g.V.values, 1.0, r.capacity_Ah, f"{cell}-d{i}")
    return tc, td
