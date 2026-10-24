"""SOH estimation from CHARGE-phase health indicators (NASA PCoE cells).

Label  : SOH = discharge capacity of the discharge that FOLLOWS the charge / 2.0 Ah (rated).
Features (charge record only, so the label's own discharge is never used):
  t_cc_s       duration of the constant-current phase (until V >= 4.19 V)
  t_cv_s       duration of the constant-voltage phase (CC end until |I| < 0.1 A)
  t_39_41_s    time for terminal voltage to rise from 3.9 V to 4.1 V during CC
Records are dropped (and listed) when the charge does not start from a discharged state
(V >= 3.9 V at CC start) or the CC phase is shorter than 300 s.  Temperature-rise features were
tested and excluded: they depend on the per-cell rest time after discharge (B0018 starts charging
warm), i.e. on protocol, not on health.
Total charged Ah is deliberately excluded: it is (capacity / coulombic efficiency), i.e. the label.
Splits: leave-one-cell-out (train on 3 cells, test on the 4th) to avoid trajectory leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RATED_AH = 2.0
FEATURES = ["t_cc_s", "t_cv_s", "t_39_41_s"]


def charge_features(g: pd.DataFrame):
    t, V, I, T = g.t_s.values, g.V.values, -g.I.values, g.T_C.values   # I here: charge current > 0
    if len(t) < 20 or I.max() < 1.0:
        return None
    on = np.nonzero(I > 1.0)[0]
    k0 = on[0]
    if V[k0] >= 3.9:
        return "charge started above 3.9 V"
    cc_end = np.nonzero((V >= 4.19) & (np.arange(len(V)) > k0))[0]
    if not len(cc_end):
        return None
    k1 = cc_end[0]
    if t[k1] - t[k0] < 300:
        return "CC phase < 300 s"
    after = np.nonzero((np.abs(I) < 0.1) & (np.arange(len(I)) > k1))[0]
    if not len(after):
        return None
    k2 = after[0]
    seg = slice(k0, k1 + 1)
    tv, vv = t[seg], V[seg]
    try:
        t39 = tv[np.nonzero(vv >= 3.9)[0][0]]
        t41 = tv[np.nonzero(vv >= 4.1)[0][0]]
    except IndexError:
        return None
    return dict(t_cc_s=t[k1] - t[k0], t_cv_s=t[k2] - t[k1], t_39_41_s=t41 - t39)


def build_dataset(processed_dir, cells=("B0005", "B0006", "B0007", "B0018")):
    cyc = pd.read_csv(f"{processed_dir}/nasa_cycles.csv")
    rows, dropped = [], []
    for c in cells:
        ch = pd.read_csv(f"{processed_dir}/nasa_{c}_charge.csv.gz")
        groups = dict(tuple(ch.groupby("charge_idx")))
        for _, r in cyc[cyc.cell == c].iterrows():
            if pd.isna(r.prev_charge_idx):
                dropped.append((c, r.discharge_idx, "no preceding charge")); continue
            f = charge_features(groups[int(r.prev_charge_idx)])
            if f is None or isinstance(f, str):
                dropped.append((c, r.discharge_idx, f or "incomplete charge record")); continue
            rows.append(dict(cell=c, discharge_idx=int(r.discharge_idx), soh=r.capacity_Ah / RATED_AH,
                             capacity_Ah=r.capacity_Ah, **f))
    return pd.DataFrame(rows), pd.DataFrame(dropped, columns=["cell", "discharge_idx", "reason"])


def models(seed=0):
    return {
        "baseline_linear_t_cc": ("t_cc_s", make_pipeline(StandardScaler(), LinearRegression())),
        "ridge_all": (FEATURES, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "random_forest": (FEATURES, RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed)),
        "gradient_boosting": (FEATURES, GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                                  learning_rate=0.05, random_state=seed)),
    }


def leave_one_cell_out(df: pd.DataFrame, seed=0):
    preds = []
    for test_cell in sorted(df.cell.unique()):
        tr, te = df[df.cell != test_cell], df[df.cell == test_cell]
        for name, (feats, mdl) in models(seed).items():
            X = lambda d: d[[feats]].values if isinstance(feats, str) else d[feats].values  # noqa: E731
            mdl.fit(X(tr), tr.soh.values)
            preds.append(pd.DataFrame({"model": name, "test_cell": test_cell,
                                       "discharge_idx": te.discharge_idx.values,
                                       "soh_true": te.soh.values, "soh_pred": mdl.predict(X(te))}))
    return pd.concat(preds, ignore_index=True)
