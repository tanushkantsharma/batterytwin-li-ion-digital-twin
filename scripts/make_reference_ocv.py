"""Extract a model-derived OCV(SOC) table from PyBaMM (Chen2020, SPM, C/50 discharge).
Output: data/params/reference_ocv_chen2020.csv  (columns soc, ocv_V)."""
from pathlib import Path
import numpy as np
import pybamm

ROOT = Path(__file__).resolve().parents[1]


def main():
    pv = pybamm.ParameterValues("Chen2020")
    model = pybamm.lithium_ion.SPM()
    exp = pybamm.Experiment(["Discharge at C/50 until 2.5 V"], period="5 minutes")
    sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp).solve(initial_soc=1.0)
    q = sol["Discharge capacity [A.h]"].entries
    ocv = sol["Bulk open-circuit voltage [V]"].entries
    soc = 1 - q / q[-1]
    grid = np.linspace(0, 1, 201)
    tab = np.interp(grid, soc[::-1], ocv[::-1])
    out = ROOT / "data" / "params" / "reference_ocv_chen2020.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.c_[grid, tab], delimiter=",", header="soc,ocv_V", comments="", fmt="%.6f")
    print(f"PyBaMM {pybamm.__version__}: C/50 capacity {q[-1]:.4f} Ah, OCV {tab[0]:.3f}-{tab[-1]:.3f} V -> {out}")


if __name__ == "__main__":
    main()
