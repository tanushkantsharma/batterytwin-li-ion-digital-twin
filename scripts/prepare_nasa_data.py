"""Download (or locate), verify and preprocess the NASA PCoE Li-ion battery aging data (B0005-7, B0018).

Usage:
    python scripts/prepare_nasa_data.py                 # tries official zip, then GitHub mirror
    python scripts/prepare_nasa_data.py --raw-dir DIR   # use already downloaded B00xx.mat files

Outputs (data/processed/):
    nasa_cycles.csv                     one row per discharge: capacity label + preceding-charge id
    nasa_<cell>_discharge.csv.gz        time series of every discharge (I>0 = discharge)
    nasa_<cell>_charge.csv.gz           time series of every charge   (I<0 = charge)
    nasa_impedance.csv                  EIS-derived Re and Rct with chronological position
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nasa"
OUT = ROOT / "data" / "processed"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
OFFICIAL_ZIP = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
MIRROR = "https://raw.githubusercontent.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset/master/{}.mat"
# SHA-256 of the files actually used to produce the results in this repository
SHA256 = json.load(open(ROOT / "data" / "nasa_sha256.json")) if (ROOT / "data" / "nasa_sha256.json").exists() else {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in CELLS if not (raw_dir / f"{c}.mat").exists()]
    if not missing:
        return
    try:
        print("trying official archive", OFFICIAL_ZIP)
        data = urllib.request.urlopen(OFFICIAL_ZIP, timeout=60).read()
        outer = zipfile.ZipFile(io.BytesIO(data))
        for name in outer.namelist():          # the archive nests per-batch zips
            if name.endswith(".zip"):
                inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
                for n in inner.namelist():
                    if Path(n).stem in missing:
                        (raw_dir / Path(n).name).write_bytes(inner.read(n))
    except Exception as e:  # noqa: BLE001
        print("official download failed:", e)
    for c in [c for c in CELLS if not (raw_dir / f"{c}.mat").exists()]:
        print("fetching mirror", MIRROR.format(c))
        urllib.request.urlretrieve(MIRROR.format(c), raw_dir / f"{c}.mat")


def verify(raw_dir: Path):
    got = {c: sha(raw_dir / f"{c}.mat") for c in CELLS}
    for c, h in got.items():
        ref = SHA256.get(c)
        status = "OK" if ref == h else ("UNVERIFIED (no reference hash)" if ref is None else "MISMATCH")
        print(f"{c}: sha256 {h[:16]}... {status}")
        if status == "MISMATCH":
            print("  WARNING: file differs from the one used for the published results")
    return got


def process(raw_dir: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    cyc_rows, imp_rows = [], []
    for c in CELLS:
        m = sio.loadmat(raw_dir / f"{c}.mat", simplify_cells=True)[c]["cycle"]
        dis, chg = [], []
        n_dis = n_chg = 0
        last_charge = None
        for pos, cyc in enumerate(m):
            d = cyc["data"]
            if cyc["type"] == "charge":
                chg.append(pd.DataFrame({"charge_idx": n_chg, "t_s": d["Time"], "V": d["Voltage_measured"],
                                         "I": -np.asarray(d["Current_measured"], float),
                                         "T_C": d["Temperature_measured"]}))
                last_charge = n_chg
                n_chg += 1
            elif cyc["type"] == "discharge":
                dis.append(pd.DataFrame({"discharge_idx": n_dis, "t_s": d["Time"], "V": d["Voltage_measured"],
                                         "I": -np.asarray(d["Current_measured"], float),
                                         "T_C": d["Temperature_measured"]}))
                cyc_rows.append(dict(cell=c, discharge_idx=n_dis, position=pos, capacity_Ah=float(d["Capacity"]),
                                     ambient_C=float(cyc["ambient_temperature"]), prev_charge_idx=last_charge))
                n_dis += 1
            elif cyc["type"] == "impedance":
                re, rct = np.real(d["Re"]), np.real(d["Rct"])
                imp_rows.append(dict(cell=c, position=pos, discharges_before=n_dis, Re_ohm=float(re), Rct_ohm=float(rct)))
        pd.concat(dis).to_csv(OUT / f"nasa_{c}_discharge.csv.gz", index=False, float_format="%.6g")
        pd.concat(chg).to_csv(OUT / f"nasa_{c}_charge.csv.gz", index=False, float_format="%.6g")
        print(f"{c}: {n_chg} charges, {n_dis} discharges")
    pd.DataFrame(cyc_rows).to_csv(OUT / "nasa_cycles.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(OUT / "nasa_impedance.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW)
    ap.add_argument("--write-hashes", action="store_true", help="record hashes of current raw files")
    a = ap.parse_args()
    fetch(a.raw_dir)
    h = verify(a.raw_dir)
    if a.write_hashes:
        json.dump(h, open(ROOT / "data" / "nasa_sha256.json", "w"), indent=2)
    process(a.raw_dir)


if __name__ == "__main__":
    sys.exit(main())
