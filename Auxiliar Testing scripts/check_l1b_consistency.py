# check_l1b_consistency.py
# Valida L1B comparando nuestra salida con la referencia:
#   - Producto "l1b_toa_{band}.nc"      (radiances)
#   - o "l1b_toa_eq_{band}.nc"           (equalized)
#
# Criterio UTS: diferencias < 0.01% para >= 2-sigma de los puntos (≈ 95.45%)

import os
import numpy as np
import xarray as xr

# --- paths (ajusta si cambian) ---
REF_DIR  = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\data\EODP-TS-L1B\output"
TEST_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\L1B_0001"

# --- qué producto quieres comparar: "l1b_toa" o "l1b_toa_eq" ---
PRODUCT = "l1b_toa_eq"     # cambia a "l1b_toa" o "l1b_toa_eq"

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

# patrón de fichero según producto
FNAME = f"{PRODUCT}" + "_{band}.nc"

THRESH_PCT = 0.01       # 0.01%
PASS_FRAC  = 0.9545     # 95.45%  (≈ 2σ)

def load_array(path):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]         # toma la primera variable
    arr = ds[var].values
    ds.close()
    return np.asarray(arr, dtype=np.float64)

def band_check(band):
    ref_path  = os.path.join(REF_DIR,  FNAME.format(band=band))
    test_path = os.path.join(TEST_DIR, FNAME.format(band=band))

    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"[{band}] Missing reference file: {ref_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"[{band}] Missing test file: {test_path}")

    ref  = load_array(ref_path)
    test = load_array(test_path)

    if ref.shape != test.shape:
        raise ValueError(f"[{band}] shape mismatch: {ref.shape} vs {test.shape}")

    # error relativo (%) robusto a ceros
    eps_scale = 1e-12 * max(1.0, np.nanmax(np.abs(ref)))
    denom = np.maximum(np.abs(ref), eps_scale)
    rel_err_pct = np.abs(test - ref) / denom * 100.0

    flat = rel_err_pct.ravel()
    frac_ok = np.count_nonzero(flat < THRESH_PCT) / flat.size

    med = float(np.nanmedian(flat))
    p99 = float(np.nanpercentile(flat, 99))

    print(f"{PRODUCT} | {band}: {frac_ok*100:.6f}% pixels < {THRESH_PCT}% "
          f"(median={med:.5f}%, p99={p99:.5f}%)")

    return frac_ok

if __name__ == "__main__":
    print(f"=== L1B CONSISTENCY VALIDATION ({PRODUCT}) ===\n")
    fracs = []
    for b in BANDS:
        fracs.append(band_check(b))

    mean_ok = float(np.mean(fracs))
    print(f"\nOverall mean compliance = {mean_ok*100:.6f}%")
    if np.all(np.array(fracs) >= PASS_FRAC):
        print("PASS (≥ 2σ = 95.45%)")
    else:
        print("FAIL (requirement: ≥ 95.45%)")
