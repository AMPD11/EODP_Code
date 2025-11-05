import os
import numpy as np
import xarray as xr

# --- paths (ajusta si cambian) ---
REF_DIR  = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\data\EODP-TS-ISM\output"
TEST_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\ISM_0001"

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
FNAME = "ism_toa_detection_{band}.nc"  # <--- cambiado a 'detection'

THRESH_PCT = 0.01      # 0.01% tolerance
PASS_FRAC  = 0.9973    # 99.73% of pixels (≈ 3σ)

def load2d(path):
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]
    arr = ds[var].values
    ds.close()
    # expected shape: (ALT, ACT) after detection stage
    return np.asarray(arr, dtype=np.float64)

def band_check(band):
    ref_path  = os.path.join(REF_DIR,  FNAME.format(band=band))
    test_path = os.path.join(TEST_DIR, FNAME.format(band=band))

    ref  = load2d(ref_path)
    test = load2d(test_path)

    if ref.shape != test.shape:
        raise ValueError(f"[{band}] shape mismatch: {ref.shape} vs {test.shape}")

    # relative error in percent; protect division by 0 con eps respecto a escala típica
    eps = 1e-12 * max(1.0, np.nanmax(np.abs(ref)))
    denom = np.maximum(np.abs(ref), eps)
    rel_err_pct = np.abs(test - ref) / denom * 100.0

    flat = rel_err_pct.ravel()
    ok = np.count_nonzero(flat < THRESH_PCT)
    frac_ok = ok / flat.size

    med = float(np.nanmedian(flat))
    p99 = float(np.nanpercentile(flat, 99))

    print(f"Band {band}: {frac_ok*100:.6f}% pixels < {THRESH_PCT}% "
          f"(median={med:.5f}%, p99={p99:.5f}%)")

    return frac_ok

if __name__ == "__main__":
    print("=== DETECTION CONSISTENCY VALIDATION (ism_toa_detection) ===\n")
    fracs = []
    for b in BANDS:
        fracs.append(band_check(b))

    mean_ok = np.mean(fracs)
    print(f"\nOverall mean compliance = {mean_ok*100:.6f}%")
    if np.all(np.array(fracs) >= PASS_FRAC):
        print("PASS (≥ 3σ = 99.73%)")
    else:
        print("FAIL (requirement: ≥ 99.73%)")
