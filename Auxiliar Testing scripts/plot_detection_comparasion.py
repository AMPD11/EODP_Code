import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- paths (ajusta si cambian) ---
REF_DIR  = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\data\EODP-TS-ISM\output"
TEST_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\ISM_0001"
OUT_DIR  = TEST_DIR  # guarda las figuras junto a tus outputs

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
FNAME = "ism_toa_detection_{band}.nc"  # <-- detection

def load_toa(path):
    ds = xr.open_dataset(path)
    # toma la primera variable de datos
    varname = list(ds.data_vars)[0]
    arr = ds[varname].values
    ds.close()
    return np.asarray(arr, dtype=np.float64)

def plot_band(band):
    ref_path  = os.path.join(REF_DIR,  FNAME.format(band=band))
    test_path = os.path.join(TEST_DIR, FNAME.format(band=band))

    ref = load_toa(ref_path)
    test = load_toa(test_path)

    # dimensión: (ALT, ACT)
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch {ref.shape} vs {test.shape} for {band}")

    alt_mid = ref.shape[0] // 2
    x = np.arange(ref.shape[1])

    y_ref  = ref[alt_mid, :]
    y_test = test[alt_mid, :]

    plt.figure(figsize=(7, 4.2), dpi=130)
    plt.plot(x, y_ref,  label="Ref TOA (detection)", linewidth=1.5)
    plt.plot(x, y_test, label="Test TOA (detection)", linewidth=1.2)
    plt.title(f"TOA detection comparison — {band}")
    plt.xlabel("ACT [-]")
    plt.ylabel("Radiance [mW/m²/sr]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png = os.path.join(OUT_DIR, f"detection_compare_{band}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for b in BANDS:
        plot_band(b)
