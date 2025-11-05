import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- paths (ajusta si cambian) ---
ISRF_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\ISM_0001"
L1B_DIR  = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\L1B_0001"
OUT_DIR  = L1B_DIR  # guarda las figuras junto al producto L1B

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
F_ISRF = "ism_toa_isrf_{band}.nc"
F_L1B  = "l1b_toa_{band}.nc"

def load_toa(path):
    ds = xr.open_dataset(path)
    # toma la primera variable de datos
    varname = list(ds.data_vars)[0]
    arr = ds[varname].values
    ds.close()
    return np.asarray(arr, dtype=np.float64)

def plot_band(band):
    isrf_path = os.path.join(ISRF_DIR, F_ISRF.format(band=band))
    l1b_path  = os.path.join(L1B_DIR,  F_L1B.format(band=band))

    toa_isrf = load_toa(isrf_path)
    toa_l1b  = load_toa(l1b_path)

    # dimensión: (ALT, ACT)
    if toa_isrf.shape != toa_l1b.shape:
        raise ValueError(f"Shape mismatch {toa_isrf.shape} vs {toa_l1b.shape} for {band}")

    alt_mid = toa_isrf.shape[0] // 2
    x = np.arange(toa_isrf.shape[1])

    y_isrf = toa_isrf[alt_mid, :]
    y_l1b  = toa_l1b[alt_mid, :]

    plt.figure(figsize=(7.4, 4.4), dpi=130)
    plt.plot(x, y_isrf, label="TOA after ISRF (reference)", linewidth=1.6)
    plt.plot(x, y_l1b,  label="Restored L1B TOA",         linewidth=1.2)
    plt.title(f"Central ALT slice — {band}")
    plt.xlabel("ACT [-]")
    plt.ylabel("Radiance [mW/m²/sr]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_png = os.path.join(OUT_DIR, f"l1b_vs_isrf_compare_{band}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for b in BANDS:
        plot_band(b)
