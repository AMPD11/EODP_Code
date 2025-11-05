import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- paths (ajusta si cambian) ---
L1B_EQ_DIR   = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\L1B_0001"
L1B_NOEQ_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\L1B_0001_NoEqualiz"
OUT_DIR      = L1B_EQ_DIR  # guarda las figuras junto al L1B con equalization

BANDS  = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]
F_EQ   = "l1b_toa_eq_{band}.nc"   # EQ ON
F_NOEQ = "l1b_toa_{band}.nc"      # EQ OFF

def load_toa(path):
    ds = xr.open_dataset(path)
    varname = list(ds.data_vars)[0]
    arr = ds[varname].values
    ds.close()
    return np.asarray(arr, dtype=np.float64)

def plot_band(band):
    path_eq   = os.path.join(L1B_EQ_DIR,   F_EQ.format(band=band))
    path_noeq = os.path.join(L1B_NOEQ_DIR, F_NOEQ.format(band=band))

    toa_eq   = load_toa(path_eq)
    toa_noeq = load_toa(path_noeq)

    if toa_eq.shape != toa_noeq.shape:
        raise ValueError(f"[{band}] shape mismatch {toa_eq.shape} vs {toa_noeq.shape}")

    alt_mid = toa_eq.shape[0] // 2
    x = np.arange(toa_eq.shape[1])

    y_eq   = toa_eq[alt_mid, :]
    y_noeq = toa_noeq[alt_mid, :]

    plt.figure(figsize=(7.6, 4.4), dpi=130)
    plt.plot(x, y_eq,   label="L1B restored (Equalization = True)", linewidth=1.6)
    plt.plot(x, y_noeq, label="L1B restored (Equalization = False)", linewidth=1.2)
    plt.title(f"Central ALT slice — {band}")
    plt.xlabel("ACT [-]")
    plt.ylabel("Radiance [mW/m²/sr]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = os.path.join(OUT_DIR, f"l1b_eqON_vs_eqOFF_{band}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for b in BANDS:
        plot_band(b)
