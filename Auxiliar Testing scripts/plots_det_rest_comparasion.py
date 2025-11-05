# plot_l1b_comparison.py
# Comparación de perfiles (ALT medio) entre referencia y test para L1B:
#  - Equalized DN:   l1b_toa_eq_{band}.nc
#  - Radiances L1B:  l1b_toa_{band}.nc
#
# Salva una figura por banda y por producto, al estilo plot_isrf_comparasion.

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- rutas (ajústalas si cambian) ---
REF_DIR  = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\data\EODP-TS-L1B\output"
TEST_DIR = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\L1B_0001"
OUT_DIR  = TEST_DIR  # guardamos figuras junto a tus outputs

BANDS = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

F_EQ  = "l1b_toa_eq_{band}.nc"   # DN
F_TOA = "l1b_toa_{band}.nc"      # mW/m²/sr

def _load_nc(path):
    ds = xr.open_dataset(path)
    varname = list(ds.data_vars)[0]        # primera variable del dataset
    arr = ds[varname].values
    ds.close()
    return np.asarray(arr, dtype=np.float64)

def _plot_profile(ref, test, band, title, ylabel, out_png):
    # Matrices (ALT, ACT). Tomamos corte por el ALT central.
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch {ref.shape} vs {test.shape} for {band}")

    alt_mid = ref.shape[0] // 2
    x = np.arange(ref.shape[1])

    y_ref  = ref[alt_mid, :]
    y_test = test[alt_mid, :]

    plt.figure(figsize=(7, 4.2), dpi=130)
    plt.plot(x, y_ref,  label="Ref",  linewidth=1.8)
    plt.plot(x, y_test, label="Test", linewidth=1.6)
    plt.title(title)
    plt.xlabel("ACT [-]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")

def _compare_product(fname_tmpl, band, title_prefix, ylabel, suffix):
    ref_path  = os.path.join(REF_DIR,  fname_tmpl.format(band=band))
    test_path = os.path.join(TEST_DIR, fname_tmpl.format(band=band))

    ref  = _load_nc(ref_path)
    test = _load_nc(test_path)

    out_png = os.path.join(OUT_DIR, f"compare_{suffix}_{band}.png")
    _plot_profile(ref, test, band,
                  title=f"{title_prefix} — {band}",
                  ylabel=ylabel,
                  out_png=out_png)

def plot_band(band):
    # 1) Equalized (DN)
    _compare_product(F_EQ, band,
                     title_prefix="L1B Equalized (DN) profile",
                     ylabel="DN",
                     suffix="l1b_toa_eq")

    # 2) L1B TOA (radiances)
    _compare_product(F_TOA, band,
                     title_prefix="L1B TOA (radiances) profile",
                     ylabel="Radiance [mW/m²/sr]",
                     suffix="l1b_toa")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for b in BANDS:
        plot_band(b)
