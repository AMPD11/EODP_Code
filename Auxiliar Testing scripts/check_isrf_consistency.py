import os
import numpy as np
import xarray as xr

# Paths
ref_dir = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\data\EODP-TS-ISM\output"
my_dir = r"C:\Users\Andrei\Documents\MiSE\3.1 Semester\Earth Observation Data Processing\EODP_Outputs\ISM_0001"

# Bands
bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

threshold = 0.01  # 0.01%

results = []

print("\n=== ISRF CONSISTENCY VALIDATION ===\n")
for b in bands:
    ref_path = os.path.join(ref_dir, f"ism_toa_isrf_{b}.nc")
    our_path = os.path.join(my_dir, f"ism_toa_isrf_{b}.nc")

    # Load data
    ref = xr.open_dataset(ref_path)["toa"].values.astype(np.float64)
    our = xr.open_dataset(our_path)["toa"].values.astype(np.float64)

    # Relative error (in %)
    eps = np.abs(our - ref) / np.maximum(ref, 1e-12) * 100.0  # avoid /0

    mask = eps < threshold
    pass_ratio = np.mean(mask) * 100.0

    results.append(pass_ratio)

    print(f"Band {b}: {pass_ratio:.6f}% pixels < {threshold}%")

overall = np.mean(results)
print(f"\nOverall mean compliance = {overall:.6f}%")

if overall > 99.73:
    print("PASS (≥ 3σ = 99.73%)")
else:
    print("FAIL (requirement: ≥ 99.73%)")