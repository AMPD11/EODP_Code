
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\eodpcode\\auxiliary'
indir = r"C:\\Users\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\data\\EODP-TS-L1B-20250925T181849Z-1-001\\EODP-TS-L1B\\input"
outdir = r"C:\\Users\\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\data\\EODP-TS-L1B-20250925T181849Z-1-001\\EODP-TS-L1B\\output"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
