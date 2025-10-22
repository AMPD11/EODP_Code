
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = "C:/Users/Andrei/Documents/MiSE/3.1 Semester/Earth Observation Data Processing/eodpcode/auxiliary"
indir  = "C:/Users/Andrei/Documents/MiSE/3.1 Semester/Earth Observation Data Processing/data/EODP-TS-L1B/input"
outdir = "C:/Users/Andrei/Documents/MiSE/3.1 Semester/Earth Observation Data Processing/EODP_Outputs/L1B_0001"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
