
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\eodpcode\\auxiliary'
indir = r"C:\\Users\\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\data\\EODP-TS-ISM\\input\\gradient_alt100_act150" # small scene
outdir = r"C:\\Users\\Andrei\\Documents\\MiSE\\3.1 Semester\\Earth Observation Data Processing\\data\\EODP-TS-ISM\\output"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
