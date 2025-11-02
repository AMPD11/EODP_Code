
from ism.src.initIsm import initIsm
from math import pi
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifft2, fft2
import numpy as np
from common.io.writeToa import writeToa
from common.io.readIsrf import readIsrf
from scipy.interpolate import interp1d, interp2d
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
from scipy.signal import convolve2d
from common.src.auxFunc import getIndexBand

class opticalPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, sgm_toa, sgm_wv, band):
        """
        The optical phase is in charge of simulating the radiance
        to irradiance conversion, the spatial filter (PSF)
        and the spectral filter (ISRF).
        :return: TOA image in irradiances [mW/m2/nm],
                    with spatial and spectral filter
        """
        self.logger.info("EODP-ALG-ISM-1000: Optical stage")

        # Calculation and application of the ISRF
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1010: Spectral modelling. ISRF")
        toa = self.spectralIntegration(sgm_toa, sgm_wv, band)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_isrf:
            saveas_str = self.globalConfig.ism_toa_isrf + band
            writeToa(self.outdir, saveas_str, toa)

        # Radiance to Irradiance conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1020: Radiances to Irradiances")
        toa = self.rad2Irrad(toa,
                             self.ismConfig.D,
                             self.ismConfig.f,
                             self.ismConfig.Tr)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        # Spatial filter
        # -------------------------------------------------------------------------------
        # Calculation and application of the system MTF
        self.logger.info("EODP-ALG-ISM-1030: Spatial modelling. PSF/MTF")
        myMtf = mtf(self.logger, self.outdir)
        Hsys = myMtf.system_mtf(toa.shape[0], toa.shape[1],
                                self.ismConfig.D, self.ismConfig.wv[getIndexBand(band)], self.ismConfig.f, self.ismConfig.pix_size,
                                self.ismConfig.kLF, self.ismConfig.wLF, self.ismConfig.kHF, self.ismConfig.wHF,
                                self.ismConfig.defocus, self.ismConfig.ksmear, self.ismConfig.kmotion,
                                self.outdir, band)

        # Apply system MTF
        toa = self.applySysMtf(toa, Hsys) # always calculated
        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        # Write output TOA & plots
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_optical_stage:
            saveas_str = self.globalConfig.ism_toa_optical + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the optical phase [mW/sr/m2]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa

    def rad2Irrad(self, toa, D, f, Tr):
        """
        Radiance to Irradiance conversion
        :param toa: Input TOA image in radiances [mW/sr/m2]
        :param D: Pupil diameter [m]
        :param f: Focal length [m]
        :param Tr: Optical transmittance [-]
        :return: TOA image in irradiances [mW/m2]
        """
        omega = (pi / 4.0) * (D / f) ** 2
        scale = Tr * omega
        toa = toa.astype(np.float64, copy=False) * scale
        return toa

    def applySysMtf(self, toa, Hsys):
        """
        Application of the system MTF to the TOA
        :param toa: Input TOA image in irradiances [mW/m2]
        :param Hsys: System MTF
        :return: TOA image in irradiances [mW/m2]
        """
        F = fft2(toa.astype(np.float64))  # uncentered spectrum (zero freq at [0,0])
        Fc = fftshift(F)  # center it to match Hsys
        if Hsys.shape != Fc.shape:
            raise ValueError(f"MTF shape {Hsys.shape} != FFT shape {Fc.shape}")
        Gc = Fc * Hsys  # filtering in centered frequency domain

        # For even-sized arrays, fftshift == ifftshift
        if toa.shape[0] % 2 or toa.shape[1] % 2:
            # If ever you run odd sizes, import ifftshift and use it here.
            raise ValueError("Odd-sized arrays require ifftshift; import it to proceed.")
        G  = np.fft.ifftshift(Gc)

        toa_ft = np.real(ifft2(G))
        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integrate the hyperspectral cube with the ISRF to get a single band.
        :param sgm_toa: 3D TOA cube [mW/m2/sr] (ALT, ACT, λ)
        :param sgm_wv: wavelengths of the TOA cube (λ vector)
        :param band: band name, e.g., "VNIR-0"
        :return: 2D TOA image [mW/m2/sr]
        """
        # Load ISRF
        isrf_dir = self.auxdir.rstrip('/\\') + '/isrf/'
        isrf_name = 'isrf_' + band
        isrf_resp, isrf_wv = readIsrf(isrf_dir, isrf_name)

        # Arrays
        sgm_toa = np.asarray(sgm_toa, dtype=np.float64)
        sgm_wv = np.asarray(sgm_wv, dtype=np.float64).squeeze()
        isrf_resp = np.asarray(isrf_resp, dtype=np.float64).squeeze()
        if isrf_wv is None:
            isrf_wv = sgm_wv.copy()
        else:
            isrf_wv = np.asarray(isrf_wv, dtype=np.float64).squeeze()

        # Make ISRF wavelengths use the same unit as sgm_wv (nm/um heuristics)
        # target: unit of sgm_wv
        max_s = float(np.nanmax(sgm_wv))
        max_i = float(np.nanmax(isrf_wv))

        # If sgm is in nm (~200..5000) and ISRF in um (~0.2..5) -> convert um->nm
        if 200.0 <= max_s <= 5000.0 and 0.1 <= max_i <= 10.0:
            isrf_wv = isrf_wv * 1000.0
        # If sgm is in um (~0.2..5) and ISRF in nm (~200..5000) -> convert nm->um
        elif 0.1 <= max_s <= 10.0 and 200.0 <= max_i <= 5000.0:
            isrf_wv = isrf_wv / 1000.0
        # else: assume same unit already

        # Clip negatives and build safe interpolation
        isrf_resp = np.clip(isrf_resp, 0.0, None)
        if not np.any(isrf_resp > 0):
            raise ValueError(f"ISRF '{band}' is empty or negative.")

        # Interpolate ISRF onto sgm_wv, zero outside overlap
        f = interp1d(isrf_wv, isrf_resp, kind="linear",
                     bounds_error=False, fill_value=0.0, assume_sorted=False)
        w = f(sgm_wv)

        # Overlap mask and renormalization on the sgm grid
        ov = w > 0.0
        if not np.any(ov):
            raise ValueError(f"No spectral overlap for {band} after unit alignment.")

        # Normalize area on the target grid (PDF-like weights)
        area = np.trapz(w[ov], sgm_wv[ov])
        if area <= 0:
            raise ValueError(f"ISRF '{band}' has zero area on the target grid.")
        w /= area

        # Weighted spectral integration: (ALT, ACT, λ) · (λ) -> (ALT, ACT)
        toa = np.tensordot(sgm_toa, w, axes=([2], [0]))

        return toa