
from ism.src.initIsm import initIsm
from math import pi
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifft2, fft2, ifftshift
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
        # Build PSF from centered MTF
        H_un = ifftshift(Hsys)  # move DC to [0,0] for IFFT
        psf0 = np.real(ifft2(H_un))  # impulse response
        psf = fftshift(psf0)  # center the PSF peak

        # Normalize PSF to unity DC gain (numerical safety)
        s = psf.sum()
        if s != 0:
            psf = psf / s

        # Linear convolution with symmetric boundary to avoid wrap-around artefacts
        toa_ft = convolve2d(toa.astype(np.float64), psf, mode='same', boundary='symm')

        # Clamp tiny negatives from numerical noise
        toa_ft = np.where(toa_ft < 0.0, 0.0, toa_ft)
        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integrate the hyperspectral cube with the ISRF to get a single band.
        :param sgm_toa: 3D TOA cube [mW/m2/sr] (ALT, ACT, λ)
        :param sgm_wv: wavelengths of the TOA cube (λ vector)
        :param band: band name, e.g., "VNIR-0"
        :return: 2D TOA image [mW/m2/sr]
        """
        # Read ISRF (response and its wavelength grid)
        isrf_dir = self.auxdir.rstrip('/\\') + '/isrf/'
        isrf_name = 'isrf_' + band
        isrf_resp, isrf_wv = readIsrf(isrf_dir, isrf_name)

        sgm_toa = np.asarray(sgm_toa, dtype=np.float64)  # (ALT, ACT, Ls)
        sgm_wv = np.asarray(sgm_wv, dtype=np.float64).squeeze()  # (Ls,)
        isrf_r = np.asarray(isrf_resp, dtype=np.float64).squeeze()

        # ISRF wavelength vector
        if isrf_wv is None:
            isrf_w = sgm_wv.copy()
        else:
            isrf_w = np.asarray(isrf_wv, dtype=np.float64).squeeze()

        # Harmonize nm/um (simple heuristic)
        max_s = float(np.nanmax(sgm_wv))
        max_i = float(np.nanmax(isrf_w))
        if 200.0 <= max_s <= 5000.0 and 0.1 <= max_i <= 10.0:
            isrf_w = isrf_w * 1000.0
        elif 0.1 <= max_s <= 10.0 and 200.0 <= max_i <= 5000.0:
            isrf_w = isrf_w / 1000.0

        # Clip negative ISRF
        isrf_r = np.clip(isrf_r, 0.0, None)

        # Interpolate the spectral cube onto the ISRF wavelength grid, per pixel
        # Build 1D interpolator over sgm_wv for each (ALT, ACT)
        # Result: cube on ISRF grid, shape (ALT, ACT, Li)
        Li = isrf_w.shape[0]
        ALT, ACT, _ = sgm_toa.shape
        toa_on_isrf = np.empty((ALT, ACT, Li), dtype=np.float64)

        # Pre-build interpolator kind
        from scipy.interpolate import interp1d
        # Vectorized over last dim by looping spatial pixels (fast enough for 100x150x600)
        for i in range(ALT):
            row = sgm_toa[i, :, :]  # (ACT, Ls)
            for j in range(ACT):
                f = interp1d(sgm_wv, row[j, :], kind='linear',
                             bounds_error=False, fill_value=0.0, assume_sorted=False)
                toa_on_isrf[i, j, :] = f(isrf_w)

        # Weighted integration on the ISRF grid, normalized by the ISRF area
        area = np.trapz(isrf_r, isrf_w)  # scalar
        if area <= 0.0:
            raise ValueError(f"ISRF '{band}' has zero area on its native grid.")

        num = np.trapz(toa_on_isrf * isrf_r[None, None, :], isrf_w, axis=2)  # (ALT, ACT)
        toa = num / area  # (ALT, ACT)

        return toa