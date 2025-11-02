
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
        G = fftshift(Gc)  # back to uncentered layout

        toa_ft = np.real(ifft2(G))
        return toa_ft

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integration with the ISRF to retrieve one band
        :param sgm_toa: Spectrally oversampled TOA cube 3D in irradiances [mW/m2]
        :param sgm_wv: wavelengths of the input TOA cube
        :param band: band
        :return: TOA image 2D in radiances [mW/m2]
        """
        isrf_dir = self.auxdir.rstrip('/\\') + '/isrf/'
        isrf_name = 'isrf_' + band

        # Leer ISRF
        isrf_resp, isrf_wv = readIsrf(isrf_dir, isrf_name)
        isrf_resp = np.asarray(isrf_resp, dtype=np.float64).squeeze()
        if isrf_wv is None:
            isrf_wv = np.asarray(sgm_wv, dtype=np.float64).squeeze()
        else:
            isrf_wv = np.asarray(isrf_wv, dtype=np.float64).squeeze()

        # --- 1) Normalizar unidades a METROS (detección heurística) ---
        def to_meters(wv):
            wv = np.asarray(wv, dtype=np.float64)
            mx = float(np.nanmax(wv))
            # Heurística:
            #   [200, 5000]     -> nanómetros
            #   [0.2, 5]        -> micrómetros
            #   [1e-7, 1e-5]    -> metros ya
            if 200.0 <= mx <= 5000.0:  # nm
                return wv * 1e-9
            if 0.2 <= mx <= 5.0:  # micras
                return wv * 1e-6
            return wv  # asumimos m

        sgm_wv_m = to_meters(sgm_wv)
        isrf_wv_m = to_meters(isrf_wv)

        # --- 2) Sanitizar ISRF (no negativos) y normalizar área ---
        isrf_resp = np.clip(isrf_resp, 0.0, None)
        if not np.any(isrf_resp > 0):
            raise ValueError(f"ISRF '{band}' vacío o negativo")
        # Normalización por integral discreta (espaciado real en su rejilla)
        area = np.trapz(isrf_resp, isrf_wv_m)
        if area <= 0:
            raise ValueError(f"ISRF '{band}' con área nula")
        isrf_resp /= area

        # --- 3) Verificar solape espectral real ---
        lo = max(np.min(isrf_wv_m), np.min(sgm_wv_m))
        hi = min(np.max(isrf_wv_m), np.max(sgm_wv_m))
        if not (lo < hi):
            raise ValueError(
                f"Sin solape espectral para {band}. "
                f"ISRF=[{np.min(isrf_wv_m):.3e},{np.max(isrf_wv_m):.3e}] m, "
                f"SGM=[{np.min(sgm_wv_m):.3e},{np.max(sgm_wv_m):.3e}] m"
            )

        # --- 4) Interpolar el ISRF a la rejilla del SGM y RENORMALIZAR solo en el solape ---
        f = interp1d(isrf_wv_m, isrf_resp, kind="linear",
                     bounds_error=False, fill_value=0.0, assume_sorted=False)
        weights = f(sgm_wv_m)  # en m
        # Cero fuera del solape, renormaliza solo con los >0
        mask = weights > 0
        ws = float(np.trapz(weights[mask], sgm_wv_m[mask])) if np.any(mask) else 0.0
        if ws <= 0:
            raise ValueError(
                f"Interpolación del ISRF a SGM dio suma nula en {band}. "
                f"Revisa unidades y rango espectral."
            )
        weights /= ws

        # --- 5) Integración espectral discreta (pesa por 'weights') ---
        toa = np.tensordot(sgm_toa.astype(np.float64), weights, axes=([2], [0]))
        return toa